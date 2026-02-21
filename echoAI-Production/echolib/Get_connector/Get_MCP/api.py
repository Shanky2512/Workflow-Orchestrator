"""
FastAPI routes for MCP connector management.
Provides HTTP API for creating, testing, and managing connectors.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from .base import TransportType
from .http_script import HTTPMCPConnector
from .sse import SSEMCPConnector
from .stdio import STDIOMCPConnector
from .validator import validate_and_normalize, ValidationError
from .storage import get_storage
from .tester import ConnectorTester, create_default_test_payload


logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Connector API",
    description="Production MCP Connector Generator API",
    version="1.0.0"
)


# Request/Response Models
class CreateConnectorRequest(BaseModel):
    """Request model for creating connectors"""
    name: str
    description: str
    transport_type: str
    auth_config: Dict[str, Any]
    endpoint_url: str
    method: str = "POST"
    headers: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    query_mapping: Dict[str, str] = {}
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout: int = 30
    retry_count: int = 0
    metadata: Dict[str, Any] = {}


class TestConnectorRequest(BaseModel):
    """Request model for testing connectors"""
    connector_id: str
    test_payload: Optional[Dict[str, Any]] = None


class ConfigureConnectorRequest(BaseModel):
    """Request model for updating connector configuration"""
    connector_id: str
    updates: Dict[str, Any]


# Helper functions
def _create_connector_from_data(data: Dict[str, Any]):
    """Factory function to create appropriate connector type"""
    transport = TransportType(data["transport_type"])

    if transport == TransportType.HTTP:

        # 🔒 FAIL FAST: prevent broken GET connectors
        if data.get("method", "POST").upper() == "GET" and not data.get("query_mapping"):
            raise HTTPException(
                status_code=400,
                detail="query_mapping is required for HTTP GET connectors"
            )

        return HTTPMCPConnector(
            name=data["name"],
            description=data["description"],
            auth_config=data["auth_config"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            endpoint_url=data["endpoint_url"],
            method=data.get("method", "POST"),
            headers=data.get("headers", {}),
            query_params=data.get("query_params", {}),
            query_mapping=data.get("query_mapping", {}),  # ✅ REQUIRED FIX
            timeout=data.get("timeout", 30),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {})
        )

    elif transport == TransportType.SSE:
        return SSEMCPConnector(
            name=data["name"],
            description=data["description"],
            auth_config=data["auth_config"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            endpoint_url=data["endpoint_url"],
            headers=data.get("headers", {}),
            reconnect=data.get("reconnect", True),
            max_reconnect_attempts=data.get("max_reconnect_attempts", 5),
            reconnect_delay=data.get("reconnect_delay", 1),
            heartbeat_timeout=data.get("heartbeat_timeout", 60),
            event_filter=data.get("event_filter"),
            metadata=data.get("metadata", {})
        )

    elif transport == TransportType.STDIO:
        return STDIOMCPConnector(
            name=data["name"],
            description=data["description"],
            auth_config=data["auth_config"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            command=data["command"],
            args=data.get("args", []),
            env_vars=data.get("env_vars", {}),
            working_dir=data.get("working_dir"),
            timeout=data.get("timeout", 30),
            shell=data.get("shell", False),
            metadata=data.get("metadata", {})
        )

    else:
        raise ValueError(f"Unsupported transport type: {transport}")


def _load_connector_from_storage(connector_id: str):
    """Load and instantiate connector from storage"""
    storage = get_storage()
    data = storage.load(connector_id)
    
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector not found: {connector_id}"
        )
    
    # Extract connector_config (handles both old and new storage formats)
    config = data.get("connector_config", data)
    
    # ✅ FIX: Handle transport_type as dict/enum
    transport_value = config.get("transport_type")
    if isinstance(transport_value, dict):
        transport_value = transport_value.get("value")
    
    transport = TransportType(transport_value)
    
    if transport == TransportType.HTTP:
        return HTTPMCPConnector.from_dict(config)
    elif transport == TransportType.SSE:
        return SSEMCPConnector.from_dict(config)
    elif transport == TransportType.STDIO:
        return STDIOMCPConnector.from_dict(config)


# API Routes
@app.post("/connector/create")
async def create_connector(request: CreateConnectorRequest):
    """
    Create a new MCP connector.
    
    Accepts free-form JSON and performs strict validation.
    """
    try:
        # Validate and normalize input
        normalized_data = validate_and_normalize(request.dict())
        
        # Create connector
        connector = _create_connector_from_data(normalized_data)
        
        # Validate connector config
        is_valid, errors = connector.validate_config()
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Connector validation failed",
                    "errors": errors
                }
            )
        
        # Save to storage
        storage = get_storage()
        storage.save(connector.serialize())
        
        return {
            "success": True,
            "connector_id": connector.connector_id,
            "name": connector.name,
            "transport_type": connector.transport_type.value,
            "status": connector.status.value,
            "message": "Connector created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Validation failed",
                "errors": e.errors
            }
        )
    except Exception as e:
        logger.error(f"Failed to create connector: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/connector/test")
async def test_connector(request: TestConnectorRequest):
    """
    Test a connector with provided payload.
    
    Executes real test against the connector.
    """
    try:
        # Load connector
        connector = _load_connector_from_storage(request.connector_id)
        
        # Get test payload
        test_payload = request.test_payload
        if not test_payload:
            test_payload = create_default_test_payload(
                connector.transport_type.value
            )
        
        # Run test
        test_result = await connector.test(test_payload)
        
        # Update storage with test result
        storage = get_storage()
        storage.update(connector.serialize())
        
        return {
        "success": test_result["success"],
        "connector_id": connector.connector_id,
        "test_result": test_result,
        "connector_status": connector.status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test connector: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/connector/save")
async def save_connector(request: CreateConnectorRequest):
    """
    Save or update a connector.
    
    Alias for create - handles both new and existing connectors.
    """
    return await create_connector(request)


@app.get("/connector/{connector_id}")
async def get_connector(connector_id: str):
    """
    Retrieve connector by ID.
    
    Returns full connector specification.
    """
    try:
        connector = _load_connector_from_storage(connector_id)
        
        return {
            "success": True,
            "connector": connector.serialize(),
            "mcp_spec": connector.get_mcp_spec()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get connector: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/connector")
async def list_connectors(transport_type: Optional[str] = None):
    """
    List all connectors.
    
    Optionally filter by transport type.
    """
    try:
        storage = get_storage()
        connectors = storage.list_all()  # ✅ No argument
        
        # Filter by transport_type if provided
        if transport_type:
            connectors = [c for c in connectors if c.get("transport_type") == transport_type]
        
        return {
            "success": True,
            "count": len(connectors),
            "connectors": connectors
        }
        
    except Exception as e:
        logger.error(f"Failed to list connectors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/connector/{connector_id}")
async def delete_connector(connector_id: str):
    """Delete a connector"""
    try:
        storage = get_storage()
        success = storage.delete(connector_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector not found: {connector_id}"
            )
        
        return {
            "success": True,
            "message": f"Connector {connector_id} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete connector: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/connector/configure")
async def configure_connector(request: ConfigureConnectorRequest):
    try:
        storage = get_storage()

        # 1️⃣ Load existing connector
        connector = _load_connector_from_storage(request.connector_id)

        updates = request.updates

        # 2️⃣ Disallow dangerous updates
        DISALLOWED_FIELDS = {"transport_type", "endpoint_url", "method"}
        for key in updates:
            if key in DISALLOWED_FIELDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Field '{key}' cannot be updated"
                )

        # 3️⃣ Apply updates (shallow merge)
        for key, value in updates.items():
            if hasattr(connector, key):
                setattr(connector, key, value)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown or immutable field: {key}"
                )

        # 4️⃣ Re-validate UPDATED connector
        is_valid, errors = connector.validate_config()
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Validation failed after update",
                    "errors": errors
                }
            )

        # 5️⃣ Save back to storage
        storage.save(connector.serialize())

        return {
            "success": True,
            "connector_id": connector.connector_id,
            "status": connector.status.value,
            "message": "Connector updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure connector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mcp-connector-api"}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "MCP Connector Generator API",
        "version": "1.0.0",
        "endpoints": {
            "create": "POST /connector/create",
            "test": "POST /connector/test",
            "get": "GET /connector/{id}",
            "list": "GET /connector",
            "delete": "DELETE /connector/{id}",
            "configure": "POST /connector/configure"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
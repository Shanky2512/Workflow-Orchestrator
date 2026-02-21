from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *
from echolib.services import ConnectorManager, MCPConnector, APIConnector
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, status
import os
import json
import uuid
from pathlib import Path
 
from fastapi import APIRouter, Depends, HTTPException
from echolib.di import container
from echolib.security import user_context
from echolib.types import *
from echolib.services import ConnectorManager, MCPConnector, APIConnector
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, status
from sqlalchemy import create_engine, text, inspect
from urllib.parse import urlparse

router = APIRouter(prefix='/connectors', tags=['Connector API'])


def mgr() -> ConnectorManager:
    """Dependency injection for ConnectorManager."""
    return container.resolve('connector.manager')


# Persist saved DB connections for Convobi "connect to this DB" buttons
_SAVED_DB_CONNECTIONS_FILE = Path(__file__).resolve().parent / "saved_db_connections.json"

# ============================================================================
# REQUEST/RESPONSE MODELS - MCP
# ============================================================================
class MCPConnectorCreateRequest(BaseModel):
    """MCP connector creation request."""
    name: str
    description: str
    transport_type: str  # "http", "sse", "stdio"
    auth_config: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
    # HTTP fields (optional)
    endpoint_url: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    query_params: Optional[Dict[str, str]] = None
    query_mapping: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None
    retry_count: Optional[int] = None
    verify_ssl: Optional[bool] = True
    
    # SSE fields (optional)
    reconnect: Optional[bool] = None
    max_reconnect_attempts: Optional[int] = None
    reconnect_delay: Optional[int] = None
    heartbeat_timeout: Optional[int] = None
    event_filter: Optional[list] = None
    
    # STDIO fields (optional)
    command: Optional[str] = None
    args: Optional[list] = None
    env_vars: Optional[Dict[str, str]] = None
    working_dir: Optional[str] = None
    shell: Optional[bool] = None
    
    # Example payload for auto-test
    example_payload: Optional[Dict[str, Any]] = None

class MCPConnectorInvokeRequest(BaseModel):
    """MCP connector invocation request."""
    connector_id: str
    payload: Optional[Dict[str, Any]] = None


class MCPConnectorUpdateRequest(BaseModel):
    """MCP connector update request."""
    connector_id: str
    updates: Dict[str, Any]



# ============================================================================
# REQUEST/RESPONSE MODELS - API
# ============================================================================

class APIConnectorCreateRequest(BaseModel):
    """Create connector request - ADD example_payload field"""
    name: str
    base_url: str
    auth_config: Dict[str, Any]
    endpoints: Optional[Dict[str, Any]] = None
    default_headers: Optional[Dict[str, str]] = None
    timeout: Optional[int] = 30
    example_payload: Dict[str, Any] = None

class APIConnectorInvokeRequest(BaseModel):
    """Invoke connector request - Keep as is"""
    connector_id: str
    payload: Optional[Dict[str, Any]] = None

class APIConnectorTestRequest(BaseModel):
    """Test connector request - NEW MODEL"""
    connector_id: str  # Changed from connector_id to connector_name for consistency
    custom_payload: Optional[Dict[str, Any]] = None

class APIConnectorUpdateRequest(BaseModel):
    connector_id: str
    updates: Dict[str, Any]

class APIConnectorDeleteRequest(BaseModel):
    connector_id: str
# ============================================================================
# MCP CONNECTOR ROUTES
# ============================================================================
@router.post('/mcp/create')
async def create_mcp_connector(
    req: MCPConnectorCreateRequest,
    manager: ConnectorManager = Depends(mgr)
):
    """Create a new MCP connector with optional auto-test."""
    result = await manager.mcp.create(req.dict(exclude_none=True))
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    return result

@router.post('/mcp/invoke')
async def invoke_mcp_connector(
    req: MCPConnectorInvokeRequest,
    manager: ConnectorManager = Depends(mgr)
):
    """Invoke an MCP connector with smart payload logic (model-based)."""

    # Use the parsed model directly—do NOT call body()/json() on the model
    result = await manager.mcp.invoke_async(
        connector_id=req.connector_id,
        payload=req.payload
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)

    return result


@router.get('/mcp/get')
async def get_mcp_connector(
    connector_id: str,
    manager: ConnectorManager = Depends(mgr)
):
    """Get MCP connector with full metadata."""
    connector = manager.mcp.get(connector_id)
    
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    
    return connector


@router.get('/mcp')
async def list_mcp_connectors(
    manager: ConnectorManager = Depends(mgr)
):
    """List all MCP connectors."""
    return manager.mcp.list()


@router.put('/mcp/update')
async def update_mcp_connector(
    req: MCPConnectorUpdateRequest,
    manager: ConnectorManager = Depends(mgr)
):
    """Update MCP connector with hybrid validation and mandatory testing."""
    result = await manager.mcp.update(req.connector_id, req.updates)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    return result

@router.delete('/mcp/delete')
async def delete_mcp_connector(
    connector_id: str,
    manager: ConnectorManager = Depends(mgr)
):
    """Delete an MCP connector."""
    result = manager.mcp.delete(connector_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    return result

# ============================================================================
# API CONNECTOR ROUTES
# ============================================================================
# ============ CREATE CONNECTOR ============

@router.post("/api/create",
    responses={
        200: {"description": "Connector created"},
        400: {
            "description": "Connector validation failed",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Connector validation failed",
                        "message": "Cannot create connector: example_payload test failed. Please check your configuration and try again.",
                        "status_code": 401,
                        "response_body": {"cod": 401, "message": "Invalid API key"},
                        "details": {
                            "type": "auth_error",
                            "category": "authentication",
                            "hint": "Check API key"
                        }
                    }
                }
            }
        }
    }
)
async def create_api_connector(
    req: APIConnectorCreateRequest,
    manager = Depends(mgr)
):
    """
    Create API connector.
    - Validates config
    - Creates connector in-memory
    - Tests example_payload
    - Persists only if test passes
    - Returns detailed error JSON on failure
    """
    result = manager.api.create(req.dict(exclude_none=True))

    if not result.get("success"):
        # Return the entire result as JSON (do NOT raise HTTPException)
        # so Swagger shows the full error payload instead of {"detail": "..."}
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result,               # keep your error fields at the root
            media_type="application/json"
        )

    return result

# ============ LIST CONNECTORS ============
@router.get('/api/list')
async def list_api_connectors(
    manager = Depends(mgr)
):
    """List all API connectors."""
    return manager.api.list()


# ============ INVOKE CONNECTOR ============
@router.post('/api/invoke')
async def invoke_api_connector(
    req: APIConnectorInvokeRequest,
    manager = Depends(mgr)
):
    """Execute API connector."""
    result = manager.api.invoke(
        req.connector_id,
        req.payload
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    
    return result

# ============ UPDATE CONNECTOR ============
@router.put('/api/update')
async def update_api_connector(
    req: APIConnectorUpdateRequest,
    manager = Depends(mgr)
):
    """
    Update API connector configuration.
    
    Request body:
    {
      "connector_id": "api_xxx",
      "updates": {
        "timeout": 60,
        "auth_config": {"key": "new_key"}
      }
    }
    
    Connector is re-validated before saving.
    """
    # Merge connector_id into updates for the service method
    payload = {"connector_id": req.connector_id, **req.updates}
    result = manager.api.update(payload)
    return result


# ============ GET CONNECTOR ============
@router.get('/api/get')
async def get_api_connector(
    connector_id: str,
    manager = Depends(mgr)
):
    """
    Get connector details - RETURNS FULL CREATION PAYLOAD.
    
    Returns:
    - connector_id
    - connector_name
    - creation_payload (entire JSON used during creation, with masked sensitive fields)
    - validation status with error details if failed
    - timestamps
    """
    connector = manager.api.get(connector_id)
    
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_id}' not found")
    
    return connector

# ============ DELETE CONNECTOR ============
@router.delete('/api/delete')
async def delete_api_connector(
    req: APIConnectorDeleteRequest,
    manager = Depends(mgr)
):
    """Delete API connector."""
    result = manager.api.delete(req.connector_id)
    return result

# ============================================================================
# DB PLUGIN ROUTES (Convobi: delete saved connection)
# ============================================================================

def _deduplicate_saved_connections(connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list with at most one entry per connection_string (first occurrence kept)."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for c in connections:
        cs = (c.get("connection_string") or "").strip()
        if cs not in seen:
            seen.add(cs)
            out.append(c)
    return out


def _load_saved_db_connections() -> List[Dict[str, Any]]:
    """Load saved DB connections from file. Returns deduplicated list by connection_string; cleans file if duplicates found."""
    try:
        if _SAVED_DB_CONNECTIONS_FILE.exists():
            with open(_SAVED_DB_CONNECTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                connections = data.get("connections", [])
                deduped = _deduplicate_saved_connections(connections)
                if len(deduped) < len(connections):
                    _save_saved_db_connections(deduped)
                return deduped
    except Exception:
        pass
    return []


def _save_saved_db_connections(connections: List[Dict[str, Any]]) -> None:
    """Persist saved DB connections to file. Always saves a deduplicated list by connection_string."""
    try:
        deduped = _deduplicate_saved_connections(connections)
        _SAVED_DB_CONNECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SAVED_DB_CONNECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({"connections": deduped}, f, indent=2)
    except Exception:
        pass


def _mask_connection_string(cs: str) -> str:
    """Mask password in connection string for display."""
    if not cs or "://" not in cs:
        return cs or ""
    try:
        parts = cs.split("://", 1)
        scheme = parts[0]
        rest = parts[1]
        if "@" in rest:
            creds, host_part = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{scheme}://{user}:****@{host_part}"
        return f"{scheme}://****"
    except Exception:
        return "***"


def _dialect_from_connection_string(connection_string: str) -> str:
    """Infer SQL dialect from connection string scheme (mirrors convobi SQLAlchemyRDBPlugin)."""
    cs = (connection_string or "").strip()
    scheme = urlparse(cs).scheme.lower()
    if scheme in {"postgresql", "postgres"}:
        return "postgres"
    if scheme in {"mysql", "mariadb"}:
        return "mysql"
    if scheme.startswith("mssql"):
        return "mssql"
    if scheme.startswith("oracle"):
        return "oracle"
    if scheme == "bigquery":
        return "bigquery"
    if scheme == "snowflake":
        return "snowflake"
    if scheme.startswith("clickhouse"):
        return "clickhouse"
    if scheme == "hive":
        return "hive"
    if scheme == "duckdb":
        return "duckdb"
    return "sqlite"


def _odbc_friendly_message(exc: Exception, connection_string: str) -> str:
    """Return a user-friendly message for ODBC/driver errors."""
    msg = str(exc).lower()
    cs = (connection_string or "").strip()
    scheme = urlparse(cs).scheme.lower() if cs else ""
    if "im002" in msg or "data source name not found" in msg or "no default driver specified" in msg:
        hint = (
            "ODBC Driver Manager could not find a driver or DSN. "
            "On Windows: install the correct ODBC driver (e.g. 'ODBC Driver 17 for SQL Server' for SQL Server). "
            "Alternatively use a driver-specific URL: for SQL Server try mssql+pymssql://user:pass@host/db (requires pymssql), "
            "or mssql+pyodbc://... with driver= in the query string. "
            "For SQLite use sqlite:///path/to/file.db (no ODBC)."
        )
        return f"Connection failed: {exc}. {hint}"
    if "mssql" in scheme or "sql server" in msg:
        if "pyodbc" in msg or "odbc" in msg:
            return (
                f"Connection failed: {exc}. "
                "For SQL Server without ODBC: use mssql+pymssql://user:password@host:port/dbname (pip install pymssql). "
                "With ODBC: install 'ODBC Driver 17 for SQL Server' and use mssql+pyodbc://..."
            )
    return f"Connection failed: {exc}"


def _test_db_connection(connection_string: str) -> Dict[str, Any]:
    """
    Test database connection and return dialect and table count.
    Returns {"ok": True, "dialect": str, "tables_count": int} or raises with message.
    """
    cs = (connection_string or "").strip()
    if not cs:
        raise ValueError("connection_string is required")
    try:
        engine = create_engine(cs, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        insp = inspect(engine)
        tables = insp.get_table_names()
        dialect = _dialect_from_connection_string(cs)
        engine.dispose()
        return {"ok": True, "dialect": dialect, "tables_count": len(tables)}
    except Exception as e:
        raise ValueError(_odbc_friendly_message(e, cs)) from e


# ============================================================================
# DB PLUGIN ROUTES (Convobi: delete saved connection only)
# ============================================================================

@router.delete("/plugin/{connection_id}", summary="Convobi Plugins Delete")
async def plugin_delete(connection_id: str):
    """
    **Convobi Plugins Delete.** Remove a saved DB connection by id from the plugin saved_connections store.
    Returns 404 if the connection_id is not found.
    """
    if not connection_id or not connection_id.strip():
        raise HTTPException(status_code=422, detail="connection_id is required")
    saved = _load_saved_db_connections()
    before = len(saved)
    saved = [c for c in saved if c.get("id") != connection_id]
    if len(saved) == before:
        raise HTTPException(status_code=404, detail=f"Saved connection '{connection_id}' not found")
    _save_saved_db_connections(saved)
    return {"ok": True, "deleted_id": connection_id}


# ============================================================================
# UNIVERSAL DATABASE CONNECTOR
# ============================================================================
# Full-featured DB connector matching the frontend form:
#   Database Type dropdown  ·  Host  ·  Port  ·  Database Name
#   Username  ·  Password  ·  Server IP Address (for firewall rules)
#   [Test Connection]  ·  [Next →]
#
# Supports:
#   • All major SQL databases (PostgreSQL, MySQL, MSSQL, Oracle, SQLite,
#     BigQuery, Snowflake, ClickHouse, Redshift, Apache Hive, DuckDB)
#   • Public databases (direct connection)
#   • Private databases (SSH tunnel via bastion / jump host)
#   • Schema introspection, table preview, query execution
# ============================================================================


from .db_connector import (
    DatabaseConnectorService,
    DBConnectionConfig,
    DB_TYPES,
)


_db_service = DatabaseConnectorService()


class DBConnectionUpdateRequest(BaseModel):
    """Request to update a saved database connection."""
    name: Optional[str] = Field(None, description="New display name")


class DBInvokeRequest(BaseModel):
    """Request to run a natural language question on a saved database connection (requires LLM)."""
    connection_id: str = Field(..., description="ID of the saved DB connection")
    question: str = Field(
        ...,
        description="Natural language question (e.g. 'What are the top 10 users by signup date?'). Requires DB_INVOKE_LLM_URL.",
    )
    limit: int = Field(100, description="Max rows to return for read queries")
    timeout: int = Field(30, description="Query timeout in seconds")


# ---------- Supported databases (drives the Database Type dropdown) ----------

@router.get(
    "/db/supported",
    summary="List Supported Database Types",
)
async def db_supported_databases():
    """
    List all supported database types with default ports and available drivers.

    The frontend uses this to populate the **Database Type** dropdown and
    auto-fill the **Port** field.

    Returns: PostgreSQL, MySQL, MariaDB, MSSQL, Oracle, SQLite,
    BigQuery, Snowflake, ClickHouse, Redshift.
    """
    return {
        "databases": _db_service.list_supported_databases(),
        "ssh_tunnel_supported": True,
    }


# ---------- Test Connection (the "Test Connection" button) ----------

@router.post(
    "/db/test",
    summary="Test Database Connection",
)
async def db_test_connection(config: DBConnectionConfig):
    """
    Test a database connection **without** saving it.

    Matches the frontend form fields exactly:

    **PostgreSQL / MySQL / ClickHouse / Apache Hive (host-based):**
    ```json
    {
      "db_type": "postgresql",
      "host": "db.example.com",
      "port": 5432,
      "database_name": "mydb",
      "username": "admin",
      "password": "secret"
    }
    ```

    **SQL Server (ODBC connection string):**
    ```json
    {
      "db_type": "mssql",
      "odbc_connection_string": "Driver={ODBC Driver 17 for SQL Server};Server=server;Database=database;UID=user;PWD=password;"
    }
    ```

    **Snowflake (account URL):**
    ```json
    {
      "db_type": "snowflake",
      "database_name": "mydb",
      "username": "admin",
      "password": "secret",
      "account_url": "your-account.snowflakecomputing.com"
    }
    ```

    **SQLite (file path):**
    ```json
    {
      "db_type": "sqlite",
      "database_file_path": "/path/to/database.sqlite"
    }
    ```

    **DuckDB (file path or :memory:):**
    ```json
    {
      "db_type": "duckdb",
      "database_file_path": "file:///path/to/database.db"
    }
    ```

    **Private database (SSH tunnel through bastion host):**
    ```json
    {
      "db_type": "mysql",
      "host": "10.0.0.5",
      "port": 3306,
      "database_name": "production",
      "username": "app_user",
      "password": "db_pass",
      "ssh_tunnel": {
        "ssh_host": "bastion.example.com",
        "ssh_port": 22,
        "ssh_username": "ubuntu",
        "ssh_private_key": "-----BEGIN RSA PRIVATE KEY-----\\n..."
      }
    }
    ```

    **Using raw connection string:**
    ```json
    {
      "connection_string": "postgresql://user:pass@host:5432/mydb"
    }
    ```
    """
    try:
        result = _db_service.test_connection(config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------- Connect & Save (the "Next" button after test passes) ----------

@router.post(
    "/db/connect",
    summary="Connect & Save Database",
)
async def db_connect_and_save(config: DBConnectionConfig):
    """
    Test connection, then **save** it for future use.

    Same payload as ``/db/test`` plus an optional ``name`` field.
    Returns ``connection_id`` to use in subsequent API calls.

    **Example:**
    ```json
    {
      "db_type": "postgresql",
      "host": "db.example.com",
      "port": 5432,
      "database_name": "analytics",
      "username": "admin",
      "password": "secret",
      "name": "Analytics DB (Production)"
    }
    ```

    **Private MSSQL via SSH:**
    ```json
    {
      "db_type": "mssql",
      "host": "192.168.1.100",
      "port": 1433,
      "database_name": "SummitAI",
      "username": "sa",
      "password": "MyStr0ngP@ss",
      "name": "SummitAI (Private)",
      "ssh_tunnel": {
        "ssh_host": "jump.corp.com",
        "ssh_username": "deploy",
        "ssh_password": "ssh_secret"
      }
    }
    ```
    """
    try:
        result = _db_service.connect_and_save(config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------- List saved connections ----------

@router.get(
    "/db/list",
    summary="List Saved Database Connections",
)
async def db_list_connections():
    """List all saved database connections (credentials masked)."""
    return {"connections": _db_service.list_connections()}


# ---------- Get connection details ----------

@router.get(
    "/db/{connection_id}",
    summary="Get Database Connection Details",
)
async def db_get_connection(connection_id: str):
    """Get details of a saved database connection by ID."""
    conn = _db_service.get_connection(connection_id)
    if not conn:
        raise HTTPException(
            status_code=404, detail=f"Connection '{connection_id}' not found"
        )
    return conn


# ---------- Update connection ----------

@router.put(
    "/db/{connection_id}",
    summary="Update Database Connection",
)
async def db_update_connection(connection_id: str, req: DBConnectionUpdateRequest):
    """Update a saved connection (rename, etc.)."""
    try:
        return _db_service.update_connection(
            connection_id, req.dict(exclude_none=True)
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Delete connection ----------

@router.delete(
    "/db/{connection_id}",
    summary="Delete Database Connection",
)
async def db_delete_connection(connection_id: str):
    """Delete a saved connection and clean up engine / SSH tunnel."""
    try:
        return _db_service.delete_connection(connection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Invoke (execute query on saved connection) ----------

@router.post(
    "/db/invoke",
    summary="Run Natural Language Question on Database Connection",
)
async def db_invoke(req: DBInvokeRequest):
    """
    Run a **natural language question** on a saved database connection.

    Requires **DB_INVOKE_LLM_URL** (and optionally **DB_INVOKE_LLM_MODEL**). The backend uses
    the connection's schema and the LLM to generate SQL, then executes it and returns
    columns/rows (read) or rows affected (write).

    **Example:**
    ```json
    { "connection_id": "a1b2c3d4", "question": "What are the top 10 users by signup date?", "limit": 100, "timeout": 30 }
    ```

    **Response** includes ``question``, ``generated_sql``, ``explanation``, and the usual ``columns``/``rows`` or ``rows_affected``.
    """
    try:
        return _db_service.invoke_with_question(
            connection_id=req.connection_id,
            question=req.question.strip(),
            limit=req.limit,
            timeout=req.timeout,
        )
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)



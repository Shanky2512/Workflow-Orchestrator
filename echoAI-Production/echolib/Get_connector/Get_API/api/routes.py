"""
FastAPI routes for connector management.
Provides REST API for creating, testing, and managing connectors.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from models.config import ConnectorConfig, ExecuteRequest
from models.responses import (
    ExecuteResponse,
    ConnectorResponse,
    ConnectorListResponse,
    ErrorResponse
)
from connectors import ConnectorFactory, BaseConnector
from storage import StorageBase
from .dependencies import get_storage


router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.post(
    "/create",
    response_model=ConnectorResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse}
    }
)
async def create_connector(
    config: ConnectorConfig,
    storage: StorageBase = Depends(get_storage)
) -> ConnectorResponse:
    """
    Create a new connector.
    
    Args:
        config: Connector configuration
        storage: Storage dependency
        
    Returns:
        ConnectorResponse with created connector details
        
    Raises:
        HTTPException: If connector ID already exists or creation fails
    """
    try:
        # Check if connector already exists
        if storage.exists(config.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Connector with ID '{config.id}' already exists"
            )
        
        # Create connector instance to validate configuration
        connector = ConnectorFactory.create(config)
        
        # Save to storage
        storage.save(config.id, connector.to_dict())
        
        # Load back to get metadata
        saved_data = storage.load(config.id)
        
        return ConnectorResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            base_url=config.base_url,
            auth_type=config.auth.get("type") if isinstance(config.auth, dict) else config.auth.type.value,
            created_at=saved_data.get("created_at"),
            updated_at=saved_data.get("updated_at")
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create connector: {str(e)}"
        )


@router.get(
    "/{connector_id}",
    response_model=ConnectorResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_connector(
    connector_id: str,
    storage: StorageBase = Depends(get_storage)
) -> ConnectorResponse:
    """
    Get connector details by ID.
    
    Args:
        connector_id: Connector identifier
        storage: Storage dependency
        
    Returns:
        ConnectorResponse with connector details
        
    Raises:
        HTTPException: If connector not found
    """
    try:
        data = storage.load(connector_id)
        config_data = data.get("config", {})
        
        return ConnectorResponse(
            id=config_data.get("id", connector_id),
            name=config_data.get("name", ""),
            description=config_data.get("description"),
            base_url=config_data.get("base_url", ""),
            auth_type=config_data.get("auth", {}).get("type", "none"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
    
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{connector_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve connector: {str(e)}"
        )


@router.get(
    "",
    response_model=ConnectorListResponse
)
async def list_connectors(
    storage: StorageBase = Depends(get_storage)
) -> ConnectorListResponse:
    """
    List all connectors.
    
    Args:
        storage: Storage dependency
        
    Returns:
        ConnectorListResponse with list of connectors
    """
    try:
        connector_ids = storage.list_all()
        connectors = []
        
        for connector_id in connector_ids:
            try:
                data = storage.load(connector_id)
                config_data = data.get("config", {})
                
                connectors.append(
                    ConnectorResponse(
                        id=config_data.get("id", connector_id),
                        name=config_data.get("name", ""),
                        description=config_data.get("description"),
                        base_url=config_data.get("base_url", ""),
                        auth_type=config_data.get("auth", {}).get("type", "none"),
                        created_at=data.get("created_at"),
                        updated_at=data.get("updated_at")
                    )
                )
            except Exception:
                continue
        
        return ConnectorListResponse(
            connectors=connectors,
            total=len(connectors)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list connectors: {str(e)}"
        )


@router.put(
    "/{connector_id}",
    response_model=ConnectorResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}}
)
async def update_connector(
    connector_id: str,
    config: ConnectorConfig,
    storage: StorageBase = Depends(get_storage)
) -> ConnectorResponse:
    """
    Update an existing connector.
    
    Args:
        connector_id: Connector identifier
        config: Updated connector configuration
        storage: Storage dependency
        
    Returns:
        ConnectorResponse with updated connector details
        
    Raises:
        HTTPException: If connector not found or update fails
    """
    try:
        # Check if connector exists
        if not storage.exists(connector_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Connector '{connector_id}' not found"
            )
        
        # Ensure IDs match
        if config.id != connector_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Connector ID in URL ({connector_id}) does not match ID in config ({config.id})"
            )
        
        # Create connector instance to validate configuration
        connector = ConnectorFactory.create(config)
        
        # Update storage
        storage.update(connector_id, connector.to_dict())
        
        # Load back to get metadata
        saved_data = storage.load(connector_id)
        
        return ConnectorResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            base_url=config.base_url,
            auth_type=config.auth.get("type") if isinstance(config.auth, dict) else config.auth.type.value,
            created_at=saved_data.get("created_at"),
            updated_at=saved_data.get("updated_at")
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update connector: {str(e)}"
        )


@router.delete(
    "/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}}
)
async def delete_connector(
    connector_id: str,
    storage: StorageBase = Depends(get_storage)
) -> None:
    """
    Delete a connector.
    
    Args:
        connector_id: Connector identifier
        storage: Storage dependency
        
    Raises:
        HTTPException: If connector not found
    """
    try:
        storage.delete(connector_id)
    
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{connector_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete connector: {str(e)}"
        )


@router.post(
    "/{connector_id}/execute",
    response_model=ExecuteResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}}
)
async def execute_connector(
    connector_id: str,
    request: ExecuteRequest,
    storage: StorageBase = Depends(get_storage)
) -> ExecuteResponse:
    """
    Execute a connector request.
    
    Args:
        connector_id: Connector identifier
        request: Execution request parameters
        storage: Storage dependency
        
    Returns:
        ExecuteResponse with request results
        
    Raises:
        HTTPException: If connector not found or execution fails
    """
    try:
        # Load connector
        data = storage.load(connector_id)
        connector = ConnectorFactory.from_dict(data)
        
        # Execute request
        response = connector.execute(
            method=request.method.value if hasattr(request.method, 'value') else str(request.method),
            endpoint=request.endpoint,
            headers=request.headers,
            query_params=request.query_params,
            body=request.body,
            timeout_override=request.timeout_override,
            stream=request.stream
        )
        
        return response
    
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{connector_id}' not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


@router.post(
    "/{connector_id}/test",
    response_model=ExecuteResponse,
    responses={404: {"model": ErrorResponse}}
)
async def test_connector(
    connector_id: str,
    storage: StorageBase = Depends(get_storage)
) -> ExecuteResponse:
    """
    Test connector connectivity.
    
    Args:
        connector_id: Connector identifier
        storage: Storage dependency
        
    Returns:
        ExecuteResponse with test results
        
    Raises:
        HTTPException: If connector not found
    """
    try:
        # Load connector
        data = storage.load(connector_id)
        connector = ConnectorFactory.from_dict(data)
        
        # Test connection
        response = connector.test_connection()
        
        return response
    
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector '{connector_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test failed: {str(e)}"
        )

"""
Response models for API operations.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ExecuteResponse(BaseModel):
    """Response from connector execution."""
    success: bool = Field(..., description="Whether the request succeeded")
    status_code: int = Field(..., description="HTTP status code")
    headers: Dict[str, str] = Field(..., description="Response headers")
    body: Any = Field(..., description="Response body (parsed if JSON)")
    raw_body: Optional[str] = Field(None, description="Raw response body")
    error: Optional[str] = Field(None, description="Error message if failed")
    elapsed_seconds: float = Field(..., description="Request duration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": {"message": "Success"},
                "raw_body": '{"message": "Success"}',
                "error": None,
                "elapsed_seconds": 0.245
            }
        }


class ConnectorResponse(BaseModel):
    """Response for connector operations."""
    id: str = Field(..., description="Connector ID")
    name: str = Field(..., description="Connector name")
    description: Optional[str] = Field(None, description="Connector description")
    base_url: str = Field(..., description="Base URL")
    auth_type: str = Field(..., description="Authentication type")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class ConnectorListResponse(BaseModel):
    """Response for listing connectors."""
    connectors: list[ConnectorResponse] = Field(..., description="List of connectors")
    total: int = Field(..., description="Total number of connectors")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation error",
                "detail": "API key cannot be empty"
            }
        }

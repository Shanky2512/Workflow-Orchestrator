"""
Abstract base class for MCP connectors.
Defines the contract that all transport-specific connectors must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from datetime import datetime


class TransportType(Enum):
    """Supported MCP transport types"""
    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


class AuthType(Enum):
    """Supported authentication mechanisms"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    CUSTOM_HEADER = "custom_header"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class ConnectorStatus(Enum):
    """Connector lifecycle status"""
    DRAFT = "draft"
    VALIDATED = "validated"
    TESTED = "tested"
    ACTIVE = "active"
    FAILED = "failed"


class BaseMCPConnector(ABC):
    """
    Abstract base for all MCP connectors.
    
    This class enforces:
    - Consistent metadata structure
    - Validation contract
    - Test execution interface
    - Serialization for persistence
    
    Child classes MUST implement transport-specific logic.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        transport_type: TransportType,
        auth_config: Dict[str, Any],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        connector_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base connector with required metadata.
        
        Args:
            name: Unique connector name (used by AI agents)
            description: What this connector does (for AI context)
            transport_type: HTTP, SSE, or STDIO
            auth_config: Authentication configuration (type + credentials)
            input_schema: JSON schema for expected inputs
            output_schema: JSON schema for expected outputs
            connector_id: Optional unique ID (generated if not provided)
            metadata: Additional metadata (tags, version, etc.)
        """
        self.connector_id = connector_id or self._generate_id()
        self.name = name
        self.description = description
        self.transport_type = transport_type
        self.auth_config = auth_config
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metadata = metadata or {}
        self.status = ConnectorStatus.DRAFT
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.test_results: List[Dict[str, Any]] = []
        
    @staticmethod
    def _generate_id() -> str:
        """Generate unique connector ID"""
        import uuid
        return f"mcp_{uuid.uuid4().hex[:12]}"
    
    @abstractmethod
    def validate_config(self) -> tuple[bool, List[str]]:
        """
        Validate connector configuration.
        
        Must check:
        - Required fields present
        - Auth config valid for transport type
        - Schema definitions valid
        - Transport-specific requirements
        
        Returns:
            (is_valid, error_messages)
        """
        pass
    
    @abstractmethod
    def build_connector(self) -> Dict[str, Any]:
        """
        Build the actual MCP connector specification.
        
        This is what gets executed by AI agents.
        Must return a complete, runnable connector config.
        
        Returns:
            Complete MCP connector specification
        """
        pass
    
    @abstractmethod
    async def test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a real test against the connector.
        
        This MUST make actual calls (HTTP request, SSE stream, subprocess).
        No mocks, no simulations.
        
        Args:
            payload: Test input matching input_schema
            
        Returns:
            {
                "success": bool,
                "output": Any,
                "error": Optional[str],
                "duration_ms": int,
                "metadata": Dict[str, Any]
            }
        """
        pass
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize connector to JSON-safe dict.
        
        Used for:
        - Saving to storage
        - API responses
        - Versioning
        """
        return {
            "connector_id": self.connector_id,
            "name": self.name,
            "description": self.description,
            "transport_type": self.transport_type.value,
            "auth_config": self.auth_config,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "test_results": self.test_results,
        }
    
    def _sanitize_auth(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive data from auth config for serialization.
        
        Replaces actual secrets with placeholders.
        Real secrets should be stored in secure vault.
        """
        sanitized = auth_config.copy()
        sensitive_keys = ["api_key", "token", "password", "secret", "client_secret"]
        
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = "***REDACTED***"
                
        # Handle nested structures
        for key, value in sanitized.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_auth(value)
                
        return sanitized
    
    def update_status(self, status: ConnectorStatus):
        """Update connector status and timestamp"""
        self.status = status
        self.updated_at = datetime.utcnow().isoformat()
    
    def add_test_result(self, result: Dict[str, Any]):
        """Record test execution result"""
        result["timestamp"] = datetime.utcnow().isoformat()
        self.test_results.append(result)
        
        # Update status based on test result
        if result.get("success"):
            self.update_status(ConnectorStatus.TESTED)
        else:
            self.update_status(ConnectorStatus.FAILED)
    
    def get_mcp_spec(self) -> Dict[str, Any]:
        """
        Generate MCP specification for AI agent consumption.
        
        This is the standardized format that AI agents expect.
        """
        connector_spec = self.build_connector()
        
        return {
            "id": self.connector_id,
            "name": self.name,
            "description": self.description,
            "transport": self.transport_type.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "connector": connector_spec,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMCPConnector':
        """
        Deserialize connector from dict.
        
        Used when loading from storage.
        Must be implemented by child classes to restore transport-specific state.
        """
        raise NotImplementedError("Child classes must implement from_dict")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.connector_id} name={self.name} status={self.status.value}>"
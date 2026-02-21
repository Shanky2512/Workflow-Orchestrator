"""
Abstract base class for all connectors.
Defines the contract that all connector implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from echolib.Get_connector.Get_API.models.config import ConnectorConfig, ExecuteRequest
from echolib.Get_connector.Get_API.models.responses import ExecuteResponse


class BaseConnector(ABC):
    """
    Abstract base class for all API connectors.
    
    All connector implementations must inherit from this class and implement
    the required methods. This ensures consistent behavior across all connectors.
    """
    
    def __init__(self, config: ConnectorConfig) -> None:
        """
        Initialize the base connector.
        
        Args:
            config: Connector configuration
        """
        self.config = config
    
    @abstractmethod
    def execute(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        timeout_override: Optional[float] = None,
        stream: bool = False
    ) -> ExecuteResponse:
        """
        Execute an API request.
        
        This is the primary method for AI agents to interact with the connector.
        All parameters are explicit and stateless.
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint path (will be appended to base_url)
            headers: Additional headers to include (optional)
            query_params: Query parameters (optional)
            body: Request body (optional)
            timeout_override: Override default timeout (optional)
            stream: Whether to stream the response (optional)
            
        Returns:
            ExecuteResponse containing status, headers, body, and metadata
            
        Raises:
            ValueError: If request parameters are invalid
            httpx.HTTPError: If request fails
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> ExecuteResponse:
        """
        Test the connector configuration.
        
        Makes a simple request to verify connectivity and authentication.
        
        Returns:
            ExecuteResponse with test results
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize connector configuration to dictionary.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConnector':
        """
        Deserialize connector from dictionary.
        
        Args:
            data: Dictionary containing connector configuration
            
        Returns:
            BaseConnector instance
            
        Raises:
            ValueError: If data is invalid
        """
        pass
    
    def get_config(self) -> ConnectorConfig:
        """
        Get the connector configuration.
        
        Returns:
            ConnectorConfig object
        """
        return self.config
    
    def update_config(self, config: ConnectorConfig) -> None:
        """
        Update the connector configuration.
        
        Args:
            config: New configuration
        """
        self.config = config

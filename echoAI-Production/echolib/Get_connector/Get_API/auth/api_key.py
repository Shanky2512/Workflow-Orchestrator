"""
API key authentication implementation.
Supports header, query parameter, and cookie placement.
"""

from typing import Dict, Any
import httpx
from echolib.Get_connector.Get_API.auth.base import AuthBase
from echolib.Get_connector.Get_API.models.config import ApiKeyLocation


class ApiKeyAuth(AuthBase):
    """
    API key authentication strategy.
    
    Supports placing the API key in:
    - HTTP headers
    - Query parameters
    - Cookies
    """
    
    def __init__(
        self,
        key: str,
        location: ApiKeyLocation,
        param_name: str
    ) -> None:
        """
        Initialize API key authentication.
        
        Args:
            key: API key value
            location: Where to place the key (header/query/cookie)
            param_name: Name of the parameter/header
            
        Raises:
            ValueError: If key is empty
        """
        if not key or not key.strip():
            raise ValueError("API key cannot be empty")
        if not param_name or not param_name.strip():
            raise ValueError("Parameter name cannot be empty")
            
        self.key = key.strip()
        self.location = location
        self.param_name = param_name.strip()
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply API key authentication.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Modified (headers, params) with API key applied
        """
        headers = headers.copy()
        params = params.copy()
        
        if self.location == ApiKeyLocation.HEADER:
            headers[self.param_name] = self.key
        elif self.location == ApiKeyLocation.QUERY:
            params[self.param_name] = self.key
        elif self.location == ApiKeyLocation.COOKIE:
            client.cookies.set(self.param_name, self.key)
        
        return headers, params
    
    def refresh_if_needed(self) -> None:
        """No refresh needed for API keys."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "api_key",
            "key": self.key,
            "location": self.location.value if hasattr(self.location, 'value') else str(self.location),
            "param_name": self.param_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiKeyAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            ApiKeyAuth instance
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["key", "location", "param_name"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(
            key=data["key"],
            location=ApiKeyLocation(data["location"]),
            param_name=data["param_name"]
        )

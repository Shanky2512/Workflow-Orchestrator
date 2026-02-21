"""
Custom header authentication implementation.
For proprietary or non-standard authentication schemes.
"""

from typing import Dict, Any
import httpx
from echolib.Get_connector.Get_API.auth.base import AuthBase


class CustomHeaderAuth(AuthBase):
    """
    Custom header authentication strategy.
    
    Allows arbitrary headers to be added for authentication.
    Useful for proprietary or non-standard auth schemes.
    """
    
    def __init__(self, headers: Dict[str, str]) -> None:
        """
        Initialize custom header authentication.
        
        Args:
            headers: Dictionary of custom headers for authentication
            
        Raises:
            ValueError: If headers is empty
        """
        if not headers:
            raise ValueError("Custom headers cannot be empty")
        
        self.headers = headers.copy()
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply custom header authentication.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Modified (headers, params) with custom auth headers added
        """
        headers = headers.copy()
        headers.update(self.headers)
        return headers, params.copy()
    
    def refresh_if_needed(self) -> None:
        """No refresh needed for static custom headers."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "custom_header",
            "headers": self.headers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomHeaderAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            CustomHeaderAuth instance
            
        Raises:
            ValueError: If headers field is missing
        """
        if "headers" not in data:
            raise ValueError("Missing required field: headers")
        
        return cls(headers=data["headers"])

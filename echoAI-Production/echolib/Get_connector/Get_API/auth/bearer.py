"""
Bearer token authentication implementation.
"""

from typing import Dict, Any
import httpx
from echolib.Get_connector.Get_API.auth.base import AuthBase


class BearerTokenAuth(AuthBase):
    """
    Bearer token authentication strategy.
    
    Adds Authorization: Bearer <token> header to requests.
    """
    
    def __init__(self, token: str) -> None:
        """
        Initialize Bearer token authentication.
        
        Args:
            token: Bearer token value
            
        Raises:
            ValueError: If token is empty
        """
        if not token or not token.strip():
            raise ValueError("Bearer token cannot be empty")
        
        self.token = token.strip()
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply Bearer token authentication.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Modified (headers, params) with Bearer token in Authorization header
        """
        headers = headers.copy()
        headers["Authorization"] = f"Bearer {self.token}"
        return headers, params.copy()
    
    def refresh_if_needed(self) -> None:
        """No refresh needed for static bearer tokens."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "bearer",
            "token": self.token
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BearerTokenAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            BearerTokenAuth instance
            
        Raises:
            ValueError: If token is missing
        """
        if "token" not in data:
            raise ValueError("Missing required field: token")
        
        return cls(token=data["token"])

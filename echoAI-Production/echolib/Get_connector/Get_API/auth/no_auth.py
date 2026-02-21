"""
No authentication implementation.
"""

from typing import Dict, Any
import httpx
from echolib.Get_connector.Get_API.auth.base import AuthBase


class NoAuth(AuthBase):
    """
    No authentication strategy.
    Passes requests through without modification.
    """
    
    def __init__(self) -> None:
        """Initialize NoAuth strategy."""
        pass
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply no authentication (pass through unchanged).
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Unchanged (headers, params)
        """
        return headers.copy(), params.copy()
    
    def refresh_if_needed(self) -> None:
        """No refresh needed for NoAuth."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {"type": "none"}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            NoAuth instance
        """
        return cls()

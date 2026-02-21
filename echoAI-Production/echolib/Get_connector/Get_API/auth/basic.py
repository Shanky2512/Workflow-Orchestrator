"""Basic authentication strategy."""
import base64
from typing import Dict, Any
from .base import AuthBase
import httpx

class BasicAuthStrategy(AuthBase):
    """
    Basic authentication using username and password.
    
    Adds Authorization header with base64-encoded credentials.
    """
    
    def __init__(self, username: str, password: str):
        """
        Initialize basic auth strategy.
        
        Args:
            username: Username for authentication
            password: Password for authentication
        """
        self.username = username
        self.password = password
    
    def apply(
    self,
    headers: Dict[str, str],
    params: Dict[str, Any],
    client: httpx.Client  # <-- add this parameter
    ) -> tuple[Dict[str, str], Dict[str, Any]]:  # <-- return the tuple per base
        # Encode credentials as base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()

        # Add Authorization header
        headers["Authorization"] = f"Basic {encoded}"

        # Return modified headers and params to honor the base contract
        return headers, params
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BasicAuthStrategy":
        """
        Create BasicAuthStrategy from configuration dict.
        
        Args:
            config: Configuration dictionary with username and password
            
        Returns:
            BasicAuthStrategy instance
            
        Raises:
            ValueError: If required fields are missing
        """
        username = config.get("username")
        password = config.get("password")
        
        if not username:
            raise ValueError("Basic auth requires 'username' field")
        
        if not password:
            raise ValueError("Basic auth requires 'password' field")
        
        return cls(username=username, password=password)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "basic",
            "username": self.username,
            "password": self.password
        }

    
    def refresh_if_needed(self) -> None:
        """
        Refresh authentication if needed.
        
        Basic auth credentials don't expire, so this is a no-op.
        """
        pass  # Basic auth doesn't need refresh

"""
Abstract base class for authentication strategies.
All auth implementations must inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import httpx


class AuthBase(ABC):
    """
    Abstract base class for authentication strategies.
    
    Each auth type implements how to apply authentication to HTTP requests.
    This follows the Strategy pattern for clean, extensible auth handling.
    """
    
    @abstractmethod
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply authentication to the request.
        
        Args:
            headers: Request headers to modify
            params: Query parameters to modify
            client: HTTP client (for accessing cookies, etc.)
            
        Returns:
            Tuple of (modified_headers, modified_params)
            
        Raises:
            ValueError: If authentication cannot be applied
        """
        pass
    
    @abstractmethod
    def refresh_if_needed(self) -> None:
        """
        Refresh authentication credentials if needed (e.g., expired tokens).
        
        Raises:
            ValueError: If refresh fails
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize auth configuration to dictionary.
        
        Returns:
            Dictionary representation of auth config
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthBase':
        """
        Deserialize auth configuration from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            AuthBase instance
            
        Raises:
            ValueError: If data is invalid
        """
        pass

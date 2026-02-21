"""
Mutual TLS (mTLS) authentication implementation.
Uses client certificates for authentication.
"""

from typing import Dict, Any, Optional
import httpx
from pathlib import Path
from echolib.Get_connector.Get_API.auth.base import AuthBase


class MTLSAuth(AuthBase):
    """
    Mutual TLS authentication strategy.
    
    Uses client certificates and private keys for authentication.
    Optionally supports custom CA bundles for certificate verification.
    """
    
    def __init__(
        self,
        cert_path: str,
        key_path: str,
        ca_bundle_path: Optional[str] = None,
        verify_ssl: bool = True
    ) -> None:
        """
        Initialize mTLS authentication.
        
        Args:
            cert_path: Path to client certificate file
            key_path: Path to client private key file
            ca_bundle_path: Path to CA bundle for verification (optional)
            verify_ssl: Whether to verify SSL certificates
            
        Raises:
            ValueError: If certificate or key files don't exist
        """
        if not cert_path or not key_path:
            raise ValueError("Both cert_path and key_path are required")
        
        cert_file = Path(cert_path)
        key_file = Path(key_path)
        
        if not cert_file.exists():
            raise ValueError(f"Certificate file not found: {cert_path}")
        if not key_file.exists():
            raise ValueError(f"Key file not found: {key_path}")
        
        if ca_bundle_path:
            ca_file = Path(ca_bundle_path)
            if not ca_file.exists():
                raise ValueError(f"CA bundle file not found: {ca_bundle_path}")
        
        self.cert_path = str(cert_file.absolute())
        self.key_path = str(key_file.absolute())
        self.ca_bundle_path = str(Path(ca_bundle_path).absolute()) if ca_bundle_path else None
        self.verify_ssl = verify_ssl
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply mTLS authentication.
        
        Note: mTLS is applied at the client level, not per-request.
        This method returns headers/params unchanged but validates configuration.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client (must be configured with mTLS cert)
            
        Returns:
            Unchanged (headers, params)
        """
        return headers.copy(), params.copy()
    
    def get_cert_tuple(self) -> tuple[str, str]:
        """
        Get certificate tuple for httpx client configuration.
        
        Returns:
            Tuple of (cert_path, key_path)
        """
        return (self.cert_path, self.key_path)
    
    def get_verify_option(self) -> bool | str:
        """
        Get SSL verification option for httpx client configuration.
        
        Returns:
            Either False (no verification), True (default verification),
            or path to CA bundle
        """
        if not self.verify_ssl:
            return False
        if self.ca_bundle_path:
            return self.ca_bundle_path
        return True
    
    def refresh_if_needed(self) -> None:
        """
        No refresh needed for mTLS.
        
        Certificate rotation would require creating a new connector instance.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "mtls",
            "cert_path": self.cert_path,
            "key_path": self.key_path,
            "ca_bundle_path": self.ca_bundle_path,
            "verify_ssl": self.verify_ssl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MTLSAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            MTLSAuth instance
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["cert_path", "key_path"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(
            cert_path=data["cert_path"],
            key_path=data["key_path"],
            ca_bundle_path=data.get("ca_bundle_path"),
            verify_ssl=data.get("verify_ssl", True)
        )

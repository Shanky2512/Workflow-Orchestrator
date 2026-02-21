"""
JWT authentication implementation.
Supports both pre-generated tokens and runtime token generation.
"""

from typing import Dict, Any, Optional
import httpx
import jwt
import time
from echolib.Get_connector.Get_API.auth.base import AuthBase


class JWTAuth(AuthBase):
    """
    JWT authentication strategy.
    
    Can either:
    1. Use a pre-generated JWT token
    2. Generate JWT tokens on-the-fly using a secret and payload
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        secret: Optional[str] = None,
        algorithm: str = "HS256",
        payload: Optional[Dict[str, Any]] = None,
        header_name: str = "Authorization",
        header_prefix: str = "Bearer"
    ) -> None:
        """
        Initialize JWT authentication.
        
        Args:
            token: Pre-generated JWT token (optional)
            secret: Secret key for JWT generation (optional)
            algorithm: JWT signing algorithm
            payload: JWT payload for generation (optional)
            header_name: Header name for JWT
            header_prefix: Prefix for the header value
            
        Raises:
            ValueError: If neither token nor (secret + payload) provided
        """
        if not token and not (secret and payload):
            raise ValueError("Either provide a token or (secret + payload) for JWT generation")
        
        self.token = token
        self.secret = secret
        self.algorithm = algorithm
        self.payload = payload or {}
        self.header_name = header_name
        self.header_prefix = header_prefix
        self._generated_token: Optional[str] = None
    
    def _generate_token(self) -> str:
        """
        Generate a JWT token from payload and secret.
        
        Returns:
            Generated JWT token
            
        Raises:
            ValueError: If generation fails
        """
        if not self.secret or not self.payload:
            raise ValueError("Cannot generate token without secret and payload")
        
        try:
            payload = self.payload.copy()
            
            # Add standard claims if not present
            if 'iat' not in payload:
                payload['iat'] = int(time.time())
            if 'exp' not in payload:
                # Default 1 hour expiry
                payload['exp'] = int(time.time()) + 3600
            
            token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
            return token
        except Exception as e:
            raise ValueError(f"Failed to generate JWT token: {str(e)}")
    
    def _get_token(self) -> str:
        """
        Get the JWT token (either pre-generated or newly generated).
        
        Returns:
            JWT token
        """
        if self.token:
            return self.token
        
        # Generate new token if needed
        if not self._generated_token:
            self._generated_token = self._generate_token()
        
        # Check if we need to regenerate (simple expiry check)
        if self.payload.get('exp'):
            try:
                jwt.decode(
                    self._generated_token,
                    self.secret,
                    algorithms=[self.algorithm]
                )
            except jwt.ExpiredSignatureError:
                self._generated_token = self._generate_token()
        
        return self._generated_token
    
    def apply(
        self,
        headers: Dict[str, str],
        params: Dict[str, Any],
        client: httpx.Client
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Apply JWT authentication.
        
        Args:
            headers: Request headers
            params: Query parameters
            client: HTTP client
            
        Returns:
            Modified (headers, params) with JWT in specified header
        """
        headers = headers.copy()
        token = self._get_token()
        
        if self.header_prefix:
            headers[self.header_name] = f"{self.header_prefix} {token}"
        else:
            headers[self.header_name] = token
        
        return headers, params.copy()
    
    def refresh_if_needed(self) -> None:
        """
        Refresh JWT token if needed.
        
        For generated tokens, this checks expiry and regenerates if needed.
        """
        if not self.token and self.secret and self.payload:
            self._generated_token = self._generate_token()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": "jwt",
            "token": self.token,
            "secret": self.secret,
            "algorithm": self.algorithm,
            "payload": self.payload,
            "header_name": self.header_name,
            "header_prefix": self.header_prefix
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JWTAuth':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary containing auth configuration
            
        Returns:
            JWTAuth instance
        """
        return cls(
            token=data.get("token"),
            secret=data.get("secret"),
            algorithm=data.get("algorithm", "HS256"),
            payload=data.get("payload"),
            header_name=data.get("header_name", "Authorization"),
            header_prefix=data.get("header_prefix", "Bearer")
        )

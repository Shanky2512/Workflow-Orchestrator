"""
Pydantic models for connector configuration and validation.
All configurations are strictly typed and validated.
"""

from typing import Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime
import hashlib
import uuid

class AuthType(str, Enum):
    """Supported authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MTLS = "mtls"
    CUSTOM_HEADER = "custom_header"


class ApiKeyLocation(str, Enum):
    """Where to place the API key."""
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"


class OAuth2GrantType(str, Enum):
    """OAuth2 grant types."""
    CLIENT_CREDENTIALS = "client_credentials"
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"


class HTTPMethod(str, Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class NoAuthConfig(BaseModel):
    """Configuration for no authentication."""
    type: Literal[AuthType.NONE] = AuthType.NONE


class ApiKeyAuthConfig(BaseModel):
    """Configuration for API key authentication."""
    type: Literal[AuthType.API_KEY] = AuthType.API_KEY
    key: str = Field(..., description="API key value")
    location: ApiKeyLocation = Field(..., description="Where to place the key")
    param_name: str = Field(..., description="Parameter/header name for the key")

    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class BasicAuthConfig(BaseModel):
    """Configuration for Basic authentication."""
    type: Literal[AuthType.BASIC] = AuthType.BASIC
    username: str = Field(..., description="Username for basic auth")
    password: str = Field(..., description="Password for basic auth")


class BearerTokenAuthConfig(BaseModel):
    """Configuration for Bearer token authentication."""
    type: Literal[AuthType.BEARER] = AuthType.BEARER
    token: str = Field(..., description="Bearer token value")

    @field_validator('token')
    @classmethod
    def validate_token(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Bearer token cannot be empty")
        return v.strip()


class JWTAuthConfig(BaseModel):
    """Configuration for JWT authentication."""
    type: Literal[AuthType.JWT] = AuthType.JWT
    token: Optional[str] = Field(None, description="Pre-generated JWT token")
    secret: Optional[str] = Field(None, description="Secret for JWT generation")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    payload: Optional[Dict[str, Any]] = Field(None, description="JWT payload if generating")
    header_name: str = Field(default="Authorization", description="Header name")
    header_prefix: str = Field(default="Bearer", description="Header prefix")

    @model_validator(mode='after')
    def validate_jwt_config(self) -> 'JWTAuthConfig':
        if not self.token and not (self.secret and self.payload):
            raise ValueError("Either provide a token or (secret + payload) for JWT generation")
        return self


class OAuth2AuthConfig(BaseModel):
    """Configuration for OAuth2 authentication."""
    type: Literal[AuthType.OAUTH2] = AuthType.OAUTH2
    grant_type: OAuth2GrantType = Field(..., description="OAuth2 grant type")
    token_url: str = Field(..., description="Token endpoint URL")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(default="", description="OAuth2 client secret")
    scope: Optional[str] = Field(None, description="Requested scopes")
    
    # For authorization code flow
    authorization_url: Optional[str] = Field(None, description="Authorization endpoint URL")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI")
    code: Optional[str] = Field(None, description="Authorization code")
    
    # For device code flow
    device_code_url: Optional[str] = Field(None, description="Device code endpoint URL")
    device_code: Optional[str] = Field(None, description="Device code from provider")
    
    # For refresh token
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    
    # Token storage
    access_token: Optional[str] = Field(None, description="Current access token")
    token_expiry: Optional[int] = Field(None, description="Token expiry timestamp")
    
    # Provider-specific overrides (NEW - makes it flexible)
    send_client_secret_in_device_code: bool = Field(
        default=False, 
        description="Whether to send client_secret in device code request (provider-specific)"
    )
    token_request_method: str = Field(
        default="POST",
        description="HTTP method for token requests"
    )
    custom_token_params: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional parameters for token requests (provider-specific)"
    )

    @model_validator(mode='after')
    def validate_oauth2_config(self) -> 'OAuth2AuthConfig':
        # Only validate what's truly required
        if self.grant_type == OAuth2GrantType.AUTHORIZATION_CODE:
            if not self.authorization_url or not self.redirect_uri:
                raise ValueError("Authorization code flow requires authorization_url and redirect_uri")
        elif self.grant_type == OAuth2GrantType.DEVICE_CODE:
            if not self.device_code_url:
                raise ValueError("Device code flow requires device_code_url")
        return self


class MTLSAuthConfig(BaseModel):
    """Configuration for mutual TLS authentication."""
    type: Literal[AuthType.MTLS] = AuthType.MTLS
    cert_path: str = Field(..., description="Path to client certificate file")
    key_path: str = Field(..., description="Path to client private key file")
    ca_bundle_path: Optional[str] = Field(None, description="Path to CA bundle for verification")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")

    @field_validator('cert_path', 'key_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Certificate/key path cannot be empty")
        return v.strip()


class CustomHeaderAuthConfig(BaseModel):
    """Configuration for custom header authentication."""
    type: Literal[AuthType.CUSTOM_HEADER] = AuthType.CUSTOM_HEADER
    headers: Dict[str, str] = Field(..., description="Custom headers for authentication")

    @field_validator('headers')
    @classmethod
    def validate_headers(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            raise ValueError("Headers cannot be empty")
        return v


# Union of all auth configs
AuthConfig = Union[
    NoAuthConfig,
    ApiKeyAuthConfig,
    BasicAuthConfig,
    BearerTokenAuthConfig,
    JWTAuthConfig,
    OAuth2AuthConfig,
    MTLSAuthConfig,
    CustomHeaderAuthConfig
]


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retries")
    backoff_factor: float = Field(default=1.0, ge=0.1, le=10.0, description="Backoff multiplier")
    retry_status_codes: list[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes to retry on"
    )


class ConnectorConfig(BaseModel):
    """Complete connector configuration."""
    id: str = Field(..., description="Unique connector identifier")
    name: str = Field(..., description="Human-readable connector name")
    description: Optional[str] = Field(None, description="Connector description")
    base_url: str = Field(..., description="Base URL for the API")
    auth: Dict[str, Any] = Field(..., description="Authentication configuration")
    default_headers: Optional[Dict[str, str]] = Field(default=None, description="Default headers")
    timeout: float = Field(default=30.0, gt=0, le=300, description="Request timeout in seconds")
    retry_config: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    allow_redirects: bool = Field(default=True, description="Whether to follow redirects")
    max_redirects: int = Field(default=10, ge=0, le=50, description="Maximum number of redirects")

    @field_validator('id', 'name')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        # Remove trailing slash for consistency
        return v.rstrip('/')
    def __init__(self, **data):
        """Auto-generate unique ID if not provided."""
        if not data.get('id'):
            raw_id = f"{data.get('name')}_{datetime.utcnow().isoformat()}_{uuid.uuid4().hex[:8]}"
            data['id'] = f"api_{hashlib.md5(raw_id.encode()).hexdigest()[:12]}"
        
        super().__init__(**data)


class ExecuteRequest(BaseModel):
    """Request model for executing a connector."""
    method: HTTPMethod = Field(..., description="HTTP method")
    endpoint: str = Field(..., description="API endpoint path")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional headers")
    query_params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")
    body: Optional[Union[Dict[str, Any], str, bytes]] = Field(default=None, description="Request body")
    timeout_override: Optional[float] = Field(default=None, gt=0, description="Override default timeout")
    stream: bool = Field(default=False, description="Whether to stream the response")

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        if not v:
            return ""
        if not v.startswith('/'):
            v = '/' + v
        return v
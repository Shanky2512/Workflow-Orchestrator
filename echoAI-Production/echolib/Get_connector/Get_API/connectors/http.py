"""
HTTP connector implementation.
Handles all HTTP-based API communication with comprehensive error handling and retry logic.
"""

from typing import Dict, Any, Optional
import httpx
import time
from .base import BaseConnector
from echolib.Get_connector.Get_API.models.config import ConnectorConfig, HTTPMethod
from echolib.Get_connector.Get_API.models.responses import ExecuteResponse
from echolib.Get_connector.Get_API.auth.basic import BasicAuthStrategy
from echolib.Get_connector.Get_API.auth import (
    AuthBase, NoAuth, ApiKeyAuth, BearerTokenAuth,
    JWTAuth, OAuth2Auth, MTLSAuth, CustomHeaderAuth
)
from echolib.Get_connector.Get_API.models.config import AuthType
import json

class HTTPConnector(BaseConnector):
    """
    HTTP connector for RESTful API communication.
    
    Features:
    - Full HTTP method support (GET, POST, PUT, PATCH, DELETE, etc.)
    - Automatic retry with exponential backoff
    - Streaming support
    - Comprehensive error handling
    - Authentication strategy pattern
    - Stateless execution for AI agent compatibility
    """
    
    def __init__(self, config: ConnectorConfig) -> None:
        """
        Initialize HTTP connector.
        
        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self.auth = self._create_auth_strategy(config.auth)
        self._client: Optional[httpx.Client] = None
    
    def _create_auth_strategy(self, auth_config: Any) -> AuthBase:
        """
        Create authentication strategy from configuration.
        Handles both Pydantic models and dicts.
        """
        # Extract type - works for both dict and Pydantic model
        if isinstance(auth_config, dict):
            auth_type = auth_config.get("type")
        else:
            auth_type = auth_config.type
        
        if auth_type == "none" or auth_type == AuthType.NONE:
            return NoAuth()
        
        elif auth_type == "api_key" or auth_type == AuthType.API_KEY:
            return ApiKeyAuth(
                key=auth_config.get("key") if isinstance(auth_config, dict) else auth_config.key,
                location=auth_config.get("location") if isinstance(auth_config, dict) else auth_config.location,
                param_name=auth_config.get("param_name") if isinstance(auth_config, dict) else auth_config.param_name
            )
        
        elif auth_type == "bearer" or auth_type == AuthType.BEARER:
            return BearerTokenAuth(
                token=auth_config.get("token") if isinstance(auth_config, dict) else auth_config.token
            )
        elif auth_type == "basic" or auth_type == AuthType.BASIC:
            return BasicAuthStrategy(
                username=auth_config.get("username") if isinstance(auth_config, dict) else auth_config.username,
                password=auth_config.get("password") if isinstance(auth_config, dict) else auth_config.password
            )

        elif auth_type == "jwt" or auth_type == AuthType.JWT:
            return JWTAuth(
                token=auth_config.get("token") if isinstance(auth_config, dict) else auth_config.token,
                secret=auth_config.get("secret") if isinstance(auth_config, dict) else auth_config.secret,
                algorithm=auth_config.get("algorithm", "HS256") if isinstance(auth_config, dict) else auth_config.algorithm,
                payload=auth_config.get("payload") if isinstance(auth_config, dict) else auth_config.payload,
                header_name=auth_config.get("header_name", "Authorization") if isinstance(auth_config, dict) else auth_config.header_name,
                header_prefix=auth_config.get("header_prefix", "Bearer") if isinstance(auth_config, dict) else auth_config.header_prefix
            )
        
        elif auth_type == "oauth2" or auth_type == AuthType.OAUTH2:
            return OAuth2Auth(
                grant_type=auth_config.get("grant_type") if isinstance(auth_config, dict) else auth_config.grant_type,
                token_url=auth_config.get("token_url") if isinstance(auth_config, dict) else auth_config.token_url,
                client_id=auth_config.get("client_id") if isinstance(auth_config, dict) else auth_config.client_id,
                client_secret=auth_config.get("client_secret", "") if isinstance(auth_config, dict) else auth_config.client_secret,
                scope=auth_config.get("scope") if isinstance(auth_config, dict) else auth_config.scope,
                authorization_url=auth_config.get("authorization_url") if isinstance(auth_config, dict) else auth_config.authorization_url,
                redirect_uri=auth_config.get("redirect_uri") if isinstance(auth_config, dict) else auth_config.redirect_uri,
                code=auth_config.get("code") if isinstance(auth_config, dict) else auth_config.code,
                refresh_token=auth_config.get("refresh_token") if isinstance(auth_config, dict) else auth_config.refresh_token,
                access_token=auth_config.get("access_token") if isinstance(auth_config, dict) else auth_config.access_token,
                token_expiry=auth_config.get("token_expiry") if isinstance(auth_config, dict) else auth_config.token_expiry
            )
        
        elif auth_type == "mtls" or auth_type == AuthType.MTLS:
            return MTLSAuth(
                cert_path=auth_config.get("cert_path") if isinstance(auth_config, dict) else auth_config.cert_path,
                key_path=auth_config.get("key_path") if isinstance(auth_config, dict) else auth_config.key_path,
                ca_bundle_path=auth_config.get("ca_bundle_path") if isinstance(auth_config, dict) else auth_config.ca_bundle_path,
                verify_ssl=auth_config.get("verify_ssl", True) if isinstance(auth_config, dict) else auth_config.verify_ssl
            )
        
        elif auth_type == "custom_header" or auth_type == AuthType.CUSTOM_HEADER:
            return CustomHeaderAuth(
                headers=auth_config.get("headers", {}) if isinstance(auth_config, dict) else auth_config.headers
            )
        
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")
    
    def _get_client(self) -> httpx.Client:
        """
        Get or create HTTP client with proper configuration.
        
        Returns:
            Configured httpx.Client
        """
        if self._client is None:
            client_kwargs = {
                "follow_redirects": self.config.allow_redirects,
                "max_redirects": self.config.max_redirects,
                "timeout": self.config.timeout,
            }
            
            # Configure mTLS if needed
            if isinstance(self.auth, MTLSAuth):
                client_kwargs["cert"] = self.auth.get_cert_tuple()
                client_kwargs["verify"] = self.auth.get_verify_option()
            else:
                client_kwargs["verify"] = self.config.verify_ssl
            
            self._client = httpx.Client(**client_kwargs)
        
        return self._client
    
    def _close_client(self) -> None:
        """Close the HTTP client if it exists."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from base URL and endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL
        """
        if not endpoint:
            return self.config.base_url
        
        endpoint = endpoint.lstrip('/')
        return f"{self.config.base_url.rstrip('/')}/{endpoint}"
    
    def _prepare_headers(
        self,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Prepare request headers.
        
        Args:
            additional_headers: Additional headers to merge
            
        Returns:
            Complete headers dictionary
        """
        headers = {}
        
        # Start with default headers
        if self.config.default_headers:
            headers.update(self.config.default_headers)
        
        # Add custom headers
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    def _execute_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
        body: Optional[Any],
        timeout: float,
        stream: bool
    ) -> httpx.Response:
        """
        Execute request with retry logic.
        
        Args:
            method: HTTP method
            url: Full URL
            headers: Request headers
            params: Query parameters
            body: Request body
            timeout: Request timeout
            stream: Whether to stream response
            
        Returns:
            httpx.Response
            
        Raises:
            httpx.HTTPError: If all retries fail
        """
        client = self._get_client()
        retry_config = self.config.retry_config
        last_exception = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                # Refresh auth if needed
                self.auth.refresh_if_needed()
                
                # Apply authentication
                auth_headers, auth_params = self.auth.apply(headers.copy(), params.copy(), client)
                
                # Prepare request kwargs
                request_kwargs = {
                    "method": method,
                    "url": url,
                    "headers": auth_headers,
                    "params": auth_params,
                    "timeout": timeout,
                }
                
                # Add body if present
                if body is not None:
                    if isinstance(body, dict):
                        request_kwargs["json"] = body
                    elif isinstance(body, str):
                        request_kwargs["content"] = body
                    elif isinstance(body, bytes):
                        request_kwargs["content"] = body
                    else:
                        request_kwargs["json"] = body
                
                # Execute request
                if stream:
                    return client.stream(**request_kwargs)
                else:
                    response = client.request(**request_kwargs)
                    
                    # Check if we should retry based on status code
                    if response.status_code in retry_config.retry_status_codes:
                        if attempt < retry_config.max_retries:
                            wait_time = retry_config.backoff_factor * (2 ** attempt)
                            time.sleep(wait_time)
                            continue
                    
                    return response
            
            except httpx.HTTPError as e:
                last_exception = e
                if attempt < retry_config.max_retries:
                    wait_time = retry_config.backoff_factor * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                raise
        
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected error in retry logic")
    
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
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint path
            headers: Additional headers (optional)
            query_params: Query parameters (optional)
            body: Request body (optional)
            timeout_override: Override default timeout (optional)
            stream: Whether to stream response (optional)
            
        Returns:
            ExecuteResponse with results
            
        Raises:
            ValueError: If parameters are invalid
        """
        start_time = time.time()
        
        try:
            # Validate method
            method = method.upper()

            # Build URL
            url = self._build_url(endpoint)
            
            # Prepare headers and params
            request_headers = self._prepare_headers(headers)
            request_params = query_params or {}
            
            # Determine timeout
            timeout = timeout_override if timeout_override is not None else self.config.timeout
            
            # Execute with retry
            response = self._execute_with_retry(
                method=method,
                url=url,
                headers=request_headers,
                params=request_params,
                body=body,
                timeout=timeout,
                stream=stream
            )
            
            # Parse response
            elapsed = time.time() - start_time
            
            # Get response body
            try:
                response_body = response.json()
            except Exception:
                response_body = response.text
            
            # Build error message for non-2xx responses
            error_message = None
            if not (200 <= response.status_code < 300):
                error_parts = [
                    f"HTTP {response.status_code}",
                    f"{method} {url}"
                ]
                if response_body:
                    if isinstance(response_body, dict):
                        error_parts.append(f"Response: {json.dumps(response_body, indent=2)}")
                    else:
                        body_preview = str(response_body)[:500]
                        error_parts.append(f"Response: {body_preview}")
                error_message = "\n".join(error_parts)
            
            return ExecuteResponse(
                success=200 <= response.status_code < 300,
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response_body,
                raw_body=response.text,
                error=error_message,
                elapsed_seconds=elapsed
            )
        
        except httpx.HTTPError as e:
            elapsed = time.time() - start_time
            
            # Build detailed error message
            error_parts = ["HTTP Error"]
            if hasattr(e, 'response') and e.response:
                error_parts.append(f"Status: {e.response.status_code}")
                error_parts.append(f"{method} {self._build_url(endpoint)}")
                try:
                    error_body = e.response.json()
                    error_parts.append(f"Response: {json.dumps(error_body, indent=2)}")
                except Exception:
                    error_parts.append(f"Response: {e.response.text[:500]}")
            else:
                error_parts.append(f"{method} {self._build_url(endpoint)}")
                error_parts.append(str(e))
            
            return ExecuteResponse(
                success=False,
                status_code=e.response.status_code if hasattr(e, 'response') and e.response else 500,
                headers={},
                body=None,
                raw_body=None,
                error="\n".join(error_parts),
                elapsed_seconds=elapsed
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            error_message = f"Unexpected error\n{method} {self._build_url(endpoint)}\n{str(e)}"
            
            return ExecuteResponse(
                success=False,
                status_code=500,
                headers={},
                body=None,
                raw_body=None,
                error=error_message,
                elapsed_seconds=elapsed
            )
    
    def test_connection(self) -> ExecuteResponse:
        """
        Test connector by making a simple request.
        
        Attempts a GET request to the base URL or root endpoint.
        
        Returns:
            ExecuteResponse with test results
        """
        return self.execute(method="GET", endpoint="/")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize connector to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "config": self.config.model_dump(),
            "auth": self.auth.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HTTPConnector':
        """
        Deserialize connector from dictionary.
        
        Args:
            data: Dictionary containing connector data
            
        Returns:
            HTTPConnector instance
            
        Raises:
            ValueError: If data is invalid
        """
        if "config" not in data:
            raise ValueError("Missing 'config' field")
        
        config = ConnectorConfig(**data["config"])
        return cls(config=config)
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._close_client()
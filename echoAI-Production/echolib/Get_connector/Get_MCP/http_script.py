"""
HTTP MCP Connector implementation.
Supports arbitrary HTTP methods, auth schemes, headers, and payloads.
"""
from typing import Dict, Any, Optional, List
import httpx
import time
from .base import BaseMCPConnector, TransportType, AuthType, ConnectorStatus
import logging

logger = logging.getLogger(__name__)


class HTTPMCPConnector(BaseMCPConnector):
    """
    HTTP transport MCP connector.
    
    Handles:
    - Any HTTP method (GET, POST, PUT, PATCH, DELETE, etc.)
    - Custom headers
    - Query parameters
    - Request body (JSON, form, raw)
    - Multiple auth schemes
    - Timeout and retry logic
    - Response validation
    """
    def __init__(
        self,
        name: str,
        description: str,
        auth_config: Dict[str, Any],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        endpoint_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        query_mapping: Optional[Dict[str, str]] = None,
        verify_ssl : bool = True,
        timeout: int = 30,
        retry_count: int = 0,
        connector_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HTTP connector.
        
        Args:
            endpoint_url: Full URL to API endpoint
            method: HTTP method (GET, POST, etc.)
            headers: Static headers to include
            query_params: Static query parameters
            query_mapping: Maps user-friendly input names to API query param names
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        super().__init__(
            name=name,
            description=description,
            transport_type=TransportType.HTTP,
            auth_config=auth_config,
            input_schema=input_schema,
            output_schema=output_schema,
            connector_id=connector_id,
            metadata=metadata
        )
        
        self.endpoint_url = endpoint_url
        self.method = method.upper()
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.query_mapping = query_mapping or {}
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.retry_count = retry_count
        
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate HTTP connector configuration"""
        errors = []
        
        # Validate URL
        if not self.endpoint_url:
            errors.append("endpoint_url is required")
        elif not self.endpoint_url.startswith(("http://", "https://")):
            errors.append("endpoint_url must start with http:// or https://")
            
        # Validate method
        valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        if self.method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}")
            
        # Validate auth config
        auth_type = self.auth_config.get("type")
        if auth_type:
            try:
                auth_enum = AuthType(auth_type)
                
                if auth_enum == AuthType.API_KEY:
                    if not self.auth_config.get("api_key"):
                        errors.append("api_key required for API_KEY auth")
                    if not self.auth_config.get("key_name"):
                        errors.append("key_name required for API_KEY auth (header or query param name)")
                        
                elif auth_enum == AuthType.BEARER:
                    if not self.auth_config.get("token"):
                        errors.append("token required for BEARER auth")
                        
                elif auth_enum == AuthType.CUSTOM_HEADER:
                    if not self.auth_config.get("header_name"):
                        errors.append("header_name required for CUSTOM_HEADER auth")
                    if not self.auth_config.get("header_value"):
                        errors.append("header_value required for CUSTOM_HEADER auth")
                        
                elif auth_enum == AuthType.BASIC:
                    if not self.auth_config.get("username"):
                        errors.append("username required for BASIC auth")
                    if not self.auth_config.get("password"):
                        errors.append("password required for BASIC auth")
                        
                elif auth_enum == AuthType.OAUTH2:
                    if not self.auth_config.get("access_token"):
                        errors.append("access_token required for OAUTH2 auth")
                        
            except ValueError:
                errors.append(f"Invalid auth type: {auth_type}")
                
        # Validate schemas
        if not self.input_schema:
            errors.append("input_schema is required")
        if not self.output_schema:
            errors.append("output_schema is required")
            
        # Validate timeout
        if self.timeout <= 0:
            errors.append("timeout must be positive")
            
        is_valid = len(errors) == 0
        if is_valid:
            self.update_status(ConnectorStatus.VALIDATED)
            
        return is_valid, errors
    
    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authentication headers based on auth_config"""
        auth_headers = {}
        auth_type = self.auth_config.get("type")
        
        if not auth_type or auth_type == "none":
            return auth_headers
            
        auth_enum = AuthType(auth_type)
        
        if auth_enum == AuthType.API_KEY:
            location = self.auth_config.get("location", "header")
            if location == "header":
                key_name = self.auth_config["key_name"]
                # Support both 'api_key' and 'key_value' field names
                api_key = self.auth_config.get("api_key") or self.auth_config.get("key_value")
                if api_key:
                    auth_headers[key_name] = api_key
                
        elif auth_enum == AuthType.BEARER:
            token = self.auth_config["token"]
            auth_headers["Authorization"] = f"Bearer {token}"
            
        elif auth_enum == AuthType.CUSTOM_HEADER:
            header_name = self.auth_config["header_name"]
            header_value = self.auth_config["header_value"]
            auth_headers[header_name] = header_value
            
        elif auth_enum == AuthType.BASIC:
            import base64
            username = self.auth_config["username"]
            password = self.auth_config["password"]
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            auth_headers["Authorization"] = f"Basic {credentials}"
            
        elif auth_enum == AuthType.OAUTH2:
            token = self.auth_config["access_token"]
            auth_headers["Authorization"] = f"Bearer {token}"
            
        return auth_headers
    
    def _build_query_params(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Build query parameters from static config and dynamic payload"""
        print("🔥 _build_query_params CALLED 🔥")
        print("QUERY_MAPPING =", self.query_mapping)
        print("PAYLOAD =", payload)

        params = self.query_params.copy()
        
        # Add API key to query if configured
        auth_type = self.auth_config.get("type")
        if auth_type == "api_key":
            location = self.auth_config.get("location", "header")
            if location == "query":
                key_name = self.auth_config["key_name"]
                # Support both 'api_key' and 'key_value' field names
                api_key = self.auth_config.get("api_key") or self.auth_config.get("key_value")
                if api_key:
                    params[key_name] = api_key
        
        # Apply query_mapping to payload before adding to params
        mapped_payload = {}

        # If query_mapping exists, ONLY mapped keys are allowed
        if self.query_mapping:
            for input_key, api_key in self.query_mapping.items():
                if input_key in payload:
                    mapped_payload[api_key] = payload[input_key]
        else:
            # Backward compatibility: no mapping defined
            for payload_key, payload_value in payload.items():
                if payload_key in ["query_params", "headers"]:
                    continue
                mapped_payload[payload_key] = payload_value

        # Add mapped payload params
        params.update(mapped_payload)
                
        # Add dynamic query params from payload if specified
        if "query_params" in payload and not self.query_mapping:
            params.update(payload["query_params"])

        
        # Re-apply auth params to ensure they override user input
        if auth_type == "api_key":
            location = self.auth_config.get("location", "header")
            if location == "query":
                key_name = self.auth_config["key_name"]
                api_key = self.auth_config.get("api_key") or self.auth_config.get("key_value")
                if api_key:
                    params[key_name] = api_key
            
        return params

    
    def build_connector(self) -> Dict[str, Any]:
        """Build HTTP MCP connector specification"""
        return {
            "type": "http",
            "endpoint": self.endpoint_url,
            "method": self.method,
            "headers": {**self.headers, **self._build_auth_headers()},
            "timeout": self.timeout,
            "retry": self.retry_count,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
    
    async def test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute real HTTP request to test connector.
        
        This makes an actual API call with the configured settings.
        """
        start_time = time.time()
        
        if "data" in payload and isinstance(payload["data"], dict):
            payload = payload["data"]

        if "input" in payload and isinstance(payload["input"], dict):
            payload = payload["input"]
        elif "test_payload" in payload and isinstance(payload["test_payload"], dict):
            payload = payload["test_payload"]

        try:
            # Use only user-provided headers
            headers = self.headers.copy() if self.headers else {}
            
            # Add authentication if configured
            if self.auth_config.get("type") and self.auth_config["type"] != "none":
                auth_headers = self._build_auth_headers()
                headers.update(auth_headers)
            
            # Build query params (applies query_mapping)
            params = self._build_query_params(payload)
            
            # Extract request body (only for methods that support body)
            body = None
            if self.method in ["POST", "PUT", "PATCH", "DELETE"]:
                body = {k: v for k, v in payload.items() if k not in ["query_params", "headers"]}
            
            # Make request
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                logger.info(f"Testing HTTP connector: {self.method} {self.endpoint_url}")
                
                response = await client.request(
                    method=self.method,
                    url=self.endpoint_url,
                    headers=headers,
                    params=params,
                    json=body if body else None
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Parse response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = response.text
                
                # Determine success
                success = response.status_code < 400
                
                result = {
                    "success": success,
                    "output": response_data,
                    "error": None if success else f"HTTP {response.status_code}",
                    "duration_ms": duration_ms,
                    "metadata": {
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "url": str(response.url),
                    }
                }
                
                self.add_test_result(result)
                return result
                
        except httpx.TimeoutException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = {
                "success": False,
                "output": None,
                "error": f"Request timeout after {self.timeout}s",
                "duration_ms": duration_ms,
                "metadata": {"exception": str(e)}
            }
            self.add_test_result(result)
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"HTTP test failed: {e}", exc_info=True)
            result = {
                "success": False,
                "output": None,
                "error": str(e),
                "duration_ms": duration_ms,
                "metadata": {"exception_type": type(e).__name__}
            }
            self.add_test_result(result)
            return result
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize HTTP connector with transport-specific fields"""
        base_data = super().serialize()
        base_data.update({
            "endpoint_url": self.endpoint_url,
            "method": self.method,
            "headers": self.headers,
            "query_params": self.query_params,
            "query_mapping": self.query_mapping,
            "verify_ssl" : self.verify_ssl,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        })
        return base_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HTTPMCPConnector':
        """Deserialize HTTP connector from dict"""
        # Restore auth config with actual secrets (should load from vault in production)
        auth_config = data["creation_payload"]["auth_config"]
        
        connector = cls(
            name=data["connector_name"],
            description=data["creation_payload"]["description"],
            auth_config=auth_config,
            input_schema=data["creation_payload"]["input_schema"],
            output_schema=data["creation_payload"]["output_schema"],
            endpoint_url=data["creation_payload"]["endpoint_url"],
            method=data["creation_payload"]["method"],
            headers=data.get("creation_payload").get("headers", {}),
            query_params=data.get("creation_payload").get("query_params", {}),
            query_mapping=data.get("creation_payload").get("query_mapping", {}),
            timeout=data.get("creation_payload").get("timeout", 30),
            retry_count=data.get("creation_payload").get("retry_count", 0),
            verify_ssl = data.get("creation_payload").get("verify_ssl", True),
            connector_id=data["connector_id"],
            metadata=data.get("metadata", {})
        )
        
        connector.status = ConnectorStatus(data["validation_status"])
        connector.created_at = data["created_at"]
        connector.updated_at = data["updated_at"]
        # connector.test_results = data.get("test_results", [])
        
        return connector
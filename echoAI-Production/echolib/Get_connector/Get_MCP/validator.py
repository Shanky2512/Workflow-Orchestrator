"""
Input validation and normalization for connector creation.
Enforces strict rules before connector instantiation.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from .base import TransportType, AuthType
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")


class ConnectorValidator:
    """
    Validates and normalizes connector input before creation.
    
    Enforces:
    - Required fields present
    - Field types correct
    - Values within acceptable ranges
    - Schema validity
    - Auth config completeness
    """
    
    @staticmethod
    def validate_base_fields(data: Dict[str, Any]) -> List[str]:
        """Validate fields common to all connectors"""
        errors = []
        
        # Name validation
        if "name" not in data or not data["name"]:
            errors.append("name is required")
        elif not isinstance(data["name"], str):
            errors.append("name must be a string")
        elif len(data["name"]) < 3:
            errors.append("name must be at least 3 characters")
        elif len(data["name"]) > 100:
            errors.append("name must be at most 100 characters")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', data["name"]):
            errors.append("name can only contain alphanumeric, underscore, and hyphen")
            
        # Description validation
        if "description" not in data or not data["description"]:
            errors.append("description is required")
        elif not isinstance(data["description"], str):
            errors.append("description must be a string")
        elif len(data["description"]) > 1000:
            errors.append("description must be at most 1000 characters")
            
        # Transport type validation
        if "transport_type" not in data:
            errors.append("transport_type is required")
        else:
            try:
                TransportType(data["transport_type"])
            except ValueError:
                valid_types = [t.value for t in TransportType]
                errors.append(f"transport_type must be one of {valid_types}")
                
        return errors
    
    @staticmethod
    def validate_auth_config(auth_config: Dict[str, Any]) -> List[str]:
        """Validate authentication configuration"""
        errors = []
        
        if not isinstance(auth_config, dict):
            errors.append("auth_config must be a dictionary")
            return errors
            
        auth_type = auth_config.get("type")
        if not auth_type:
            # No auth is valid
            return errors
            
        try:
            auth_enum = AuthType(auth_type)
        except ValueError:
            valid_types = [t.value for t in AuthType]
            errors.append(f"auth type must be one of {valid_types}")
            return errors
        
        # Validate based on auth type
        if auth_enum == AuthType.API_KEY:
            if "api_key" not in auth_config:
                errors.append("api_key required for API_KEY auth")
            elif not auth_config["api_key"]:
                errors.append("api_key cannot be empty")
                
            if "key_name" not in auth_config:
                errors.append("key_name required for API_KEY auth")
            elif not auth_config["key_name"]:
                errors.append("key_name cannot be empty")
                
            location = auth_config.get("location", "header")
            if location not in ["header", "query"]:
                errors.append("location must be 'header' or 'query' for API_KEY auth")
                
        elif auth_enum == AuthType.BEARER:
            if "token" not in auth_config:
                errors.append("token required for BEARER auth")
            elif not auth_config["token"]:
                errors.append("token cannot be empty")
                
        elif auth_enum == AuthType.CUSTOM_HEADER:
            if "header_name" not in auth_config:
                errors.append("header_name required for CUSTOM_HEADER auth")
            if "header_value" not in auth_config:
                errors.append("header_value required for CUSTOM_HEADER auth")
                
        elif auth_enum == AuthType.BASIC:
            if "username" not in auth_config:
                errors.append("username required for BASIC auth")
            if "password" not in auth_config:
                errors.append("password required for BASIC auth")
                
        elif auth_enum == AuthType.OAUTH2:
            if "access_token" not in auth_config:
                errors.append("access_token required for OAUTH2 auth")
            elif not auth_config["access_token"]:
                errors.append("access_token cannot be empty")
                
        return errors
    
    @staticmethod
    def validate_json_schema(schema: Any, schema_name: str) -> List[str]:
        """
        Validate that schema is a valid JSON schema.
        
        Simplified validation - in production, use jsonschema library.
        """
        errors = []
        
        if not isinstance(schema, dict):
            errors.append(f"{schema_name} must be a dictionary")
            return errors
            
        # Check for basic schema structure
        if "type" not in schema and "properties" not in schema:
            errors.append(f"{schema_name} must have 'type' or 'properties'")
            
        return errors
    
    @staticmethod
    def validate_http_config(data: Dict[str, Any]) -> List[str]:
        """Validate HTTP-specific configuration"""
        errors = []
        
        # Endpoint URL
        if "endpoint_url" not in data:
            errors.append("endpoint_url is required for HTTP connector")
        elif not data["endpoint_url"].startswith(("http://", "https://")):
            errors.append("endpoint_url must start with http:// or https://")
            
        # HTTP method
        method = data.get("method", "POST").upper()
        valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        if method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}")
            
        # Headers
        if "headers" in data and not isinstance(data["headers"], dict):
            errors.append("headers must be a dictionary")
            
        # Query params
        if "query_params" in data and not isinstance(data["query_params"], dict):
            errors.append("query_params must be a dictionary")
            
        # Timeout
        timeout = data.get("timeout", 30)
        if not isinstance(timeout, int) or timeout <= 0:
            errors.append("timeout must be a positive integer")
            
        return errors
    
    @staticmethod
    def validate_sse_config(data: Dict[str, Any]) -> List[str]:
        """Validate SSE-specific configuration"""
        errors = []
        
        # Endpoint URL
        if "endpoint_url" not in data:
            errors.append("endpoint_url is required for SSE connector")
        elif not data["endpoint_url"].startswith(("http://", "https://")):
            errors.append("endpoint_url must start with http:// or https://")
            
        # Headers
        if "headers" in data and not isinstance(data["headers"], dict):
            errors.append("headers must be a dictionary")
            
        # Reconnection config
        max_attempts = data.get("max_reconnect_attempts", 5)
        if not isinstance(max_attempts, int) or max_attempts < 0:
            errors.append("max_reconnect_attempts must be a non-negative integer")
            
        reconnect_delay = data.get("reconnect_delay", 1)
        if not isinstance(reconnect_delay, int) or reconnect_delay <= 0:
            errors.append("reconnect_delay must be a positive integer")
            
        # Event filter
        if "event_filter" in data:
            event_filter = data["event_filter"]
            if event_filter is not None and not isinstance(event_filter, list):
                errors.append("event_filter must be a list or null")
            elif isinstance(event_filter, list):
                if not all(isinstance(e, str) for e in event_filter):
                    errors.append("event_filter must contain only strings")
                    
        return errors
    
    @staticmethod
    def validate_stdio_config(data: Dict[str, Any]) -> List[str]:
        """Validate STDIO-specific configuration"""
        errors = []
        
        # Command
        if "command" not in data:
            errors.append("command is required for STDIO connector")
        elif not data["command"]:
            errors.append("command cannot be empty")
            
        # Args
        if "args" in data:
            if not isinstance(data["args"], list):
                errors.append("args must be a list")
            elif not all(isinstance(arg, str) for arg in data["args"]):
                errors.append("args must contain only strings")
                
        # Environment variables
        if "env_vars" in data:
            if not isinstance(data["env_vars"], dict):
                errors.append("env_vars must be a dictionary")
            elif not all(isinstance(k, str) and isinstance(v, str) for k, v in data["env_vars"].items()):
                errors.append("env_vars must contain only string key-value pairs")
                
        # Timeout
        timeout = data.get("timeout", 30)
        if not isinstance(timeout, int) or timeout <= 0:
            errors.append("timeout must be a positive integer")
            
        return errors
    
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete connector configuration.
        
        Returns:
            (is_valid, error_messages)
        """
        all_errors = []
        
        # Validate base fields
        all_errors.extend(cls.validate_base_fields(data))
        
        # If transport_type is invalid, can't validate further
        if "transport_type" not in data:
            return False, all_errors
            
        try:
            transport = TransportType(data["transport_type"])
        except ValueError:
            return False, all_errors
        
        # Validate auth config
        auth_config = data.get("auth_config", {})
        all_errors.extend(cls.validate_auth_config(auth_config))
        
        # Validate schemas
        input_schema = data.get("input_schema", {})
        output_schema = data.get("output_schema", {})
        all_errors.extend(cls.validate_json_schema(input_schema, "input_schema"))
        all_errors.extend(cls.validate_json_schema(output_schema, "output_schema"))
        
        # Validate transport-specific config
        if transport == TransportType.HTTP:
            all_errors.extend(cls.validate_http_config(data))
        elif transport == TransportType.SSE:
            all_errors.extend(cls.validate_sse_config(data))
        elif transport == TransportType.STDIO:
            all_errors.extend(cls.validate_stdio_config(data))
            
        is_valid = len(all_errors) == 0
        return is_valid, all_errors
    
    @staticmethod
    def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize input data.
        
        - Trim whitespace
        - Set defaults
        - Convert types
        """
        normalized = data.copy()
        
        # Normalize strings
        if "name" in normalized:
            normalized["name"] = normalized["name"].strip()
        if "description" in normalized:
            normalized["description"] = normalized["description"].strip()
            
        # Set defaults
        if "auth_config" not in normalized:
            normalized["auth_config"] = {"type": "none"}
            
        if "metadata" not in normalized:
            normalized["metadata"] = {}
            
        # Transport-specific defaults
        transport = normalized.get("transport_type")
        
        if transport == "http":
            normalized.setdefault("method", "POST")
            normalized.setdefault("headers", {})
            normalized.setdefault("query_params", {})
            normalized.setdefault("timeout", 30)
            normalized.setdefault("retry_count", 0)
            
        elif transport == "sse":
            normalized.setdefault("headers", {})
            normalized.setdefault("reconnect", True)
            normalized.setdefault("max_reconnect_attempts", 5)
            normalized.setdefault("reconnect_delay", 1)
            normalized.setdefault("heartbeat_timeout", 60)
            
        elif transport == "stdio":
            normalized.setdefault("args", [])
            normalized.setdefault("env_vars", {})
            normalized.setdefault("timeout", 30)
            normalized.setdefault("shell", False)
            
        return normalized


def validate_and_normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate and normalize in one step.
    
    Raises ValidationError if validation fails.
    """
    # Normalize first
    normalized = ConnectorValidator.normalize(data)
    
    # Then validate
    is_valid, errors = ConnectorValidator.validate(normalized)
    
    if not is_valid:
        raise ValidationError(errors)
        
    return normalized
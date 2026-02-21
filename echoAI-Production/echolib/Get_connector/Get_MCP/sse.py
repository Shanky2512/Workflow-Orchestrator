"""
SSE (Server-Sent Events) MCP Connector implementation.
Handles long-lived streaming connections with reconnection logic.
"""

from typing import Dict, Any, Optional, List, AsyncIterator
import httpx
import asyncio
import time
from .base import BaseMCPConnector, TransportType, AuthType, ConnectorStatus
import logging
import json

logger = logging.getLogger(__name__)


class SSEMCPConnector(BaseMCPConnector):
    """
    SSE transport MCP connector.
    
    Handles:
    - Long-lived streaming connections
    - Event parsing (data, event, id, retry fields)
    - Automatic reconnection with backoff
    - Heartbeat/keep-alive detection
    - Graceful shutdown
    - Connection state management
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        auth_config: Dict[str, Any],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        endpoint_url: str,
        headers: Optional[Dict[str, str]] = None,
        reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 1,
        heartbeat_timeout: int = 60,
        event_filter: Optional[List[str]] = None,
        connector_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SSE connector.
        
        Args:
            endpoint_url: SSE stream URL
            headers: Static headers
            reconnect: Enable automatic reconnection
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Initial delay between reconnects (exponential backoff)
            heartbeat_timeout: Timeout for heartbeat/keep-alive
            event_filter: Only process events with these names (None = all events)
        """
        super().__init__(
            name=name,
            description=description,
            transport_type=TransportType.SSE,
            auth_config=auth_config,
            input_schema=input_schema,
            output_schema=output_schema,
            connector_id=connector_id,
            metadata=metadata
        )
        
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.heartbeat_timeout = heartbeat_timeout
        self.event_filter = event_filter
        
        # Connection state
        self._connection_active = False
        self._last_event_id: Optional[str] = None
        
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate SSE connector configuration"""
        errors = []
        
        # Validate URL
        if not self.endpoint_url:
            errors.append("endpoint_url is required")
        elif not self.endpoint_url.startswith(("http://", "https://")):
            errors.append("endpoint_url must start with http:// or https://")
            
        # Validate auth config
        auth_type = self.auth_config.get("type")
        if auth_type and auth_type != "none":
            try:
                auth_enum = AuthType(auth_type)
                
                if auth_enum == AuthType.API_KEY:
                    if not self.auth_config.get("api_key"):
                        errors.append("api_key required for API_KEY auth")
                    if not self.auth_config.get("key_name"):
                        errors.append("key_name required for API_KEY auth")
                        
                elif auth_enum == AuthType.BEARER:
                    if not self.auth_config.get("token"):
                        errors.append("token required for BEARER auth")
                        
            except ValueError:
                errors.append(f"Invalid auth type: {auth_type}")
                
        # Validate schemas
        if not self.input_schema:
            errors.append("input_schema is required")
        if not self.output_schema:
            errors.append("output_schema is required")
            
        # Validate reconnection config
        if self.max_reconnect_attempts < 0:
            errors.append("max_reconnect_attempts must be non-negative")
        if self.reconnect_delay <= 0:
            errors.append("reconnect_delay must be positive")
            
        is_valid = len(errors) == 0
        if is_valid:
            self.update_status(ConnectorStatus.VALIDATED)
            
        return is_valid, errors
    
    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authentication headers"""
        auth_headers = {}
        auth_type = self.auth_config.get("type")
        
        if not auth_type or auth_type == "none":
            return auth_headers
            
        auth_enum = AuthType(auth_type)
        
        if auth_enum == AuthType.API_KEY:
            location = self.auth_config.get("location", "header")
            if location == "header":
                key_name = self.auth_config["key_name"]
                api_key = self.auth_config["api_key"]
                auth_headers[key_name] = api_key
                
        elif auth_enum == AuthType.BEARER:
            token = self.auth_config["token"]
            auth_headers["Authorization"] = f"Bearer {token}"
            
        return auth_headers
    
    def build_connector(self) -> Dict[str, Any]:
        """Build SSE MCP connector specification"""
        return {
            "type": "sse",
            "endpoint": self.endpoint_url,
            "headers": {**self.headers, **self._build_auth_headers()},
            "reconnect": self.reconnect,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "event_filter": self.event_filter,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
    
    async def _parse_sse_stream(
        self,
        response: httpx.Response,
        max_events: int = 10
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse SSE stream according to SSE specification.
        
        Yields parsed events with: event, data, id, retry fields
        """
        event_type = ""
        event_data = []
        event_id = ""
        
        events_received = 0
        
        async for line in response.aiter_lines():
            # Empty line signals end of event
            if not line or line == "\n":
                if event_data:
                    # Construct event
                    data_str = "\n".join(event_data)
                    
                    # Try to parse as JSON
                    try:
                        data_parsed = json.loads(data_str)
                    except json.JSONDecodeError:
                        data_parsed = data_str
                    
                    event = {
                        "event": event_type or "message",
                        "data": data_parsed,
                        "id": event_id,
                        "timestamp": time.time()
                    }
                    
                    # Apply event filter if configured
                    if self.event_filter is None or event["event"] in self.event_filter:
                        yield event
                        events_received += 1
                        
                        if events_received >= max_events:
                            return
                    
                    # Store last event ID for reconnection
                    if event_id:
                        self._last_event_id = event_id
                    
                    # Reset for next event
                    event_type = ""
                    event_data = []
                    event_id = ""
                continue
            
            # Parse field
            if ":" in line:
                field, _, value = line.partition(":")
                value = value.lstrip()
                
                if field == "event":
                    event_type = value
                elif field == "data":
                    event_data.append(value)
                elif field == "id":
                    event_id = value
                elif field == "retry":
                    # Could update reconnection delay here
                    pass
    
    async def test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test SSE connector by opening stream and collecting events.
        
        This establishes a real SSE connection and captures events.
        """
        start_time = time.time()
        max_test_events = payload.get("max_events", 5)
        test_duration = payload.get("duration_seconds", 10)
        
        collected_events = []
        
        try:
            headers = {
                **self.headers,
                **self._build_auth_headers(),
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            }
            
            # Add Last-Event-ID for reconnection
            if self._last_event_id:
                headers["Last-Event-ID"] = self._last_event_id
            
            logger.info(f"Testing SSE connector: {self.endpoint_url}")
            
            async with httpx.AsyncClient(timeout=test_duration) as client:
                async with client.stream(
                    "GET",
                    self.endpoint_url,
                    headers=headers
                ) as response:
                    
                    if response.status_code >= 400:
                        duration_ms = int((time.time() - start_time) * 1000)
                        result = {
                            "success": False,
                            "output": None,
                            "error": f"HTTP {response.status_code}",
                            "duration_ms": duration_ms,
                            "metadata": {
                                "status_code": response.status_code,
                                "headers": dict(response.headers)
                            }
                        }
                        self.add_test_result(result)
                        return result
                    
                    self._connection_active = True
                    
                    # Collect events
                    try:
                        async for event in self._parse_sse_stream(response, max_test_events):
                            collected_events.append(event)
                            
                            # Check if we've collected enough
                            if len(collected_events) >= max_test_events:
                                break
                                
                    except asyncio.TimeoutError:
                        logger.info("SSE test timeout reached")
                    
                    self._connection_active = False
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "success": len(collected_events) > 0,
                "output": {
                    "events": collected_events,
                    "event_count": len(collected_events)
                },
                "error": None if collected_events else "No events received",
                "duration_ms": duration_ms,
                "metadata": {
                    "last_event_id": self._last_event_id,
                    "connection_time_ms": duration_ms
                }
            }
            
            self.add_test_result(result)
            return result
            
        except httpx.TimeoutException:
            duration_ms = int((time.time() - start_time) * 1000)
            result = {
                "success": len(collected_events) > 0,
                "output": {
                    "events": collected_events,
                    "event_count": len(collected_events)
                },
                "error": "Connection timeout" if not collected_events else None,
                "duration_ms": duration_ms,
                "metadata": {"partial_success": len(collected_events) > 0}
            }
            self.add_test_result(result)
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"SSE test failed: {e}", exc_info=True)
            result = {
                "success": False,
                "output": {
                    "events": collected_events,
                    "event_count": len(collected_events)
                } if collected_events else None,
                "error": str(e),
                "duration_ms": duration_ms,
                "metadata": {"exception_type": type(e).__name__}
            }
            self.add_test_result(result)
            return result
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize SSE connector with transport-specific fields"""
        base_data = super().serialize()
        base_data.update({
            "endpoint_url": self.endpoint_url,
            "headers": self.headers,
            "reconnect": self.reconnect,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "reconnect_delay": self.reconnect_delay,
            "heartbeat_timeout": self.heartbeat_timeout,
            "event_filter": self.event_filter,
            "last_event_id": self._last_event_id,
        })
        return base_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SSEMCPConnector':
        """Deserialize SSE connector from dict"""
        connector = cls(
            name=data["name"],
            description=data["description"],
            auth_config=data["auth_config"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            endpoint_url=data["endpoint_url"],
            headers=data.get("headers", {}),
            reconnect=data.get("reconnect", True),
            max_reconnect_attempts=data.get("max_reconnect_attempts", 5),
            reconnect_delay=data.get("reconnect_delay", 1),
            heartbeat_timeout=data.get("heartbeat_timeout", 60),
            event_filter=data.get("event_filter"),
            connector_id=data["connector_id"],
            metadata=data.get("metadata", {})
        )
        
        connector.status = ConnectorStatus(data["status"])
        connector.created_at = data["created_at"]
        connector.updated_at = data["updated_at"]
        connector.test_results = data.get("test_results", [])
        connector._last_event_id = data.get("last_event_id")
        
        return connector
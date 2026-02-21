"""
MCP Connector Generator Package
Production-grade MCP connector creation and management for AI agents.
"""

from .base import (
    BaseMCPConnector,
    TransportType,
    AuthType,
    ConnectorStatus
)

from .http_script import HTTPMCPConnector
from .sse import SSEMCPConnector
from .stdio import STDIOMCPConnector

from .validator import (
    ConnectorValidator,
    ValidationError,
    validate_and_normalize
)

from .storage import (
    ConnectorStorage,
    StorageError,
    get_storage
)

from .tester import (
    ConnectorTester,
    TestResult,
    TestSuite,
    create_default_test_payload
)

__version__ = "1.0.0"
__author__ = "MCP Connector Team"

__all__ = [
    # Base classes and enums
    "BaseMCPConnector",
    "TransportType",
    "AuthType",
    "ConnectorStatus",
    
    # Connector implementations
    "HTTPMCPConnector",
    "SSEMCPConnector",
    "STDIOMCPConnector",
    
    # Validation
    "ConnectorValidator",
    "ValidationError",
    "validate_and_normalize",
    
    # Storage
    "ConnectorStorage",
    "StorageError",
    "get_storage",
    
    # Testing
    "ConnectorTester",
    "TestResult",
    "TestSuite",
    "create_default_test_payload",
]
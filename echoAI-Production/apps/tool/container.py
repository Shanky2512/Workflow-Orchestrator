"""
Tool System Dependency Injection Container

This module configures and registers all tool system components
with the EchoAI dependency injection container.

Components registered:
- tool.storage: ToolStorage instance for persistence
- tool.registry: ToolRegistry instance for tool management
- tool.service: ToolService instance (backward compatibility)
"""

import logging
from pathlib import Path

from echolib.di import container
from echolib.services import ToolService
from echolib.repositories.tool_repo import ToolRepository

from .storage import ToolStorage
from .registry import ToolRegistry
from .executor import ToolExecutor

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration Constants
# ==============================================================================

# Base path for the echoAI application
_ECHOAI_ROOT = Path(__file__).parent.parent.parent

# Tool storage directory (apps/storage/tools/)
TOOLS_STORAGE_DIR = _ECHOAI_ROOT / "apps" / "storage" / "tools"

# Directories to scan for tool manifests (AgentTools folder)
TOOLS_DISCOVERY_DIRS = [
    _ECHOAI_ROOT / "AgentTools"
]

# ==============================================================================
# Component Initialization
# ==============================================================================

def _init_tool_storage() -> ToolStorage:
    """
    Initialize the ToolStorage component.

    Returns:
        ToolStorage: Configured storage instance
    """
    try:
        storage = ToolStorage(storage_dir=TOOLS_STORAGE_DIR)
        logger.info(f"ToolStorage initialized at {TOOLS_STORAGE_DIR}")
        return storage
    except Exception as e:
        logger.error(f"Failed to initialize ToolStorage: {e}")
        raise


def _init_tool_registry(storage: ToolStorage) -> ToolRegistry:
    """
    Initialize the ToolRegistry component.

    Args:
        storage: ToolStorage instance for persistence

    Returns:
        ToolRegistry: Configured registry instance
    """
    try:
        # Convert paths to ensure cross-platform compatibility
        discovery_dirs = [Path(d) for d in TOOLS_DISCOVERY_DIRS if Path(d).exists()]

        # ToolRepository for DB dual-write (stateless, safe to instantiate)
        tool_repo = ToolRepository()

        registry = ToolRegistry(
            storage=storage,
            discovery_dirs=discovery_dirs,
            tool_repo=tool_repo
        )
        logger.info(
            f"ToolRegistry initialized with {registry.count()} tools, "
            f"{len(discovery_dirs)} discovery directories, DB dual-write enabled"
        )
        return registry
    except Exception as e:
        logger.error(f"Failed to initialize ToolRegistry: {e}")
        raise


# ==============================================================================
# Singleton Instances
# ==============================================================================

# Initialize components (lazy loading pattern)
_storage: ToolStorage = None
_registry: ToolRegistry = None
_executor: ToolExecutor = None
_tool_service: ToolService = None


def get_storage() -> ToolStorage:
    """
    Get or create the ToolStorage singleton.

    Returns:
        ToolStorage: Shared storage instance
    """
    global _storage
    if _storage is None:
        _storage = _init_tool_storage()
    return _storage


def get_registry() -> ToolRegistry:
    """
    Get or create the ToolRegistry singleton.

    Returns:
        ToolRegistry: Shared registry instance
    """
    global _registry
    if _registry is None:
        _registry = _init_tool_registry(get_storage())
    return _registry


def get_executor() -> ToolExecutor:
    """
    Get or create the ToolExecutor singleton.

    Returns:
        ToolExecutor: Shared executor instance
    """
    global _executor
    if _executor is None:
        # Resolve ConnectorManager for managed tool execution (API/MCP connectors)
        connector_manager = None
        try:
            connector_manager = container.resolve('connector.manager')
        except (KeyError, Exception):
            logger.debug("ConnectorManager not yet available for ToolExecutor; will lazy-resolve")

        _executor = ToolExecutor(
            registry=get_registry(),
            default_timeout=60,
            connector_manager=connector_manager
        )
        logger.info("ToolExecutor initialized")
    return _executor


def get_tool_service() -> ToolService:
    """
    Get or create the ToolService singleton (backward compatibility).

    Returns:
        ToolService: Shared service instance
    """
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service


# ==============================================================================
# DI Container Registration
# ==============================================================================

# Register components with the container
# Using factory functions to enable lazy initialization

container.register('tool.storage', get_storage)
container.register('tool.registry', get_registry)
container.register('tool.executor', get_executor)
container.register('tool.service', get_tool_service)

logger.debug("Tool system components registered in DI container")

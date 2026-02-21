"""
Tool API Routes

This module provides FastAPI routes for tool management and execution
in the EchoAI system.

Routes:
    POST /register - Register a new tool
    GET /list - List all registered tools
    POST /invoke/{name} - Invoke tool by name (backward compatibility)
    POST /invoke/id/{tool_id} - Invoke tool by ID (preferred)
    GET /{tool_id} - Get tool definition by ID
    POST /discover - Trigger tool discovery from AgentTools folder
    GET /agent/{agent_id} - Get tools assigned to an agent
    DELETE /{tool_id} - Delete a tool
"""

from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from echolib.di import container
from echolib.security import get_current_user, require_user
from echolib.types import ToolDef, ToolType, UserContext

from .registry import ToolRegistry
from .executor import ToolExecutor

router = APIRouter(prefix='/tools', tags=['ToolApi'])


# ==============================================================================
# Dependency Injection Helpers
# ==============================================================================

def get_registry() -> ToolRegistry:
    """
    Get the ToolRegistry singleton from the DI container.

    Returns:
        ToolRegistry: Shared registry instance
    """
    return container.resolve('tool.registry')


def get_executor() -> ToolExecutor:
    """
    Get the ToolExecutor singleton from the DI container.

    Returns:
        ToolExecutor: Shared executor instance
    """
    return container.resolve('tool.executor')


# ==============================================================================
# Tool Registration Routes
# ==============================================================================

@router.post('/register')
async def register(tool: ToolDef) -> Dict[str, Any]:
    """
    Register a new tool or update an existing one.

    Args:
        tool: ToolDef with tool definition

    Returns:
        Dict with tool_id, status, and message

    Raises:
        HTTPException: If registration fails
    """
    try:
        result = get_registry().register(tool)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.delete('/{tool_id}')
async def delete_tool(tool_id: str) -> Dict[str, Any]:
    """
    Delete a tool from the registry.

    Args:
        tool_id: Unique identifier of the tool to delete

    Returns:
        Dict with status and message

    Raises:
        HTTPException: If deletion fails
    """
    try:
        success = get_registry().delete(tool_id)
        if success:
            return {
                "status": "deleted",
                "tool_id": tool_id,
                "message": f"Tool '{tool_id}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_id}' not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# ==============================================================================
# Tool Listing Routes
# ==============================================================================

@router.get('/list')
async def list_tools() -> List[Dict[str, Any]]:
    """
    List all registered tools.

    Returns:
        List of tool summaries (tool_id, name, tool_type, status)
    """
    try:
        tools = get_registry().list_all()
        return [
            {
                "tool_id": t.tool_id,
                "name": t.name,
                "description": t.description,
                "tool_type": t.tool_type.value if hasattr(t.tool_type, 'value') else str(t.tool_type),
                "status": t.status,
                "version": t.version,
                "tags": t.tags
            }
            for t in tools
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get('/list/type/{tool_type}')
async def list_tools_by_type(tool_type: str) -> List[Dict[str, Any]]:
    """
    List tools filtered by type.

    Args:
        tool_type: Tool type (local, mcp, api, crewai)

    Returns:
        List of tools matching the specified type
    """
    try:
        # Convert string to ToolType enum
        try:
            type_enum = ToolType(tool_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool type: {tool_type}. Must be one of: local, mcp, api, crewai"
            )

        tools = get_registry().list_by_type(type_enum)
        return [
            {
                "tool_id": t.tool_id,
                "name": t.name,
                "description": t.description,
                "tool_type": t.tool_type.value,
                "status": t.status
            }
            for t in tools
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get('/search')
async def search_tools(q: str) -> List[Dict[str, Any]]:
    """
    Search tools by name or description.

    Args:
        q: Search query string

    Returns:
        List of matching tools
    """
    try:
        tools = get_registry().search(q)
        return [
            {
                "tool_id": t.tool_id,
                "name": t.name,
                "description": t.description,
                "tool_type": t.tool_type.value if hasattr(t.tool_type, 'value') else str(t.tool_type),
                "status": t.status
            }
            for t in tools
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ==============================================================================
# Tool Retrieval Routes
# ==============================================================================

@router.get('/{tool_id}')
async def get_tool(tool_id: str) -> Dict[str, Any]:
    """
    Get tool definition by ID.

    Args:
        tool_id: Unique identifier of the tool

    Returns:
        Full tool definition

    Raises:
        HTTPException: If tool not found
    """
    try:
        tool = get_registry().get(tool_id)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_id}' not found"
            )
        return tool.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tool: {str(e)}")


# ==============================================================================
# Tool Invocation Routes
# ==============================================================================

@router.post('/invoke/{name}')
async def invoke(name: str, args: dict) -> Dict[str, Any]:
    """
    Invoke a tool by name (backward compatibility).

    This route maintains backward compatibility with the original API.
    For new code, prefer using /invoke/id/{tool_id}.

    Args:
        name: Tool name
        args: Input arguments for the tool

    Returns:
        ToolResult as dict

    Raises:
        HTTPException: If tool not found or invocation fails
    """
    try:
        registry = get_registry()

        # Find tool by name
        tool = registry.get_by_name(name)
        if not tool:
            # Also try treating name as tool_id for flexibility
            tool = registry.get(name)

        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{name}' not found"
            )

        # Invoke using executor
        executor = get_executor()
        result = await executor.invoke(tool.tool_id, args)
        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/invoke/id/{tool_id}')
async def invoke_by_id(tool_id: str, args: dict) -> Dict[str, Any]:
    """
    Invoke a tool by ID (preferred method).

    This is the recommended way to invoke tools as it uses the
    unique tool_id for lookup.

    Args:
        tool_id: Unique tool identifier
        args: Input arguments for the tool

    Returns:
        ToolResult as dict

    Raises:
        HTTPException: If tool not found or invocation fails
    """
    try:
        executor = get_executor()
        result = await executor.invoke(tool_id, args)

        # Check if invocation succeeded
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=result.error or "Tool invocation failed"
            )

        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# Tool Discovery Routes
# ==============================================================================

@router.post('/discover')
async def discover_tools() -> Dict[str, Any]:
    """
    Trigger tool discovery from AgentTools folder.

    Scans configured discovery directories for tool_manifest.json files
    and registers the discovered tools.

    Returns:
        Dict with status, discovered count, and list of tool IDs
    """
    try:
        registry = get_registry()
        discovered = registry.discover_local_tools()
        return {
            "status": "success",
            "discovered_count": len(discovered),
            "tools": [t.tool_id for t in discovered]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tool discovery failed: {str(e)}"
        )


@router.post('/discover/connectors')
async def discover_connectors() -> Dict[str, Any]:
    """
    Sync registered MCP connectors as tools.

    Gets all connectors from ConnectorManager and registers them
    as tools with tool_type=MCP in the ToolRegistry. This operation
    is idempotent - connectors that already have corresponding tools
    are skipped.

    Returns:
        Dict with:
            - status: "success" or "error"
            - synced_count: Number of newly synced tools
            - tools: List of newly synced tool_ids
            - skipped: List of tool_ids that already existed
            - errors: List of error messages for failed syncs
            - message: Optional error message if operation failed
    """
    try:
        registry = get_registry()
        result = registry.sync_connectors_as_tools()

        # Transform result to match API response format
        return {
            "status": result["status"],
            "synced_count": len(result["synced"]),
            "tools": result["synced"],
            "skipped": result["skipped"],
            "errors": result["errors"],
            "message": result.get("message")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Connector discovery failed: {str(e)}"
        )


# ==============================================================================
# Agent-Tool Integration Routes
# ==============================================================================

@router.get('/agent/{agent_id}')
async def get_agent_tools(agent_id: str) -> Dict[str, Any]:
    """
    Get tools assigned to an agent.

    Looks up the agent in the agent registry and returns
    all tools assigned to it.

    Args:
        agent_id: Agent identifier

    Returns:
        Dict with agent_id, tool_count, and list of tool definitions

    Raises:
        HTTPException: If agent not found
    """
    try:
        # Get agent from agent registry
        agent_registry = container.resolve('agent.registry')
        agent = agent_registry.get_agent(agent_id)

        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' not found"
            )

        # Get tool IDs from agent
        tool_ids = agent.get("tools", [])

        # Get tool definitions from tool registry
        tool_registry = get_registry()
        tools = tool_registry.get_tools_for_agent(tool_ids)

        return {
            "agent_id": agent_id,
            "agent_name": agent.get("name", ""),
            "tool_count": len(tools),
            "tools": [t.model_dump() for t in tools]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent tools: {str(e)}"
        )


# ==============================================================================
# Utility Routes
# ==============================================================================

@router.get('/health')
async def health_check() -> Dict[str, Any]:
    """
    Health check for the tool system.

    Returns:
        Dict with status and component health
    """
    try:
        registry = get_registry()
        executor = get_executor()

        return {
            "status": "healthy",
            "registry": {
                "tool_count": registry.count(),
                "discovery_dirs": len(registry.discovery_dirs)
            },
            "executor": {
                "cached_instances": len(executor.get_cached_instances()),
                "default_timeout": executor.default_timeout
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post('/cache/clear')
async def clear_cache() -> Dict[str, Any]:
    """
    Clear the tool executor's instance cache.

    Useful for reloading tool implementations during development.

    Returns:
        Dict with number of cleared instances
    """
    try:
        executor = get_executor()
        count = executor.clear_instance_cache()
        return {
            "status": "success",
            "cleared_instances": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post('/validate/{tool_id}')
async def validate_tool_input(tool_id: str, input_data: dict) -> Dict[str, Any]:
    """
    Validate input data against a tool's schema without executing.

    Useful for frontend validation before submission.

    Args:
        tool_id: Tool identifier
        input_data: Input data to validate

    Returns:
        Dict with validation status

    Raises:
        HTTPException: If tool not found or validation fails
    """
    try:
        registry = get_registry()

        # Check if tool exists
        tool = registry.get(tool_id)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_id}' not found"
            )

        # Validate input
        try:
            registry.validate_tool_input(tool_id, input_data)
            return {
                "valid": True,
                "tool_id": tool_id,
                "message": "Input validation passed"
            }
        except ValueError as e:
            return {
                "valid": False,
                "tool_id": tool_id,
                "error": str(e)
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )

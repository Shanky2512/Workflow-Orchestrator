"""
Tool Registry Module

This module provides the ToolRegistry class for managing tool registration,
discovery, and retrieval in the EchoAI system. It handles both local tools
(Python functions) and external tools (MCP servers, API endpoints).

The registry maintains an index of all available tools and provides methods
for querying, filtering, and validating tools for agent use.
"""

import asyncio
import json
import logging
import threading
import uuid as _uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from echolib.types import ToolDef, ToolType
from echolib.repositories.tool_repo import ToolRepository
from echolib.database import get_db_session
from echolib.repositories.base import DEFAULT_USER_ID

from .storage import ToolStorage

logger = logging.getLogger(__name__)

# ─── DB Dual-Write Constants ───
# Fixed UUID namespace for deterministic tool_id → UUID conversion.
# Tool.tool_id (DB) is UUID but ToolDef.tool_id is str (e.g. "tool_api_jira").
# uuid5(TOOL_NS, str) ensures the same string always maps to the same UUID.
_TOOL_NS = _uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# System user ID for tools synced at startup (no HTTP/user context).
# Reuses the DEFAULT_USER_ID already defined in echolib.repositories.base.
SYSTEM_USER_ID = DEFAULT_USER_ID


def _tool_id_to_uuid(tool_id_str: str) -> _uuid.UUID:
    """Deterministic UUID from a string tool_id. Idempotent and reversible."""
    return _uuid.uuid5(_TOOL_NS, tool_id_str)


class ToolRegistry:
    """
    Central registry for managing tools across the EchoAI platform.

    The ToolRegistry is responsible for:
    - Registering new tools (local, MCP, API)
    - Discovering available tools from configured locations
    - Providing tool metadata and schemas
    - Filtering tools by type, capability, or agent requirements
    - Validating tool inputs against schemas

    Attributes:
        storage (ToolStorage): Persistence layer for tool definitions
        discovery_dirs (List[Path]): Directories to scan for local tools
        _cache (Dict[str, ToolDef]): In-memory cache of registered tools
    """

    def __init__(
        self,
        storage: ToolStorage,
        discovery_dirs: Optional[List[Path]] = None,
        tool_repo: Optional[ToolRepository] = None
    ):
        """
        Initialize the ToolRegistry.

        Args:
            storage: ToolStorage instance for persistence
            discovery_dirs: Optional list of directories to scan for tool manifests
            tool_repo: Optional ToolRepository for DB dual-write persistence.
                       If provided, tools are written to both filesystem AND database.
                       DB writes are non-blocking — failures never block filesystem ops.
        """
        self.storage = storage
        self.discovery_dirs = discovery_dirs or []
        self._cache: Dict[str, ToolDef] = {}
        self._tool_repo: Optional[ToolRepository] = tool_repo

        # Load all existing tools into cache
        self._load_all()

        # Auto-sync connectors (API + MCP) as tools on startup
        try:
            sync_result = self.sync_connectors_as_tools()
            synced_count = len(sync_result.get("synced", []))
            if synced_count > 0:
                logger.info(f"Auto-synced {synced_count} connectors as tools on startup")
        except Exception as e:
            logger.warning(f"Connector auto-sync on startup skipped: {e}")

        logger.info(
            f"ToolRegistry initialized with {len(self._cache)} tools, "
            f"{len(self.discovery_dirs)} discovery directories"
            f"{', DB dual-write enabled' if self._tool_repo else ''}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # DB Dual-Write Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _schedule_db_write(self, coro) -> None:
        """
        Schedule an async DB coroutine from sync context. Non-blocking, fail-safe.

        Inside FastAPI (running event loop): uses loop.create_task() — fire-and-forget.
        Outside event loop (startup/CLI): spawns a daemon thread with its own loop.
        """
        if self._tool_repo is None:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._safe_db_op(coro))
        except RuntimeError:
            # No running event loop — spawn one in a daemon thread
            threading.Thread(
                target=lambda: asyncio.run(self._safe_db_op(coro)),
                daemon=True
            ).start()

    async def _safe_db_op(self, coro) -> None:
        """Execute an async DB operation with error handling. Never raises."""
        try:
            await coro
        except Exception as e:
            logger.warning(f"DB dual-write failed (non-blocking): {e}")

    async def _persist_tool_to_db(self, tool: ToolDef, user_id: str) -> None:
        """
        Persist a ToolDef to the database via ToolRepository.

        Handles upsert: checks existence first, then creates or updates.
        Maps ToolDef fields to Tool model columns with UUID conversion.
        """
        tool_uuid = str(_tool_id_to_uuid(tool.tool_id))

        tool_data = {
            "tool_id": _tool_id_to_uuid(tool.tool_id),
            "name": tool.name,
            "description": tool.description,
            "tool_type": tool.tool_type.value,
            "definition": tool.model_dump(mode="json"),
            "status": tool.status or "active",
            "version": tool.version or "1.0",
            "tags": tool.tags or [],
        }

        async with get_db_session() as db:
            existing = await self._tool_repo.get_by_id(db, tool_uuid, user_id)
            if existing:
                # Update — remove tool_id from updates (it's the PK, not updatable)
                update_data = {k: v for k, v in tool_data.items() if k != "tool_id"}
                await self._tool_repo.update(db, tool_uuid, user_id, update_data)
                logger.debug(f"DB dual-write: updated tool '{tool.tool_id}'")
            else:
                await self._tool_repo.create(db, user_id, tool_data)
                logger.debug(f"DB dual-write: created tool '{tool.tool_id}'")

    async def _delete_tool_from_db(self, tool_id: str, user_id: str) -> None:
        """Soft-delete a tool from the database via ToolRepository."""
        tool_uuid = str(_tool_id_to_uuid(tool_id))

        async with get_db_session() as db:
            deleted = await self._tool_repo.delete(db, tool_uuid, user_id)
            if deleted:
                logger.debug(f"DB dual-write: soft-deleted tool '{tool_id}'")
            else:
                logger.debug(f"DB dual-write: tool '{tool_id}' not found in DB (OK)")

    # ──────────────────────────────────────────────────────────────────────

    def register(self, tool: ToolDef, user_id: Optional[str] = None) -> Dict[str, str]:
        """
        Register a new tool or update an existing one.

        Persists to filesystem (primary) and database (secondary, non-blocking).
        DB failure never blocks registration — filesystem is the source of truth.

        Args:
            tool: ToolDef instance to register
            user_id: Optional owner user ID for DB persistence.
                     Defaults to SYSTEM_USER_ID for startup sync / CLI usage.

        Returns:
            Dict with registration result (tool_id, status, message)

        Raises:
            ValueError: If tool definition is invalid
        """
        # Validate required fields
        if not tool.name:
            raise ValueError("Tool must have a name")

        if not tool.description:
            raise ValueError("Tool must have a description")

        # Ensure tool_id is set
        if not tool.tool_id:
            tool_id = f"tool_{tool.name.lower().replace(' ', '_').replace('-', '_')}"
            # Create a new tool with the generated ID
            tool = ToolDef(
                tool_id=tool_id,
                name=tool.name,
                description=tool.description,
                tool_type=tool.tool_type,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                execution_config=tool.execution_config,
                version=tool.version,
                tags=tool.tags,
                status=tool.status,
                metadata=tool.metadata
            )

        # Check if updating existing tool
        is_update = tool.tool_id in self._cache

        try:
            # Persist to filesystem (primary store)
            self.storage.save_tool(tool)

            # Update cache
            self._cache[tool.tool_id] = tool

            status = "updated" if is_update else "registered"
            logger.info(f"Tool '{tool.tool_id}' {status}")

            # Dual-write to DB (non-blocking, fail-safe)
            if self._tool_repo is not None:
                resolved_user_id = user_id or SYSTEM_USER_ID
                self._schedule_db_write(
                    self._persist_tool_to_db(tool, resolved_user_id)
                )

            return {
                "tool_id": tool.tool_id,
                "status": status,
                "message": f"Tool '{tool.name}' {status} successfully"
            }

        except Exception as e:
            logger.error(f"Failed to register tool '{tool.name}': {e}")
            raise ValueError(f"Failed to register tool: {e}")

    def get(self, tool_id: str) -> Optional[ToolDef]:
        """
        Retrieve a tool definition by its ID.

        Args:
            tool_id: Unique identifier of the tool

        Returns:
            ToolDef instance if found, None otherwise
        """
        # Try cache first
        if tool_id in self._cache:
            return self._cache[tool_id]

        # Try loading from storage (in case cache is stale)
        tool = self.storage.load_tool(tool_id)
        if tool:
            self._cache[tool_id] = tool
            return tool

        logger.debug(f"Tool '{tool_id}' not found")
        return None

    def get_by_name(self, name: str) -> Optional[ToolDef]:
        """
        Retrieve a tool definition by its name.

        Args:
            name: Human-readable name of the tool

        Returns:
            ToolDef instance if found, None otherwise
        """
        name_lower = name.lower()
        for tool in self._cache.values():
            if tool.name.lower() == name_lower:
                return tool
        return None

    def get_tool_id_by_name(self, name: str) -> Optional[str]:
        """
        Get tool_id by tool name.

        This method supports frontend tool resolution where tools
        are referenced by name.

        Args:
            name: Human-readable name of the tool

        Returns:
            tool_id if found, None otherwise
        """
        tool = self.get_by_name(name)
        return tool.tool_id if tool else None

    def list_all(self) -> List[ToolDef]:
        """
        List all registered tools.

        Returns:
            List of all ToolDef instances in the registry
        """
        return list(self._cache.values())

    def list_by_type(self, tool_type: ToolType) -> List[ToolDef]:
        """
        List all tools of a specific type.

        Args:
            tool_type: ToolType enum value to filter by

        Returns:
            List of ToolDef instances matching the specified type
        """
        return [
            tool for tool in self._cache.values()
            if tool.tool_type == tool_type
        ]

    def list_by_status(self, status: str) -> List[ToolDef]:
        """
        List all tools with a specific status.

        Args:
            status: Status string to filter by (active, deprecated, disabled)

        Returns:
            List of ToolDef instances with the specified status
        """
        return [
            tool for tool in self._cache.values()
            if tool.status == status.lower()
        ]

    def list_by_tags(self, tags: List[str]) -> List[ToolDef]:
        """
        List tools that have any of the specified tags.

        Args:
            tags: List of tags to filter by

        Returns:
            List of ToolDef instances that have at least one matching tag
        """
        tags_lower = {tag.lower() for tag in tags}
        return [
            tool for tool in self._cache.values()
            if any(tag.lower() in tags_lower for tag in tool.tags)
        ]

    def delete(self, tool_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a tool from the registry.

        Removes from filesystem (primary) and soft-deletes in DB (secondary, non-blocking).

        Args:
            tool_id: Unique identifier of the tool to delete
            user_id: Optional owner user ID for DB soft-delete.
                     Defaults to SYSTEM_USER_ID.

        Returns:
            True if deletion successful, False if tool not found
        """
        if tool_id not in self._cache:
            logger.warning(f"Tool '{tool_id}' not found for deletion")
            return False

        try:
            # Delete from filesystem (primary store)
            success = self.storage.delete_tool(tool_id)

            if success:
                # Remove from cache
                del self._cache[tool_id]
                logger.info(f"Tool '{tool_id}' deleted from registry")

                # Dual-delete from DB (non-blocking, fail-safe)
                if self._tool_repo is not None:
                    resolved_user_id = user_id or SYSTEM_USER_ID
                    self._schedule_db_write(
                        self._delete_tool_from_db(tool_id, resolved_user_id)
                    )

            return success

        except Exception as e:
            logger.error(f"Failed to delete tool '{tool_id}': {e}")
            return False

    def discover_local_tools(self) -> List[ToolDef]:
        """
        Discover and register tools from configured discovery directories.

        Scans each directory in discovery_dirs for tool_manifest.json files
        and automatically registers the discovered tools.

        Discovery is idempotent - tools that are already registered will be
        skipped to prevent duplicates. Errors in individual tool manifests
        will not stop discovery of other tools.

        Returns:
            List of newly discovered ToolDef instances
        """
        discovered = []

        if not self.discovery_dirs:
            logger.info("No discovery directories configured")
            return discovered

        for discovery_dir in self.discovery_dirs:
            discovery_path = Path(discovery_dir)

            if not discovery_path.exists():
                logger.warning(f"Discovery directory does not exist: {discovery_path}")
                continue

            logger.info(f"Scanning directory for tools: {discovery_path}")

            # Scan for tool_manifest.json in each subdirectory
            for tool_dir in discovery_path.iterdir():
                if not tool_dir.is_dir():
                    continue

                manifest_path = tool_dir / "tool_manifest.json"
                if not manifest_path.exists():
                    logger.debug(f"No manifest found in {tool_dir.name}")
                    continue

                try:
                    # Load and parse manifest
                    tool = self._load_manifest(manifest_path)
                    if not tool:
                        continue

                    # Check if tool already registered (idempotent discovery)
                    existing = self.get(tool.tool_id)
                    if existing:
                        logger.info(f"Tool {tool.tool_id} already registered, skipping")
                        continue

                    # Register the discovered tool
                    self.register(tool)
                    discovered.append(tool)
                    logger.info(
                        f"Discovered and registered tool: {tool.tool_id} ({tool.name})"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load tool from {manifest_path}: {e}"
                    )
                    continue

        logger.info(f"Discovery complete. Found {len(discovered)} new tools")
        return discovered

    def _load_manifest(self, manifest_path: Path) -> Optional[ToolDef]:
        """
        Load a tool definition from a manifest file.

        Args:
            manifest_path: Path to tool_manifest.json file

        Returns:
            ToolDef instance if valid, None otherwise

        Raises:
            ValueError: If manifest is invalid (handled internally, returns None)
        """
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert tool_type string to enum if present
            if 'tool_type' in data and isinstance(data['tool_type'], str):
                tool_type_str = data['tool_type'].upper()
                # Try to get enum value, default to LOCAL if not found
                try:
                    data['tool_type'] = ToolType[tool_type_str]
                except KeyError:
                    # Fallback: try by value (e.g., "local" -> ToolType.LOCAL)
                    try:
                        data['tool_type'] = ToolType(data['tool_type'].lower())
                    except ValueError:
                        logger.warning(
                            f"Unknown tool_type '{data['tool_type']}' in {manifest_path}, "
                            f"defaulting to LOCAL"
                        )
                        data['tool_type'] = ToolType.LOCAL

            tool_def = ToolDef(**data)
            return tool_def

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in manifest {manifest_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse manifest {manifest_path}: {e}")
            return None

    def get_tools_for_agent(self, tool_ids: List[str]) -> List[ToolDef]:
        """
        Get all tools required by an agent.

        Args:
            tool_ids: List of tool identifiers the agent needs

        Returns:
            List of ToolDef instances for the requested tools

        Note:
            Missing tools are logged as warnings but don't raise errors.
            This allows workflows to continue with available tools.
        """
        tools = []

        for tool_id in tool_ids:
            tool = self.get(tool_id)
            if tool:
                # Only include active tools
                if tool.status == "active":
                    tools.append(tool)
                else:
                    logger.warning(
                        f"Tool '{tool_id}' is {tool.status}, skipping"
                    )
            else:
                # Try finding by name as fallback (for frontend compatibility)
                tool = self.get_by_name(tool_id)
                if tool and tool.status == "active":
                    tools.append(tool)
                else:
                    logger.warning(f"Tool '{tool_id}' not found for agent")

        return tools

    def validate_tool_input(self, tool_id: str, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against a tool's input schema.

        Uses JSON Schema validation to check if input_data conforms to
        the tool's declared input_schema.

        Args:
            tool_id: ID of the tool to validate against
            input_data: Input data to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails with details
            KeyError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise KeyError(f"Tool '{tool_id}' not found")

        # If no input schema defined, accept any input
        if not tool.input_schema:
            return True

        try:
            # Use jsonschema for validation
            import jsonschema
            jsonschema.validate(instance=input_data, schema=tool.input_schema)
            return True

        except jsonschema.ValidationError as e:
            error_msg = f"Input validation failed for tool '{tool_id}': {e.message}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        except jsonschema.SchemaError as e:
            error_msg = f"Invalid schema for tool '{tool_id}': {e.message}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_tool_output(self, tool_id: str, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data against a tool's output schema.

        Args:
            tool_id: ID of the tool to validate against
            output_data: Output data to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
            KeyError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise KeyError(f"Tool '{tool_id}' not found")

        # If no output schema defined, accept any output
        if not tool.output_schema:
            return True

        try:
            import jsonschema
            jsonschema.validate(instance=output_data, schema=tool.output_schema)
            return True

        except jsonschema.ValidationError as e:
            error_msg = f"Output validation failed for tool '{tool_id}': {e.message}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

    def _load_all(self) -> None:
        """
        Load all tools from storage into the cache.

        Called during initialization to populate the in-memory cache.
        """
        try:
            tools = self.storage.load_all()
            for tool in tools:
                self._cache[tool.tool_id] = tool

            logger.debug(f"Loaded {len(tools)} tools into cache")

        except Exception as e:
            logger.error(f"Failed to load tools from storage: {e}")
            # Continue with empty cache rather than failing initialization

    def refresh_cache(self) -> int:
        """
        Refresh the in-memory cache from storage and re-sync connectors.

        Returns:
            Number of tools loaded (including synced connector tools)
        """
        self._cache.clear()
        self._load_all()

        # Re-sync connectors to pick up newly registered API/MCP connectors
        try:
            self.sync_connectors_as_tools()
        except Exception as e:
            logger.warning(f"Connector re-sync during refresh skipped: {e}")

        return len(self._cache)

    def count(self) -> int:
        """
        Get the number of registered tools.

        Returns:
            Total count of tools in registry
        """
        return len(self._cache)

    def search(self, query: str) -> List[ToolDef]:
        """
        Search tools by name or description.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching ToolDef instances
        """
        query_lower = query.lower()
        return [
            tool for tool in self._cache.values()
            if query_lower in tool.name.lower()
            or query_lower in tool.description.lower()
        ]

    def sync_connectors_as_tools(self) -> Dict[str, Any]:
        """
        Sync registered API and MCP connectors as tools in the registry.

        Queries ConnectorManager.api.list() and ConnectorManager.mcp.list()
        separately (ConnectorManager has no unified .list()), then converts
        each connector dict into a ToolDef with:
        - Correct tool_type: ToolType.API for API connectors, ToolType.MCP for MCP
        - input_schema derived from the connector's creation_payload
        - Auto-generated system prompt for agent consumption
        - execution_config that stores the connector_id for runtime lookup

        This operation is idempotent — connectors with existing tools are skipped.

        Returns:
            Dict with status, synced, skipped, and errors lists.
        """
        from echolib.di import container

        synced: List[str] = []
        skipped: List[str] = []
        errors: List[str] = []

        # ─── Resolve ConnectorManager from DI ───
        try:
            connector_manager = container.resolve('connector.manager')
        except KeyError:
            logger.warning("ConnectorManager not available in DI container")
            return {
                "status": "error",
                "synced": [],
                "skipped": [],
                "errors": [],
                "message": "ConnectorManager not available. Ensure 'connector.manager' is registered in the DI container."
            }

        # ─── Fetch ALL connectors: API + MCP separately ───
        all_connectors: List[Dict[str, Any]] = []

        # API connectors — cm.api.list() → {"success": True, "count": N, "connectors": [...]}
        try:
            api_result = connector_manager.api.list()
            if api_result.get("success"):
                for conn in api_result.get("connectors", []):
                    conn["_connector_source"] = "api"
                all_connectors.extend(api_result.get("connectors", []))
                logger.info(f"Fetched {api_result.get('count', 0)} API connectors")
        except Exception as e:
            error_msg = f"Failed to list API connectors: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        # MCP connectors — cm.mcp.list() → {"success": True, "count": N, "connectors": [...]}
        try:
            mcp_result = connector_manager.mcp.list()
            if mcp_result.get("success"):
                for conn in mcp_result.get("connectors", []):
                    conn["_connector_source"] = "mcp"
                all_connectors.extend(mcp_result.get("connectors", []))
                logger.info(f"Fetched {mcp_result.get('count', 0)} MCP connectors")
        except Exception as e:
            error_msg = f"Failed to list MCP connectors: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        logger.info(f"Total connectors to sync: {len(all_connectors)}")

        # ─── Convert each connector dict into a ToolDef ───
        for connector in all_connectors:
            try:
                source = connector.get("_connector_source", "api")
                connector_id = connector.get("connector_id", "")
                connector_name = connector.get("connector_name") or connector.get("name", connector_id)
                connector_desc = connector.get("description", "") or connector.get("creation_payload", {}).get("description", "")

                # Build a stable, unique tool_id
                safe_name = connector_name.lower().replace(" ", "_").replace("-", "_")
                tool_id = f"tool_{source}_{safe_name}"

                # Idempotent check — skip if tool exists AND connector_id unchanged
                existing = self.get(tool_id)
                if existing:
                    existing_conn_id = (existing.execution_config or {}).get("connector_id", "")
                    if existing_conn_id == connector_id:
                        logger.debug(f"Tool {tool_id} already exists, skipping")
                        skipped.append(tool_id)
                        continue
                    # Connector definition changed — fall through to re-register (updates both FS + DB)

                # ── Schema Mapping ──
                # Derive input_schema from the connector's creation_payload
                creation_payload = connector.get("creation_payload", {})
                raw_input_schema = creation_payload.get("input_schema", {})
                example_payload = creation_payload.get("example_payload", {})

                if raw_input_schema and isinstance(raw_input_schema, dict):
                    input_schema = raw_input_schema
                elif example_payload and isinstance(example_payload, dict):
                    # Infer a basic schema from the example payload keys
                    props = {
                        k: {"type": _infer_json_type(v)}
                        for k, v in example_payload.items()
                    }
                    input_schema = {
                        "type": "object",
                        "properties": props,
                    }
                else:
                    input_schema = {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "object", "description": "Request payload"}
                        }
                    }

                output_schema = creation_payload.get("output_schema", {"type": "object"})

                # ── System Prompt Generation ──
                system_prompt = (
                    f"This tool allows you to interact with {connector_name}. "
                    f"{connector_desc + '. ' if connector_desc else ''}"
                    f"Use it when you need to call the {connector_name} "
                    f"{'API' if source == 'api' else 'MCP service'}."
                )

                # ── Tool Type ──
                tool_type = ToolType.API if source == "api" else ToolType.MCP

                # ── Tags ──
                tags = [source, "connector", "synced"]
                if connector.get("transport_type"):
                    tags.append(connector["transport_type"])

                # ── Create ToolDef ──
                tool = ToolDef(
                    tool_id=tool_id,
                    name=connector_name,
                    description=system_prompt,
                    tool_type=tool_type,
                    input_schema=input_schema,
                    output_schema=output_schema if isinstance(output_schema, dict) else {"type": "object"},
                    execution_config={
                        "connector_id": connector_id,
                        "connector_source": source,
                        "source": "connector_sync",
                    },
                    status="active",
                    tags=tags,
                    metadata={
                        "connector_id": connector_id,
                        "connector_name": connector_name,
                        "connector_source": source,
                        "system_prompt": system_prompt,
                    }
                )

                self.register(tool, user_id=SYSTEM_USER_ID)
                synced.append(tool_id)
                logger.info(f"Synced {source.upper()} connector '{connector_name}' → tool '{tool_id}'")

            except Exception as e:
                cname = connector.get("name", connector.get("connector_id", "unknown"))
                error_msg = f"Failed to sync connector '{cname}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        logger.info(
            f"Connector sync complete: {len(synced)} synced, "
            f"{len(skipped)} skipped, {len(errors)} errors"
        )

        return {
            "status": "success",
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
        }


def _infer_json_type(value: Any) -> str:
    """
    Infer a JSON Schema type string from a Python value.

    Used when generating input_schema from example_payload.
    """
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return "string"

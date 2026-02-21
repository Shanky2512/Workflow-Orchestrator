"""
Node Mapper: Bidirectional conversion between frontend canvas and backend workflow schema.
Handles all 16 node types with layout persistence and connection preservation.
"""
import json
import os
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from echolib.utils import new_id


def normalize_agent_config(agent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure agent dict has a top-level 'config' key derived from metadata.

    Handles agents saved before the node_mapper fix where config was
    only stored in metadata.*_config with no top-level 'config' key.
    Idempotent — safe to call on already-normalized agents.

    Args:
        agent: Agent definition dict (modified in-place and returned)

    Returns:
        The same agent dict, with 'config' key guaranteed present for
        API, Code, and MCP node types.
    """
    # Skip if config already present and non-empty
    if agent.get("config") and isinstance(agent["config"], dict) and len(agent["config"]) > 0:
        return agent

    metadata = agent.get("metadata") or {}
    node_type = metadata.get("node_type", "")

    if node_type == "Code":
        agent["config"] = metadata.get("code_config", {})
    elif node_type == "API":
        agent["config"] = metadata.get("api_config", {})
    elif node_type == "MCP":
        agent["config"] = metadata.get("mcp_config", {})
    elif node_type == "Conditional":
        agent["config"] = {"branches": metadata.get("branches", [])}
    elif node_type == "Loop":
        agent["config"] = metadata.get("loop_config", {})
    elif node_type == "HITL":
        agent["config"] = metadata.get("hitl_config", {})
    elif node_type == "Map":
        agent["config"] = metadata.get("map_config", {})
    elif node_type == "Self-Review":
        agent["config"] = metadata.get("review_config", {})

    return agent


# Module-level cache for tool registry
_TOOL_REGISTRY_CACHE: Dict[str, Dict[str, Any]] = {}
_TOOL_REGISTRY_LOADED: bool = False


def _get_tools_storage_path() -> str:
    """Get path to tools storage directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from visualization/ to apps/storage/tools/
    return os.path.join(current_dir, "..", "..", "storage", "tools")


def _load_tool_registry() -> Dict[str, Dict[str, Any]]:
    """
    Load tool registry from tool_index.json and individual tool files.
    Returns dict mapping tool_id -> {name, description, tool_type, tags, ...}
    """
    global _TOOL_REGISTRY_CACHE, _TOOL_REGISTRY_LOADED

    if _TOOL_REGISTRY_LOADED and _TOOL_REGISTRY_CACHE:
        return _TOOL_REGISTRY_CACHE

    tools_path = _get_tools_storage_path()
    index_path = os.path.join(tools_path, "tool_index.json")

    try:
        # Load tool index
        if not os.path.exists(index_path):
            return {}

        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        tools_dict = index_data.get("tools", {})

        # Load full details from individual tool files
        for tool_id, tool_info in tools_dict.items():
            tool_file = os.path.join(tools_path, f"{tool_id}.json")
            if os.path.exists(tool_file):
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        full_tool = json.load(f)
                    # Merge full details
                    _TOOL_REGISTRY_CACHE[tool_id] = {
                        "tool_id": tool_id,
                        "name": full_tool.get("name", tool_info.get("name", tool_id)),
                        "description": full_tool.get("description", ""),
                        "tool_type": full_tool.get("tool_type", tool_info.get("tool_type", "local")),
                        "tags": full_tool.get("tags", tool_info.get("tags", [])),
                        "version": full_tool.get("version", tool_info.get("version", "1.0")),
                        "status": tool_info.get("status", "active")
                    }
                except Exception:
                    # Use index info as fallback
                    _TOOL_REGISTRY_CACHE[tool_id] = {
                        "tool_id": tool_id,
                        "name": tool_info.get("name", tool_id),
                        "description": "",
                        "tool_type": tool_info.get("tool_type", "local"),
                        "tags": tool_info.get("tags", []),
                        "version": tool_info.get("version", "1.0"),
                        "status": tool_info.get("status", "active")
                    }
            else:
                # Use index info
                _TOOL_REGISTRY_CACHE[tool_id] = {
                    "tool_id": tool_id,
                    "name": tool_info.get("name", tool_id),
                    "description": "",
                    "tool_type": tool_info.get("tool_type", "local"),
                    "tags": tool_info.get("tags", []),
                    "version": tool_info.get("version", "1.0"),
                    "status": tool_info.get("status", "active")
                }

        _TOOL_REGISTRY_LOADED = True
        return _TOOL_REGISTRY_CACHE

    except Exception as e:
        print(f"Warning: Could not load tool registry: {e}")
        return {}


class NodeMapper:
    """
    Bidirectional mapper for frontend canvas ↔ backend workflow schema.
    """

    # Node type to color mapping
    NODE_COLORS = {
        "Start": "#10b981",
        "End": "#64748b",
        "Agent": "#f59e0b",
        "Subagent": "#f59e0b",
        "Prompt": "#ec4899",
        "Conditional": "#8b5cf6",
        "Loop": "#8b5cf6",
        "Map": "#8b5cf6",
        "Self-Review": "#06b6d4",
        "HITL": "#06b6d4",
        "API": "#3b82f6",
        "MCP": "#3b82f6",
        "Code": "#10b981",
        "Template": "#10b981",
        "Failsafe": "#ef4444",
        "Merge": "#64748b"
    }

    # Node type to icon mapping
    NODE_ICONS = {
        "Start": "▶️",
        "End": "⏹️",
        "Agent": "🔶",
        "Subagent": "👥",
        "Prompt": "💬",
        "Conditional": "🔀",
        "Loop": "🔄",
        "Map": "🔁",
        "Self-Review": "✅",
        "HITL": "👤",
        "API": "🌐",
        "MCP": "🔌",
        "Code": "💻",
        "Template": "📝",
        "Failsafe": "🛡️",
        "Merge": "⚡"
    }

    def __init__(self, tool_registry=None):
        """
        Initialize mapper.

        Args:
            tool_registry: Tool registry for resolving tool names → IDs
        """
        self.tool_registry = tool_registry

    # ==================== FRONTEND → BACKEND ====================

    def map_frontend_to_backend(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        workflow_name: Optional[str] = None,
        auto_generate_name: bool = True,
        execution_model: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Convert frontend canvas to backend workflow JSON.

        Args:
            canvas_nodes: Frontend node list
            connections: Frontend connection list
            workflow_name: Explicit workflow name (optional)
            auto_generate_name: Generate name from first agent if not provided
            workflow_id: Existing workflow ID to preserve (optional)

        Returns:
            Tuple of (workflow_dict, agents_dict)

        Raises:
            ValueError: If validation fails
        """
        # Validate Start node
        self._validate_start_node(canvas_nodes)

        # Use existing workflow ID if provided, otherwise generate new
        if not workflow_id:
            workflow_id = new_id("wf_")

        # Determine workflow name
        if not workflow_name and auto_generate_name:
            workflow_name = self._auto_generate_workflow_name(canvas_nodes)
        elif not workflow_name:
            workflow_name = "Untitled Workflow"

        # Convert nodes to agents
        agents_dict = {}
        agent_ids = []
        id_mapping = {}  # frontend_id → backend_agent_id

        for node in canvas_nodes:
            # PRESERVE existing agent_id if present (from backend_id or config)
            # This prevents duplicate agent creation when re-saving workflows
            existing_agent_id = node.get("backend_id") or node.get("config", {}).get("agent_id")

            if existing_agent_id and isinstance(existing_agent_id, str) and existing_agent_id.startswith("agt_"):
                # Use existing agent_id - UPDATE instead of CREATE
                agent_id = existing_agent_id
            else:
                # Generate new ID only for truly new agents
                agent_id = new_id("agt_")

            agent = self._convert_node_to_agent(node)
            agent["agent_id"] = agent_id  # Set the agent_id (existing or new)
            agents_dict[agent_id] = agent
            agent_ids.append(agent_id)
            id_mapping[node["id"]] = agent_id

        # Convert connections (preserve for arrow rendering)
        backend_connections = []
        for conn in connections:
            connection = {
                "from": id_mapping[conn["from"]],
                "to": id_mapping[conn["to"]]
            }
            # Only add condition if it exists (optional field, must be string if present, not None)
            if conn.get("condition"):
                connection["condition"] = conn.get("condition")
            backend_connections.append(connection)

        # Always infer execution model from structure
        # (frontend value may be incorrect for imported workflows that default to 'sequential')
        execution_model = self._infer_execution_model(canvas_nodes, connections)

        # Extract state schema from Start/End nodes
        state_schema = self._extract_state_schema(canvas_nodes)

        # Extract HITL configuration
        hitl_config = self._extract_hitl(canvas_nodes, id_mapping)

        # Build hierarchy if hierarchical (omit if not hierarchical)
        hierarchy_config = {}
        if execution_model == "hierarchical":
            hierarchy_config = self._build_hierarchy(canvas_nodes, connections, id_mapping)

        # Resolve branch targetNodeId from frontend canvas IDs to backend agent IDs
        for agent_id, agent in agents_dict.items():
            branches = agent.get("metadata", {}).get("branches", [])
            for branch in branches:
                frontend_target = branch.get("targetNodeId")
                if frontend_target is not None:
                    resolved = id_mapping.get(str(frontend_target))
                    if resolved is None:
                        # Try numeric lookup (frontend IDs may be int or str)
                        resolved = id_mapping.get(int(frontend_target)) if str(frontend_target).isdigit() else None
                    if resolved:
                        branch["targetNodeId"] = resolved

        # Build workflow with EMBEDDED agent configurations (not just IDs)
        # This ensures all agent config is preserved in the workflow JSON itself
        embedded_agents = []
        for agent_id in agent_ids:
            agent_config = agents_dict[agent_id].copy()
            # Ensure agent_id is set in the embedded config
            agent_config["agent_id"] = agent_id
            embedded_agents.append(agent_config)

        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "description": f"Created from canvas on {datetime.utcnow().strftime('%Y-%m-%d')}",
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": embedded_agents,  # Full agent objects, not just IDs
            "connections": backend_connections,
            "state_schema": state_schema,
            "metadata": {
                "created_by": "workflow_builder",
                "created_at": datetime.utcnow().isoformat(),
                "canvas_layout": {
                    "width": 5000,
                    "height": 5000
                }
            }
        }

        # Add optional fields only if they have content (must be objects, not None)
        if hierarchy_config:
            workflow["hierarchy"] = hierarchy_config
        if hitl_config and hitl_config.get("enabled"):
            workflow["human_in_loop"] = hitl_config

        return workflow, agents_dict

    def _validate_start_node(self, canvas_nodes: List[Dict[str, Any]]) -> None:
        """
        Validate that Start node exists and is properly positioned.

        Raises:
            ValueError: If Start node missing or invalid
        """
        start_nodes = [n for n in canvas_nodes if n.get("type") == "Start"]

        if not start_nodes:
            raise ValueError("Workflow must have a Start node")

        if len(start_nodes) > 1:
            raise ValueError("Workflow can only have one Start node")

        # Start should be the entry point (no incoming connections would be validated later)

    def _auto_generate_workflow_name(self, canvas_nodes: List[Dict[str, Any]]) -> str:
        """
        Auto-generate workflow name from first agent.

        Args:
            canvas_nodes: Frontend node list

        Returns:
            Generated workflow name
        """
        # Try to find first meaningful agent
        for node in canvas_nodes:
            if node.get("type") in ["Agent", "Subagent"]:
                agent_name = node.get("name", "Agent")
                return f"{agent_name} Workflow"

        # Fallback
        return f"Workflow {datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    def _convert_node_to_agent(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert frontend node to backend agent definition.
        Handles all 16 node types.

        Uses the standard agent storage schema:
        {
            "agent_id", "name", "role", "description", "prompt", "icon",
            "model" (string), "tools" (array), "variables" (array),
            "settings" (object), "permissions", "metadata"
        }

        Args:
            node: Frontend node

        Returns:
            Backend agent definition matching storage schema
        """
        node_type = node.get("type", "Agent")
        config = node.get("config", {})

        # Extract prompt - check both direct config and nested structures
        prompt = config.get("prompt", "")
        if not prompt:
            prompt = config.get("systemPrompt", "") or config.get("instructions", "")

        # Extract goal - use config goal or derive from name/role
        goal = config.get("goal", "")
        if not goal and node_type in ["Agent", "Subagent"]:
            node_name = node.get("name", node_type)
            goal = f"Complete the task as {node_name}"

        # Extract model config - handle both object and string formats
        model_config = config.get("model", {})
        if isinstance(model_config, str):
            model_name = model_config
            provider = ""
        else:
            model_name = model_config.get("modelName") or model_config.get("model", "")
            provider = model_config.get("provider", "")

        # Extract settings from config
        max_iterations = config.get("maxIterations", 5)
        temperature = config.get("temperature", 0.7)
        top_p = config.get("topP", 0.9)
        max_tokens = config.get("maxTokens", 4000)

        # Build agent with CORRECT storage schema
        agent = {
            "agent_id": "",  # Will be set by caller
            "name": node.get("name", node_type),
            "role": config.get("role") or self._get_node_role(node_type),
            "description": config.get("description") or goal or f"{node_type} node",
            "prompt": prompt,
            "icon": node.get("icon", self.NODE_ICONS.get(node_type, "🔶")),
            "model": model_name,  # String, not object - matches storage schema
            "tools": self._resolve_tools(config.get("tools", [])),
            "variables": config.get("variables", []),
            "settings": {
                "temperature": temperature,
                "max_token": max_tokens,
                "top_p": top_p,
                "max_iteration": max_iterations,
                "provider": provider  # Store provider in settings
            },
            "permissions": {
                "can_call_agents": node_type == "Subagent",
                "allowed_agents": []
            },
            "metadata": {
                "node_type": node_type,
                "goal": goal,  # Store goal in metadata for retrieval
                "ui_layout": {
                    "x": node.get("x", 300),
                    "y": node.get("y", 200),
                    "icon": node.get("icon", self.NODE_ICONS.get(node_type, "🔶")),
                    "color": node.get("color", self.NODE_COLORS.get(node_type, "#64748b"))
                },
                "created_at": datetime.utcnow().isoformat()
            },
            # Keep these for workflow execution compatibility
            "input_schema": [],
            "output_schema": []
        }

        # Node-type specific handling
        if node_type in ["Agent", "Subagent", "Prompt"]:
            agent["tools"] = self._resolve_tools(config.get("tools", []))

        elif node_type == "Start":
            agent["input_schema"] = [
                var.get("name") for var in config.get("inputVariables", [])
            ]

        elif node_type == "End":
            agent["output_schema"] = [
                var.get("name") for var in config.get("outputVariables", [])
            ]

        elif node_type == "Conditional":
            agent["metadata"]["branches"] = config.get("branches", [])

        elif node_type == "Loop":
            agent["metadata"]["loop_config"] = {
                "loopType": config.get("loopType", "for-each"),
                "arrayVariable": config.get("arrayVariable"),
                "maxIterations": config.get("maxIterations", 100)
            }

        elif node_type == "Map":
            agent["metadata"]["map_config"] = {
                "operation": config.get("operation"),
                "maxConcurrency": config.get("maxConcurrency", 5)
            }

        elif node_type == "HITL":
            agent["metadata"]["hitl_config"] = {
                "title": config.get("title", ""),
                "message": config.get("message", ""),
                "priority": config.get("priority", "medium"),
                "allowEdit": config.get("allowEdit", True),
                "allowDefer": config.get("allowDefer", False)
            }

        elif node_type == "API":
            agent["metadata"]["api_config"] = {
                "method": config.get("method", "GET"),
                "url": config.get("url", ""),
                "headers": config.get("headers", {}),
                "auth": config.get("auth", {})
            }

        elif node_type == "MCP":
            # Parse argumentsRaw (JSON textarea string) → arguments dict
            args_raw = config.get("argumentsRaw", "")
            if isinstance(args_raw, str) and args_raw.strip():
                try:
                    parsed_args = json.loads(args_raw)
                except Exception:
                    parsed_args = {}
            else:
                parsed_args = config.get("arguments") or config.get("payload") or {}

            # Derive transport_type from serverId
            server_id = config.get("serverId", "")
            if server_id == "azure":
                transport_type = "stdio"
            else:
                transport_type = config.get("transport_type", "http")

            agent["metadata"]["mcp_config"] = {
                "serverId": server_id,
                "serverName": config.get("serverName", ""),
                "toolName": config.get("toolName", ""),
                "endpoint_url": config.get("endpoint_url") or config.get("url", ""),
                "method": config.get("method", "POST"),
                "headers": config.get("headers", {}),
                "auth_config": config.get("auth_config") or config.get("auth", {}),
                "connector_id": config.get("connector_id", ""),
                "transport_type": transport_type,
                "command": config.get("command", ""),
                "timeout": config.get("timeout", 30),
                "arguments": parsed_args,
            }

        elif node_type == "Code":
            agent["metadata"]["code_config"] = {
                "language": config.get("language", "python"),
                "code": config.get("code", ""),
                "packages": config.get("packages", "")
            }

        elif node_type == "Self-Review":
            agent["metadata"]["review_config"] = {
                "checkCompleteness": config.get("checkCompleteness", True),
                "checkAccuracy": config.get("checkAccuracy", True),
                "confidenceThreshold": config.get("confidenceThreshold", 0.8)
            }

        # Populate top-level "config" for compiler compatibility
        # The compiler reads node_config.get("config", node_config) to extract
        # node-type-specific settings (code, url, method, etc.)
        if node_type == "Code":
            agent["config"] = agent["metadata"].get("code_config", {})
        elif node_type == "API":
            agent["config"] = agent["metadata"].get("api_config", {})
        elif node_type == "MCP":
            agent["config"] = agent["metadata"].get("mcp_config", {})

        return agent

    def _get_node_role(self, node_type: str) -> str:
        """Get role description for node type."""
        roles = {
            "Start": "Workflow entry point",
            "End": "Workflow exit point",
            "Agent": "Autonomous AI agent",
            "Subagent": "Specialist delegation",
            "Prompt": "Direct LLM call",
            "Conditional": "Conditional branching",
            "Loop": "Iteration logic",
            "Map": "Parallel execution",
            "Self-Review": "Quality validation",
            "HITL": "Human approval gate",
            "API": "HTTP request",
            "MCP": "MCP tool execution",
            "Code": "Code execution",
            "Template": "String templating",
            "Failsafe": "Error handling",
            "Merge": "Branch merging"
        }
        return roles.get(node_type, "Processing node")

    def _extract_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract LLM configuration from node config.
        Preserves full model object including provider and displayName.

        Handles multiple input formats:
        1. Reference JSON format: {"model": {"provider": "anthropic", "modelName": "claude-opus-4-20250514", "displayName": "Claude Opus 4"}}
        2. Canvas format: {"model": {"modelName": "gpt-4o-mini"}}
        3. Flat format: {"model": "gpt-4o-mini"}
        """
        model_config = config.get("model", {})

        # Handle flat model string format
        if isinstance(model_config, str):
            return {
                "model": model_config,
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("maxTokens", 4000)
            }

        # Handle object model format - preserve full structure
        # Extract model name from either "modelName" or "model" field
        model_name = model_config.get("modelName") or model_config.get("model", "gpt-4o-mini")

        # Build LLM config with all available fields
        llm_config = {
            "model": model_name,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("maxTokens", 4000)
        }

        # Preserve provider if present (for reference JSON format)
        if model_config.get("provider"):
            llm_config["provider"] = model_config["provider"]

        # Preserve displayName if present
        if model_config.get("displayName"):
            llm_config["display_name"] = model_config["displayName"]

        return llm_config

    def _resolve_tools(self, frontend_tools: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """
        Resolve tool names to tool IDs (pre-registered).

        Handles two input formats:
        1. String format (imported workflows): ["Document Analysis", "OCR"]
        2. Dict format (canvas UI): [{"name": "...", "tool_id": "..."}]

        Args:
            frontend_tools: List of frontend tool configs (strings or dicts)

        Returns:
            List of tool IDs
        """
        tool_ids = []

        for tool in frontend_tools:
            # Handle string format (imported workflows)
            if isinstance(tool, str):
                # Try to resolve via tool registry first
                if self.tool_registry:
                    tool_id = self.tool_registry.get_tool_id_by_name(tool)
                    if tool_id:
                        tool_ids.append(tool_id)
                        continue
                # Fallback: normalize string name as placeholder ID
                tool_ids.append(tool.lower().replace(" ", "_"))
                continue

            # Handle dict format (canvas UI) - existing logic
            if isinstance(tool, dict):
                # Priority 1: Use tool_id if already provided (from registered tools)
                if tool.get("tool_id"):
                    tool_ids.append(tool["tool_id"])
                    continue

                # Priority 2: Resolve by name using tool registry
                tool_name = tool.get("name", "")
                if self.tool_registry:
                    tool_id = self.tool_registry.get_tool_id_by_name(tool_name)
                    if tool_id:
                        tool_ids.append(tool_id)
                        continue

                # Priority 3: For builtin types (code, subworkflow, mcp_server), use type as marker
                tool_type = tool.get("type", "")
                if tool_type in ["code", "subworkflow", "subworkflow_deployment", "mcp_server"]:
                    tool_ids.append(f"builtin_{tool_type}")
                    continue

                # Fallback: use name as placeholder
                if tool_name:
                    tool_ids.append(tool_name.lower().replace(" ", "_"))

        return tool_ids

    def _infer_execution_model(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> str:
        """
        Infer execution model from canvas structure.

        Returns:
            "sequential" | "parallel" | "hierarchical" | "hybrid"
        """
        node_types = [n.get("type") for n in canvas_nodes]

        # Check for hierarchical pattern (Subagent delegation)
        if "Subagent" in node_types:
            return "hierarchical"

        # Check for parallel execution
        if "Map" in node_types:
            return "parallel" if "Conditional" not in node_types else "hybrid"

        # Check for multiple branches from single node (parallel)
        outgoing_counts = {}
        for conn in connections:
            from_id = conn["from"]
            outgoing_counts[from_id] = outgoing_counts.get(from_id, 0) + 1

        has_parallel = any(count > 1 for count in outgoing_counts.values())
        has_merge = "Merge" in node_types
        has_conditional = "Conditional" in node_types or "Loop" in node_types

        # Workflow is hybrid if it has parallel branches AND:
        # - Merge nodes (parallel sections that converge)
        # - OR Conditional/Loop nodes (branching logic)
        if has_parallel and (has_conditional or has_merge):
            return "hybrid"
        elif has_parallel:
            return "parallel"
        else:
            return "sequential"

    def _extract_state_schema(self, canvas_nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract state schema from Start/End nodes."""
        state_schema = {}

        for node in canvas_nodes:
            if node.get("type") == "Start":
                for var in node.get("config", {}).get("inputVariables", []):
                    state_schema[var.get("name")] = var.get("type", "string")

            elif node.get("type") == "End":
                for var in node.get("config", {}).get("outputVariables", []):
                    state_schema[var.get("name")] = var.get("type", "string")

        return state_schema

    def _extract_hitl(
        self,
        canvas_nodes: List[Dict[str, Any]],
        id_mapping: Dict[int, str]
    ) -> Dict[str, Any]:
        """Extract human-in-the-loop configuration."""
        hitl_nodes = [n for n in canvas_nodes if n.get("type") == "HITL"]

        if not hitl_nodes:
            return {"enabled": False, "review_points": []}

        return {
            "enabled": True,
            "review_points": [id_mapping[n["id"]] for n in hitl_nodes]
        }

    def _build_hierarchy(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        id_mapping: Dict[int, str]
    ) -> Optional[Dict[str, Any]]:
        """Build hierarchy structure for hierarchical workflows."""
        # Find master agent (first Subagent or first Agent)
        master_node = None
        for node in canvas_nodes:
            if node.get("type") in ["Agent", "Subagent"]:
                master_node = node
                break

        if not master_node:
            return None

        master_id = id_mapping[master_node["id"]]

        # Find all downstream agents
        downstream = []
        for conn in connections:
            if id_mapping.get(conn["from"]) == master_id:
                downstream.append(id_mapping[conn["to"]])

        return {
            "master_agent": master_id,
            "delegation_order": downstream
        }

    # ==================== BACKEND → FRONTEND ====================

    def map_backend_to_frontend(
        self,
        workflow: Dict[str, Any],
        agents_dict: Dict[str, Dict[str, Any]],
        include_start_end: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert backend workflow JSON to frontend canvas nodes.

        Args:
            workflow: Backend workflow definition
            agents_dict: Backend agents dictionary (can be empty if agents are embedded)
            include_start_end: If True, automatically adds Start and End nodes

        Returns:
            Tuple of (canvas_nodes, connections)
        """
        canvas_nodes = []
        id_mapping = {}  # backend agent_id → frontend id
        current_id = 1
        spacing = 280
        base_y = 200

        # Handle both formats:
        # 1. New format: agents is a list of full agent objects
        # 2. Old format: agents is a list of string IDs
        workflow_agents = workflow.get("agents", [])

        # Build effective agents dict - merge embedded agents with provided agents_dict
        effective_agents = dict(agents_dict) if agents_dict else {}

        # If agents are embedded objects, add them to effective_agents
        for agent_entry in workflow_agents:
            if isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
                if agent_id:
                    effective_agents[agent_id] = agent_entry

        # Check if Start/End nodes already exist in embedded agents
        # This prevents duplicate nodes when loading saved workflows
        existing_start_agent_id = None
        existing_end_agent_id = None
        for agent_entry in workflow_agents:
            if isinstance(agent_entry, dict):
                metadata = agent_entry.get("metadata") or {}
                node_type = metadata.get("node_type")
                if node_type == "Start":
                    existing_start_agent_id = agent_entry.get("agent_id")
                elif node_type == "End":
                    existing_end_agent_id = agent_entry.get("agent_id")

        # Add Start node if requested AND not already present in agents
        start_node_id = None
        if include_start_end and not existing_start_agent_id:
            start_node_id = current_id
            start_node = {
                "id": start_node_id,
                "type": "Start",
                "name": "Workflow Start",
                "label": "Start",
                "x": 100,
                "y": base_y,
                "icon": self.NODE_ICONS.get("Start", "▶️"),
                "color": self.NODE_COLORS.get("Start", "#22c55e"),
                "config": {
                    "inputVariables": [],
                    "description": "Workflow entry point"
                },
                "status": "idle"
            }
            canvas_nodes.append(start_node)
            current_id += 1

        # Convert agents to canvas nodes
        agent_node_ids = []  # Track agent node IDs for connection building
        for idx, agent_entry in enumerate(workflow_agents):
            # Determine agent_id based on format
            if isinstance(agent_entry, str):
                agent_id = agent_entry
            elif isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
            else:
                continue
            agent = effective_agents.get(agent_id)
            if not agent:
                continue

            frontend_id = current_id
            id_mapping[agent_id] = frontend_id
            agent_node_ids.append(frontend_id)
            current_id += 1

            # Extract metadata safely
            metadata = agent.get("metadata") or {}
            ui_layout = metadata.get("ui_layout") or {}

            # Calculate position - after Start node if present
            x_offset = 1 if include_start_end else 0
            x_pos = 100 + ((idx + x_offset) * spacing)

            # Extract node type (with fallback)
            node_type = metadata.get("node_type") or "Agent"

            # Extract icon - check root level first, then ui_layout
            icon = agent.get("icon") or ui_layout.get("icon") or self.NODE_ICONS.get(node_type, "🔶")

            # Extract color from ui_layout
            color = ui_layout.get("color") or self.NODE_COLORS.get(node_type, "#64748b")

            node = {
                "id": frontend_id,
                "type": node_type,
                "name": agent.get("name") or node_type,
                "x": ui_layout.get("x") or x_pos,
                "y": ui_layout.get("y") or base_y,
                "icon": icon,
                "color": color,
                "config": self._extract_node_config(agent, node_type),
                "backend_id": agent_id,
                "agentId": agent_id,
                "status": "idle"
            }
            canvas_nodes.append(node)

        # Add End node if requested AND not already present in agents
        end_node_id = None
        if include_start_end and not existing_end_agent_id:
            end_node_id = current_id
            # Calculate x offset accounting for whether we added a Start node
            x_offset = 1 if (include_start_end and not existing_start_agent_id) else 0
            end_x = 100 + ((len(workflow_agents) + x_offset) * spacing)
            end_node = {
                "id": end_node_id,
                "type": "End",
                "name": "Workflow End",
                "label": "End",
                "x": end_x,
                "y": base_y,
                "icon": self.NODE_ICONS.get("End", "⏹️"),
                "color": self.NODE_COLORS.get("End", "#ef4444"),
                "config": {
                    "outputVariables": [],
                    "description": "Workflow exit point"
                },
                "status": "idle"
            }
            canvas_nodes.append(end_node)
            current_id += 1

        # Build connections
        connections = []
        conn_id = 1

        # If Start/End nodes already exist in agents, use workflow connections
        # (they were saved with the workflow and have correct references)
        if existing_start_agent_id or existing_end_agent_id:
            # Use the existing workflow connections with id_mapping
            for idx, conn in enumerate(workflow.get("connections", [])):
                from_id = id_mapping.get(conn["from"])
                to_id = id_mapping.get(conn["to"])

                if from_id and to_id:
                    connections.append({
                        "id": conn.get("id", idx + 1),
                        "from": from_id,
                        "to": to_id,
                        "condition": conn.get("condition")
                    })

        elif include_start_end and agent_node_ids:
            # Fresh workflow: generate connections with injected Start/End
            # Start -> first agent
            connections.append({
                "id": conn_id,
                "from": start_node_id,
                "to": agent_node_ids[0]
            })
            conn_id += 1

            # Agent to agent connections (sequential for now)
            for i in range(len(agent_node_ids) - 1):
                connections.append({
                    "id": conn_id,
                    "from": agent_node_ids[i],
                    "to": agent_node_ids[i + 1]
                })
                conn_id += 1

            # Last agent -> End
            connections.append({
                "id": conn_id,
                "from": agent_node_ids[-1],
                "to": end_node_id
            })
            conn_id += 1

        elif include_start_end and not agent_node_ids:
            # No agents - direct Start -> End
            connections.append({
                "id": conn_id,
                "from": start_node_id,
                "to": end_node_id
            })

        else:
            # Original behavior: use workflow connections
            for idx, conn in enumerate(workflow.get("connections", [])):
                from_id = id_mapping.get(conn["from"])
                to_id = id_mapping.get(conn["to"])

                if from_id and to_id:
                    connections.append({
                        "id": conn.get("id", idx + 1),
                        "from": from_id,
                        "to": to_id,
                        "condition": conn.get("condition")
                    })

        return canvas_nodes, connections

    def _generate_auto_layout(self, index: int, total: int) -> Dict[str, int]:
        """
        Generate auto-layout positions for nodes without ui_layout.
        Uses horizontal flow with vertical spacing.
        """
        x = 200 + (index * 250)
        y = 200 + ((index % 3) * 150)  # Stagger vertically

        return {"x": x, "y": y}

    def _extract_node_config(
        self,
        agent: Dict[str, Any],
        node_type: str
    ) -> Dict[str, Any]:
        """
        Extract frontend node config from backend agent.
        Reverse of _convert_node_to_agent.

        Supports both storage schemas:
        - New: model (string), settings (object)
        - Old: llm (object), constraints (object)
        """
        config = {}
        metadata = agent.get("metadata") or {}
        settings = agent.get("settings") or {}

        # Extract prompt and goal for Agent/Subagent nodes
        if node_type in ["Agent", "Subagent", "Prompt"]:
            if agent.get("prompt"):
                config["prompt"] = agent["prompt"]
            # Goal may be in agent directly or in metadata
            goal = agent.get("goal") or metadata.get("goal", "")
            if goal:
                config["goal"] = goal
            if agent.get("description"):
                config["description"] = agent["description"]

        # Extract model config - support BOTH storage schemas
        # Schema 1 (NEW): "model" as string + "settings" object
        # Schema 2 (OLD): "llm" as object with model, temperature, etc.

        if "model" in agent and isinstance(agent.get("model"), str) and agent.get("model"):
            # NEW SCHEMA: model is string, settings is object
            model_name = str(agent["model"])  # Ensure string
            provider = str(settings.get("provider", "") or "")

            config["model"] = {
                "modelName": model_name,
                "displayName": model_name.replace("-", " ").title() if model_name else "Default Model",
                "provider": provider
            }
            config["temperature"] = float(settings.get("temperature", 0.7) or 0.7)
            config["maxTokens"] = int(settings.get("max_token", 4000) or 4000)
            config["topP"] = float(settings.get("top_p", 0.9) or 0.9)
            config["maxIterations"] = int(settings.get("max_iteration", 5) or 5)

        elif "llm" in agent and isinstance(agent.get("llm"), dict):
            # OLD SCHEMA: llm is object
            llm = agent["llm"] or {}
            config["model"] = {
                "modelName": llm.get("model") or "gpt-4o-mini",
                "displayName": llm.get("display_name") or llm.get("model") or "GPT-4o Mini",
                "provider": llm.get("provider") or ""
            }
            config["temperature"] = float(llm.get("temperature") or 0.7)
            config["maxTokens"] = int(llm.get("max_tokens") or 4000)

            # Extract maxIterations from constraints (old schema)
            constraints = agent.get("constraints") or {}
            config["maxIterations"] = int(constraints.get("max_steps") or 5)

        else:
            # No model config - use defaults
            config["model"] = {
                "modelName": "",
                "displayName": "LLM Manager Default",
                "provider": ""
            }
            config["temperature"] = 0.7
            config["maxTokens"] = 4000
            config["maxIterations"] = 5

        # Extract tools - always present (handle None safely)
        raw_tools = agent.get("tools") or []
        config["tools"] = self._tools_ids_to_names(raw_tools)

        # Extract variables (handle None safely)
        config["variables"] = agent.get("variables") or []

        # Node-type specific
        if node_type == "Start":
            input_schema = agent.get("input_schema") or []
            config["inputVariables"] = [
                {"name": str(name), "type": "string", "required": True}
                for name in input_schema if name
            ]

        elif node_type == "End":
            output_schema = agent.get("output_schema") or []
            config["outputVariables"] = [
                {"name": str(name), "type": "string"}
                for name in output_schema if name
            ]

        elif node_type == "Conditional":
            config["branches"] = metadata.get("branches") or []

        elif node_type == "Loop":
            config.update(metadata.get("loop_config") or {})

        elif node_type == "Map":
            config.update(metadata.get("map_config") or {})

        elif node_type == "HITL":
            config.update(metadata.get("hitl_config") or {})

        elif node_type == "API":
            config.update(metadata.get("api_config") or {})

        elif node_type == "MCP":
            mcp_cfg = metadata.get("mcp_config") or {}
            config.update(mcp_cfg)

            # Reconstruct serverId for the frontend dropdown if not already stored
            if not config.get("serverId"):
                if config.get("command") or config.get("transport_type") == "stdio":
                    config["serverId"] = "azure"
                elif config.get("connector_id"):
                    config["serverId"] = "managed"
                elif config.get("endpoint_url") or config.get("customUrl"):
                    config["serverId"] = "custom"

            # Reconstruct argumentsRaw (JSON string for the textarea)
            if not config.get("argumentsRaw"):
                args = config.get("arguments") or {}
                if args:
                    try:
                        config["argumentsRaw"] = json.dumps(args, indent=2)
                    except Exception:
                        config["argumentsRaw"] = ""
                else:
                    config["argumentsRaw"] = ""

        elif node_type == "Code":
            config.update(metadata.get("code_config") or {})

        elif node_type == "Self-Review":
            config.update(metadata.get("review_config") or {})

        return config

    def _tools_ids_to_names(self, tool_ids: List[str]) -> List[Dict[str, Any]]:
        """Convert tool IDs back to frontend tool configs.

        Looks up actual tool names from the tool registry.
        Preserves tool_id for round-trip save/load operations.
        Handles various input formats defensively.
        """
        tools = []

        # Handle None or non-list inputs
        if not tool_ids or not isinstance(tool_ids, list):
            return tools

        # Load tool registry to get actual names
        tool_registry = _load_tool_registry()

        for tool_id in tool_ids:
            # Handle string tool IDs
            if isinstance(tool_id, str):
                # Look up from registry for actual name
                registered_tool = tool_registry.get(tool_id)
                if registered_tool:
                    # Use actual registered tool info
                    tools.append({
                        "id": new_id("tool_"),
                        "tool_id": tool_id,
                        "name": registered_tool.get("name", tool_id),
                        "type": registered_tool.get("tool_type", "local"),
                        "description": registered_tool.get("description", ""),
                        "enabled": True,
                        "config": {}
                    })
                else:
                    # Fallback for unknown tools (not in registry)
                    tools.append({
                        "id": new_id("tool_"),
                        "tool_id": tool_id,
                        "name": tool_id.replace("_", " ").title(),
                        "type": "tools",
                        "enabled": True,
                        "config": {}
                    })
            # Handle dict tool configs (already converted)
            elif isinstance(tool_id, dict):
                tools.append({
                    "id": tool_id.get("id") or new_id("tool_"),
                    "tool_id": tool_id.get("tool_id", ""),
                    "name": tool_id.get("name", "Unknown Tool"),
                    "type": tool_id.get("type", "tools"),
                    "enabled": tool_id.get("enabled", True),
                    "config": tool_id.get("config", {})
                })
            # Skip invalid entries

        return tools

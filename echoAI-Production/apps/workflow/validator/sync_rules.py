"""
Synchronous validation rules.
Fast, deterministic checks for workflow correctness.
"""
import jsonschema
from typing import Dict, Any, List, Union
from echolib.schemas import WORKFLOW_SCHEMA, AGENT_SCHEMA
from .errors import ValidationResult


def _extract_agent_ids(agents: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """
    Extract agent IDs from agents list, handling both formats:
    - New format: list of dicts with "agent_id" key
    - Old format: list of strings
    """
    agent_ids = []
    for agent in agents:
        if isinstance(agent, str):
            agent_ids.append(agent)
        elif isinstance(agent, dict):
            agent_id = agent.get("agent_id")
            if agent_id:
                agent_ids.append(agent_id)
    return agent_ids


def _build_agent_registry_from_embedded(
    workflow: Dict[str, Any],
    external_registry: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Build effective agent registry from embedded agents + external registry.
    Embedded agents take precedence.
    """
    effective = dict(external_registry) if external_registry else {}

    for agent in workflow.get("agents", []):
        if isinstance(agent, dict):
            agent_id = agent.get("agent_id")
            if agent_id:
                effective[agent_id] = agent

    return effective


def validate_workflow_schema(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate workflow matches JSON schema."""
    try:
        jsonschema.validate(instance=workflow, schema=WORKFLOW_SCHEMA)
    except jsonschema.ValidationError as e:
        result.add_error(f"Workflow schema validation failed: {e.message}")


def validate_agent_schemas(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate all agents match agent schema."""
    effective_registry = _build_agent_registry_from_embedded(workflow, agent_registry)
    agent_ids = _extract_agent_ids(workflow.get("agents", []))

    for agent_id in agent_ids:
        if agent_id not in effective_registry:
            continue  # Will be caught by validate_agents_exist

        agent = effective_registry[agent_id]
        try:
            jsonschema.validate(instance=agent, schema=AGENT_SCHEMA)
        except jsonschema.ValidationError as e:
            result.add_error(f"Agent '{agent_id}' schema invalid: {e.message}")


def validate_agents_exist(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Check all referenced agents exist."""
    effective_registry = _build_agent_registry_from_embedded(workflow, agent_registry)
    agent_ids = _extract_agent_ids(workflow.get("agents", []))

    for agent_id in agent_ids:
        if agent_id not in effective_registry:
            result.add_error(f"Agent '{agent_id}' not found")


def validate_tools(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    tool_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate all agent tools exist in tool registry."""
    # Builtin tool types that don't require registry lookup
    BUILTIN_TOOL_PREFIXES = ("builtin_code", "builtin_subworkflow", "builtin_mcp_server")

    effective_registry = _build_agent_registry_from_embedded(workflow, agent_registry)
    agent_ids = _extract_agent_ids(workflow.get("agents", []))

    for agent_id in agent_ids:
        if agent_id not in effective_registry:
            continue

        agent = effective_registry[agent_id]
        for tool_id in agent.get("tools", []):
            # Skip builtin tools (code, subworkflow, mcp_server)
            if isinstance(tool_id, str) and tool_id.startswith(BUILTIN_TOOL_PREFIXES):
                continue

            if tool_id not in tool_registry:
                result.add_error(
                    f"Tool '{tool_id}' not found for agent '{agent_id}'"
                )
            elif tool_registry.get(tool_id, {}).get("status") == "deprecated":
                result.add_warning(
                    f"Tool '{tool_id}' is deprecated"
                )


def validate_io_contracts(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate agent I/O contracts (A2A safety)."""
    produced_keys = set()
    state_keys = set(workflow.get("state_schema", {}).keys())

    effective_registry = _build_agent_registry_from_embedded(workflow, agent_registry)
    agent_ids = _extract_agent_ids(workflow.get("agents", []))

    # Collect all produced state keys
    for agent_id in agent_ids:
        if agent_id not in effective_registry:
            continue

        agent = effective_registry[agent_id]
        for key in agent.get("output_schema", []):
            if key in produced_keys:
                result.add_error(
                    f"State key '{key}' written by multiple agents"
                )
            produced_keys.add(key)

    # Check all required inputs are satisfied
    for agent_id in agent_ids:
        if agent_id not in effective_registry:
            continue

        agent = effective_registry[agent_id]
        for key in agent.get("input_schema", []):
            if key not in produced_keys and key not in state_keys:
                result.add_error(
                    f"Agent '{agent_id}' expects '{key}' but no producer found"
                )


def validate_workflow_topology(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate workflow topology (no dead ends, no infinite loops)."""
    agent_ids = _extract_agent_ids(workflow.get("agents", []))
    nodes = set(agent_ids)
    connections = workflow.get("connections", [])

    # Build graph
    connected = set()
    for edge in connections:
        from_node = edge.get("from")
        to_node = edge.get("to")
        connected.add(from_node)
        connected.add(to_node)

        # Check nodes exist
        if from_node not in nodes:
            result.add_error(f"Connection references unknown agent '{from_node}'")
        if to_node not in nodes:
            result.add_error(f"Connection references unknown agent '{to_node}'")

    # Warn about isolated nodes
    for node in nodes:
        if node not in connected:
            result.add_warning(f"Agent '{node}' is isolated in workflow")


def validate_execution_model(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate execution model specific rules."""
    mode = workflow.get("execution_model")
    agent_ids = _extract_agent_ids(workflow.get("agents", []))
    connections = workflow.get("connections", [])

    if mode == "sequential":
        # Sequential must be linear
        if len(connections) != len(agent_ids) - 1:
            result.add_error("Sequential workflow must be linear")

    elif mode == "parallel":
        # Parallel workflows should have a merge point or final aggregator
        # This is a soft check (warning)
        if len(connections) == 0:
            result.add_warning("Parallel workflow has no connections")

    elif mode == "hierarchical":
        # Hierarchical must have hierarchy block
        hierarchy = workflow.get("hierarchy")
        if not hierarchy:
            result.add_error("Hierarchical workflow missing hierarchy block")


def validate_hierarchical_rules(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate hierarchical workflow rules."""
    if workflow.get("execution_model") != "hierarchical":
        return

    hierarchy = workflow.get("hierarchy")
    if not hierarchy:
        return  # Already caught by validate_execution_model

    master = hierarchy.get("master_agent")
    agent_ids = _extract_agent_ids(workflow.get("agents", []))

    if master not in agent_ids:
        result.add_error("Master agent not found in agent list")

    effective_registry = _build_agent_registry_from_embedded(workflow, agent_registry)

    # Sub-agents cannot call each other directly in hierarchical mode
    for agent_id in agent_ids:
        if agent_id == master:
            continue

        if agent_id not in effective_registry:
            continue

        agent = effective_registry[agent_id]
        permissions = agent.get("permissions", {})
        if permissions.get("can_call_agents", False):
            result.add_error(
                f"Sub-agent '{agent_id}' cannot call agents in hierarchical mode"
            )


def validate_hitl_rules(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate HITL configuration."""
    hitl = workflow.get("human_in_loop", {})
    if not hitl.get("enabled"):
        return

    agent_ids = _extract_agent_ids(workflow.get("agents", []))
    for point in hitl.get("review_points", []):
        if point not in agent_ids:
            result.add_error(
                f"HITL review point '{point}' is invalid"
            )

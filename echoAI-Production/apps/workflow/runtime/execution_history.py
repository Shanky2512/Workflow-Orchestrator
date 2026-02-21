"""
Per-node execution history tracker for Smart Replay.

Tracks input/output pairs for each node execution within a workflow run.
This data is stored in the workflow state so it persists via the LangGraph
checkpointer (MemorySaver or PostgresSaver).

Used by EditResolutionEngine to determine which node to restart from
when a human edits the workflow during a HITL pause.

State key: "_execution_history"
Format: list of {node_id, input, output, timestamp} dicts.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# State key used to store execution history in LangGraph workflow state
EXECUTION_HISTORY_KEY = "_execution_history"


def record_node_execution(
    state: Dict[str, Any],
    node_id: str,
    node_input: Dict[str, Any],
    node_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Record a node's input/output in the workflow state execution history.

    This should be called after each node completes execution.
    The history is stored in state under the EXECUTION_HISTORY_KEY.

    Args:
        state: Current workflow state dict (will be mutated).
        node_id: Identifier of the node that executed.
        node_input: Input data the node received.
        node_output: Output data the node produced.

    Returns:
        The updated state dict with the new history entry appended.
    """
    history = state.get(EXECUTION_HISTORY_KEY, [])

    entry = {
        "node_id": node_id,
        "input": _safe_serialize(node_input),
        "output": _safe_serialize(node_output),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    history.append(entry)
    state[EXECUTION_HISTORY_KEY] = history

    logger.debug(
        f"Recorded execution history for node '{node_id}' "
        f"(total entries: {len(history)})"
    )

    return state


def get_execution_history(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve the full execution history from workflow state.

    Args:
        state: Current workflow state dict.

    Returns:
        List of execution history entries, ordered by execution time.
    """
    return state.get(EXECUTION_HISTORY_KEY, [])


def get_node_history(
    state: Dict[str, Any],
    node_id: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve execution history for a specific node.

    A node may have multiple entries if it was executed more than once
    (e.g., after a Smart Replay restart).

    Args:
        state: Current workflow state dict.
        node_id: Node identifier to filter by.

    Returns:
        List of execution history entries for the specified node.
    """
    history = get_execution_history(state)
    return [entry for entry in history if entry["node_id"] == node_id]


def get_last_node_output(
    state: Dict[str, Any],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent output from a specific node.

    Useful for Smart Replay to determine what data a node last produced.

    Args:
        state: Current workflow state dict.
        node_id: Node identifier.

    Returns:
        The output dict from the most recent execution, or None if not found.
    """
    entries = get_node_history(state, node_id)
    if entries:
        return entries[-1].get("output")
    return None


def get_executed_node_ids(state: Dict[str, Any]) -> List[str]:
    """
    Get ordered list of node IDs that have been executed.

    Returns unique node IDs in the order they were first executed.

    Args:
        state: Current workflow state dict.

    Returns:
        Ordered list of node IDs.
    """
    history = get_execution_history(state)
    seen = set()
    ordered = []
    for entry in history:
        nid = entry["node_id"]
        if nid not in seen:
            seen.add(nid)
            ordered.append(nid)
    return ordered


def _safe_serialize(data: Any) -> Any:
    """
    Safely serialize data for storage in execution history.

    Handles non-serializable objects by converting them to strings.
    Truncates very large values to prevent state bloat.

    Args:
        data: Data to serialize.

    Returns:
        Serializable representation of the data.
    """
    if data is None:
        return None

    if isinstance(data, (str, int, float, bool)):
        if isinstance(data, str) and len(data) > 5000:
            return data[:5000] + "... [truncated]"
        return data

    if isinstance(data, dict):
        return {
            str(k): _safe_serialize(v)
            for k, v in data.items()
        }

    if isinstance(data, (list, tuple)):
        return [_safe_serialize(item) for item in data[:100]]  # Cap at 100 items

    # Fallback: convert to string
    try:
        return str(data)[:2000]
    except Exception:
        return "<non-serializable>"

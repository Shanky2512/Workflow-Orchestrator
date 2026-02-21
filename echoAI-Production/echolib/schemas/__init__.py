"""
Schema definitions for the Echo AI orchestrator.
Contains JSON schemas and Pydantic models for validation.
"""
import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema by name."""
    schema_path = SCHEMA_DIR / f"{schema_name}.json"
    with open(schema_path) as f:
        return json.load(f)


# Pre-load schemas for validation
WORKFLOW_SCHEMA = load_schema("workflow_schema")
AGENT_SCHEMA = load_schema("agent_schema")
TOOL_SCHEMA = load_schema("tool_schema")
GRAPH_SCHEMA = load_schema("graph_schema")

__all__ = [
    "WORKFLOW_SCHEMA",
    "AGENT_SCHEMA",
    "TOOL_SCHEMA",
    "GRAPH_SCHEMA",
    "load_schema",
]

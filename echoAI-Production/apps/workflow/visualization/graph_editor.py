"""
Graph editor.
Applies UI graph edits back to workflow JSON.
"""
from typing import Dict, Any, List


class GraphEditor:
    """
    Applies graph edits to workflow definitions.
    Supports Human-in-the-Loop workflow editing.
    """

    def __init__(self):
        """Initialize graph editor."""
        pass

    def apply_node_changes(
        self,
        workflow: Dict[str, Any],
        node_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply node changes from UI to workflow.

        Args:
            workflow: Current workflow definition
            node_changes: List of node modifications

        Returns:
            Updated workflow
        """
        for change in node_changes:
            change_type = change.get("type")
            node_id = change.get("node_id")

            if change_type == "add":
                self._add_node(workflow, node_id, change)
            elif change_type == "remove":
                self._remove_node(workflow, node_id)
            elif change_type == "update":
                self._update_node(workflow, node_id, change)

        return workflow

    def apply_edge_changes(
        self,
        workflow: Dict[str, Any],
        edge_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply edge changes from UI to workflow.

        Args:
            workflow: Current workflow definition
            edge_changes: List of edge modifications

        Returns:
            Updated workflow
        """
        connections = workflow.get("connections", [])

        for change in edge_changes:
            change_type = change.get("type")

            if change_type == "add":
                connections.append({
                    "from": change.get("source"),
                    "to": change.get("target"),
                    "condition": change.get("condition")
                })
            elif change_type == "remove":
                source = change.get("source")
                target = change.get("target")
                connections = [
                    c for c in connections
                    if not (c.get("from") == source and c.get("to") == target)
                ]

        workflow["connections"] = connections
        return workflow

    def _add_node(
        self,
        workflow: Dict[str, Any],
        node_id: str,
        change: Dict[str, Any]
    ) -> None:
        """Add node to workflow."""
        agents = workflow.get("agents", [])

        # Extract existing agent IDs to check for duplicates
        existing_ids = []
        for agent in agents:
            if isinstance(agent, str):
                existing_ids.append(agent)
            elif isinstance(agent, dict):
                aid = agent.get("agent_id")
                if aid:
                    existing_ids.append(aid)

        if node_id not in existing_ids:
            agents.append(node_id)
        workflow["agents"] = agents

    def _remove_node(self, workflow: Dict[str, Any], node_id: str) -> None:
        """Remove node from workflow."""
        agents = workflow.get("agents", [])

        # Handle both embedded agent objects and ID-referenced agents
        new_agents = []
        for agent in agents:
            if isinstance(agent, str):
                if agent != node_id:
                    new_agents.append(agent)
            elif isinstance(agent, dict):
                if agent.get("agent_id") != node_id:
                    new_agents.append(agent)

        workflow["agents"] = new_agents

        # Remove related connections
        connections = workflow.get("connections", [])
        connections = [
            c for c in connections
            if c.get("from") != node_id and c.get("to") != node_id
        ]
        workflow["connections"] = connections

    def _update_node(
        self,
        workflow: Dict[str, Any],
        node_id: str,
        change: Dict[str, Any]
    ) -> None:
        """Update node metadata."""
        # Node metadata updates would affect agent registry, not workflow
        # This is a placeholder for future implementation
        pass

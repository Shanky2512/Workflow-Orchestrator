"""
Graph mapper.
Converts workflow JSON to graph nodes and edges for visualization.
"""
from typing import Dict, Any, List


class GraphMapper:
    """
    Maps workflow definitions to graph representations.
    """

    def __init__(self, storage=None):
        """
        Initialize graph mapper.

        Args:
            storage: Optional WorkflowStorage instance
        """
        self.storage = storage

    def workflow_to_graph(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert workflow to graph nodes and edges.

        Args:
            workflow: Workflow definition
            agent_registry: Optional agent definitions

        Returns:
            Graph with nodes and edges
        """
        if agent_registry is None:
            agent_registry = {}

        nodes = self._create_nodes(workflow, agent_registry)
        edges = self._create_edges(workflow)

        return {
            "nodes": nodes,
            "edges": edges
        }

    def _create_nodes(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create graph nodes from workflow agents.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            List of graph nodes
        """
        nodes = []
        execution_model = workflow.get("execution_model")
        hierarchy = workflow.get("hierarchy", {})
        master_agent = hierarchy.get("master_agent")

        # Build effective registry from embedded agents + external registry
        effective_registry = dict(agent_registry) if agent_registry else {}
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
                if aid:
                    effective_registry[aid] = agent_entry

        # Extract agent IDs (handle both embedded objects and string IDs)
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, str):
                agent_id = agent_entry
            elif isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
            else:
                continue

            agent = effective_registry.get(agent_id, {})

            # Determine node type
            if execution_model == "hierarchical" and agent_id == master_agent:
                node_type = "master_agent"
            else:
                node_type = "agent"

            nodes.append({
                "id": agent_id,
                "label": agent.get("name", agent_id),
                "type": node_type,
                "metadata": {
                    "role": agent.get("role"),
                    "llm": agent.get("llm", {}),
                    "tools": agent.get("tools", [])
                }
            })

        return nodes

    def _create_edges(self, workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create graph edges from workflow connections.

        Args:
            workflow: Workflow definition

        Returns:
            List of graph edges
        """
        edges = []

        for connection in workflow.get("connections", []):
            edges.append({
                "source": connection.get("from"),
                "target": connection.get("to"),
                "condition": connection.get("condition")
            })

        return edges

    def get_graph(self, workflow_id: str, state: str = "temp") -> Dict[str, Any]:
        """
        Get graph representation for a stored workflow.

        Args:
            workflow_id: Workflow identifier
            state: Workflow state (temp, final, etc.)

        Returns:
            Graph representation
        """
        if not self.storage:
            raise ValueError("Storage not configured")

        workflow = self.storage.load_workflow(workflow_id, state)
        return self.workflow_to_graph(workflow)

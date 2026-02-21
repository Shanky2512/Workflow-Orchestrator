"""
Agent permissions.
Enforces agent-to-agent communication rules and hierarchical constraints.
"""
from typing import Dict, Any, List, Optional


class AgentPermissions:
    """
    Agent permissions service.
    Manages and enforces agent communication rules.
    """

    def __init__(self):
        """Initialize permissions service."""
        self._rules_cache = {}

    def can_call_agent(
        self,
        caller_id: str,
        target_id: str,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Check if caller can communicate with target agent.

        Args:
            caller_id: Calling agent ID
            target_id: Target agent ID
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            True if allowed, False otherwise
        """
        execution_model = workflow.get("execution_model")

        # Hierarchical workflows have strict rules
        if execution_model == "hierarchical":
            return self._check_hierarchical_permission(
                caller_id, target_id, workflow, agent_registry
            )

        # Sequential workflows have implicit ordering
        if execution_model == "sequential":
            return self._check_sequential_permission(
                caller_id, target_id, workflow
            )

        # Parallel workflows allow communication
        if execution_model == "parallel":
            return True

        # Default: check agent-level permissions
        return self._check_agent_permission(
            caller_id, target_id, agent_registry
        )

    def _check_hierarchical_permission(
        self,
        caller_id: str,
        target_id: str,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Check permission in hierarchical workflow.

        Rules:
        - Master agent can call any sub-agent
        - Sub-agents cannot call each other directly
        - Sub-agents can only respond to master
        """
        hierarchy = workflow.get("hierarchy", {})
        master_agent = hierarchy.get("master_agent")

        # Master can call anyone
        if caller_id == master_agent:
            return True

        # Sub-agents can only call master
        if target_id == master_agent:
            return True

        # Sub-agent trying to call another sub-agent: DENY
        return False

    def _check_sequential_permission(
        self,
        caller_id: str,
        target_id: str,
        workflow: Dict[str, Any]
    ) -> bool:
        """
        Check permission in sequential workflow.

        Rules:
        - Agent can only call next agent in sequence
        """
        connections = workflow.get("connections", [])

        for conn in connections:
            if conn.get("from") == caller_id and conn.get("to") == target_id:
                return True

        return False

    def _check_agent_permission(
        self,
        caller_id: str,
        target_id: str,
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Check agent-level permissions.

        Args:
            caller_id: Calling agent ID
            target_id: Target agent ID
            agent_registry: Agent definitions

        Returns:
            True if allowed
        """
        caller = agent_registry.get(caller_id, {})
        permissions = caller.get("permissions", {})

        # Check if agent can call other agents at all
        if not permissions.get("can_call_agents", False):
            return False

        # Check allowed agents list
        allowed_agents = permissions.get("allowed_agents", [])
        if allowed_agents and target_id not in allowed_agents:
            return False

        return True

    def validate_workflow_permissions(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate all permissions in a workflow.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            List of permission violation errors
        """
        errors = []
        execution_model = workflow.get("execution_model")

        if execution_model == "hierarchical":
            errors.extend(self._validate_hierarchical_permissions(
                workflow, agent_registry
            ))

        return errors

    def _validate_hierarchical_permissions(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate hierarchical workflow permissions.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            List of errors
        """
        errors = []
        hierarchy = workflow.get("hierarchy", {})
        master_agent = hierarchy.get("master_agent")
        agents = workflow.get("agents", [])

        # Check that sub-agents don't have can_call_agents permission
        for agent_id in agents:
            if agent_id == master_agent:
                continue

            agent = agent_registry.get(agent_id, {})
            permissions = agent.get("permissions", {})

            if permissions.get("can_call_agents", False):
                errors.append(
                    f"Sub-agent '{agent_id}' cannot have can_call_agents in hierarchical workflow"
                )

        return errors

    def get_allowed_targets(
        self,
        agent_id: str,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Get list of agents that the given agent can call.

        Args:
            agent_id: Agent identifier
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            List of allowed target agent IDs
        """
        execution_model = workflow.get("execution_model")

        if execution_model == "hierarchical":
            hierarchy = workflow.get("hierarchy", {})
            master_agent = hierarchy.get("master_agent")

            if agent_id == master_agent:
                # Master can call all sub-agents
                return [a for a in workflow.get("agents", []) if a != master_agent]
            else:
                # Sub-agent can only call master
                return [master_agent]

        elif execution_model == "sequential":
            # Can only call next in sequence
            connections = workflow.get("connections", [])
            return [
                conn.get("to")
                for conn in connections
                if conn.get("from") == agent_id
            ]

        elif execution_model == "parallel":
            # Can call any other agent
            return [a for a in workflow.get("agents", []) if a != agent_id]

        else:
            # Check agent-level permissions
            agent = agent_registry.get(agent_id, {})
            permissions = agent.get("permissions", {})

            if not permissions.get("can_call_agents", False):
                return []

            allowed = permissions.get("allowed_agents", [])
            if allowed:
                return allowed

            # Default: can call any agent
            return [a for a in workflow.get("agents", []) if a != agent_id]

    def enforce_permission(
        self,
        caller_id: str,
        target_id: str,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Enforce permission check with exception on violation.

        Args:
            caller_id: Calling agent ID
            target_id: Target agent ID
            workflow: Workflow definition
            agent_registry: Agent definitions

        Raises:
            PermissionError: If permission denied
        """
        if not self.can_call_agent(caller_id, target_id, workflow, agent_registry):
            raise PermissionError(
                f"Agent '{caller_id}' not allowed to call agent '{target_id}'"
            )

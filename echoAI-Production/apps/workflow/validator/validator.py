"""
Main workflow validator.
Orchestrates sync and async validation rules.
"""
from typing import Dict, Any, Callable, Optional
from .errors import ValidationResult
from .sync_rules import (
    validate_workflow_schema,
    validate_agents_exist,
    validate_agent_schemas,
    validate_tools,
    validate_io_contracts,
    validate_workflow_topology,
    validate_execution_model,
    validate_hierarchical_rules,
    validate_hitl_rules,
)
from .async_rules import validate_runtime_async


class WorkflowValidator:
    """
    Workflow validation service.
    Performs sync and async validation checks.
    """

    def __init__(
        self,
        tool_registry: Dict[str, Dict[str, Any]] = None,
        tool_registry_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None
    ):
        """
        Initialize validator.

        Args:
            tool_registry: Static tool registry for validation
            tool_registry_provider: Callable that returns fresh tool registry (preferred)
        """
        self._tool_registry = tool_registry or {}
        self._tool_registry_provider = tool_registry_provider

    @property
    def tool_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get current tool registry (dynamic if provider is set)."""
        if self._tool_registry_provider:
            try:
                return self._tool_registry_provider()
            except Exception:
                return self._tool_registry
        return self._tool_registry

    async def validate_workflow(
        self,
        workflow_json: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """
        Full validation pipeline (sync + async).

        Args:
            workflow_json: Workflow definition
            agent_registry: Agent definitions keyed by agent_id

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # ---------- SYNC PHASE ----------
        validate_workflow_schema(workflow_json, result)
        if not result.is_valid():
            return result

        validate_agents_exist(workflow_json, agent_registry, result)
        validate_agent_schemas(workflow_json, agent_registry, result)
        validate_tools(workflow_json, agent_registry, self.tool_registry, result)
        validate_io_contracts(workflow_json, agent_registry, result)
        validate_workflow_topology(workflow_json, result)
        validate_execution_model(workflow_json, result)
        validate_hierarchical_rules(workflow_json, agent_registry, result)
        validate_hitl_rules(workflow_json, result)

        if not result.is_valid():
            return result

        # ---------- ASYNC PHASE ----------
        await validate_runtime_async(workflow_json, agent_registry, result)

        return result

    def validate_draft(
        self,
        workflow_json: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """
        Sync-only validation for draft workflows (before HITL).

        Args:
            workflow_json: Workflow definition
            agent_registry: Agent definitions

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        validate_workflow_schema(workflow_json, result)
        validate_agents_exist(workflow_json, agent_registry, result)
        validate_agent_schemas(workflow_json, agent_registry, result)
        validate_tools(workflow_json, agent_registry, self.tool_registry, result)
        validate_io_contracts(workflow_json, agent_registry, result)
        validate_workflow_topology(workflow_json, result)
        validate_execution_model(workflow_json, result)
        validate_hierarchical_rules(workflow_json, agent_registry, result)
        validate_hitl_rules(workflow_json, result)

        return result

    async def validate_final(
        self,
        workflow_json: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """
        Full validation for final workflows (after HITL).

        Args:
            workflow_json: Workflow definition
            agent_registry: Agent definitions

        Returns:
            ValidationResult
        """
        return await self.validate_workflow(workflow_json, agent_registry)

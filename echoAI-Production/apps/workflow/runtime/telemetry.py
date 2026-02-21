"""
Runtime telemetry.
OpenTelemetry instrumentation for workflow execution.
"""
from typing import Dict, Any, Optional
from contextlib import contextmanager


class WorkflowTelemetry:
    """
    Telemetry service for workflow execution.
    Provides OpenTelemetry-compatible tracing.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize telemetry.

        Args:
            enabled: Whether telemetry is enabled
        """
        self.enabled = enabled
        self.spans: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def workflow_span(
        self,
        workflow_id: str,
        version: str,
        execution_model: str
    ):
        """
        Create workflow-level span.

        Args:
            workflow_id: Workflow identifier
            version: Workflow version
            execution_model: Execution model

        Yields:
            Span context
        """
        if not self.enabled:
            yield None
            return

        span_id = f"workflow_{workflow_id}"

        # Start span
        span = {
            "span_id": span_id,
            "workflow_id": workflow_id,
            "version": version,
            "execution_model": execution_model,
            "start_time": None,  # TODO: Add actual timing
            "end_time": None
        }

        self.spans[span_id] = span

        try:
            yield span
        finally:
            # End span
            span["end_time"] = None  # TODO: Add actual timing

    @contextmanager
    def agent_span(
        self,
        agent_id: str,
        agent_role: str,
        llm_provider: str,
        llm_model: str
    ):
        """
        Create agent-level span.

        Args:
            agent_id: Agent identifier
            agent_role: Agent role
            llm_provider: LLM provider
            llm_model: LLM model

        Yields:
            Span context
        """
        if not self.enabled:
            yield None
            return

        span_id = f"agent_{agent_id}"

        # Start span
        span = {
            "span_id": span_id,
            "agent_id": agent_id,
            "agent_role": agent_role,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "start_time": None,  # TODO: Add actual timing
            "end_time": None,
            "tokens_in": 0,
            "tokens_out": 0
        }

        self.spans[span_id] = span

        try:
            yield span
        finally:
            # End span
            span["end_time"] = None  # TODO: Add actual timing

    @contextmanager
    def tool_span(self, tool_id: str, mcp_server: str):
        """
        Create tool-level span.

        Args:
            tool_id: Tool identifier
            mcp_server: MCP server

        Yields:
            Span context
        """
        if not self.enabled:
            yield None
            return

        span_id = f"tool_{tool_id}"

        # Start span
        span = {
            "span_id": span_id,
            "tool_id": tool_id,
            "mcp_server": mcp_server,
            "start_time": None,
            "end_time": None,
            "success": None
        }

        self.spans[span_id] = span

        try:
            yield span
            span["success"] = True
        except Exception as e:
            span["success"] = False
            span["error"] = str(e)
            raise
        finally:
            # End span
            span["end_time"] = None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected telemetry metrics.

        Returns:
            Telemetry metrics
        """
        return {
            "enabled": self.enabled,
            "spans": self.spans
        }

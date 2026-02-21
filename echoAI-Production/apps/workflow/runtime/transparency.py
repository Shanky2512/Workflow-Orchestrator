"""
Execution Transparency Module

Provides step-level visibility into workflow execution.
This module implements ephemeral in-memory tracking of execution steps
with WebSocket event emission for real-time visibility.

Key design principles:
- NO raw chain-of-thought exposure - only structured execution steps
- Ephemeral in-memory state (no persistent storage)
- 60-second cleanup after run completion
- Sanitized input/output summaries (truncated, no internal reasoning)
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of execution steps in a workflow."""
    AGENT = "agent"                          # Single agent execution
    TOOL_CALL = "tool_call"                  # Tool invocation
    LLM_CALL = "llm_call"                    # Direct LLM call
    TRANSFORM = "transform"                   # Data transformation
    COORDINATOR = "coordinator"               # Parallel/hybrid coordinator
    MERGE = "merge"                          # Parallel results merge
    PARALLEL_CREW = "parallel_crew"          # Parallel crew execution
    HIERARCHICAL_CREW = "hierarchical_crew"  # Hierarchical crew execution


class StepStatus(str, Enum):
    """Status of an execution step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(str, Enum):
    """Types of execution events emitted via WebSocket."""
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_OUTPUT = "step_output"      # For intermediate results (optional)
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass
class ExecutionStep:
    """
    Single step in workflow execution.

    Represents a discrete unit of work (agent, tool call, etc.)
    with sanitized input/output summaries for transparency.
    """
    step_id: str
    step_name: str
    step_type: StepType
    status: StepStatus = StepStatus.PENDING
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class RunContext:
    """
    In-memory execution context for a single workflow run.

    Tracks all steps, their order, and the final output.
    This is ephemeral - cleaned up 60 seconds after run completion.
    """
    run_id: str
    workflow_id: str
    steps: Dict[str, ExecutionStep] = field(default_factory=dict)
    step_order: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    final_output: Optional[str] = None


@dataclass
class ExecutionEvent:
    """
    WebSocket event payload.

    Represents an event to be broadcast to connected clients.
    """
    event_type: EventType
    run_id: str
    step_id: Optional[str]
    timestamp: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event": self.event_type.value,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "payload": self.payload
        }


class StepTracker:
    """
    Tracks step execution state in memory.

    Provides methods to:
    - Create and manage run contexts
    - Track step lifecycle (start, complete, fail)
    - Auto-cleanup after 60 seconds
    """

    def __init__(self):
        self._runs: Dict[str, RunContext] = {}

    def create_run(self, run_id: str, workflow_id: str) -> RunContext:
        """
        Create a new run context for tracking execution.

        Args:
            run_id: Unique run identifier
            workflow_id: ID of the workflow being executed

        Returns:
            New RunContext instance
        """
        context = RunContext(run_id=run_id, workflow_id=workflow_id)
        self._runs[run_id] = context
        logger.debug(f"Created run context for {run_id}")
        return context

    def get_run(self, run_id: str) -> Optional[RunContext]:
        """
        Get run context by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunContext if found, None otherwise
        """
        return self._runs.get(run_id)

    def start_step(
        self,
        run_id: str,
        step_id: str,
        step_name: str,
        step_type: StepType,
        input_summary: Optional[Dict[str, Any]] = None
    ) -> ExecutionStep:
        """
        Start tracking a new step.

        Args:
            run_id: Run identifier
            step_id: Unique step identifier
            step_name: Human-readable step name
            step_type: Type of step (agent, tool_call, etc.)
            input_summary: Sanitized input data (no CoT)

        Returns:
            New ExecutionStep instance

        Raises:
            ValueError: If run context not found
        """
        context = self._runs.get(run_id)
        if not context:
            # Create context if not exists (for backward compatibility)
            logger.warning(f"Run {run_id} not found, creating new context")
            context = self.create_run(run_id, "unknown")

        step = ExecutionStep(
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            status=StepStatus.RUNNING,
            input_summary=input_summary,
            started_at=datetime.utcnow()
        )
        context.steps[step_id] = step
        context.step_order.append(step_id)
        logger.debug(f"Started step {step_id} ({step_name}) in run {run_id}")
        return step

    def complete_step(
        self,
        run_id: str,
        step_id: str,
        output_summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a step as completed.

        Args:
            run_id: Run identifier
            step_id: Step identifier
            output_summary: Sanitized output data (no CoT)
        """
        context = self._runs.get(run_id)
        if context and step_id in context.steps:
            step = context.steps[step_id]
            step.status = StepStatus.COMPLETED
            step.output_summary = output_summary
            step.completed_at = datetime.utcnow()
            logger.debug(f"Completed step {step_id} in run {run_id}")

    def fail_step(self, run_id: str, step_id: str, error: str) -> None:
        """
        Mark a step as failed.

        Args:
            run_id: Run identifier
            step_id: Step identifier
            error: Error message
        """
        context = self._runs.get(run_id)
        if context and step_id in context.steps:
            step = context.steps[step_id]
            step.status = StepStatus.FAILED
            step.error = error
            step.completed_at = datetime.utcnow()
            logger.debug(f"Failed step {step_id} in run {run_id}: {error}")

    def complete_run(self, run_id: str, final_output: Optional[str] = None) -> None:
        """
        Mark run as completed and schedule cleanup.

        Args:
            run_id: Run identifier
            final_output: Final response text
        """
        context = self._runs.get(run_id)
        if context:
            context.completed_at = datetime.utcnow()
            context.final_output = final_output
            logger.debug(f"Completed run {run_id}")

        # Schedule cleanup after 60 seconds
        self._schedule_cleanup(run_id)

    def fail_run(self, run_id: str) -> None:
        """
        Mark run as failed and schedule cleanup.

        Args:
            run_id: Run identifier
        """
        context = self._runs.get(run_id)
        if context:
            context.completed_at = datetime.utcnow()
            logger.debug(f"Failed run {run_id}")

        # Schedule cleanup after 60 seconds
        self._schedule_cleanup(run_id)

    def _schedule_cleanup(self, run_id: str) -> None:
        """
        Schedule cleanup of run context after 60 seconds.

        Args:
            run_id: Run identifier to clean up
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_later(60, lambda: self._cleanup(run_id))
            else:
                # Fallback: create task for cleanup
                asyncio.create_task(self._async_cleanup(run_id))
        except RuntimeError:
            # No event loop - cleanup immediately or skip
            logger.warning(f"No event loop for cleanup scheduling of run {run_id}")

    async def _async_cleanup(self, run_id: str) -> None:
        """Async cleanup after delay."""
        await asyncio.sleep(60)
        self._cleanup(run_id)

    def _cleanup(self, run_id: str) -> None:
        """
        Remove run context from memory.

        Args:
            run_id: Run identifier to remove
        """
        if run_id in self._runs:
            del self._runs[run_id]
            logger.debug(f"Cleaned up run context for {run_id}")


# Singleton instance for global access
step_tracker = StepTracker()

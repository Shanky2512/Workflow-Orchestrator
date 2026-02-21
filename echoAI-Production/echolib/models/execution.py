"""
EchoAI Execution and HITL Models

Stores workflow execution runs and human-in-the-loop checkpoints.
Execution events are NOT stored here - they are ephemeral via WebSocket
per the transparent_plan.md specification.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Integer, Text, ForeignKey, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class Execution(BaseModel):
    """
    Workflow execution run model.

    Tracks workflow execution runs with status, input/output, and metrics.
    Step-level events are NOT stored here (they go via WebSocket for real-time
    transparency and are ephemeral with 60s TTL).

    Attributes:
        run_id: Primary key UUID
        workflow_id: Reference to workflow being executed
        user_id: User who initiated the execution
        session_id: Optional chat session context
        execution_mode: How the workflow was run (draft/test/final)
        workflow_version: Version string of the workflow
        status: Current execution status
        input_payload: Input data provided to the workflow
        output: Final output from the workflow
        error_message: Error details if failed
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp
        duration_ms: Total execution duration in milliseconds
        agent_count: Number of agents in the workflow

    Status State Machine:
        queued -> running -> completed
                       |
                       +-> hitl_waiting -> hitl_approved -> completed
                       |              |
                       |              +-> hitl_rejected -> completed (with rejection)
                       |
                       +-> failed
                       |
                       +-> cancelled

    Indexes:
        - idx_executions_workflow: Executions for a workflow
        - idx_executions_user: User's executions
        - idx_executions_session: Executions from a session
        - idx_executions_status: Active/waiting executions
        - idx_executions_user_recent: Recent executions for user
    """
    __tablename__ = "executions"

    # Primary key
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    # References
    workflow_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id"),
        nullable=False,
        index=True
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id"),
        nullable=False,
        index=True
    )

    session_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id"),
        nullable=True
    )

    # Execution info
    execution_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False  # draft/test/final
    )

    workflow_version: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(30),
        default="queued",
        server_default="queued",
        nullable=False
    )

    # Input/Output
    input_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    output: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.utcnow(),
        server_default="CURRENT_TIMESTAMP",
        nullable=False
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # Metrics
    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )

    agent_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "execution_mode IN ('draft', 'test', 'final')",
            name="valid_exec_mode"
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'hitl_waiting', 'hitl_approved', "
            "'hitl_rejected', 'completed', 'failed', 'cancelled')",
            name="valid_exec_status"
        ),
        Index("idx_executions_session", "session_id", postgresql_where="session_id IS NOT NULL"),
        Index(
            "idx_executions_status",
            "status",
            postgresql_where="status IN ('running', 'hitl_waiting')"
        ),
        Index("idx_executions_user_recent", "user_id", "started_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert execution to dictionary representation.

        Returns:
            Dict containing execution data.
        """
        return {
            "run_id": str(self.run_id),
            "workflow_id": str(self.workflow_id),
            "user_id": str(self.user_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "execution_mode": self.execution_mode,
            "workflow_version": self.workflow_version,
            "status": self.status,
            "input_payload": self.input_payload or {},
            "output": self.output,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "agent_count": self.agent_count,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HITLCheckpoint(BaseModel):
    """
    Human-in-the-loop checkpoint model.

    Stores HITL checkpoint state when executions are paused for human review.
    Contains the agent output, execution context, and decision history.

    Attributes:
        checkpoint_id: Primary key UUID
        run_id: Reference to the paused execution
        status: Checkpoint status (waiting_for_human/approved/rejected/modified/deferred)
        blocked_at: Agent ID where execution paused
        agent_output: Output from the blocked agent
        tools_used: Tools used by the agent
        execution_metrics: Timing and performance metrics
        state_snapshot: Full state at checkpoint time
        previous_decisions: History of decisions for this checkpoint
        resolved_at: When the checkpoint was resolved
        resolved_by: User who resolved it
        resolution: How it was resolved (approve/reject/modify/defer/rerun)
        resolution_data: Additional resolution data (e.g., modifications)

    Status Values:
        - waiting_for_human: Awaiting human decision
        - approved: Human approved continuation
        - rejected: Human rejected, execution stops
        - modified: Human modified output and approved
        - deferred: Decision deferred for later

    Resolution Values:
        - approve: Continue execution as-is
        - reject: Stop execution
        - modify: Continue with modified output
        - defer: Pause for later decision
        - rerun: Re-execute the blocked agent

    Indexes:
        - idx_hitl_run: Checkpoints for an execution
        - idx_hitl_pending: Waiting checkpoints
    """
    __tablename__ = "hitl_checkpoints"

    # Primary key
    checkpoint_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    # Reference to execution
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("executions.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(30),
        default="waiting_for_human",
        server_default="waiting_for_human",
        nullable=False
    )

    # Where execution paused
    blocked_at: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False  # agent_id
    )

    # Context snapshot
    agent_output: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    tools_used: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    execution_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    state_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    # Decision history
    previous_decisions: Mapped[List[Dict[str, Any]]] = mapped_column(
        ARRAY(JSONB),
        default=list,
        server_default="{}",
        nullable=False
    )

    # Resolution tracking
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    resolved_by: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id"),
        nullable=True
    )

    resolution: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True
    )

    resolution_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True
    )

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('waiting_for_human', 'approved', 'rejected', 'modified', 'deferred')",
            name="valid_hitl_status"
        ),
        CheckConstraint(
            "resolution IN ('approve', 'reject', 'modify', 'defer', 'rerun') OR resolution IS NULL",
            name="valid_resolution"
        ),
        Index(
            "idx_hitl_pending",
            "status",
            postgresql_where="status = 'waiting_for_human'"
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert checkpoint to dictionary representation.

        Returns:
            Dict containing checkpoint data.
        """
        return {
            "checkpoint_id": str(self.checkpoint_id),
            "run_id": str(self.run_id),
            "status": self.status,
            "blocked_at": str(self.blocked_at),
            "agent_output": self.agent_output,
            "tools_used": self.tools_used,
            "execution_metrics": self.execution_metrics,
            "state_snapshot": self.state_snapshot,
            "previous_decisions": self.previous_decisions or [],
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": str(self.resolved_by) if self.resolved_by else None,
            "resolution": self.resolution,
            "resolution_data": self.resolution_data,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

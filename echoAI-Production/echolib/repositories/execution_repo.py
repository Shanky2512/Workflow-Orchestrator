"""
EchoAI Execution Repository

Provides data access for workflow executions and HITL checkpoints.
Executions track workflow runs; HITL checkpoints manage human approval gates.

Note: Step-level execution events are NOT stored in the database.
They are streamed via WebSocket and are ephemeral (60s TTL).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import Execution, HITLCheckpoint


class ExecutionRepository:
    """
    Repository for Execution and HITLCheckpoint entity operations.

    Provides:
        - Execution CRUD and status management
        - HITL checkpoint creation and resolution
        - Queries for running/pending executions

    Execution Status Flow:
        queued -> running -> completed
                       |
                       +-> hitl_waiting -> hitl_approved -> completed
                       |              |
                       |              +-> hitl_rejected -> completed (with rejection)
                       |
                       +-> failed
                       |
                       +-> cancelled
    """

    # ==================== Execution Operations ====================

    async def create_execution(
        self,
        db: AsyncSession,
        user_id: str,
        workflow_id: str,
        execution_mode: str,
        input_payload: Dict[str, Any],
        session_id: Optional[str] = None,
        workflow_version: Optional[str] = None,
        agent_count: Optional[int] = None
    ) -> Execution:
        """
        Create a new workflow execution record.

        Args:
            db: Async database session
            user_id: User who initiated the execution
            workflow_id: Workflow being executed
            execution_mode: How the workflow is run (draft/test/final)
            input_payload: Input data for the workflow
            session_id: Optional chat session context
            workflow_version: Optional version string
            agent_count: Optional number of agents in workflow

        Returns:
            The created Execution instance
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        session_uuid = None
        if session_id:
            session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id

        execution = Execution(
            workflow_id=workflow_uuid,
            user_id=user_uuid,
            session_id=session_uuid,
            execution_mode=execution_mode,
            workflow_version=workflow_version,
            status="queued",
            input_payload=input_payload,
            started_at=datetime.now(timezone.utc),
            agent_count=agent_count
        )

        db.add(execution)
        await db.flush()
        await db.refresh(execution)
        return execution

    async def get_execution(
        self,
        db: AsyncSession,
        run_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Execution]:
        """
        Get an execution by ID.

        Args:
            db: Async database session
            run_id: Execution primary key
            user_id: Optional user ID for scoping

        Returns:
            Execution instance if found, None otherwise
        """
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id

        stmt = select(Execution).where(Execution.run_id == run_uuid)

        if user_id:
            user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
            stmt = stmt.where(Execution.user_id == user_uuid)

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        db: AsyncSession,
        run_id: str,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[Execution]:
        """
        Update execution status and optionally output/error.

        Args:
            db: Async database session
            run_id: Execution primary key
            status: New status value
            output: Optional final output data
            error: Optional error message

        Returns:
            Updated Execution if found, None otherwise
        """
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id

        update_values: Dict[str, Any] = {"status": status}

        if output is not None:
            update_values["output"] = output

        if error is not None:
            update_values["error_message"] = error

        # Set completed_at and duration for terminal states
        if status in ("completed", "failed", "cancelled", "hitl_rejected"):
            now = datetime.now(timezone.utc)
            update_values["completed_at"] = now

            # Calculate duration if we have started_at
            exec_record = await self.get_execution(db, run_id)
            if exec_record and exec_record.started_at:
                duration = (now - exec_record.started_at).total_seconds() * 1000
                update_values["duration_ms"] = int(duration)

        stmt = (
            update(Execution)
            .where(Execution.run_id == run_uuid)
            .values(**update_values)
            .returning(Execution)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
        return updated

    async def get_by_workflow(
        self,
        db: AsyncSession,
        workflow_id: str,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Execution]:
        """
        Get executions for a workflow.

        Args:
            db: Async database session
            workflow_id: Workflow ID
            user_id: Owner user ID for scoping
            limit: Maximum results to return
            offset: Number of records to skip

        Returns:
            List of Execution instances for the workflow
        """
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            select(Execution)
            .where(
                Execution.workflow_id == workflow_uuid,
                Execution.user_id == user_uuid
            )
            .order_by(Execution.started_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_running(
        self,
        db: AsyncSession,
        user_id: str
    ) -> List[Execution]:
        """
        Get all running or waiting executions for a user.

        Args:
            db: Async database session
            user_id: User ID

        Returns:
            List of Execution instances that are in progress
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        active_statuses = ("queued", "running", "hitl_waiting", "hitl_approved")

        stmt = (
            select(Execution)
            .where(
                Execution.user_id == user_uuid,
                Execution.status.in_(active_statuses)
            )
            .order_by(Execution.started_at.desc())
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_session(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 50
    ) -> List[Execution]:
        """
        Get executions from a specific chat session.

        Args:
            db: Async database session
            session_id: Chat session ID
            limit: Maximum results to return

        Returns:
            List of Execution instances from the session
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id

        stmt = (
            select(Execution)
            .where(Execution.session_id == session_uuid)
            .order_by(Execution.started_at.desc())
            .limit(limit)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ==================== HITL Checkpoint Operations ====================

    async def create_checkpoint(
        self,
        db: AsyncSession,
        run_id: str,
        blocked_at: str,
        agent_output: Optional[Dict[str, Any]] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
        tools_used: Optional[Dict[str, Any]] = None,
        execution_metrics: Optional[Dict[str, Any]] = None
    ) -> HITLCheckpoint:
        """
        Create a HITL checkpoint for an execution.

        Called when execution reaches an agent marked for human review.

        Args:
            db: Async database session
            run_id: Execution that is paused
            blocked_at: Agent ID where execution paused
            agent_output: Output from the blocked agent
            state_snapshot: Full execution state at checkpoint
            tools_used: Tools used by the agent
            execution_metrics: Timing/performance metrics

        Returns:
            The created HITLCheckpoint instance
        """
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id
        blocked_at_uuid = UUID(blocked_at) if isinstance(blocked_at, str) else blocked_at

        checkpoint = HITLCheckpoint(
            run_id=run_uuid,
            status="waiting_for_human",
            blocked_at=blocked_at_uuid,
            agent_output=agent_output,
            state_snapshot=state_snapshot,
            tools_used=tools_used,
            execution_metrics=execution_metrics,
            previous_decisions=[]
        )

        db.add(checkpoint)
        await db.flush()
        await db.refresh(checkpoint)

        # Also update execution status
        await self.update_status(db, run_id, "hitl_waiting")

        return checkpoint

    async def get_checkpoint(
        self,
        db: AsyncSession,
        checkpoint_id: str
    ) -> Optional[HITLCheckpoint]:
        """
        Get a HITL checkpoint by ID.

        Args:
            db: Async database session
            checkpoint_id: Checkpoint primary key

        Returns:
            HITLCheckpoint instance if found, None otherwise
        """
        checkpoint_uuid = UUID(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id

        stmt = select(HITLCheckpoint).where(
            HITLCheckpoint.checkpoint_id == checkpoint_uuid
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def resolve_checkpoint(
        self,
        db: AsyncSession,
        checkpoint_id: str,
        user_id: str,
        resolution: str,
        resolution_data: Optional[Dict[str, Any]] = None
    ) -> Optional[HITLCheckpoint]:
        """
        Resolve a HITL checkpoint with human decision.

        Args:
            db: Async database session
            checkpoint_id: Checkpoint to resolve
            user_id: User who made the decision
            resolution: Resolution type (approve/reject/modify/defer/rerun)
            resolution_data: Additional resolution data (e.g., modifications)

        Returns:
            Updated HITLCheckpoint if found, None otherwise
        """
        checkpoint_uuid = UUID(checkpoint_id) if isinstance(checkpoint_id, str) else checkpoint_id
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        # Map resolution to status
        status_map = {
            "approve": "approved",
            "reject": "rejected",
            "modify": "modified",
            "defer": "deferred",
            "rerun": "waiting_for_human"  # Stays waiting for re-execution
        }

        new_status = status_map.get(resolution, "approved")

        # Get current checkpoint for previous_decisions update
        checkpoint = await self.get_checkpoint(db, checkpoint_id)
        if not checkpoint:
            return None

        # Add to previous decisions
        previous = list(checkpoint.previous_decisions or [])
        previous.append({
            "resolution": resolution,
            "resolved_by": str(user_uuid),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolution_data": resolution_data
        })

        stmt = (
            update(HITLCheckpoint)
            .where(HITLCheckpoint.checkpoint_id == checkpoint_uuid)
            .values(
                status=new_status,
                resolved_at=datetime.now(timezone.utc),
                resolved_by=user_uuid,
                resolution=resolution,
                resolution_data=resolution_data,
                previous_decisions=previous
            )
            .returning(HITLCheckpoint)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)

            # Update execution status based on resolution
            if resolution == "approve" or resolution == "modify":
                await self.update_status(db, str(checkpoint.run_id), "hitl_approved")
            elif resolution == "reject":
                await self.update_status(db, str(checkpoint.run_id), "hitl_rejected")

        return updated

    async def get_pending_checkpoints(
        self,
        db: AsyncSession,
        user_id: str
    ) -> List[HITLCheckpoint]:
        """
        Get all pending HITL checkpoints for a user.

        Args:
            db: Async database session
            user_id: User ID

        Returns:
            List of HITLCheckpoint instances waiting for human decision
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        # Join with executions to filter by user
        stmt = (
            select(HITLCheckpoint)
            .join(Execution, HITLCheckpoint.run_id == Execution.run_id)
            .where(
                Execution.user_id == user_uuid,
                HITLCheckpoint.status == "waiting_for_human"
            )
            .order_by(HITLCheckpoint.created_at.desc())
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_checkpoints_by_run(
        self,
        db: AsyncSession,
        run_id: str
    ) -> List[HITLCheckpoint]:
        """
        Get all checkpoints for an execution.

        Args:
            db: Async database session
            run_id: Execution ID

        Returns:
            List of HITLCheckpoint instances for the execution
        """
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id

        stmt = (
            select(HITLCheckpoint)
            .where(HITLCheckpoint.run_id == run_uuid)
            .order_by(HITLCheckpoint.created_at)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_checkpoint(
        self,
        db: AsyncSession,
        run_id: str
    ) -> Optional[HITLCheckpoint]:
        """
        Get the most recent checkpoint for an execution.

        Args:
            db: Async database session
            run_id: Execution ID

        Returns:
            Most recent HITLCheckpoint if any, None otherwise
        """
        run_uuid = UUID(run_id) if isinstance(run_id, str) else run_id

        stmt = (
            select(HITLCheckpoint)
            .where(HITLCheckpoint.run_id == run_uuid)
            .order_by(HITLCheckpoint.created_at.desc())
            .limit(1)
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

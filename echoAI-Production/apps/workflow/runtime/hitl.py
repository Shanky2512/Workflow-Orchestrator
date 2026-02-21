"""
Human-in-the-Loop (HITL) Runtime Support

HITL is a first-class execution boundary with:
- Pause (deterministic checkpoint)
- Context (full decision info)
- Controlled action (approve/reject/modify/defer)
- Re-validation (after changes)
- Audit (immutable trail)

This is NOT a UI feature - it's a workflow state enforced by runtime.

Architecture:
1. Design-time: Workflow declares HITL points
2. Compile-time: Validator enforces HITL rules
3. Run-time: Orchestrator manages HITL state
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path

from echolib.utils import new_id

logger = logging.getLogger(__name__)


# ============================================================================
# HITL STATE MACHINE
# ============================================================================

class HITLState(str, Enum):
    """
    HITL execution states

    State transitions:
    RUNNING → WAITING_FOR_HUMAN → APPROVED/REJECTED/MODIFIED/DEFERRED → RESUMED/TERMINATED
    """
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"
    RESUMED = "resumed"
    TERMINATED = "terminated"


class HITLAction(str, Enum):
    """Allowed human actions during HITL"""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"
    RERUN = "rerun"


# ============================================================================
# HITL DATA MODELS
# ============================================================================

@dataclass
class HITLContext:
    """
    Full decision context presented to human

    Attributes:
        workflow_id: Workflow being executed
        run_id: Execution run ID
        blocked_at: Where execution is paused (agent ID or step)
        agent_output: Output from the agent that triggered HITL
        tools_used: List of tools invoked
        execution_metrics: Cost, duration, tokens
        state_snapshot: Current workflow state
        previous_decisions: History of HITL decisions in this run
    """
    workflow_id: str
    run_id: str
    blocked_at: str
    agent_output: Optional[Dict[str, Any]] = None
    tools_used: List[str] = field(default_factory=list)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    previous_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "blocked_at": self.blocked_at,
            "agent_output": self.agent_output,
            "tools_used": self.tools_used,
            "execution_metrics": self.execution_metrics,
            "state_snapshot": self.state_snapshot,
            "previous_decisions": self.previous_decisions
        }


@dataclass
class HITLDecision:
    """
    Human decision record (audit trail)

    Attributes:
        decision_id: Unique decision ID
        run_id: Execution run ID
        action: Action taken
        actor: Who made the decision
        timestamp: When decision was made
        context_snapshot: State at decision time
        changes: What was changed (for modifications)
        rationale: Optional human comment
    """
    decision_id: str
    run_id: str
    action: HITLAction
    actor: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_snapshot: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "action": self.action.value,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "context_snapshot": self.context_snapshot,
            "changes": self.changes,
            "rationale": self.rationale
        }


@dataclass
class HITLCheckpoint:
    """
    HITL execution checkpoint

    Stored before entering WAITING_FOR_HUMAN state.
    Enables deterministic resume after human action.

    Attributes:
        checkpoint_id: Unique checkpoint ID
        run_id: Execution run ID
        state: HITL state
        workflow_state: Complete workflow state
        execution_position: Where execution paused
        created_at: Checkpoint creation time
        timeout_at: When this checkpoint expires
    """
    checkpoint_id: str
    run_id: str
    state: HITLState
    workflow_state: Dict[str, Any]
    execution_position: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if checkpoint has expired"""
        if not self.timeout_at:
            return False
        return datetime.utcnow() > self.timeout_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "state": self.state.value,
            "workflow_state": self.workflow_state,
            "execution_position": self.execution_position,
            "created_at": self.created_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None
        }


# ============================================================================
# HITL MANAGER (Runtime Orchestrator)
# ============================================================================

class HITLManager:
    """
    Manages HITL lifecycle for workflow executions

    Responsibilities:
    - Create HITL checkpoints
    - Store execution context
    - Record human decisions
    - Manage timeouts
    - Enforce validation after modifications
    - Maintain audit trail

    Integration with LangGraph:
    - Uses LangGraph checkpointing for state persistence
    - Stateless - can survive restarts

    Usage:
        hitl = HITLManager()

        # In workflow - request approval
        hitl.request_approval(
            run_id="exec_123",
            workflow_id="wf_456",
            blocked_at="agent:analyst",
            context=context_data
        )

        # Human approves via API
        hitl.approve(run_id="exec_123", actor="user@example.com")

        # Check status
        status = hitl.get_status("exec_123")
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize HITL manager

        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "storage" / "hitl"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._checkpoints: Dict[str, HITLCheckpoint] = {}
        self._contexts: Dict[str, HITLContext] = {}
        self._decisions: Dict[str, List[HITLDecision]] = {}
        self._states: Dict[str, HITLState] = {}

        # Load existing checkpoints
        self._load_checkpoints()

    # ========================================================================
    # CORE HITL OPERATIONS
    # ========================================================================

    def request_approval(
        self,
        run_id: str,
        workflow_id: str,
        blocked_at: str,
        context: Dict[str, Any],
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Request human approval (pauses execution)

        This is called from within workflow execution to pause and wait
        for human decision.

        Args:
            run_id: Execution run ID
            workflow_id: Workflow ID
            blocked_at: Where execution is paused (e.g., "agent:analyst")
            context: Full decision context
            timeout_seconds: Optional timeout

        Returns:
            Interrupt info for frontend

        Behavior:
            1. Create checkpoint
            2. Store context
            3. Set state = WAITING_FOR_HUMAN
            4. Return interrupt info
        """
        # Create checkpoint
        checkpoint = self._create_checkpoint(
            run_id=run_id,
            state=HITLState.WAITING_FOR_HUMAN,
            workflow_state=context.get("state_snapshot", {}),
            execution_position=blocked_at,
            timeout_seconds=timeout_seconds
        )

        # Store context — include hitl_config inside state_snapshot so
        # derive_allowed_decisions can retrieve it in the /hitl/decide route.
        snapshot = dict(context.get("state_snapshot", {}))
        if "hitl_config" in context:
            snapshot["hitl_config"] = context["hitl_config"]

        hitl_context = HITLContext(
            workflow_id=workflow_id,
            run_id=run_id,
            blocked_at=blocked_at,
            agent_output=context.get("agent_output"),
            tools_used=context.get("tools_used", []),
            execution_metrics=context.get("execution_metrics", {}),
            state_snapshot=snapshot,
            previous_decisions=self._get_decisions(run_id)
        )

        self._contexts[run_id] = hitl_context
        self._states[run_id] = HITLState.WAITING_FOR_HUMAN

        # Save to disk
        self._save_checkpoint(checkpoint)
        self._save_context(run_id, hitl_context)

        # Return interrupt payload
        return {
            "interrupted": True,
            "run_id": run_id,
            "workflow_id": workflow_id,
            "blocked_at": blocked_at,
            "checkpoint_id": checkpoint.checkpoint_id,
            "context": hitl_context.to_dict(),
            "allowed_actions": [a.value for a in HITLAction],
            "status": "waiting_for_human"
        }

    def approve(self, run_id: str, actor: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        """
        Approve execution (resume without changes)

        Args:
            run_id: Execution run ID
            actor: Who approved
            rationale: Optional comment

        Returns:
            Resume info
        """
        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.APPROVED

        # Record decision
        self._record_decision(
            run_id=run_id,
            action=HITLAction.APPROVE,
            actor=actor,
            rationale=rationale
        )

        return {
            "action": "approve",
            "run_id": run_id,
            "actor": actor,
            "status": "approved",
            "can_resume": True
        }

    def reject(self, run_id: str, actor: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        """
        Reject execution (terminate workflow)

        Args:
            run_id: Execution run ID
            actor: Who rejected
            rationale: Optional comment

        Returns:
            Termination info
        """
        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.REJECTED

        # Record decision
        self._record_decision(
            run_id=run_id,
            action=HITLAction.REJECT,
            actor=actor,
            rationale=rationale
        )

        return {
            "action": "reject",
            "run_id": run_id,
            "actor": actor,
            "status": "rejected",
            "can_resume": False,
            "terminated": True
        }

    def modify(
        self,
        run_id: str,
        actor: str,
        changes: Dict[str, Any],
        rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Modify workflow/agent configuration (TEMP workflows only)

        Args:
            run_id: Execution run ID
            actor: Who made changes
            changes: Changes to apply
            rationale: Optional comment

        Returns:
            Modification info

        Behavior:
            1. Apply changes
            2. Invalidate validation
            3. Set state = MODIFIED
            4. Require re-validation before resume
        """
        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.MODIFIED

        # Record decision with changes
        self._record_decision(
            run_id=run_id,
            action=HITLAction.MODIFY,
            actor=actor,
            changes=changes,
            rationale=rationale
        )

        return {
            "action": "modify",
            "run_id": run_id,
            "actor": actor,
            "status": "modified",
            "changes": changes,
            "validation_required": True,
            "can_resume": False  # Must re-validate first
        }

    def defer(
        self,
        run_id: str,
        actor: str,
        defer_until: Optional[datetime] = None,
        rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Defer decision (postpone approval)

        Args:
            run_id: Execution run ID
            actor: Who deferred
            defer_until: When to revisit
            rationale: Optional comment

        Returns:
            Defer info
        """
        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.DEFERRED

        # Update checkpoint timeout
        if run_id in self._checkpoints and defer_until:
            self._checkpoints[run_id].timeout_at = defer_until

        # Record decision
        self._record_decision(
            run_id=run_id,
            action=HITLAction.DEFER,
            actor=actor,
            rationale=rationale
        )

        return {
            "action": "defer",
            "run_id": run_id,
            "actor": actor,
            "status": "deferred",
            "defer_until": defer_until.isoformat() if defer_until else None,
            "can_resume": False
        }

    def handle_timeout(self, run_id: str) -> Dict[str, Any]:
        """
        Handle HITL timeout (auto-reject)

        Args:
            run_id: Execution run ID

        Returns:
            Timeout result
        """
        checkpoint = self._checkpoints.get(run_id)
        if checkpoint and checkpoint.is_expired():
            # Update state
            self._states[run_id] = HITLState.REJECTED

            # Record decision
            self._record_decision(
                run_id=run_id,
                action=HITLAction.REJECT,
                actor="system",
                rationale="HITL timeout - auto-rejected"
            )

            return {
                "action": "reject",
                "run_id": run_id,
                "status": "timeout",
                "terminated": True
            }

        return {"status": "not_expired"}

    # ========================================================================
    # STATUS & CONTEXT QUERIES
    # ========================================================================

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get HITL status for run

        Returns:
            {
                "run_id": "...",
                "state": "waiting_for_human",
                "blocked_at": "agent:analyst",
                "allowed_actions": ["approve", "reject", ...]
            }
        """
        state = self._states.get(run_id, HITLState.RUNNING)
        context = self._contexts.get(run_id)

        return {
            "run_id": run_id,
            "state": state.value,
            "blocked_at": context.blocked_at if context else None,
            "allowed_actions": [a.value for a in HITLAction],
            "has_pending_review": state == HITLState.WAITING_FOR_HUMAN
        }

    def get_context(self, run_id: str) -> Optional[HITLContext]:
        """Get full HITL context for run"""
        return self._contexts.get(run_id)

    def get_decisions(self, run_id: str) -> List[HITLDecision]:
        """Get all decisions for run (audit trail)"""
        return self._decisions.get(run_id, [])

    def list_pending_reviews(self) -> List[Dict[str, Any]]:
        """List all runs waiting for human review"""
        pending = []
        for run_id, state in self._states.items():
            if state == HITLState.WAITING_FOR_HUMAN:
                context = self._contexts.get(run_id)
                if context:
                    pending.append({
                        "run_id": run_id,
                        "workflow_id": context.workflow_id,
                        "blocked_at": context.blocked_at,
                        "created_at": self._checkpoints[run_id].created_at.isoformat() if run_id in self._checkpoints else None
                    })
        return pending

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _create_checkpoint(
        self,
        run_id: str,
        state: HITLState,
        workflow_state: Dict[str, Any],
        execution_position: str,
        timeout_seconds: Optional[int] = None
    ) -> HITLCheckpoint:
        """Create HITL checkpoint"""
        checkpoint_id = new_id("hitl_cp_")

        timeout_at = None
        if timeout_seconds:
            timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        checkpoint = HITLCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            state=state,
            workflow_state=workflow_state,
            execution_position=execution_position,
            timeout_at=timeout_at
        )

        self._checkpoints[run_id] = checkpoint

        return checkpoint

    def _record_decision(
        self,
        run_id: str,
        action: HITLAction,
        actor: str,
        changes: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None
    ):
        """Record human decision (audit trail)"""
        decision_id = new_id("hitl_dec_")

        context = self._contexts.get(run_id)

        decision = HITLDecision(
            decision_id=decision_id,
            run_id=run_id,
            action=action,
            actor=actor,
            context_snapshot=context.to_dict() if context else None,
            changes=changes,
            rationale=rationale
        )

        if run_id not in self._decisions:
            self._decisions[run_id] = []

        self._decisions[run_id].append(decision)

        # Save to disk
        self._save_decision(run_id, decision)

    def _get_decisions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get decision history for context"""
        decisions = self._decisions.get(run_id, [])
        return [d.to_dict() for d in decisions]

    def _validate_state(self, run_id: str, expected_state: HITLState):
        """Validate current HITL state"""
        current_state = self._states.get(run_id)
        if current_state != expected_state:
            raise ValueError(
                f"Invalid HITL state for {run_id}: "
                f"expected {expected_state.value}, got {current_state.value if current_state else 'NONE'}"
            )

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _save_checkpoint(self, checkpoint: HITLCheckpoint):
        """Save checkpoint to disk"""
        file_path = self.storage_dir / f"{checkpoint.run_id}_checkpoint.json"
        with open(file_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def _save_context(self, run_id: str, context: HITLContext):
        """Save context to disk"""
        file_path = self.storage_dir / f"{run_id}_context.json"
        with open(file_path, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)

    def _save_decision(self, run_id: str, decision: HITLDecision):
        """Save decision to disk (append to audit log)"""
        file_path = self.storage_dir / f"{run_id}_decisions.jsonl"
        with open(file_path, 'a') as f:
            f.write(json.dumps(decision.to_dict()) + '\n')

    def _load_checkpoints(self):
        """Load existing checkpoints from disk"""
        if not self.storage_dir.exists():
            return

        for file_path in self.storage_dir.glob("*_checkpoint.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                run_id = data["run_id"]
                checkpoint = HITLCheckpoint(
                    checkpoint_id=data["checkpoint_id"],
                    run_id=run_id,
                    state=HITLState(data["state"]),
                    workflow_state=data["workflow_state"],
                    execution_position=data["execution_position"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    timeout_at=datetime.fromisoformat(data["timeout_at"]) if data.get("timeout_at") else None
                )

                self._checkpoints[run_id] = checkpoint
                self._states[run_id] = checkpoint.state

            except Exception as e:
                print(f"Error loading checkpoint {file_path}: {e}")


# ============================================================================
# HITL ALLOWED DECISIONS DERIVATION
# ============================================================================

def derive_allowed_decisions(hitl_config: dict) -> list:
    """
    Derive allowed_decisions list from hitl_config.

    Supports BOTH formats:
    - New format: hitl_config has "allowed_decisions" list directly
    - Legacy format: hitl_config has "allowEdit" / "allowDefer" booleans

    Approve and Reject are ALWAYS allowed (non-negotiable safety constraint).

    Args:
        hitl_config: The hitl_config dict from node metadata.

    Returns:
        List of allowed action strings, e.g. ["approve", "reject", "edit", "defer"]
    """
    if not hitl_config:
        return ["approve", "reject"]

    # If new format exists, use it directly
    if "allowed_decisions" in hitl_config:
        decisions = list(hitl_config["allowed_decisions"])
        # Enforce: approve and reject are always present
        for required in ("approve", "reject"):
            if required not in decisions:
                decisions.append(required)
        return decisions

    # Legacy boolean format -> derive list
    decisions = ["approve", "reject"]  # Always allowed
    if hitl_config.get("allowEdit", True):
        decisions.append("edit")
    if hitl_config.get("allowDefer", False):
        decisions.append("defer")
    return decisions


# ============================================================================
# HITL DB MANAGER (Database-Backed Persistence)
# ============================================================================

class HITLDBManager(HITLManager):
    """
    Database-backed HITL manager that extends HITLManager.

    Adds persistent storage via PostgreSQL for HITL state, checkpoints,
    contexts, and decisions. This ensures state survives process restarts
    and is consistent across multiple Celery workers.

    Drop-in replacement for HITLManager — all existing methods work unchanged.
    The file-based persistence from HITLManager is retained as a fallback
    and for backward compatibility. DB persistence is layered on top.

    Usage:
        hitl = HITLDBManager()

        # All existing HITLManager methods work identically:
        hitl.request_approval(run_id, workflow_id, blocked_at, context)
        hitl.approve(run_id, actor)
        hitl.get_status(run_id)

        # New method for triggering resume via Celery:
        hitl.trigger_resume(run_id, action, payload)
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize HITLDBManager.

        Calls parent HITLManager.__init__ to preserve file-based persistence.
        Adds database connection setup for persistent storage.

        Args:
            storage_dir: Directory for file-based storage (fallback).
        """
        super().__init__(storage_dir=storage_dir)

        # Database persistence flag - set to True once DB is verified available
        self._db_available = False
        self._init_db_persistence()

    def _init_db_persistence(self):
        """
        Initialize database persistence layer.

        Attempts to verify database connectivity. If the database is not
        available, falls back to file-based persistence (parent class behavior).
        """
        try:
            from echolib.config import settings
            # Verify we have a database URL configured
            if settings.database_url:
                self._db_available = True
                logger.info("HITLDBManager: Database persistence enabled")
            else:
                logger.warning(
                    "HITLDBManager: No database_url configured, "
                    "using file-based persistence only"
                )
        except Exception as e:
            logger.warning(
                f"HITLDBManager: Database init failed ({e}), "
                f"using file-based persistence only"
            )

    def request_approval(
        self,
        run_id: str,
        workflow_id: str,
        blocked_at: str,
        context: Dict[str, Any],
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Request human approval with DB persistence.

        Extends parent method by also persisting to database.

        Args:
            run_id: Execution run ID
            workflow_id: Workflow ID
            blocked_at: Where execution is paused
            context: Full decision context
            timeout_seconds: Optional timeout

        Returns:
            Interrupt info for frontend
        """
        # Call parent for file-based persistence + in-memory state
        result = super().request_approval(
            run_id=run_id,
            workflow_id=workflow_id,
            blocked_at=blocked_at,
            context=context,
            timeout_seconds=timeout_seconds,
        )

        # Persist to DB if available
        if self._db_available:
            self._persist_hitl_state_to_db(run_id, "waiting_for_human", result)

        return result

    def approve(self, run_id: str, actor: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        """Approve with DB persistence."""
        result = super().approve(run_id=run_id, actor=actor, rationale=rationale)
        if self._db_available:
            self._persist_hitl_state_to_db(run_id, "approved", result)
        return result

    def reject(self, run_id: str, actor: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        """Reject with DB persistence."""
        result = super().reject(run_id=run_id, actor=actor, rationale=rationale)
        if self._db_available:
            self._persist_hitl_state_to_db(run_id, "rejected", result)
        return result

    def modify(
        self,
        run_id: str,
        actor: str,
        changes: Dict[str, Any],
        rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """Modify with DB persistence."""
        result = super().modify(run_id=run_id, actor=actor, changes=changes, rationale=rationale)
        if self._db_available:
            self._persist_hitl_state_to_db(run_id, "modified", result)
        return result

    def defer(
        self,
        run_id: str,
        actor: str,
        defer_until: Optional[datetime] = None,
        rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """Defer with DB persistence."""
        result = super().defer(run_id=run_id, actor=actor, defer_until=defer_until, rationale=rationale)
        if self._db_available:
            self._persist_hitl_state_to_db(run_id, "deferred", result)
        return result

    def trigger_resume(
        self,
        run_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger workflow resume after a HITL decision.

        Dispatches a Celery task to resume the workflow from its
        interrupt point using LangGraph's Command(resume=...).

        Args:
            run_id: The run_id of the paused workflow.
            action: The human decision ("approve", "reject", "edit", "defer").
            payload: Additional data for the resume (edit content, etc.).

        Returns:
            Dict with task dispatch status.
        """
        if action == "reject":
            logger.info(f"HITL reject for run '{run_id}' - no resume needed")
            return {
                "run_id": run_id,
                "action": action,
                "status": "terminated",
                "resume_dispatched": False,
            }

        if action == "defer":
            logger.info(f"HITL defer for run '{run_id}' - no resume needed")
            return {
                "run_id": run_id,
                "action": action,
                "status": "deferred",
                "resume_dispatched": False,
            }

        # For approve and edit, dispatch resume task
        logger.info(f"Dispatching resume task for run '{run_id}', action='{action}'")

        # Resolve workflow_id from context
        context = self._contexts.get(run_id)
        workflow_id = context.workflow_id if context else None

        # Try Celery first, fall back to direct resume
        try:
            from echolib.celery.celery_tasks import resume_workflow_task

            task_result = resume_workflow_task.delay(
                run_id=run_id,
                action=action,
                payload=payload or {},
                workflow_id=workflow_id,
            )

            return {
                "run_id": run_id,
                "action": action,
                "status": "resume_dispatched",
                "resume_dispatched": True,
                "task_id": task_result.id,
            }

        except Exception as e:
            # Celery not installed, Redis not running, or any dispatch failure
            # — fall back to direct resume by caller
            logger.info(
                f"Celery dispatch unavailable for run '{run_id}' ({type(e).__name__}: {e}). "
                f"Falling back to direct resume."
            )
            return {
                "run_id": run_id,
                "action": action,
                "workflow_id": workflow_id,
                "status": "resume_pending",
                "resume_dispatched": False,
                "payload": payload or {},
            }

    def _persist_hitl_state_to_db(
        self,
        run_id: str,
        state: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Persist HITL state to database.

        Uses a lightweight approach: stores the HITL state as a JSON record
        in the database. This does not require new tables — it leverages
        the existing echoAI database infrastructure.

        Args:
            run_id: Execution run ID.
            state: Current HITL state string.
            data: Full state data to persist.
        """
        try:
            import asyncio
            from echolib.database import get_db_session
            from sqlalchemy import text
            from datetime import datetime as dt, timezone

            record = {
                "run_id": run_id,
                "state": state,
                "data": json.dumps(data, default=str),
                "updated_at": dt.now(timezone.utc),
            }

            async def _persist():
                async with get_db_session() as db:
                    # Upsert HITL state into a lightweight tracking approach
                    # using the existing session metadata or a dedicated query
                    await db.execute(
                        text(
                            "INSERT INTO hitl_states (run_id, state, data, updated_at) "
                            "VALUES (:run_id, :state, :data, :updated_at) "
                            "ON CONFLICT (run_id) DO UPDATE SET "
                            "state = :state, data = :data, updated_at = :updated_at"
                        ),
                        record,
                    )
                    await db.commit()

            # Schedule on the main event loop.
            # This code typically runs in a background thread (via asyncio.to_thread),
            # so asyncpg connections bound to the main loop would fail if we use
            # asyncio.run() (creates a new loop). Instead, use run_coroutine_threadsafe
            # to dispatch onto the main loop where asyncpg connections live.
            try:
                loop = asyncio.get_running_loop()
                # We're directly in an async context — create_task works
                loop.create_task(_persist())
            except RuntimeError:
                # No running loop in this thread — we're in a background thread.
                # Get the main event loop and schedule there.
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(_persist(), loop)
                        future.result(timeout=10)  # Wait with timeout
                    else:
                        loop.run_until_complete(_persist())
                except RuntimeError:
                    # Last resort: no event loop at all — skip DB persist
                    logger.warning(
                        f"HITLDBManager: No event loop available for DB persist "
                        f"(run '{run_id}'). File-based fallback active."
                    )

        except Exception as e:
            # DB persistence failure is non-fatal — file-based persistence
            # from parent class serves as fallback
            logger.warning(
                f"HITLDBManager: DB persist failed for run '{run_id}': {e}. "
                f"File-based persistence is still active."
            )

"""
Edit Resolution Engine and Workflow Replay Engine for Smart Replay.

When a human reviewer edits data at a HITL checkpoint, the system must
determine WHERE to restart execution. This module analyzes the edit
against the workflow's node graph to make that determination.

Scenarios:
    1. Future only: Edit affects only downstream nodes -> resume from next node.
    2. Immediate prev: Edit corrects the previous node's output -> re-execute previous + continue.
    3. Chain correction: Edit changes data from an upstream source -> re-execute from that node.
    4. Root change: Edit changes the original goal -> restart workflow with merged context.

Architecture:
    - EditResolutionEngine: Analyzes edit content vs node graph to determine restart_from_node_id.
    - WorkflowReplayEngine: Triggers resume with context merge using the resolved restart point.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EditScope(str, Enum):
    """Classification of edit scope relative to workflow graph."""
    FUTURE_ONLY = "future_only"       # Edit only affects downstream nodes
    IMMEDIATE_PREV = "immediate_prev" # Edit corrects immediate predecessor output
    CHAIN_CORRECTION = "chain_correction"  # Edit changes an upstream data source
    ROOT_CHANGE = "root_change"       # Edit changes the original goal/input


@dataclass
class EditResolution:
    """
    Result of edit resolution analysis.

    Attributes:
        scope: Classification of the edit scope.
        restart_from_node_id: Node ID to restart execution from.
            None means resume from the next node (future_only).
        context_merge: Whether to merge edit content into execution context.
        explanation: Human-readable explanation of the resolution.
    """
    scope: EditScope
    restart_from_node_id: Optional[str]
    context_merge: bool
    explanation: str


class EditResolutionEngine:
    """
    Analyzes an edit prompt against the workflow node graph to determine
    the optimal restart point for Smart Replay.

    The engine examines:
    - Which node the HITL pause occurred at
    - What the edit content references (node outputs, inputs, goals)
    - The workflow's connection graph to find the affected node

    Usage:
        engine = EditResolutionEngine()
        resolution = engine.resolve(
            edit_content={"prompt": "Use fresh data from the API call"},
            hitl_node_id="hitl_review_1",
            workflow=workflow_def,
            execution_history=history
        )
        # resolution.restart_from_node_id -> "api_call_1"
        # resolution.scope -> EditScope.CHAIN_CORRECTION
    """

    def resolve(
        self,
        edit_content: Dict[str, Any],
        hitl_node_id: str,
        workflow: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
    ) -> EditResolution:
        """
        Determine where to restart the workflow based on the edit.

        Args:
            edit_content: The edit data from the human reviewer.
                Expected keys: "prompt" (edit instructions), "changes" (specific field changes).
            hitl_node_id: The node ID where the HITL pause occurred.
            workflow: Full workflow definition (agents, connections).
            execution_history: Per-node execution history from state.

        Returns:
            EditResolution with the determined restart point and scope.
        """
        edit_prompt = edit_content.get("prompt", "")
        changes = edit_content.get("changes", {})

        # Build adjacency structures
        connections = workflow.get("connections", [])
        predecessors = self._build_predecessor_map(connections)
        node_ids = self._get_ordered_node_ids(workflow)
        executed_ids = [e["node_id"] for e in execution_history]

        # Get the immediate predecessor of the HITL node
        immediate_prev = predecessors.get(hitl_node_id)

        # Scenario 4: Root change — edit references original input or goal
        if self._is_root_change(edit_prompt, changes, node_ids):
            first_node = node_ids[0] if node_ids else None
            return EditResolution(
                scope=EditScope.ROOT_CHANGE,
                restart_from_node_id=first_node,
                context_merge=True,
                explanation=(
                    "Edit changes the original goal or input. "
                    "Restarting workflow from the beginning with merged context."
                ),
            )

        # Scenario 3: Chain correction — edit references a specific upstream node
        referenced_node = self._find_referenced_node(
            edit_prompt, changes, executed_ids, hitl_node_id
        )
        if referenced_node and referenced_node != immediate_prev:
            return EditResolution(
                scope=EditScope.CHAIN_CORRECTION,
                restart_from_node_id=referenced_node,
                context_merge=True,
                explanation=(
                    f"Edit references upstream node '{referenced_node}'. "
                    f"Re-executing from that node forward."
                ),
            )

        # Scenario 2: Immediate prev — edit corrects the previous node's output
        if immediate_prev and self._affects_previous_output(
            edit_prompt, changes, immediate_prev, execution_history
        ):
            return EditResolution(
                scope=EditScope.IMMEDIATE_PREV,
                restart_from_node_id=immediate_prev,
                context_merge=False,
                explanation=(
                    f"Edit corrects output from immediate predecessor '{immediate_prev}'. "
                    f"Re-executing that node and continuing."
                ),
            )

        # Scenario 1: Future only — edit only affects downstream processing
        return EditResolution(
            scope=EditScope.FUTURE_ONLY,
            restart_from_node_id=None,  # Resume from next node
            context_merge=True,
            explanation=(
                "Edit only affects downstream processing. "
                "Resuming from the next node with edit applied to context."
            ),
        )

    def _build_predecessor_map(
        self, connections: List[Dict[str, Any]]
    ) -> Dict[str, Optional[str]]:
        """Build a map of node_id -> immediate predecessor node_id."""
        predecessors: Dict[str, Optional[str]] = {}
        for conn in connections:
            from_node = conn.get("from")
            to_node = conn.get("to")
            if from_node and to_node:
                predecessors[to_node] = from_node
        return predecessors

    def _get_ordered_node_ids(self, workflow: Dict[str, Any]) -> List[str]:
        """Extract ordered node IDs from workflow definition."""
        node_ids = []
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, str):
                node_ids.append(agent_entry)
            elif isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
                if aid:
                    node_ids.append(aid)
        return node_ids

    def _is_root_change(
        self,
        edit_prompt: str,
        changes: Dict[str, Any],
        node_ids: List[str],
    ) -> bool:
        """
        Detect if the edit changes the original goal or root input.

        Heuristic: looks for keywords indicating goal/input changes.
        """
        root_indicators = [
            "change goal", "change the goal",
            "new objective", "different objective",
            "start over", "restart",
            "change input", "change the input",
            "original request", "change the task",
            "different task", "new task",
        ]
        prompt_lower = edit_prompt.lower()
        for indicator in root_indicators:
            if indicator in prompt_lower:
                return True

        # Check if changes target root-level keys
        root_keys = {"user_input", "original_user_input", "task_description", "goal"}
        if root_keys.intersection(changes.keys()):
            return True

        return False

    def _find_referenced_node(
        self,
        edit_prompt: str,
        changes: Dict[str, Any],
        executed_ids: List[str],
        hitl_node_id: str,
    ) -> Optional[str]:
        """
        Find if the edit references a specific upstream node by ID or name.

        Searches the edit prompt for node IDs that appear in the execution history.
        """
        prompt_lower = edit_prompt.lower()

        # Check if any executed node ID is mentioned in the edit prompt
        for node_id in executed_ids:
            if node_id == hitl_node_id:
                continue
            if node_id.lower() in prompt_lower:
                return node_id

        # Check if changes dict references a specific node
        target_node = changes.get("target_node") or changes.get("restart_from")
        if target_node and target_node in executed_ids:
            return target_node

        return None

    def _affects_previous_output(
        self,
        edit_prompt: str,
        changes: Dict[str, Any],
        immediate_prev: str,
        execution_history: List[Dict[str, Any]],
    ) -> bool:
        """
        Determine if the edit is correcting the immediate predecessor's output.

        Heuristic: looks for keywords like "fix", "correct", "redo", "retry"
        or if changes keys overlap with the previous node's output keys.
        """
        correction_indicators = [
            "fix", "correct", "redo", "retry", "wrong",
            "error", "mistake", "redo the", "try again",
        ]
        prompt_lower = edit_prompt.lower()
        for indicator in correction_indicators:
            if indicator in prompt_lower:
                return True

        # Check if changes keys overlap with previous node's output keys
        prev_entries = [
            e for e in execution_history if e["node_id"] == immediate_prev
        ]
        if prev_entries:
            last_output = prev_entries[-1].get("output", {})
            if isinstance(last_output, dict):
                overlap = set(changes.keys()).intersection(last_output.keys())
                if overlap:
                    return True

        return False


class WorkflowReplayEngine:
    """
    Triggers workflow resume with context merge after edit resolution.

    Given an EditResolution, this engine constructs the appropriate
    resume payload and dispatches execution via the workflow executor
    or Celery task.

    Usage:
        replay = WorkflowReplayEngine()
        result = replay.trigger_replay(
            run_id="run_123",
            workflow_id="wf_456",
            resolution=resolution,
            edit_content=edit_data,
            original_state=state_snapshot
        )
    """

    def build_replay_payload(
        self,
        resolution: EditResolution,
        edit_content: Dict[str, Any],
        original_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the payload for workflow replay/resume.

        Args:
            resolution: The edit resolution analysis result.
            edit_content: The human's edit data.
            original_state: The workflow state at the time of HITL pause.

        Returns:
            Payload dict suitable for resume_workflow or execute_workflow.
        """
        payload = {
            "action": "edit",
            "edit_scope": resolution.scope.value,
            "restart_from_node_id": resolution.restart_from_node_id,
            "edit_content": edit_content,
            "explanation": resolution.explanation,
        }

        if resolution.context_merge:
            # Merge edit content into the execution context
            merged_context = dict(original_state)

            # Apply specific field changes from the edit
            changes = edit_content.get("changes", {})
            for key, value in changes.items():
                merged_context[key] = value

            # Inject edit prompt as feedback so agents know about the correction
            edit_prompt = edit_content.get("prompt", "")
            if edit_prompt:
                merged_context["_edit_feedback"] = edit_prompt
                merged_context["_edit_instruction"] = (
                    f"IMPORTANT: The human reviewer provided the following feedback. "
                    f"Incorporate this into your processing: {edit_prompt}"
                )

            payload["merged_context"] = merged_context

        return payload

    def trigger_replay(
        self,
        run_id: str,
        workflow_id: str,
        resolution: EditResolution,
        edit_content: Dict[str, Any],
        original_state: Dict[str, Any],
        execution_mode: str = "draft",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger workflow replay based on edit resolution.

        For ROOT_CHANGE scope, re-executes the entire workflow.
        For other scopes, resumes from the determined restart point.

        Args:
            run_id: Original run ID.
            workflow_id: Workflow identifier.
            resolution: Edit resolution result.
            edit_content: Human's edit data.
            original_state: State at HITL pause.
            execution_mode: "draft", "test", or "final".
            version: Version for final mode.

        Returns:
            Dict with replay status and details.
        """
        payload = self.build_replay_payload(
            resolution, edit_content, original_state
        )

        if resolution.scope == EditScope.ROOT_CHANGE:
            # Full restart — use Celery task if available
            logger.info(
                f"Triggering full workflow restart for run '{run_id}' "
                f"(root change detected)"
            )
            try:
                from echolib.celery.celery_tasks import execute_workflow_task
                task_result = execute_workflow_task.delay(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    input_data=payload.get("merged_context", {}),
                )
                return {
                    "run_id": run_id,
                    "status": "replaying",
                    "scope": resolution.scope.value,
                    "task_id": task_result.id,
                    "explanation": resolution.explanation,
                }
            except Exception as e:
                logger.warning(
                    f"Celery task dispatch failed, returning payload for "
                    f"manual execution: {e}"
                )
                return {
                    "run_id": run_id,
                    "status": "replay_pending",
                    "scope": resolution.scope.value,
                    "payload": payload,
                    "explanation": resolution.explanation,
                }
        else:
            # Partial replay — resume from restart point
            logger.info(
                f"Triggering partial replay for run '{run_id}' "
                f"from node '{resolution.restart_from_node_id}' "
                f"(scope: {resolution.scope.value})"
            )
            try:
                from echolib.celery.celery_tasks import resume_workflow_task
                task_result = resume_workflow_task.delay(
                    run_id=run_id,
                    action="edit",
                    payload=payload,
                )
                return {
                    "run_id": run_id,
                    "status": "replaying",
                    "scope": resolution.scope.value,
                    "restart_from": resolution.restart_from_node_id,
                    "task_id": task_result.id,
                    "explanation": resolution.explanation,
                }
            except Exception as e:
                logger.warning(
                    f"Celery task dispatch failed, returning payload for "
                    f"manual execution: {e}"
                )
                return {
                    "run_id": run_id,
                    "status": "replay_pending",
                    "scope": resolution.scope.value,
                    "restart_from": resolution.restart_from_node_id,
                    "payload": payload,
                    "explanation": resolution.explanation,
                }

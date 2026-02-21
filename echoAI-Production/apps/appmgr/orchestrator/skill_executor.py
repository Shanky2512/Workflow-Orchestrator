"""
EchoAI Orchestrator -- Skill Executor

Executes the orchestrator's execution plan by invoking the appropriate
skill for each step:

    - Workflows: resolved via DI container -> WorkflowExecutor.execute_workflow()
    - Agents: StandaloneAgentExecutor.execute()

Execution Strategies:
    - single:     Execute one skill, return output
    - sequential: Loop steps in order; output N -> input N+1
    - parallel:   Group by parallel_group; asyncio.gather() per group
    - hybrid:     Topological sort on depends_on; independent steps run in parallel
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .agent_executor import StandaloneAgentExecutor

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    Executes an orchestrator plan by dispatching to workflow and agent executors.
    """

    def __init__(self) -> None:
        self._agent_executor = StandaloneAgentExecutor()

    async def execute_plan(
        self,
        db: AsyncSession,
        execution_plan: List[Dict[str, Any]],
        execution_strategy: str,
        user_input: str,
        app_config: Optional[Dict[str, Any]] = None,
        document_context: Optional[str] = None,
        sse_queue: Optional[asyncio.Queue] = None,
    ) -> Dict[str, Any]:
        """
        Execute the orchestrator's plan.

        Note: Agent-vs-workflow deduplication is handled upstream in
        SkillManifestBuilder._build_from_db().  Agents that are already
        embedded inside a linked workflow are excluded from the manifest,
        so the orchestrator LLM will never generate plans that invoke
        the same agent both standalone and within a workflow.

        Args:
            db: Async database session.
            execution_plan: List of step dicts from the orchestrator output.
            execution_strategy: "single" | "sequential" | "parallel" | "hybrid"
            user_input: The original/enhanced user input.
            app_config: Optional application config dict (not used directly
                        but available for future extensibility).
            document_context: Optional document context retrieved from
                session-scoped RAG index. When provided, it is prepended
                to each step's input so workflows and agents receive the
                relevant document chunks.

        Returns:
            Dict with keys:
                results: {output_key: result_text} for each step
                final_output: str -- the final response text
                execution_log: list of step log dicts
        """
        # Store document_context on instance for use in _execute_step
        self._document_context = document_context
        # Store sse_queue on instance for use in _execute_agent
        self._sse_queue = sse_queue

        if not execution_plan:
            return {
                "results": {},
                "final_output": "",
                "execution_log": [],
            }

        strategy = execution_strategy.lower()

        if strategy == "single":
            return await self._execute_single(
                db, execution_plan, user_input
            )
        elif strategy == "sequential":
            return await self._execute_sequential(
                db, execution_plan, user_input
            )
        elif strategy == "parallel":
            return await self._execute_parallel(
                db, execution_plan, user_input
            )
        elif strategy == "hybrid":
            return await self._execute_hybrid(
                db, execution_plan, user_input
            )
        else:
            logger.warning(
                "Unknown execution strategy '%s'; falling back to sequential",
                strategy,
            )
            return await self._execute_sequential(
                db, execution_plan, user_input
            )

    # ------------------------------------------------------------------
    # Single
    # ------------------------------------------------------------------

    async def _execute_single(
        self,
        db: AsyncSession,
        plan: List[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        """Execute a single skill."""
        step = plan[0]
        result, log_entry = await self._execute_step(
            db, step, user_input, {}
        )

        # HITL interrupt check
        if isinstance(result, dict) and result.get("hitl_interrupted"):
            return {
                "hitl_interrupted": True,
                "completed_steps": [],
                "interrupted_at_step": step.get("step", 0),
                "interrupt_payload": result,
                "run_id": result.get("run_id"),
                "workflow_id": result.get("workflow_id"),
                "results": {},
                "final_output": "",
                "execution_log": [log_entry],
            }

        output_key = step.get("output_key", "step_1_output")
        return {
            "results": {output_key: result},
            "final_output": result,
            "execution_log": [log_entry],
        }

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------

    async def _execute_sequential(
        self,
        db: AsyncSession,
        plan: List[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        """Execute steps one after another; output N feeds input N+1."""
        results: Dict[str, str] = {}
        execution_log: List[Dict[str, Any]] = []
        current_input = user_input
        completed_steps: List[Dict[str, Any]] = []

        for step in sorted(plan, key=lambda s: s.get("step", 0)):
            input_source = step.get("input_source", "user_input")
            if input_source.startswith("step_") and input_source.endswith("_output"):
                # Use previous step's output
                current_input = results.get(input_source, current_input)
            elif input_source == "merged":
                # Merge all previous results
                current_input = "\n\n".join(results.values()) if results else user_input

            result, log_entry = await self._execute_step(
                db, step, current_input, results
            )

            # HITL interrupt check -- stop sequential execution and
            # propagate the interrupt along with completed prior steps.
            if isinstance(result, dict) and result.get("hitl_interrupted"):
                execution_log.append(log_entry)
                remaining_steps = [
                    s for s in sorted(plan, key=lambda s: s.get("step", 0))
                    if s.get("step", 0) > step.get("step", 0)
                ]
                return {
                    "hitl_interrupted": True,
                    "completed_steps": completed_steps,
                    "interrupted_at_step": step.get("step", 0),
                    "interrupt_payload": result,
                    "run_id": result.get("run_id"),
                    "workflow_id": result.get("workflow_id"),
                    "remaining_plan": remaining_steps,
                    "results": results,
                    "final_output": list(results.values())[-1] if results else "",
                    "execution_log": execution_log,
                }

            output_key = step.get("output_key", f"step_{step.get('step', 0)}_output")
            results[output_key] = result
            execution_log.append(log_entry)
            completed_steps.append({
                "step": step.get("step", 0),
                "skill_name": step.get("skill_name", step.get("skill_id", "")),
                "output": result,
            })
            current_input = result  # default chain

        final_output = list(results.values())[-1] if results else ""
        return {
            "results": results,
            "final_output": final_output,
            "execution_log": execution_log,
        }

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------

    async def _execute_parallel(
        self,
        db: AsyncSession,
        plan: List[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        """Execute steps grouped by parallel_group concurrently.

        When any step in a parallel group returns a HITL interrupt,
        we collect outputs from all completed steps (this group +
        prior groups) and propagate the interrupt upward.  Only the
        *first* HITL interrupt encountered in a group is surfaced;
        others are logged but not tracked (a single HITL pause is
        enough to halt the pipeline).
        """
        groups: Dict[Optional[Any], List[Dict[str, Any]]] = defaultdict(list)
        for step in plan:
            pg = step.get("parallel_group")
            groups[pg].append(step)

        results: Dict[str, str] = {}
        execution_log: List[Dict[str, Any]] = []
        completed_steps: List[Dict[str, Any]] = []

        for group_key in sorted(groups.keys(), key=lambda k: str(k)):
            group_steps = groups[group_key]
            tasks = [
                self._execute_step(db, step, user_input, results)
                for step in group_steps
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            hitl_interrupt = None  # first HITL interrupt in this group

            for step, result_or_exc in zip(group_steps, group_results):
                output_key = step.get("output_key", f"step_{step.get('step', 0)}_output")
                if isinstance(result_or_exc, Exception):
                    logger.error(
                        "Parallel step %s failed: %s", step.get("step"), result_or_exc
                    )
                    results[output_key] = f"[Error: {result_or_exc}]"
                    execution_log.append({
                        "step": step.get("step"),
                        "skill_id": step.get("skill_id"),
                        "status": "failed",
                        "error": str(result_or_exc),
                    })
                else:
                    result_text, log_entry = result_or_exc
                    execution_log.append(log_entry)

                    # Check for HITL interrupt
                    if isinstance(result_text, dict) and result_text.get("hitl_interrupted"):
                        if hitl_interrupt is None:
                            hitl_interrupt = result_text
                            hitl_interrupt["_step"] = step
                        else:
                            logger.warning(
                                "Multiple HITL interrupts in parallel group %s; "
                                "only the first is propagated.", group_key,
                            )
                    else:
                        results[output_key] = result_text
                        completed_steps.append({
                            "step": step.get("step", 0),
                            "skill_name": step.get("skill_name", step.get("skill_id", "")),
                            "output": result_text,
                        })

            # If a HITL interrupt occurred in this group, stop processing
            if hitl_interrupt is not None:
                interrupted_step = hitl_interrupt.pop("_step", {})
                remaining_group_keys = [
                    k for k in sorted(groups.keys(), key=lambda k: str(k))
                    if str(k) > str(group_key)
                ]
                remaining_steps = []
                for rk in remaining_group_keys:
                    remaining_steps.extend(groups[rk])

                return {
                    "hitl_interrupted": True,
                    "completed_steps": completed_steps,
                    "interrupted_at_step": interrupted_step.get("step", 0),
                    "interrupt_payload": hitl_interrupt,
                    "run_id": hitl_interrupt.get("run_id"),
                    "workflow_id": hitl_interrupt.get("workflow_id"),
                    "remaining_plan": remaining_steps,
                    "results": results,
                    "final_output": list(results.values())[-1] if results else "",
                    "execution_log": execution_log,
                }

        final_output = "\n\n---\n\n".join(results.values()) if results else ""
        return {
            "results": results,
            "final_output": final_output,
            "execution_log": execution_log,
        }

    # ------------------------------------------------------------------
    # Hybrid (topological sort)
    # ------------------------------------------------------------------

    async def _execute_hybrid(
        self,
        db: AsyncSession,
        plan: List[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Execute steps based on dependency graph.

        Steps with no unresolved dependencies run in parallel.
        Steps that depend on others wait for their dependencies.

        When any step returns a HITL interrupt, execution halts
        immediately.  Completed step outputs and remaining plan
        steps are collected and propagated upward.
        """
        results: Dict[str, str] = {}
        execution_log: List[Dict[str, Any]] = []
        completed_steps: List[Dict[str, Any]] = []

        # Build dependency map
        step_map: Dict[int, Dict[str, Any]] = {}
        for step in plan:
            step_num = step.get("step", 0)
            step_map[step_num] = step

        completed: set = set()
        remaining = set(step_map.keys())

        while remaining:
            # Find steps whose dependencies are all completed
            ready = []
            for step_num in remaining:
                step = step_map[step_num]
                deps = step.get("depends_on", [])
                if all(d in completed for d in deps):
                    ready.append(step_num)

            if not ready:
                # No progress -- break to avoid infinite loop (circular deps)
                logger.error(
                    "Hybrid execution stalled: circular dependencies detected "
                    "among steps %s", remaining,
                )
                for step_num in remaining:
                    step = step_map[step_num]
                    output_key = step.get("output_key", f"step_{step_num}_output")
                    results[output_key] = "[Error: circular dependency]"
                    execution_log.append({
                        "step": step_num,
                        "skill_id": step.get("skill_id"),
                        "status": "failed",
                        "error": "Circular dependency detected",
                    })
                break

            # Execute ready steps in parallel
            tasks = []
            for step_num in ready:
                step = step_map[step_num]
                input_source = step.get("input_source", "user_input")
                step_input = user_input
                if input_source.startswith("step_") and input_source.endswith("_output"):
                    step_input = results.get(input_source, user_input)
                elif input_source == "merged":
                    # Merge results from dependencies
                    dep_results = []
                    for d in step.get("depends_on", []):
                        dep_step = step_map.get(d)
                        if dep_step:
                            dep_key = dep_step.get("output_key", f"step_{d}_output")
                            if dep_key in results:
                                dep_results.append(results[dep_key])
                    step_input = "\n\n".join(dep_results) if dep_results else user_input

                tasks.append(
                    (step_num, self._execute_step(db, step, step_input, results))
                )

            coros = [t[1] for t in tasks]
            gathered = await asyncio.gather(*coros, return_exceptions=True)

            hitl_interrupt = None  # first HITL interrupt in this batch

            for (step_num, _), result_or_exc in zip(tasks, gathered):
                step = step_map[step_num]
                output_key = step.get("output_key", f"step_{step_num}_output")
                if isinstance(result_or_exc, Exception):
                    logger.error("Hybrid step %s failed: %s", step_num, result_or_exc)
                    results[output_key] = f"[Error: {result_or_exc}]"
                    execution_log.append({
                        "step": step_num,
                        "skill_id": step.get("skill_id"),
                        "status": "failed",
                        "error": str(result_or_exc),
                    })
                else:
                    result_text, log_entry = result_or_exc
                    execution_log.append(log_entry)

                    # Check for HITL interrupt
                    if isinstance(result_text, dict) and result_text.get("hitl_interrupted"):
                        if hitl_interrupt is None:
                            hitl_interrupt = result_text
                            hitl_interrupt["_step_num"] = step_num
                        else:
                            logger.warning(
                                "Multiple HITL interrupts in hybrid batch; "
                                "only the first is propagated.",
                            )
                    else:
                        results[output_key] = result_text
                        completed_steps.append({
                            "step": step_num,
                            "skill_name": step.get("skill_name", step.get("skill_id", "")),
                            "output": result_text,
                        })

                completed.add(step_num)
                remaining.discard(step_num)

            # If a HITL interrupt occurred in this batch, stop processing
            if hitl_interrupt is not None:
                interrupted_step_num = hitl_interrupt.pop("_step_num", 0)
                remaining_steps = [
                    step_map[sn] for sn in sorted(remaining)
                ]
                return {
                    "hitl_interrupted": True,
                    "completed_steps": completed_steps,
                    "interrupted_at_step": interrupted_step_num,
                    "interrupt_payload": hitl_interrupt,
                    "run_id": hitl_interrupt.get("run_id"),
                    "workflow_id": hitl_interrupt.get("workflow_id"),
                    "remaining_plan": remaining_steps,
                    "results": results,
                    "final_output": list(results.values())[-1] if results else "",
                    "execution_log": execution_log,
                }

        final_output = list(results.values())[-1] if results else ""
        return {
            "results": results,
            "final_output": final_output,
            "execution_log": execution_log,
        }

    # ------------------------------------------------------------------
    # Single-step execution dispatcher
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        db: AsyncSession,
        step: Dict[str, Any],
        step_input: str,
        previous_results: Dict[str, str],
    ) -> tuple:
        """
        Execute a single step from the plan.

        Returns:
            Tuple of (result_text_or_hitl_dict, log_entry: dict)

            When a workflow step contains HITL nodes and the workflow
            pauses at an interrupt, result_text_or_hitl_dict will be a
            dict with ``hitl_interrupted == True`` instead of a plain
            string.  Callers (execute_plan strategies) must check for
            this and propagate the interrupt upward.
        """
        skill_type = step.get("skill_type", "workflow")
        skill_id = step.get("skill_id", "")
        skill_name = step.get("skill_name", skill_id)
        step_num = step.get("step", 0)

        # Inject document context into step input if available
        document_context = getattr(self, "_document_context", None)
        if document_context:
            step_input = f"{step_input}\n\nUPLOADED DOCUMENT CONTEXT:\n{document_context}"

        start_time = time.monotonic()
        result_text: Any = ""
        status = "completed"
        error_msg = None

        try:
            if skill_type == "workflow":
                # HITL-aware path: check for HITL nodes first
                hitl_result = await self._execute_workflow_hitl_aware(
                    db, skill_id, step_input
                )
                if isinstance(hitl_result, dict) and hitl_result.get("hitl_interrupted"):
                    # Workflow paused at HITL node -- propagate upward
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    log_entry = {
                        "step": step_num,
                        "skill_id": skill_id,
                        "skill_type": skill_type,
                        "skill_name": skill_name,
                        "status": "interrupted",
                        "duration_ms": duration_ms,
                        "error": None,
                    }
                    return hitl_result, log_entry
                # Normal completion -- hitl_result is a plain string
                result_text = hitl_result
            elif skill_type == "agent":
                result_text = await self._execute_agent(
                    db, skill_id, step_input, previous_results
                )
            else:
                raise ValueError(f"Unknown skill_type: {skill_type}")

        except Exception as exc:
            status = "failed"
            error_msg = str(exc)
            result_text = f"[Skill execution failed: {error_msg}]"
            logger.exception(
                "Step %s (%s: %s) failed", step_num, skill_type, skill_name
            )

        duration_ms = int((time.monotonic() - start_time) * 1000)

        log_entry = {
            "step": step_num,
            "skill_id": skill_id,
            "skill_type": skill_type,
            "skill_name": skill_name,
            "status": status,
            "duration_ms": duration_ms,
            "error": error_msg,
        }

        return result_text, log_entry

    # ------------------------------------------------------------------
    # Workflow execution (black box via WorkflowExecutor)
    # ------------------------------------------------------------------

    async def _execute_workflow(
        self, db: AsyncSession, workflow_id: str, user_input: str
    ) -> str:
        """
        Execute a workflow by resolving WorkflowExecutor from DI and
        calling execute_workflow() as a black box.

        WorkflowExecutor.execute_workflow() is synchronous (it internally
        calls LangGraph's invoke), so we wrap it in asyncio.to_thread().
        """
        executor = self._resolve_workflow_executor()

        # Resolve the filesystem workflow_id (wf_...) from the DB UUID
        fs_workflow_id = await self._resolve_fs_workflow_id(db, workflow_id)

        input_payload = {
            "user_input": user_input,
            "message": user_input,
        }

        # execute_workflow is synchronous
        result = await asyncio.to_thread(
            executor.execute_workflow,
            fs_workflow_id,
            "draft",
            None,
            input_payload,
        )

        if result.get("status") == "failed":
            raise RuntimeError(
                f"Workflow execution failed: {result.get('error', 'unknown')}"
            )

        # Extract output text
        output = result.get("output", {})
        # Try common output keys
        output_text = (
            output.get("crew_result")
            or output.get("result")
            or output.get("parallel_output")
            or output.get("hierarchical_output")
            or output.get("merged_output")
        )
        if not output_text:
            # Fallback: stringify the output
            if isinstance(output, dict):
                # Look through all string values
                for v in output.values():
                    if isinstance(v, str) and len(v) > 10:
                        output_text = v
                        break
            if not output_text:
                output_text = str(output)

        # Clean up common formatting artifacts from LLM / workflow output:
        # 1. Replace literal two-character sequences "\" + "n" with real newlines
        output_text = output_text.replace("\\n", "\n")
        # 2. Strip leading "String\n\n" or similar type-name prefixes
        #    that some LLM wrappers prepend to outputs
        import re
        output_text = re.sub(r"^String\s*\n", "", output_text, count=1)
        # 3. Collapse excessive blank lines (3+ newlines -> 2)
        output_text = re.sub(r"\n{3,}", "\n\n", output_text)
        # 4. Strip leading/trailing whitespace
        output_text = output_text.strip()

        return output_text

    # ------------------------------------------------------------------
    # HITL-aware workflow execution (additive — does NOT modify
    # _execute_workflow or any other existing method)
    # ------------------------------------------------------------------

    async def _execute_workflow_hitl_aware(
        self, db: AsyncSession, workflow_id: str, user_input: str
    ) -> Any:
        """
        Execute a workflow with HITL awareness.

        Checks whether the workflow definition contains any HITL nodes.
        If yes, calls ``execute_workflow_hitl()`` which can return an
        interrupt.  If no HITL nodes are present, delegates to the
        original ``_execute_workflow()`` unchanged.

        Returns:
            str  -- normal output text (no HITL, or HITL workflow that
                    completed without interrupting).
            dict -- ``{"hitl_interrupted": True, ...}`` when the workflow
                    paused at a HITL node.
        """
        has_hitl = await self._workflow_has_hitl_nodes(db, workflow_id)

        if not has_hitl:
            # No HITL nodes — delegate to existing method unchanged
            return await self._execute_workflow(db, workflow_id, user_input)

        # --- HITL-aware path ---
        executor = self._resolve_workflow_executor()
        fs_workflow_id = await self._resolve_fs_workflow_id(db, workflow_id)

        input_payload = {
            "user_input": user_input,
            "message": user_input,
        }

        # execute_workflow_hitl is synchronous (LangGraph invoke)
        result = await asyncio.to_thread(
            executor.execute_workflow_hitl,
            fs_workflow_id,
            "draft",
            None,
            input_payload,
        )

        if result.get("status") == "failed":
            raise RuntimeError(
                f"Workflow execution failed: {result.get('error', 'unknown')}"
            )

        if result.get("status") == "interrupted":
            # Workflow paused at a HITL node
            logger.info(
                "Workflow %s interrupted at HITL node (run_id=%s)",
                workflow_id, result.get("run_id"),
            )
            return {
                "hitl_interrupted": True,
                "run_id": result.get("run_id", ""),
                "workflow_id": workflow_id,
                "fs_workflow_id": fs_workflow_id,
                "interrupt": result.get("interrupt", {}),
                "partial_output": result.get("output", {}),
            }

        # Completed normally — extract text output using the same logic
        # as _execute_workflow
        return self._extract_output_text(result.get("output", {}))

    async def _workflow_has_hitl_nodes(
        self, db: AsyncSession, workflow_id: str
    ) -> bool:
        """
        Check whether a workflow definition contains any HITL nodes.

        Loads the workflow ``definition`` JSONB from the database and
        inspects each agent entry for ``metadata.node_type == "HITL"``.

        Args:
            db: Async database session.
            workflow_id: DB UUID or filesystem ID of the workflow.

        Returns:
            True if at least one HITL node is present.
        """
        from echolib.models.workflow import Workflow
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        wf_uuid = safe_uuid(workflow_id)
        if wf_uuid is None:
            # Not a valid UUID — cannot query DB
            return False

        stmt = select(Workflow.definition).where(
            Workflow.workflow_id == wf_uuid
        )
        row = await db.execute(stmt)
        definition = row.scalar_one_or_none()

        if not definition or not isinstance(definition, dict):
            return False

        for agent in definition.get("agents", []):
            if isinstance(agent, dict):
                metadata = agent.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("node_type") == "HITL":
                    return True

        return False

    @staticmethod
    def _extract_output_text(output: Any) -> str:
        """
        Extract human-readable text from a workflow output dict.

        Mirrors the extraction logic in ``_execute_workflow()`` so that
        HITL-aware completions produce identical output formatting.
        """
        import re

        if not isinstance(output, dict):
            return str(output).strip()

        output_text = (
            output.get("crew_result")
            or output.get("result")
            or output.get("parallel_output")
            or output.get("hierarchical_output")
            or output.get("merged_output")
        )
        if not output_text:
            for v in output.values():
                if isinstance(v, str) and len(v) > 10:
                    output_text = v
                    break
            if not output_text:
                output_text = str(output)

        output_text = output_text.replace("\\n", "\n")
        output_text = re.sub(r"^String\s*\n", "", output_text, count=1)
        output_text = re.sub(r"\n{3,}", "\n\n", output_text)
        output_text = output_text.strip()
        return output_text

    @staticmethod
    async def _resolve_fs_workflow_id(
        db: AsyncSession, db_workflow_id: str
    ) -> str:
        """
        Resolve the filesystem workflow_id (wf_...) from the database UUID.

        The skill manifest uses the DB UUID as skill_id, but the filesystem
        storage uses the internal wf_ prefixed ID from the workflow definition.
        """
        from echolib.models.workflow import Workflow
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        wf_uuid = safe_uuid(db_workflow_id)
        if wf_uuid is None:
            # Already a non-UUID id (e.g. wf_...), use as-is
            return db_workflow_id

        stmt = select(Workflow.definition).where(
            Workflow.workflow_id == wf_uuid
        )
        result = await db.execute(stmt)
        definition = result.scalar_one_or_none()

        if definition and isinstance(definition, dict):
            fs_id = definition.get("workflow_id")
            if fs_id and fs_id.startswith("wf_"):
                # Definition already stores the correct filesystem ID
                logger.info(
                    "Resolved DB workflow UUID %s -> filesystem ID %s",
                    db_workflow_id, fs_id,
                )
                return fs_id

        # Definition missing, or its workflow_id is a raw UUID (not wf_ prefixed).
        # Convert UUID to wf_ format: wf_ + uuid without hyphens
        fs_id = f"wf_{db_workflow_id.replace('-', '')}"
        logger.info(
            "Resolved DB workflow UUID %s -> filesystem ID %s (via conversion)",
            db_workflow_id, fs_id,
        )
        return fs_id

    @staticmethod
    def _resolve_workflow_executor():
        """
        Resolve WorkflowExecutor from the DI container.

        The workflow module registers it as 'workflow.executor'.
        """
        from echolib.di import container

        try:
            return container.resolve("workflow.executor")
        except KeyError:
            logger.error(
                "WorkflowExecutor not registered in DI container. "
                "Ensure the workflow module is initialized."
            )
            raise RuntimeError(
                "WorkflowExecutor not available. "
                "The workflow module may not be initialized."
            )

    # ------------------------------------------------------------------
    # Agent execution (standalone)
    # ------------------------------------------------------------------

    async def _execute_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        user_input: str,
        previous_results: Dict[str, str],
    ) -> str:
        """Execute an agent via StandaloneAgentExecutor."""
        context = None
        if previous_results:
            # Pass the most recent result as context
            last_result = list(previous_results.values())[-1] if previous_results else ""
            context = {"previous_output": last_result}

        result = await self._agent_executor.execute(
            db=db,
            agent_id=agent_id,
            user_input=user_input,
            context=context,
            sse_queue=getattr(self, "_sse_queue", None),
        )

        return result.get("result", "")

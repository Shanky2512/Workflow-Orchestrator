"""
EchoAI Orchestrator -- Orchestration Pipeline

The top-level coordinator that ties all orchestration steps together
for a single chat turn:

    1.  Load application from DB
    2.  Pre-guardrails check on raw user message
    3.  Build persona prompt
    4.  Enhance prompt
    5.  Build skill manifest
    6.  Get conversation history
    7.  Check HITL state (handle clarification flow)
    8.  Call orchestrator LLM for execution plan
    9.  Handle HITL response (clarify / fallback / execute)
    10. Execute skills
    11. Post-guardrails on output
    12. Save messages to DB
    13. Save execution trace
    14. Update session state back to awaiting_input
    15. Return response
"""

import asyncio
import copy
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .prompt_enhancer import PromptEnhancer
from .skill_manifest import SkillManifestBuilder
from .orchestrator import Orchestrator
from .guardrails import GuardrailsEngine
from .persona import PersonaFormatter
from .hitl import HITLManager
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)


class OrchestrationPipeline:
    """
    Full orchestration pipeline coordinator.

    Composes all orchestrator components into a single ``run()`` method
    that processes one chat turn end-to-end.
    """

    def __init__(self) -> None:
        self._prompt_enhancer = PromptEnhancer()
        self._manifest_builder = SkillManifestBuilder()
        self._orchestrator = Orchestrator()
        self._guardrails = GuardrailsEngine()
        self._persona_formatter = PersonaFormatter()
        self._hitl_manager = HITLManager()
        self._skill_executor = SkillExecutor()

    async def run(
        self,
        db: AsyncSession,
        application_id: str,
        chat_session_id: Optional[str],
        user_message: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Execute one full orchestration turn.

        Args:
            db: Async database session (auto-commits on exit).
            application_id: Application UUID string.
            chat_session_id: Existing session UUID, or None to create new.
            user_message: Raw user input text.
            user_id: User UUID string.

        Returns:
            Dict with keys:
                response (str)          -- the final assistant message
                session_id (str)        -- chat session UUID
                conversation_state (str) -- current HITL state
                execution_trace (dict)  -- audit trail data
        """
        from echolib.repositories.application_repo import ApplicationRepository
        from echolib.repositories.application_chat_repo import ApplicationChatRepository

        app_repo = ApplicationRepository()
        chat_repo = ApplicationChatRepository()

        pipeline_start = time.monotonic()
        trace_data: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # 1. Load application
        # ------------------------------------------------------------------
        app = await app_repo.get_application(db, application_id, user_id)
        if app is None:
            return self._error_response(
                "Application not found or access denied.",
                chat_session_id,
            )

        if app.status != "published":
            return self._error_response(
                "Application is not published. Only published applications can be used for chat.",
                chat_session_id,
            )

        # Record app LLM configuration in trace for frontend visibility
        trace_data["app_llm_config"] = [
            {"llm_id": link.llm_id, "role": link.role, "name": link.name}
            for link in (app.llm_links or [])
        ]

        # ------------------------------------------------------------------
        # 2. Ensure chat session exists
        # ------------------------------------------------------------------
        is_new_session = False
        if chat_session_id:
            session = await chat_repo.get_session(db, chat_session_id, user_id)
            if session is None:
                return self._error_response(
                    "Chat session not found.", chat_session_id
                )
            if session.is_closed:
                return self._error_response(
                    "Chat session is closed.", chat_session_id
                )
        else:
            session = await chat_repo.create_session(
                db, application_id, user_id,
                title=user_message[:80] if user_message else "New Chat",
            )
            chat_session_id = str(session.chat_session_id)
            is_new_session = True

            # Inject welcome_prompt as the first assistant message in the session
            if app.welcome_prompt:
                await chat_repo.add_message(
                    db, chat_session_id, "assistant", app.welcome_prompt,
                    execution_trace={"type": "welcome_prompt"},
                )

        # ------------------------------------------------------------------
        # 3. Pre-guardrails
        # ------------------------------------------------------------------
        guardrail_categories = await self._get_guardrail_category_names(db, app)
        pre_guard = self._guardrails.pre_process(
            user_message,
            guardrail_categories=guardrail_categories,
            guardrail_text=app.guardrail_text,
        )
        trace_data["guardrail_input"] = {
            "is_safe": pre_guard.is_safe,
            "violations": pre_guard.violations,
            "categories_triggered": pre_guard.categories_triggered,
        }

        if not pre_guard.is_safe:
            violation_msg = (
                "Your message was blocked by safety filters. "
                f"Violations: {', '.join(pre_guard.violations)}"
            )
            # Save user message and violation response
            await chat_repo.add_message(
                db, chat_session_id, "user", user_message,
                guardrail_flags=trace_data["guardrail_input"],
            )
            await chat_repo.add_message(
                db, chat_session_id, "assistant", violation_msg,
            )
            return {
                "response": violation_msg,
                "session_id": chat_session_id,
                "conversation_state": "awaiting_input",
                "execution_trace": trace_data,
            }

        # ------------------------------------------------------------------
        # 4. Check HITL state -- if awaiting_clarification, build combined prompt
        # ------------------------------------------------------------------
        effective_prompt = user_message
        is_clarification = False

        # Guard: if session is paused at a HITL node (workflow interrupt),
        # reject regular chat messages.  The user must use the
        # POST .../chat/hitl-decide endpoint to approve/reject/edit/defer.
        _session_ctx_data = session.context_data or {}
        if (
            session.conversation_state == "awaiting_clarification"
            and _session_ctx_data.get("hitl_state")
        ):
            hitl_ctx = _session_ctx_data["hitl_state"].get("interrupt_payload", {})
            return {
                "response": (
                    "This session is paused for human review of a workflow. "
                    "Please use the review panel to approve, reject, edit, "
                    "or defer before continuing the conversation."
                ),
                "session_id": chat_session_id,
                "conversation_state": "awaiting_hitl_review",
                "execution_trace": {
                    "hitl_context": {
                        "title": hitl_ctx.get("title"),
                        "message": hitl_ctx.get("message"),
                        "priority": hitl_ctx.get("priority"),
                        "allowed_decisions": hitl_ctx.get(
                            "allowed_decisions", ["approve", "reject"]
                        ),
                        "run_id": _session_ctx_data["hitl_state"].get("run_id", ""),
                    }
                },
            }

        if session.conversation_state == "awaiting_clarification":
            clarification_ctx = await self._hitl_manager.get_clarification_context(
                db, chat_session_id
            )
            if clarification_ctx:
                effective_prompt = await self._hitl_manager.build_clarification_prompt(
                    original_prompt=clarification_ctx.get("original_prompt", user_message),
                    clarification_answer=user_message,
                    partial_analysis=clarification_ctx.get("partial_analysis", ""),
                    conversation_window=clarification_ctx.get("conversation_window"),
                )
                is_clarification = True

        # ------------------------------------------------------------------
        # 5. Build persona prompt
        # ------------------------------------------------------------------
        persona_prompt = await self._persona_formatter.build_persona_prompt(
            db,
            persona_id=str(app.persona_id) if app.persona_id else None,
            persona_text=app.persona_text,
        )

        # ------------------------------------------------------------------
        # 5a. Resolve tags, designations, business units, data sources
        # ------------------------------------------------------------------
        app_tags = await self._get_tag_names(db, app)
        app_designations = await self._get_designation_names(db, app)
        app_business_units = await self._get_business_unit_names(db, app)
        app_data_sources = await self._get_data_source_names(db, app)
        trace_data["app_tags"] = app_tags
        trace_data["app_designations"] = app_designations
        trace_data["app_business_units"] = app_business_units
        trace_data["app_data_sources"] = app_data_sources

        # ------------------------------------------------------------------
        # 6. Enhance prompt
        # ------------------------------------------------------------------
        enhancement = await self._prompt_enhancer.enhance(effective_prompt)
        enhanced_prompt = enhancement.get("enhanced_prompt", effective_prompt)
        trace_data["enhanced_prompt"] = enhanced_prompt
        trace_data["detected_intent"] = enhancement.get("detected_intent")
        trace_data["intent_confidence"] = enhancement.get("confidence")

        # ------------------------------------------------------------------
        # 6a. Document retrieval (session-scoped RAG)
        # ------------------------------------------------------------------
        document_context = None
        try:
            from echolib.di import container as di_container
            rag_manager = di_container.resolve('rag.session_manager')
            if rag_manager.has_documents(chat_session_id):
                document_context = rag_manager.retrieve(
                    chat_session_id, enhanced_prompt, top_k=10
                )
        except KeyError:
            # rag.session_manager not registered -- RAG module not loaded
            logger.debug("rag.session_manager not registered; skipping document retrieval")
        except Exception:
            logger.exception("Document retrieval failed for session %s", chat_session_id)

        trace_data["document_context_used"] = document_context is not None

        # ------------------------------------------------------------------
        # 7. Build skill manifest
        # ------------------------------------------------------------------
        skill_manifest = await self._manifest_builder.build_manifest(
            db, application_id
        )
        trace_data["skill_count"] = len(skill_manifest.get("skills", []))

        # ------------------------------------------------------------------
        # 8. Get conversation history
        # ------------------------------------------------------------------
        # Fetch last 8 messages (4 user + 4 assistant exchanges)
        # This preserves conversation context while limiting token usage
        history_messages = await chat_repo.get_messages(
            db, chat_session_id, limit=8
        )
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history_messages
        ]

        # ------------------------------------------------------------------
        # 9. Call orchestrator LLM
        # ------------------------------------------------------------------
        guardrail_rules_text = app.guardrail_text or ""

        orchestrator_output = await self._orchestrator.plan(
            enhanced_prompt=enhanced_prompt,
            skill_manifest=skill_manifest,
            conversation_history=conversation_history,
            persona_prompt=persona_prompt,
            guardrail_rules=guardrail_rules_text,
            document_context=document_context,
            app_tags=app_tags,
            app_designations=app_designations,
            app_business_units=app_business_units,
            app_name=app.name,
            app_description=app.description,
            app_disclaimer=app.disclaimer,
            app_data_sources=app_data_sources,
        )
        trace_data["orchestrator_plan"] = orchestrator_output

        # ------------------------------------------------------------------
        # 10. Handle HITL response
        # ------------------------------------------------------------------
        hitl_action = await self._hitl_manager.handle_orchestrator_response(
            db, chat_session_id, orchestrator_output, enhanced_prompt,
            conversation_history=conversation_history,
        )

        action = hitl_action.get("action")

        # Helper to build session metadata dict when it's a new session
        def _session_meta() -> Optional[Dict[str, Any]]:
            if not is_new_session:
                return None
            return {
                "app_name": app.name,
                "app_description": app.description,
                "app_logo_url": app.logo_url,
                "welcome_prompt": app.welcome_prompt,
                "disclaimer": app.disclaimer,
                "starter_questions": app.starter_questions or [],
            }

        if action == "clarify":
            clarification_msg = hitl_action.get("message", "Could you clarify?")
            # Save messages
            await chat_repo.add_message(
                db, chat_session_id, "user", user_message,
                enhanced_prompt=enhanced_prompt,
            )
            await chat_repo.add_message(
                db, chat_session_id, "assistant", clarification_msg,
                execution_trace={"type": "clarification"},
            )
            clarify_result: Dict[str, Any] = {
                "response": clarification_msg,
                "session_id": chat_session_id,
                "conversation_state": "awaiting_clarification",
                "execution_trace": trace_data,
            }
            meta = _session_meta()
            if meta:
                clarify_result["session_metadata"] = meta
            return clarify_result

        if action == "fallback":
            fallback_msg = hitl_action.get("message", "")
            # Prepend app's sorry_message if available
            sorry = app.sorry_message or ""
            if sorry and fallback_msg:
                full_fallback = f"{sorry}\n\n{fallback_msg}"
            elif sorry:
                full_fallback = sorry
            else:
                full_fallback = fallback_msg

            # Append skill capability summary
            skills = skill_manifest.get("skills", [])
            if skills:
                skill_summary = ", ".join(
                    f"{s['name']} ({s['skill_type']})" for s in skills
                )
                full_fallback += (
                    f"\n\nI can help you with: {skill_summary}. "
                    "Please ask me something related to these capabilities."
                )

            # Save messages
            await chat_repo.add_message(
                db, chat_session_id, "user", user_message,
                enhanced_prompt=enhanced_prompt,
            )
            await chat_repo.add_message(
                db, chat_session_id, "assistant", full_fallback,
                execution_trace={"type": "fallback", "fallback": True},
            )
            fallback_result: Dict[str, Any] = {
                "response": full_fallback,
                "session_id": chat_session_id,
                "conversation_state": "awaiting_input",
                "execution_trace": trace_data,
            }
            meta = _session_meta()
            if meta:
                fallback_result["session_metadata"] = meta
            return fallback_result

        # ------------------------------------------------------------------
        # 11. Execute skills
        # ------------------------------------------------------------------
        plan = orchestrator_output.get("execution_plan", [])
        strategy = orchestrator_output.get("execution_strategy", "single")

        execution_result = await self._skill_executor.execute_plan(
            db=db,
            execution_plan=plan,
            execution_strategy=strategy,
            user_input=enhanced_prompt,
            document_context=document_context,
        )

        # ------------------------------------------------------------------
        # 11a. Check for HITL interrupt from workflow execution
        # ------------------------------------------------------------------
        if execution_result.get("hitl_interrupted"):
            hitl_interrupt = execution_result.get("interrupt_payload", {})
            interrupt_ctx = hitl_interrupt.get("interrupt", {})

            # Build partial response from completed steps
            completed_steps = execution_result.get("completed_steps", [])
            partial_parts = []
            for cs in completed_steps:
                partial_parts.append(
                    f"**Step {cs.get('step', '?')} ({cs.get('skill_name', 'unknown')}):** "
                    f"{cs.get('output', '')}"
                )
            partial_text = "\n\n".join(partial_parts) if partial_parts else ""
            if partial_text:
                partial_text += "\n\n---\n\n"
            partial_text += "Workflow paused for human review."

            # Store HITL context in session context_data for later resume
            hitl_state_data = {
                "hitl_state": {
                    "run_id": execution_result.get("run_id", ""),
                    "workflow_id": execution_result.get("workflow_id", ""),
                    "fs_workflow_id": hitl_interrupt.get("fs_workflow_id", ""),
                    "interrupt_payload": interrupt_ctx,
                    "completed_steps": completed_steps,
                    "remaining_plan": execution_result.get("remaining_plan", []),
                    "execution_plan": plan,
                    "execution_strategy": strategy,
                    "enhanced_prompt": enhanced_prompt,
                }
            }

            # Derive allowed decisions from interrupt context
            allowed_decisions = interrupt_ctx.get(
                "allowed_decisions", ["approve", "reject"]
            )

            # Save user message
            await chat_repo.add_message(
                db, chat_session_id, "user", user_message,
                enhanced_prompt=enhanced_prompt,
            )
            # Save partial assistant message
            await chat_repo.add_message(
                db, chat_session_id, "assistant", partial_text,
                execution_trace={
                    "type": "hitl_interrupt",
                    "hitl_context": {
                        "title": interrupt_ctx.get("title"),
                        "message": interrupt_ctx.get("message"),
                        "priority": interrupt_ctx.get("priority"),
                        "allowed_decisions": allowed_decisions,
                        "run_id": execution_result.get("run_id", ""),
                        "node_outputs": interrupt_ctx.get("node_outputs"),
                    },
                },
            )

            # Update session state to awaiting_clarification (DB constraint
            # limits valid values; we use awaiting_clarification and
            # distinguish HITL via context_data.hitl_state).
            await chat_repo.update_session_state(
                db,
                chat_session_id,
                conversation_state="awaiting_clarification",
                context_data=hitl_state_data,
            )

            trace_data["execution_log"] = execution_result.get("execution_log", [])
            trace_data["hitl_context"] = {
                "title": interrupt_ctx.get("title"),
                "message": interrupt_ctx.get("message"),
                "priority": interrupt_ctx.get("priority"),
                "allowed_decisions": allowed_decisions,
                "run_id": execution_result.get("run_id", ""),
                "node_outputs": interrupt_ctx.get("node_outputs"),
            }

            hitl_result: Dict[str, Any] = {
                "response": partial_text,
                "session_id": chat_session_id,
                "conversation_state": "awaiting_hitl_review",
                "execution_trace": trace_data,
            }
            meta = _session_meta()
            if meta:
                hitl_result["session_metadata"] = meta
            return hitl_result

        final_output = execution_result.get("final_output", "")
        trace_data["execution_result"] = execution_result.get("results", {})
        trace_data["execution_log"] = execution_result.get("execution_log", [])
        trace_data["skills_invoked"] = execution_result.get("execution_log", [])

        # ------------------------------------------------------------------
        # 12. Post-guardrails on output
        # ------------------------------------------------------------------
        post_guard = self._guardrails.post_process(
            final_output,
            guardrail_categories=guardrail_categories,
            guardrail_text=app.guardrail_text,
        )
        trace_data["guardrail_output"] = {
            "is_safe": post_guard.is_safe,
            "violations": post_guard.violations,
            "categories_triggered": post_guard.categories_triggered,
        }
        final_output = post_guard.sanitized_text

        # ------------------------------------------------------------------
        # 13. Save messages to DB
        # ------------------------------------------------------------------
        user_msg_record = await chat_repo.add_message(
            db, chat_session_id, "user", user_message,
            enhanced_prompt=enhanced_prompt,
        )

        assistant_msg_record = await chat_repo.add_message(
            db, chat_session_id, "assistant", final_output,
            execution_trace=trace_data,
            guardrail_flags=trace_data.get("guardrail_output"),
        )

        # ------------------------------------------------------------------
        # 14. Save execution trace
        # ------------------------------------------------------------------
        total_duration_ms = int((time.monotonic() - pipeline_start) * 1000)
        trace_data["total_duration_ms"] = total_duration_ms

        try:
            await chat_repo.create_execution_trace(
                db,
                application_id=application_id,
                chat_session_id=chat_session_id,
                user_message=user_message,
                enhanced_prompt=enhanced_prompt,
                orchestrator_plan=orchestrator_output,
                execution_result=execution_result.get("results"),
                skills_invoked=execution_result.get("execution_log"),
                guardrail_input=trace_data.get("guardrail_input"),
                guardrail_output=trace_data.get("guardrail_output"),
                total_duration_ms=total_duration_ms,
                status="completed",
                message_id=str(user_msg_record.message_id),
            )
        except Exception:
            logger.exception("Failed to save execution trace")

        # ------------------------------------------------------------------
        # 15. Update session state back to awaiting_input
        # ------------------------------------------------------------------
        await chat_repo.update_session_state(
            db, chat_session_id, conversation_state="awaiting_input"
        )

        # ------------------------------------------------------------------
        # 16. Return response
        # ------------------------------------------------------------------
        result = {
            "response": final_output,
            "session_id": chat_session_id,
            "conversation_state": "awaiting_input",
            "execution_trace": trace_data,
        }

        # Include session metadata on new session creation so the frontend
        # can render welcome_prompt, disclaimer, starter_questions, etc.
        meta = _session_meta()
        if meta:
            result["session_metadata"] = meta

        return result

    # ------------------------------------------------------------------
    # Streaming variant -- A2UI SSE
    # ------------------------------------------------------------------

    # Step definitions for A2UI progress surface
    _PIPELINE_STEPS = [
        {"id": "step_guardrails",  "icon": "shield",         "label": "Checking guardrails",  "status": "pending", "detail": ""},
        {"id": "step_enhance",     "icon": "auto_fix_high",  "label": "Enhancing prompt",     "status": "pending", "detail": ""},
        {"id": "step_rag",         "icon": "search",         "label": "Retrieving context",   "status": "pending", "detail": ""},
        {"id": "step_plan",        "icon": "account_tree",   "label": "Planning execution",   "status": "pending", "detail": ""},
        {"id": "step_execute",     "icon": "play_circle",    "label": "Executing skills",     "status": "pending", "detail": ""},
        {"id": "step_post_guard",  "icon": "verified",       "label": "Validating output",    "status": "pending", "detail": ""},
    ]

    async def run_stream(
        self,
        sse_queue: asyncio.Queue,
        run_id: str,
        db: AsyncSession,
        app_id: str,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        uploaded_file_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Execute the full orchestration pipeline while emitting A2UI v0.8
        JSONL events to *sse_queue* at each stage.

        This mirrors the logic in ``run()`` but adds streaming progress
        updates.  The original ``run()`` method is untouched.

        When complete, puts ``None`` sentinel into the queue so the SSE
        generator knows the stream has ended.

        Args:
            sse_queue: asyncio.Queue to receive A2UI JSONL strings.
            run_id: Unique execution identifier for this request.
            db: Async database session.
            app_id: Application UUID string.
            user_input: Raw user input text.
            session_id: Existing session UUID, or None to create new.
            user_id: User UUID string.
            uploaded_file_ids: Optional list of uploaded file UUIDs.
        """
        from echolib.a2ui import A2UIStreamBuilder
        from apps.workflow.runtime.ws_manager import ws_manager

        builder = A2UIStreamBuilder()
        steps = copy.deepcopy(self._PIPELINE_STEPS)

        async def _emit(line) -> None:
            try:
                sse_queue.put_nowait(line)
            except asyncio.QueueFull:
                pass

        async def _update_step(step_id: str, status: str, detail: str = "") -> None:
            for s in steps:
                if s["id"] == step_id:
                    s["status"] = status
                    s["detail"] = detail
                    break
            await _emit(builder.step_update(run_id, steps))

        try:
            from echolib.repositories.application_repo import ApplicationRepository
            from echolib.repositories.application_chat_repo import ApplicationChatRepository

            app_repo = ApplicationRepository()
            chat_repo = ApplicationChatRepository()

            pipeline_start = time.monotonic()
            trace_data: Dict[str, Any] = {}

            # ---- Load application ----
            app = await app_repo.get_application(db, app_id, user_id or "")
            app_name = app.name if app else "Application"

            # Emit initial surface
            for line in builder.build_initial_surface(run_id, app_name):
                await _emit(line)

            if app is None:
                await _update_step("step_guardrails", "failed", "Application not found")
                await _emit(builder.final_output_update(run_id, "Application not found or access denied."))
                return

            if app.status != "published":
                await _update_step("step_guardrails", "failed", "Application not published")
                await _emit(builder.final_output_update(run_id, "Application is not published."))
                return

            # ---- Ensure chat session ----
            if session_id:
                session = await chat_repo.get_session(db, session_id, user_id or "")
                if session is None or session.is_closed:
                    await _update_step("step_guardrails", "failed", "Session invalid")
                    await _emit(builder.final_output_update(run_id, "Chat session not found or closed."))
                    return
            else:
                session = await chat_repo.create_session(
                    db, app_id, user_id or "",
                    title=user_input[:80] if user_input else "New Chat",
                )
                session_id = str(session.chat_session_id)
                if app.welcome_prompt:
                    await chat_repo.add_message(
                        db, session_id, "assistant", app.welcome_prompt,
                        execution_trace={"type": "welcome_prompt"},
                    )

            # ---- Step 1: Pre-guardrails ----
            await _update_step("step_guardrails", "running")
            guardrail_categories = await self._get_guardrail_category_names(db, app)
            pre_guard = self._guardrails.pre_process(
                user_input,
                guardrail_categories=guardrail_categories,
                guardrail_text=app.guardrail_text,
            )
            trace_data["guardrail_input"] = {
                "is_safe": pre_guard.is_safe,
                "violations": pre_guard.violations,
                "categories_triggered": pre_guard.categories_triggered,
            }

            if not pre_guard.is_safe:
                await _update_step("step_guardrails", "failed", "Input blocked by safety filters")
                violation_msg = (
                    "Your message was blocked by safety filters. "
                    f"Violations: {', '.join(pre_guard.violations)}"
                )
                await chat_repo.add_message(db, session_id, "user", user_input, guardrail_flags=trace_data["guardrail_input"])
                await chat_repo.add_message(db, session_id, "assistant", violation_msg)
                await _emit(builder.final_output_update(run_id, violation_msg))
                return

            await _update_step("step_guardrails", "completed")

            # ---- HITL state check ----
            effective_prompt = user_input
            _session_ctx_data = session.context_data or {}
            if (
                session.conversation_state == "awaiting_clarification"
                and _session_ctx_data.get("hitl_state")
            ):
                await _emit(builder.final_output_update(
                    run_id,
                    "This session is paused for human review. Use the review panel to continue.",
                ))
                return

            if session.conversation_state == "awaiting_clarification":
                clarification_ctx = await self._hitl_manager.get_clarification_context(db, session_id)
                if clarification_ctx:
                    effective_prompt = await self._hitl_manager.build_clarification_prompt(
                        original_prompt=clarification_ctx.get("original_prompt", user_input),
                        clarification_answer=user_input,
                        partial_analysis=clarification_ctx.get("partial_analysis", ""),
                        conversation_window=clarification_ctx.get("conversation_window"),
                    )

            # ---- Step 2: Enhance prompt ----
            await _update_step("step_enhance", "running")
            persona_prompt = await self._persona_formatter.build_persona_prompt(
                db,
                persona_id=str(app.persona_id) if app.persona_id else None,
                persona_text=app.persona_text,
            )
            enhancement = await self._prompt_enhancer.enhance(effective_prompt)
            enhanced_prompt = enhancement.get("enhanced_prompt", effective_prompt)
            trace_data["enhanced_prompt"] = enhanced_prompt
            await _update_step("step_enhance", "completed")

            # ---- Step 3: RAG retrieval ----
            await _update_step("step_rag", "running")
            document_context = None
            try:
                from echolib.di import container as di_container
                rag_manager = di_container.resolve('rag.session_manager')
                if rag_manager.has_documents(session_id):
                    document_context = rag_manager.retrieve(session_id, enhanced_prompt, top_k=10)
            except KeyError:
                logger.debug("rag.session_manager not registered; skipping document retrieval")
            except Exception:
                logger.exception("Document retrieval failed for session %s", session_id)
            await _update_step("step_rag", "completed", "Documents found" if document_context else "No documents")

            # ---- Step 4: Plan execution ----
            await _update_step("step_plan", "running")
            skill_manifest = await self._manifest_builder.build_manifest(db, app_id)
            trace_data["skill_count"] = len(skill_manifest.get("skills", []))

            history_messages = await chat_repo.get_messages(db, session_id, limit=8)
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history_messages
            ]

            app_tags = await self._get_tag_names(db, app)
            app_designations = await self._get_designation_names(db, app)
            app_business_units = await self._get_business_unit_names(db, app)
            app_data_sources = await self._get_data_source_names(db, app)
            guardrail_rules_text = app.guardrail_text or ""

            orchestrator_output = await self._orchestrator.plan(
                enhanced_prompt=enhanced_prompt,
                skill_manifest=skill_manifest,
                conversation_history=conversation_history,
                persona_prompt=persona_prompt,
                guardrail_rules=guardrail_rules_text,
                document_context=document_context,
                app_tags=app_tags,
                app_designations=app_designations,
                app_business_units=app_business_units,
                app_name=app.name,
                app_description=app.description,
                app_disclaimer=app.disclaimer,
                app_data_sources=app_data_sources,
            )
            trace_data["orchestrator_plan"] = orchestrator_output
            await _update_step("step_plan", "completed")

            # ---- Handle HITL response ----
            hitl_action = await self._hitl_manager.handle_orchestrator_response(
                db, session_id, orchestrator_output, enhanced_prompt,
                conversation_history=conversation_history,
            )
            action = hitl_action.get("action")

            if action == "clarify":
                clarification_msg = hitl_action.get("message", "Could you clarify?")
                await chat_repo.add_message(db, session_id, "user", user_input, enhanced_prompt=enhanced_prompt)
                await chat_repo.add_message(db, session_id, "assistant", clarification_msg, execution_trace={"type": "clarification"})
                await _update_step("step_execute", "skipped", "Clarification needed")
                await _update_step("step_post_guard", "skipped")
                await _emit(builder.final_output_update(run_id, clarification_msg))
                return

            if action == "fallback":
                fallback_msg = hitl_action.get("message", "")
                sorry = app.sorry_message or ""
                if sorry and fallback_msg:
                    full_fallback = f"{sorry}\n\n{fallback_msg}"
                elif sorry:
                    full_fallback = sorry
                else:
                    full_fallback = fallback_msg
                skills = skill_manifest.get("skills", [])
                if skills:
                    skill_summary = ", ".join(f"{s['name']} ({s['skill_type']})" for s in skills)
                    full_fallback += f"\n\nI can help you with: {skill_summary}. Please ask me something related to these capabilities."
                await chat_repo.add_message(db, session_id, "user", user_input, enhanced_prompt=enhanced_prompt)
                await chat_repo.add_message(db, session_id, "assistant", full_fallback, execution_trace={"type": "fallback", "fallback": True})
                await _update_step("step_execute", "skipped", "Fallback response")
                await _update_step("step_post_guard", "skipped")
                await _emit(builder.final_output_update(run_id, full_fallback))
                return

            # ---- Step 5: Execute skills ----
            await _update_step("step_execute", "running")
            plan = orchestrator_output.get("execution_plan", [])
            strategy = orchestrator_output.get("execution_strategy", "single")

            # Subscribe queue to ws_manager so workflow transparency events
            # are forwarded to the SSE stream during skill execution.
            ws_manager.subscribe_queue(run_id, sse_queue)

            try:
                execution_result = await self._skill_executor.execute_plan(
                    db=db,
                    execution_plan=plan,
                    execution_strategy=strategy,
                    user_input=enhanced_prompt,
                    document_context=document_context,
                )
            finally:
                ws_manager.unsubscribe_queue(run_id, sse_queue)

            # Check for HITL interrupt
            if execution_result.get("hitl_interrupted"):
                await _update_step("step_execute", "interrupted", "Paused for human review")
                await _update_step("step_post_guard", "skipped")
                await _emit(builder.final_output_update(run_id, "Workflow paused for human review."))
                return

            await _update_step("step_execute", "completed")

            final_output = execution_result.get("final_output", "")
            trace_data["execution_result"] = execution_result.get("results", {})
            trace_data["execution_log"] = execution_result.get("execution_log", [])

            # ---- Step 6: Post-guardrails ----
            await _update_step("step_post_guard", "running")
            post_guard = self._guardrails.post_process(
                final_output,
                guardrail_categories=guardrail_categories,
                guardrail_text=app.guardrail_text,
            )
            trace_data["guardrail_output"] = {
                "is_safe": post_guard.is_safe,
                "violations": post_guard.violations,
                "categories_triggered": post_guard.categories_triggered,
            }
            final_output = post_guard.sanitized_text
            await _update_step("step_post_guard", "completed")

            # ---- Save messages ----
            await chat_repo.add_message(db, session_id, "user", user_input, enhanced_prompt=enhanced_prompt)
            await chat_repo.add_message(
                db, session_id, "assistant", final_output,
                execution_trace=trace_data,
                guardrail_flags=trace_data.get("guardrail_output"),
            )

            # ---- Save execution trace ----
            total_duration_ms = int((time.monotonic() - pipeline_start) * 1000)
            trace_data["total_duration_ms"] = total_duration_ms
            try:
                await chat_repo.create_execution_trace(
                    db,
                    application_id=app_id,
                    chat_session_id=session_id,
                    user_message=user_input,
                    enhanced_prompt=enhanced_prompt,
                    orchestrator_plan=orchestrator_output,
                    execution_result=execution_result.get("results"),
                    skills_invoked=execution_result.get("execution_log"),
                    guardrail_input=trace_data.get("guardrail_input"),
                    guardrail_output=trace_data.get("guardrail_output"),
                    total_duration_ms=total_duration_ms,
                    status="completed",
                )
            except Exception:
                logger.exception("Failed to save execution trace")

            await chat_repo.update_session_state(db, session_id, conversation_state="awaiting_input")

            # ---- Emit final output ----
            await _emit(builder.final_output_update(run_id, final_output))

        except Exception as exc:
            logger.exception("run_stream failed for run_id=%s", run_id)
            # Try to mark the last running step as failed
            for s in reversed(steps):
                if s["status"] == "running":
                    s["status"] = "failed"
                    s["detail"] = str(exc)[:120]
                    break
            try:
                await _emit(builder.step_update(run_id, steps))
                await _emit(builder.final_output_update(run_id, f"Pipeline error: {exc}"))
            except Exception:
                pass
        finally:
            # Always put sentinel so SSE generator terminates
            try:
                await sse_queue.put(None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_guardrail_category_names(db: AsyncSession, app) -> List[str]:
        """Extract guardrail category names from app's guardrail links."""
        if not app.guardrail_links:
            return []

        from echolib.models.application_lookups import AppGuardrailCategory
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        category_ids = []
        for link in app.guardrail_links:
            uid = safe_uuid(str(link.guardrail_category_id))
            if uid:
                category_ids.append(uid)

        if not category_ids:
            return []

        stmt = select(AppGuardrailCategory).where(
            AppGuardrailCategory.guardrail_category_id.in_(category_ids)
        )
        result = await db.execute(stmt)
        categories = result.scalars().all()
        return [cat.name for cat in categories]

    @staticmethod
    async def _get_tag_names(db: AsyncSession, app) -> List[str]:
        """Extract tag names from app's tag links."""
        if not app.tag_links:
            return []

        from echolib.models.application_lookups import AppTag
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        tag_ids = []
        for link in app.tag_links:
            uid = safe_uuid(str(link.tag_id))
            if uid:
                tag_ids.append(uid)

        if not tag_ids:
            return []

        stmt = select(AppTag).where(AppTag.tag_id.in_(tag_ids))
        result = await db.execute(stmt)
        tags = result.scalars().all()
        return [t.name for t in tags]

    @staticmethod
    async def _get_designation_names(db: AsyncSession, app) -> List[str]:
        """Extract designation names from app's designation links."""
        if not app.designation_links:
            return []

        from echolib.models.application_lookups import AppDesignation
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        ids = []
        for link in app.designation_links:
            uid = safe_uuid(str(link.designation_id))
            if uid:
                ids.append(uid)

        if not ids:
            return []

        stmt = select(AppDesignation).where(AppDesignation.designation_id.in_(ids))
        result = await db.execute(stmt)
        return [d.name for d in result.scalars().all()]

    @staticmethod
    async def _get_business_unit_names(db: AsyncSession, app) -> List[str]:
        """Extract business unit names from app's business unit links."""
        if not app.business_unit_links:
            return []

        from echolib.models.application_lookups import AppBusinessUnit
        from echolib.repositories.base import safe_uuid
        from sqlalchemy import select

        ids = []
        for link in app.business_unit_links:
            uid = safe_uuid(str(link.business_unit_id))
            if uid:
                ids.append(uid)

        if not ids:
            return []

        stmt = select(AppBusinessUnit).where(AppBusinessUnit.business_unit_id.in_(ids))
        result = await db.execute(stmt)
        return [bu.name for bu in result.scalars().all()]

    @staticmethod
    async def _get_data_source_names(db: AsyncSession, app) -> List[str]:
        """Extract data source names from app's data source links."""
        if not app.data_source_links:
            return []

        from echolib.models.application_lookups import AppDataSource
        from sqlalchemy import select

        ds_ids = [link.data_source_id for link in app.data_source_links]
        if not ds_ids:
            return []

        stmt = select(AppDataSource).where(
            AppDataSource.data_source_id.in_(ds_ids)
        )
        result = await db.execute(stmt)
        return [ds.name for ds in result.scalars().all()]

    @staticmethod
    def _error_response(
        message: str, session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Build a standard error response dict."""
        return {
            "response": message,
            "session_id": session_id or "",
            "conversation_state": "awaiting_input",
            "execution_trace": {"error": message},
        }

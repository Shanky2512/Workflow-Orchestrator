"""
EchoAI Orchestrator -- Human-in-the-Loop (HITL) Manager

Manages conversation state transitions for the HITL flow:

    awaiting_input --> executing --> awaiting_input              (normal)
    awaiting_input --> awaiting_clarification --> awaiting_input (clarification)

When the orchestrator sets ``needs_clarification=true``, this module:
    1. Stores the original prompt and partial analysis in session context_data
    2. Transitions session to ``awaiting_clarification``
    3. Returns a clarification question to the user

When the user replies during ``awaiting_clarification``:
    1. Retrieves stored context
    2. Combines original prompt with the clarification answer
    3. Returns a merged prompt for the orchestrator to re-run

When the orchestrator sets ``fallback_message``:
    1. Returns the app's sorry_message plus a capability summary
    2. Session stays in ``awaiting_input``
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class HITLManager:
    """
    Stateless HITL manager.

    All state is persisted in the ``conversation_state`` and
    ``context_data`` columns of the ``application_chat_sessions`` table
    via ``ApplicationChatRepository``.
    """

    async def handle_orchestrator_response(
        self,
        db: AsyncSession,
        chat_session_id: str,
        orchestrator_output: Dict[str, Any],
        original_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect the orchestrator output and decide the next action.

        Args:
            db: Async database session.
            chat_session_id: Chat session UUID string.
            orchestrator_output: Parsed orchestrator JSON plan.
            original_prompt: The enhanced prompt that was sent to the orchestrator.
            conversation_history: Current conversation window (last 4 exchanges /
                8 messages). Stored in context_data during clarification so the
                orchestrator retains conversational context on re-run.

        Returns:
            Action dict with keys:
                action: "execute" | "clarify" | "fallback"
                message: Optional human-facing message (for clarify/fallback)
                plan: The execution plan (only when action="execute")
        """
        from echolib.repositories.application_chat_repo import ApplicationChatRepository

        chat_repo = ApplicationChatRepository()

        # --- Clarification needed ---
        if orchestrator_output.get("needs_clarification"):
            question = (
                orchestrator_output.get("clarification_question")
                or "Could you please provide more details about your request?"
            )

            # Store context for when user replies, including the current
            # conversation window so the orchestrator retains full context
            # when re-running after the user answers the clarification.
            context = {
                "original_prompt": original_prompt,
                "partial_analysis": orchestrator_output.get("reasoning", ""),
                "clarification_question": question,
                "conversation_window": conversation_history or [],
            }

            await chat_repo.update_session_state(
                db,
                chat_session_id,
                conversation_state="awaiting_clarification",
                context_data=context,
            )

            return {
                "action": "clarify",
                "message": question,
                "plan": None,
            }

        # --- Fallback (out-of-scope) ---
        fallback_msg = orchestrator_output.get("fallback_message")
        if fallback_msg:
            # Session stays in awaiting_input (no state transition)
            return {
                "action": "fallback",
                "message": fallback_msg,
                "plan": None,
            }

        # --- Normal execution ---
        await chat_repo.update_session_state(
            db,
            chat_session_id,
            conversation_state="executing",
        )

        return {
            "action": "execute",
            "message": None,
            "plan": orchestrator_output,
        }

    async def get_clarification_context(
        self,
        db: AsyncSession,
        chat_session_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve stored clarification context for a session currently
        in the ``awaiting_clarification`` state.

        Args:
            db: Async database session.
            chat_session_id: Chat session UUID string.

        Returns:
            Dict with keys:
                original_prompt (str)
                partial_analysis (str)
                clarification_question (str)
            or empty dict if not in clarification state.
        """
        from echolib.repositories.application_chat_repo import ApplicationChatRepository
        from echolib.repositories.base import safe_uuid

        chat_repo = ApplicationChatRepository()
        sess_uuid_str = chat_session_id

        # We need to load the session to read context_data.
        # Use a direct query instead of get_session (which needs user_id).
        from sqlalchemy import select
        from echolib.models.application_chat import ApplicationChatSession

        sess_uuid = safe_uuid(sess_uuid_str)
        if sess_uuid is None:
            return {}

        stmt = select(ApplicationChatSession).where(
            ApplicationChatSession.chat_session_id == sess_uuid
        )
        result = await db.execute(stmt)
        session_obj = result.scalar_one_or_none()

        if session_obj is None:
            return {}

        if session_obj.conversation_state != "awaiting_clarification":
            return {}

        return session_obj.context_data or {}

    async def build_clarification_prompt(
        self,
        original_prompt: str,
        clarification_answer: str,
        partial_analysis: str,
        conversation_window: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Combine the original prompt with the user's clarification answer
        into a single prompt for the orchestrator to re-run.

        When a conversation_window is available (stored during the initial
        clarification request), it is prepended so the orchestrator LLM
        retains the full conversational context.

        Args:
            original_prompt: The original enhanced prompt.
            clarification_answer: The user's answer to the clarification question.
            partial_analysis: The orchestrator's reasoning from the first pass.
            conversation_window: Snapshot of the last 4 exchanges (up to 8
                messages) captured when the clarification was first triggered.
                Defaults to None for backward compatibility.

        Returns:
            Combined prompt string.
        """
        parts: List[str] = []

        # Include prior conversation context if available
        if conversation_window:
            history_lines = []
            for msg in conversation_window:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate to keep the prompt within budget
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"[{role}]: {content}")
            parts.append(
                "Previous conversation context:\n" + "\n".join(history_lines)
            )

        parts.append(f"Original request: {original_prompt}")
        if partial_analysis:
            parts.append(f"My initial analysis: {partial_analysis}")
        parts.append(f"User's clarification: {clarification_answer}")
        parts.append(
            "Based on the above conversation and clarification, "
            "please provide a complete plan."
        )
        return "\n\n".join(parts)

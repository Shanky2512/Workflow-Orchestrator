"""
Workbench Mode -- Chat Service

Passthrough chat service for Workbench mode.
Frontend handles LLM calls via its own 3rd-party endpoints.
Backend provides session management and message relay only.

This service must NEVER import from:
    - apps.appmgr.orchestrator
    - apps.workflow
    - apps.agent
    - llm_manager
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class WorkbenchChatService:
    """
    Workbench chat passthrough service.

    NO orchestrator, NO prompt enhancer, NO skill executor,
    NO guardrails, NO LLM calls from backend.
    """

    def chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """
        Process a workbench chat message.

        The backend acts as a passthrough -- it accepts the message,
        ensures a session exists, and returns a response envelope.
        The frontend is responsible for LLM interaction via its own
        3rd-party Workbench endpoints.

        Args:
            message: The user message text.
            session_id: Existing session ID, or None to create a new one.

        Returns:
            Dict with response, session_id, mode, and timestamp.
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Workbench: created new session {session_id}")

        return {
            "response": message,
            "session_id": session_id,
            "mode": "workbench",
            "timestamp": datetime.now(timezone.utc),
        }

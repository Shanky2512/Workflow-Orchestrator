"""
Workbench Mode -- Manager

Manages the Workbench lifecycle and delegates to individual services.
Acts as the single coordination point for all Workbench operations
without touching any orchestrator logic.
"""

import logging
from typing import Optional

from .services.chat_service import WorkbenchChatService
from .services.translation_service import TranslationService

logger = logging.getLogger(__name__)


class WorkbenchManager:
    """
    Manages Workbench lifecycle and delegates to services.

    This is the central entry point for all Workbench operations.
    Future Workbench tools/services can be registered here.
    """

    def __init__(
        self,
        chat_service: WorkbenchChatService,
        translation_service: TranslationService,
    ):
        self._chat_service = chat_service
        self._translation_service = translation_service

    def is_workbench_enabled(self) -> bool:
        """Check if Workbench mode is available."""
        return True

    def available_services(self) -> list[str]:
        """List available Workbench services."""
        return ["chat", "translate", "detect"]

    def handle_workbench_chat(
        self, message: str, session_id: Optional[str] = None
    ) -> dict:
        """Delegate chat to WorkbenchChatService."""
        return self._chat_service.chat(message, session_id)

    def handle_translation(self, text: str):
        """Delegate translation to TranslationService."""
        return self._translation_service.translate(text)

    def handle_detection(self, text: str):
        """Delegate language detection to TranslationService."""
        return self._translation_service.detect(text)

"""
Workbench Mode -- DI Container Registration

Registers all Workbench providers into the global ``echolib.di.container``.

Providers:
    workbench.translation  -- TranslationService (Azure Translator API)
    workbench.chat         -- WorkbenchChatService (passthrough chat)
    workbench.manager      -- WorkbenchManager (lifecycle coordinator)
"""

import os

from echolib.di import container
from .services.translation_service import TranslationService
from .services.chat_service import WorkbenchChatService
from .manager import WorkbenchManager

# ── Translation Service (Azure Translator API) ────────────────────────────

_translation_service = TranslationService(
    base_url=os.environ.get("TRANSLATION_BASE_URL", "https://api.workbench.kpmg/translator/azure/text"),
    api_key=os.environ.get("TRANSLATION_API_KEY", "7bd51d543bbb464da97d724e4cec241e"),
    charge_code=os.environ.get("x-kpmg-charge-code", "0000000"),
    api_version=os.environ.get("TRANSLATION_API_VERSION", "3.0"),
)

container.register('workbench.translation', lambda: _translation_service)

# ── Chat Service ───────────────────────────────────────────────────────────

_chat_service = WorkbenchChatService()

container.register('workbench.chat', lambda: _chat_service)

# ── Workbench Manager ─────────────────────────────────────────────────────

_manager = WorkbenchManager(
    chat_service=_chat_service,
    translation_service=_translation_service,
)

container.register('workbench.manager', lambda: _manager)

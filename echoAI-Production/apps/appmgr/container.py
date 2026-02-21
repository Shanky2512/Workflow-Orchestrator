"""
EchoAI Application Manager -- DI Container Registration

Registers providers for the Application Orchestrator module into
the global ``echolib.di.container``.

Providers:
    app.store              -- Legacy in-memory app store (kept for backwards compat)
    app.repo               -- ApplicationRepository (async PostgreSQL CRUD)
    app.chat_repo          -- ApplicationChatRepository (async chat/session CRUD)
"""

from echolib.di import container
from echolib.types import App
from echolib.repositories.application_repo import ApplicationRepository
from echolib.repositories.application_chat_repo import ApplicationChatRepository

# ---------------------------------------------------------------------------
# Legacy in-memory store (preserved for backwards compatibility)
# ---------------------------------------------------------------------------
_app_store: dict[str, App] = {}


def app_store():
    return _app_store


container.register('app.store', app_store)

# ---------------------------------------------------------------------------
# PostgreSQL repositories (new, Phase 1+2)
# ---------------------------------------------------------------------------
_application_repo = ApplicationRepository()
_application_chat_repo = ApplicationChatRepository()

container.register('app.repo', lambda: _application_repo)
container.register('app.chat_repo', lambda: _application_chat_repo)

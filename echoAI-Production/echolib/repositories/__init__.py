"""
EchoAI Repository Layer

Provides data access abstraction with user scoping for multi-tenancy.
All repositories filter by user_id to ensure data isolation between users.

Usage:
    from echolib.repositories import (
        UserRepository,
        AgentRepository,
        WorkflowRepository,
        SessionRepository,
        SessionToolConfigRepository,
        ToolRepository,
        ExecutionRepository,
        ApplicationRepository,
        ApplicationChatRepository,
    )

    # Create repository instances
    user_repo = UserRepository()
    agent_repo = AgentRepository()
    app_repo = ApplicationRepository()
    app_chat_repo = ApplicationChatRepository()

    # Use with async session
    async with get_db_session() as db:
        user = await user_repo.get_by_email(db, "user@example.com")
        agents = await agent_repo.list_by_user(db, user.user_id, limit=10, offset=0)
        apps, total = await app_repo.list_applications(db, str(user.user_id))
"""

from .base import BaseRepository
from .user_repo import UserRepository
from .agent_repo import AgentRepository
from .workflow_repo import WorkflowRepository
from .session_repo import SessionRepository
from .session_tool_config_repo import SessionToolConfigRepository
from .tool_repo import ToolRepository
from .execution_repo import ExecutionRepository
from .application_repo import ApplicationRepository
from .application_chat_repo import ApplicationChatRepository

__all__ = [
    # Base
    "BaseRepository",
    # Entity repositories
    "UserRepository",
    "AgentRepository",
    "WorkflowRepository",
    "SessionRepository",
    "SessionToolConfigRepository",
    "ToolRepository",
    "ExecutionRepository",
    # Application Orchestrator repositories
    "ApplicationRepository",
    "ApplicationChatRepository",
]

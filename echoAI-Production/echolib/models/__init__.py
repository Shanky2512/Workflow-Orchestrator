"""
EchoAI SQLAlchemy Models

This module exports all database models for the EchoAI platform.
Models follow the simplified schema design where complete JSON definitions
are stored in JSONB columns and only key fields are extracted for indexing.

Usage:
    from echolib.models import User, Agent, Workflow, ChatSession
    from echolib.models import Application, ApplicationChatSession

    # Create a new user
    user = User(email="user@example.com", display_name="John Doe")

    # Access workflow definition as dict
    workflow_dict = workflow.to_dict()  # Returns the definition JSONB

Models:
    - User: Authenticated user accounts
    - Agent: AI agent definitions (synced from workflow saves)
    - Workflow: Workflow definitions with embedded agents
    - WorkflowVersion: Immutable workflow version snapshots
    - ChatSession: Chat sessions with embedded messages
    - SessionToolConfig: Tool configuration overrides per session
    - Tool: User-owned tool definitions
    - Execution: Workflow execution runs
    - HITLCheckpoint: Human-in-the-loop checkpoint state
    - Application: AI application definitions (orchestrator module)
    - ApplicationLlmLink: App-to-LLM association
    - ApplicationSkillLink: App-to-skill (workflow/agent) association
    - ApplicationDataSourceLink: App-to-data-source association
    - ApplicationDesignationLink: App-to-designation association
    - ApplicationBusinessUnitLink: App-to-business-unit association
    - ApplicationTagLink: App-to-tag association
    - ApplicationGuardrailLink: App-to-guardrail-category association
    - ApplicationChatSession: Chat sessions for applications
    - ApplicationChatMessage: Messages within application chat sessions
    - ApplicationDocument: Uploaded documents for application RAG
    - ApplicationExecutionTrace: Orchestration audit trail
    - AppPersona: Persona lookup catalog
    - AppGuardrailCategory: Guardrail category lookup catalog
    - AppDesignation: Designation lookup catalog
    - AppBusinessUnit: Business unit lookup catalog
    - AppTag: Tag lookup catalog
    - AppLlm: LLM configuration lookup catalog
    - AppDataSource: Data source lookup catalog
"""

# Import Base from database for Alembic migrations
from ..database import Base

from .base import BaseModel as EchoBaseModel
from .user import User
from .agent import Agent
from .workflow import Workflow, WorkflowVersion
from .session import ChatSession
from .session_tool_config import SessionToolConfig
from .tool import Tool
from .execution import Execution, HITLCheckpoint

# Application Orchestrator models
from .application_lookups import (
    AppPersona,
    AppGuardrailCategory,
    AppDesignation,
    AppBusinessUnit,
    AppTag,
    AppLlm,
    AppDataSource,
)
from .application import (
    Application,
    ApplicationLlmLink,
    ApplicationSkillLink,
    ApplicationDataSourceLink,
    ApplicationDesignationLink,
    ApplicationBusinessUnitLink,
    ApplicationTagLink,
    ApplicationGuardrailLink,
)
from .application_chat import (
    ApplicationChatSession,
    ApplicationChatMessage,
    ApplicationDocument,
    ApplicationExecutionTrace,
)

__all__ = [
    # Base class for Alembic
    "Base",
    # Base model with common fields
    "EchoBaseModel",
    # Core models
    "User",
    "Agent",
    "Workflow",
    "WorkflowVersion",
    "ChatSession",
    "SessionToolConfig",
    "Tool",
    "Execution",
    "HITLCheckpoint",
    # Application Orchestrator — lookup / catalog tables
    "AppPersona",
    "AppGuardrailCategory",
    "AppDesignation",
    "AppBusinessUnit",
    "AppTag",
    "AppLlm",
    "AppDataSource",
    # Application Orchestrator — core application + link tables
    "Application",
    "ApplicationLlmLink",
    "ApplicationSkillLink",
    "ApplicationDataSourceLink",
    "ApplicationDesignationLink",
    "ApplicationBusinessUnitLink",
    "ApplicationTagLink",
    "ApplicationGuardrailLink",
    # Application Orchestrator — chat + execution models
    "ApplicationChatSession",
    "ApplicationChatMessage",
    "ApplicationDocument",
    "ApplicationExecutionTrace",
]

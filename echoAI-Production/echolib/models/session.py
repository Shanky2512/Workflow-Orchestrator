"""
EchoAI Chat Session Model

Stores user chat sessions with embedded messages as JSONB.
This provides a ChatGPT-like experience with persistent conversations.

Design Principle:
- Messages are embedded in the session as a JSONB array
- Single query retrieves entire conversation
- LLM can receive whole chat context in one read
- Atomic session + messages save
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, ForeignKey, Index, CheckConstraint, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class ChatSession(BaseModel):
    """
    Chat session model with embedded messages.

    Stores user chat sessions that can be bound to different contexts
    (general chat, workflow testing, agent chat, workflow design).

    Attributes:
        session_id: Primary key UUID
        user_id: Owner user (FK to users)
        title: Session title for display
        context_type: What the session is bound to (general/workflow/agent/workflow_design)
        context_id: Optional UUID of the bound entity (workflow_id or agent_id)
        workflow_mode: For workflow contexts - draft/test/final
        workflow_version: For workflow contexts - version string
        messages: JSONB array of chat messages (embedded)
        selected_tool_ids: Array of tool UUIDs selected for this session
        context_data: Additional context data
        variables: Session variables
        state_schema: State schema for workflow contexts
        run_ids: Array of execution run IDs from this session
        last_activity: Last message/activity timestamp

    Messages JSONB Structure:
        [
            {
                "id": "msg_abc123",
                "role": "user" | "assistant" | "system",
                "content": "...",
                "timestamp": "2026-02-02T16:53:08.084505Z",
                "agent_id": "agt_xxx" | null,
                "run_id": "run_xyz" | null,
                "metadata": {...}
            },
            ...
        ]

    Context Types:
        - general: Free-form chat, no binding (context_id = NULL)
        - workflow: Testing a specific workflow (context_id = workflow_id)
        - agent: Direct chat with an agent (context_id = agent_id)
        - workflow_design: Iterating on workflow design (context_id = workflow_id)

    Indexes:
        - idx_sessions_user: Fast user lookup
        - idx_sessions_user_active: Active sessions ordered by activity
        - idx_sessions_context: Filter by context type and ID
    """
    __tablename__ = "chat_sessions"

    # Primary key
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    # Owner (FK to users)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Session identity
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Chat",
        server_default="New Chat",
        nullable=False
    )

    # Context binding (flexible)
    context_type: Mapped[str] = mapped_column(
        String(20),
        default="general",
        server_default="general",
        nullable=False
    )

    context_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True  # NULL for general chats
    )

    # Workflow-specific state
    workflow_mode: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True  # draft/test/final for workflow contexts
    )

    workflow_version: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )

    # EMBEDDED CHAT MESSAGES (LLM-friendly)
    messages: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        server_default="[]",
        nullable=False
    )

    # Tool selection (array of UUIDs)
    selected_tool_ids: Mapped[List[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        server_default="{}",
        nullable=False
    )

    # Additional context and state
    context_data: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    variables: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    state_schema: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    # Execution tracking
    run_ids: Mapped[List[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        server_default="{}",
        nullable=False
    )

    # Activity tracking
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default="CURRENT_TIMESTAMP",
        nullable=False
    )

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False
    )

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "context_type IN ('general', 'workflow', 'agent', 'workflow_design')",
            name="valid_context_type"
        ),
        Index("idx_sessions_user_active", "user_id", "last_activity"),
        Index("idx_sessions_context", "context_type", "context_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary with all fields.

        Returns:
            Dict containing complete session data including messages.
        """
        return {
            "session_id": str(self.session_id),
            "user_id": str(self.user_id),
            "title": self.title,
            "context_type": self.context_type,
            "context_id": str(self.context_id) if self.context_id else None,
            "workflow_mode": self.workflow_mode,
            "workflow_version": self.workflow_version,
            "messages": self.messages or [],
            "selected_tool_ids": [str(tid) for tid in self.selected_tool_ids] if self.selected_tool_ids else [],
            "context_data": self.context_data or {},
            "variables": self.variables or {},
            "state_schema": self.state_schema or {},
            "run_ids": [str(rid) for rid in self.run_ids] if self.run_ids else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "is_deleted": self.is_deleted
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Return session summary without messages.

        Useful for listing sessions without loading full conversation history.

        Returns:
            Dict with session metadata only (no messages).
        """
        return {
            "session_id": str(self.session_id),
            "user_id": str(self.user_id),
            "title": self.title,
            "context_type": self.context_type,
            "context_id": str(self.context_id) if self.context_id else None,
            "workflow_mode": self.workflow_mode,
            "message_count": len(self.messages) if self.messages else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }

    def add_message(
        self,
        role: str,
        content: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to the session.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            agent_id: Optional agent ID that produced the message
            run_id: Optional execution run ID
            metadata: Optional additional metadata

        Returns:
            The newly created message dict.
        """
        message = {
            "id": f"msg_{uuid4().hex[:12]}",
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata or {}
        }

        if self.messages is None:
            self.messages = []
        self.messages.append(message)
        self.last_activity = datetime.now(timezone.utc)

        return message

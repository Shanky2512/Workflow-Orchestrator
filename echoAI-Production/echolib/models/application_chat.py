"""
EchoAI Application Chat Models

Models for the chat subsystem of the AI Application Orchestrator:
- ApplicationChatSession: Conversation sessions bound to an application + user
- ApplicationChatMessage: Individual messages within a session
- ApplicationDocument: Uploaded documents for RAG (application or session scoped)
- ApplicationExecutionTrace: Audit trail for orchestration runs

Design Principles:
- UUID primary keys consistent with existing models
- Foreign keys cascade on application deletion
- JSONB columns for flexible structured data (execution traces, guardrail flags)
- Conversation state machine tracked on the session level
- Separate messages table (not embedded JSONB) for efficient querying and indexing
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
    Boolean,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


# ---------------------------------------------------------------------------
# Application Chat Session
# ---------------------------------------------------------------------------

class ApplicationChatSession(Base):
    """
    A chat session between a user and an application.

    Each session tracks the conversation state for HITL (human-in-the-loop)
    flows, the LLM used, and any context data needed for clarification.

    Conversation States:
        - awaiting_input: waiting for user message
        - awaiting_clarification: orchestrator asked a clarifying question
        - executing: orchestrator is running skills

    Attributes:
        chat_session_id: Primary key UUID
        application_id: FK to applications table
        user_id: FK to users table (session owner)
        title: Optional session title (auto-generated or user-set)
        conversation_state: Current HITL state
        llm_id: Which LLM was used for this session
        is_closed: Whether the session has been ended
        context_data: JSONB for pending clarification context, partial plans, etc.
        started_at: Session creation timestamp
        updated_at: Last activity timestamp

    Indexes:
        - idx_app_chat_sessions_user: (application_id, user_id)
    """
    __tablename__ = "application_chat_sessions"

    chat_session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id"),
        nullable=False,
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    conversation_state: Mapped[str] = mapped_column(
        String(30),
        default="awaiting_input",
        server_default="awaiting_input",
        nullable=False,
    )
    llm_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )
    is_closed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )
    context_data: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        server_default="'{}'::jsonb",
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    messages: Mapped[List["ApplicationChatMessage"]] = relationship(
        "ApplicationChatMessage",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="ApplicationChatMessage.created_at",
    )
    documents: Mapped[List["ApplicationDocument"]] = relationship(
        "ApplicationDocument",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="noload",
        foreign_keys="ApplicationDocument.chat_session_id",
    )

    __table_args__ = (
        CheckConstraint(
            "conversation_state IN ('awaiting_input', 'awaiting_clarification', 'executing')",
            name="valid_conversation_state",
        ),
        Index(
            "idx_app_chat_sessions_user",
            "application_id",
            "user_id",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return session data as a dictionary."""
        return {
            "chat_session_id": str(self.chat_session_id),
            "application_id": str(self.application_id),
            "user_id": str(self.user_id),
            "title": self.title,
            "conversation_state": self.conversation_state,
            "llm_id": str(self.llm_id) if self.llm_id else None,
            "is_closed": self.is_closed,
            "context_data": self.context_data or {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------------
# Application Chat Message
# ---------------------------------------------------------------------------

class ApplicationChatMessage(Base):
    """
    A single message within an application chat session.

    Messages are stored in a separate table (not embedded JSONB) to allow
    efficient querying, indexing, and pagination of conversation history.

    Attributes:
        message_id: Primary key UUID
        chat_session_id: FK to application_chat_sessions
        role: Message role -- 'user', 'assistant', or 'system'
        content: Message text content
        enhanced_prompt: Prompt-enhanced version of user input (null for assistant/system)
        execution_trace: JSONB with orchestrator reasoning, skills invoked, timing
        guardrail_flags: JSONB with any guardrail violations detected
        created_at: Message creation timestamp

    Indexes:
        - idx_app_chat_messages_session: (chat_session_id, created_at)
    """
    __tablename__ = "application_chat_messages"

    message_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    chat_session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("application_chat_sessions.chat_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    enhanced_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    execution_trace: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    guardrail_flags: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "idx_app_chat_messages_session",
            "chat_session_id",
            "created_at",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return message data as a dictionary."""
        return {
            "message_id": str(self.message_id),
            "chat_session_id": str(self.chat_session_id),
            "role": self.role,
            "content": self.content,
            "enhanced_prompt": self.enhanced_prompt,
            "execution_trace": self.execution_trace,
            "guardrail_flags": self.guardrail_flags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Application Document (RAG / file uploads)
# ---------------------------------------------------------------------------

class ApplicationDocument(Base):
    """
    An uploaded document associated with an application, optionally
    bound to a specific chat session.

    Used for RAG (Retrieval-Augmented Generation) when upload_enabled
    is true on the application.

    Processing States:
        - pending: uploaded, not yet processed
        - processing: being chunked / embedded
        - ready: fully processed and queryable
        - failed: processing encountered an error

    Attributes:
        document_id: Primary key UUID
        application_id: FK to applications table
        chat_session_id: Optional FK to chat session (null if app-level upload)
        original_filename: Name of the file as uploaded by the user
        stored_filename: Internal storage filename
        public_url: Accessible URL for the document
        mime_type: MIME type of the file
        size_bytes: File size in bytes
        processing_status: Current processing state
        chunk_count: Number of chunks after processing
        metadata: JSONB for additional metadata
        created_at: Upload timestamp
    """
    __tablename__ = "application_documents"

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        nullable=False,
    )
    chat_session_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("application_chat_sessions.chat_session_id"),
        nullable=True,
    )
    original_filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    stored_filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    public_url: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    size_bytes: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
    )
    processing_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default="pending",
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    doc_metadata: Mapped[Optional[Dict]] = mapped_column(
        "metadata",
        JSONB,
        server_default="'{}'::jsonb",
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('pending', 'processing', 'ready', 'failed')",
            name="valid_doc_processing_status",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return document data as a dictionary."""
        return {
            "document_id": str(self.document_id),
            "application_id": str(self.application_id),
            "chat_session_id": str(self.chat_session_id) if self.chat_session_id else None,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "public_url": self.public_url,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "processing_status": self.processing_status,
            "chunk_count": self.chunk_count,
            "metadata": self.doc_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Application Execution Trace (Audit Trail)
# ---------------------------------------------------------------------------

class ApplicationExecutionTrace(Base):
    """
    Audit trail for a single orchestration execution.

    Records the full pipeline: user message -> prompt enhancement ->
    orchestrator plan -> skill execution -> guardrail results -> timing.

    Execution Statuses:
        - pending: trace created, execution not yet started
        - running: skills are currently executing
        - completed: execution finished successfully
        - failed: execution encountered an error

    Attributes:
        trace_id: Primary key UUID
        application_id: FK to applications table
        chat_session_id: FK to chat session
        message_id: Optional FK to the specific user message that triggered this
        user_message: The original user input text
        enhanced_prompt: The prompt after enhancement
        orchestrator_plan: Full orchestrator output JSON
        execution_result: Results from each skill execution
        skills_invoked: List of skill invocation details with timing
        guardrail_input: Pre-processing guardrail results
        guardrail_output: Post-processing guardrail results
        total_duration_ms: End-to-end execution time in milliseconds
        status: Execution status
        error_message: Error details if status = 'failed'
        created_at: Trace creation timestamp

    Indexes:
        - idx_app_exec_traces_session: (chat_session_id, created_at)
    """
    __tablename__ = "application_execution_traces"

    trace_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        nullable=False,
    )
    chat_session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("application_chat_sessions.chat_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("application_chat_messages.message_id"),
        nullable=True,
    )
    user_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    enhanced_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    orchestrator_plan: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    execution_result: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    skills_invoked: Mapped[Optional[List]] = mapped_column(
        JSONB,
        nullable=True,
    )
    guardrail_input: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    guardrail_output: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    total_duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default="pending",
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="valid_exec_trace_status",
        ),
        Index(
            "idx_app_exec_traces_session",
            "chat_session_id",
            "created_at",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return execution trace data as a dictionary."""
        return {
            "trace_id": str(self.trace_id),
            "application_id": str(self.application_id),
            "chat_session_id": str(self.chat_session_id),
            "message_id": str(self.message_id) if self.message_id else None,
            "user_message": self.user_message,
            "enhanced_prompt": self.enhanced_prompt,
            "orchestrator_plan": self.orchestrator_plan,
            "execution_result": self.execution_result,
            "skills_invoked": self.skills_invoked,
            "guardrail_input": self.guardrail_input,
            "guardrail_output": self.guardrail_output,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

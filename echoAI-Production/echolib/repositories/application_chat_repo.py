"""
EchoAI Application Chat Repository

Async data access for the application chat subsystem:
    - Chat sessions (create, get, list, update state, close)
    - Chat messages (add, list)
    - Execution traces (create, update)
    - Documents (create)

All queries enforce user_id scoping where applicable.

Pattern follows existing repository conventions:
    - ``db: AsyncSession`` as first argument
    - ``safe_uuid()`` for UUID validation
    - ``session.flush()`` after mutations
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import (
    ApplicationChatMessage,
    ApplicationChatSession,
    ApplicationDocument,
    ApplicationExecutionTrace,
)
from .base import safe_uuid


logger = logging.getLogger(__name__)


class ApplicationChatRepository:
    """
    Repository for application chat sessions, messages, documents,
    and execution traces.
    """

    # ------------------------------------------------------------------
    # CHAT SESSION — CREATE
    # ------------------------------------------------------------------

    async def create_session(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
        title: Optional[str] = None,
        llm_id: Optional[str] = None,
    ) -> ApplicationChatSession:
        """
        Create a new chat session for an application.

        Args:
            db: Async database session.
            application_id: FK to applications table.
            user_id: FK to users table (session owner).
            title: Optional session title.
            llm_id: Optional LLM UUID used for this session.

        Returns:
            The newly created ApplicationChatSession instance.

        Raises:
            ValueError: If application_id or user_id is not a valid UUID.
        """
        app_uuid = safe_uuid(application_id)
        user_uuid = safe_uuid(user_id)
        if app_uuid is None or user_uuid is None:
            raise ValueError(
                f"Invalid UUID: application_id={application_id}, user_id={user_id}"
            )

        llm_uuid = safe_uuid(llm_id) if llm_id else None

        session_obj = ApplicationChatSession(
            application_id=app_uuid,
            user_id=user_uuid,
            title=title,
            llm_id=llm_uuid,
        )
        db.add(session_obj)
        await db.flush()
        await db.refresh(session_obj)
        return session_obj

    # ------------------------------------------------------------------
    # CHAT SESSION — GET
    # ------------------------------------------------------------------

    async def get_session(
        self,
        db: AsyncSession,
        chat_session_id: str,
        user_id: str,
    ) -> Optional[ApplicationChatSession]:
        """
        Get a chat session by ID, scoped to user.

        Args:
            db: Async database session.
            chat_session_id: Session primary key.
            user_id: Owner user ID for scoping.

        Returns:
            ApplicationChatSession if found and owned by user, None otherwise.
        """
        sess_uuid = safe_uuid(chat_session_id)
        user_uuid = safe_uuid(user_id)
        if sess_uuid is None or user_uuid is None:
            return None

        stmt = (
            select(ApplicationChatSession)
            .where(
                ApplicationChatSession.chat_session_id == sess_uuid,
                ApplicationChatSession.user_id == user_uuid,
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # CHAT SESSION — LIST
    # ------------------------------------------------------------------

    async def list_sessions(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
        limit: int = 50,
    ) -> List[ApplicationChatSession]:
        """
        List chat sessions for an application, ordered by updated_at DESC.

        Args:
            db: Async database session.
            application_id: FK to applications table.
            user_id: Owner user ID for scoping.
            limit: Maximum number of sessions to return.

        Returns:
            List of ApplicationChatSession instances.
        """
        app_uuid = safe_uuid(application_id)
        user_uuid = safe_uuid(user_id)
        if app_uuid is None or user_uuid is None:
            return []

        stmt = (
            select(ApplicationChatSession)
            .where(
                ApplicationChatSession.application_id == app_uuid,
                ApplicationChatSession.user_id == user_uuid,
            )
            .order_by(ApplicationChatSession.updated_at.desc())
            .limit(limit)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # CHAT SESSION — UPDATE STATE
    # ------------------------------------------------------------------

    async def update_session_state(
        self,
        db: AsyncSession,
        chat_session_id: str,
        conversation_state: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the conversation state (and optional context_data) of a chat session.

        Args:
            db: Async database session.
            chat_session_id: Session primary key.
            conversation_state: New state value
                                (awaiting_input / awaiting_clarification / executing).
            context_data: Optional JSONB context to store
                          (e.g. pending clarification info).
        """
        sess_uuid = safe_uuid(chat_session_id)
        if sess_uuid is None:
            return

        values: Dict[str, Any] = {"conversation_state": conversation_state}
        if context_data is not None:
            values["context_data"] = context_data

        stmt = (
            update(ApplicationChatSession)
            .where(ApplicationChatSession.chat_session_id == sess_uuid)
            .values(**values)
        )

        await db.execute(stmt)
        await db.flush()

    # ------------------------------------------------------------------
    # CHAT SESSION — CLOSE
    # ------------------------------------------------------------------

    async def close_session(
        self,
        db: AsyncSession,
        chat_session_id: str,
    ) -> None:
        """
        Mark a chat session as closed.

        Args:
            db: Async database session.
            chat_session_id: Session primary key.
        """
        sess_uuid = safe_uuid(chat_session_id)
        if sess_uuid is None:
            return

        stmt = (
            update(ApplicationChatSession)
            .where(ApplicationChatSession.chat_session_id == sess_uuid)
            .values(is_closed=True)
        )

        await db.execute(stmt)
        await db.flush()

    # ------------------------------------------------------------------
    # CHAT MESSAGE — ADD
    # ------------------------------------------------------------------

    async def add_message(
        self,
        db: AsyncSession,
        chat_session_id: str,
        role: str,
        content: str,
        enhanced_prompt: Optional[str] = None,
        execution_trace: Optional[Dict[str, Any]] = None,
        guardrail_flags: Optional[Dict[str, Any]] = None,
    ) -> ApplicationChatMessage:
        """
        Add a message to a chat session.

        Args:
            db: Async database session.
            chat_session_id: FK to application_chat_sessions.
            role: Message role (user / assistant / system).
            content: Message text content.
            enhanced_prompt: Prompt-enhanced version of user input.
            execution_trace: JSONB with orchestrator reasoning and timing.
            guardrail_flags: JSONB with any guardrail violations.

        Returns:
            The newly created ApplicationChatMessage instance.

        Raises:
            ValueError: If chat_session_id is not a valid UUID.
        """
        sess_uuid = safe_uuid(chat_session_id)
        if sess_uuid is None:
            raise ValueError(f"Invalid chat_session_id: {chat_session_id}")

        message = ApplicationChatMessage(
            chat_session_id=sess_uuid,
            role=role,
            content=content,
            enhanced_prompt=enhanced_prompt,
            execution_trace=execution_trace,
            guardrail_flags=guardrail_flags,
        )
        db.add(message)
        await db.flush()
        await db.refresh(message)
        return message

    # ------------------------------------------------------------------
    # CHAT MESSAGE — GET (list for a session)
    # ------------------------------------------------------------------

    async def get_messages(
        self,
        db: AsyncSession,
        chat_session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ApplicationChatMessage]:
        """
        Get messages for a chat session, ordered by created_at ASC.

        Args:
            db: Async database session.
            chat_session_id: FK to application_chat_sessions.
            limit: Maximum messages to return.
            offset: Number of records to skip.

        Returns:
            List of ApplicationChatMessage instances in chronological order.
        """
        sess_uuid = safe_uuid(chat_session_id)
        if sess_uuid is None:
            return []

        stmt = (
            select(ApplicationChatMessage)
            .where(ApplicationChatMessage.chat_session_id == sess_uuid)
            .order_by(ApplicationChatMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # EXECUTION TRACE — CREATE
    # ------------------------------------------------------------------

    async def create_execution_trace(
        self,
        db: AsyncSession,
        application_id: str,
        chat_session_id: str,
        user_message: str,
        **kwargs: Any,
    ) -> ApplicationExecutionTrace:
        """
        Create an execution trace record.

        Args:
            db: Async database session.
            application_id: FK to applications table.
            chat_session_id: FK to application_chat_sessions.
            user_message: The original user input text.
            **kwargs: Optional fields (enhanced_prompt, orchestrator_plan,
                      message_id, etc.).

        Returns:
            The newly created ApplicationExecutionTrace instance.

        Raises:
            ValueError: If application_id or chat_session_id is not a valid UUID.
        """
        app_uuid = safe_uuid(application_id)
        sess_uuid = safe_uuid(chat_session_id)
        if app_uuid is None or sess_uuid is None:
            raise ValueError(
                f"Invalid UUID: application_id={application_id}, "
                f"chat_session_id={chat_session_id}"
            )

        # Convert message_id to UUID if present
        if "message_id" in kwargs and kwargs["message_id"] is not None:
            kwargs["message_id"] = safe_uuid(kwargs["message_id"])

        trace = ApplicationExecutionTrace(
            application_id=app_uuid,
            chat_session_id=sess_uuid,
            user_message=user_message,
            **kwargs,
        )
        db.add(trace)
        await db.flush()
        await db.refresh(trace)
        return trace

    # ------------------------------------------------------------------
    # EXECUTION TRACE — UPDATE
    # ------------------------------------------------------------------

    async def update_execution_trace(
        self,
        db: AsyncSession,
        trace_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Update an execution trace (status, results, timing, etc.).

        Args:
            db: Async database session.
            trace_id: Execution trace primary key.
            **kwargs: Fields to update (status, execution_result,
                      skills_invoked, total_duration_ms, error_message, etc.).
        """
        t_uuid = safe_uuid(trace_id)
        if t_uuid is None:
            return

        if not kwargs:
            return

        # Convert message_id to UUID if present
        if "message_id" in kwargs and kwargs["message_id"] is not None:
            kwargs["message_id"] = safe_uuid(kwargs["message_id"])

        stmt = (
            update(ApplicationExecutionTrace)
            .where(ApplicationExecutionTrace.trace_id == t_uuid)
            .values(**kwargs)
        )

        await db.execute(stmt)
        await db.flush()

    # ------------------------------------------------------------------
    # DOCUMENT — GET SESSION DOCUMENTS
    # ------------------------------------------------------------------

    async def get_session_documents(
        self,
        db: AsyncSession,
        chat_session_id: str,
        status: Optional[str] = None,
    ) -> List[ApplicationDocument]:
        """
        Get all documents for a session, optionally filtered by status.

        Args:
            db: Async database session.
            chat_session_id: FK to application_chat_sessions.
            status: Optional processing status filter
                    (pending / processing / ready / failed).

        Returns:
            List of ApplicationDocument instances for the session.
        """
        sess_uuid = safe_uuid(chat_session_id)
        if sess_uuid is None:
            return []

        stmt = select(ApplicationDocument).where(
            ApplicationDocument.chat_session_id == sess_uuid,
        )
        if status is not None:
            stmt = stmt.where(ApplicationDocument.processing_status == status)

        stmt = stmt.order_by(ApplicationDocument.created_at.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # DOCUMENT — UPDATE STATUS
    # ------------------------------------------------------------------

    async def update_document_status(
        self,
        db: AsyncSession,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
    ) -> None:
        """
        Update document processing_status and chunk_count.

        Args:
            db: Async database session.
            document_id: Document primary key.
            status: New processing status
                    (pending / processing / ready / failed).
            chunk_count: Optional number of chunks created during processing.
        """
        doc_uuid = safe_uuid(document_id)
        if doc_uuid is None:
            return

        values: Dict[str, Any] = {"processing_status": status}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count

        stmt = (
            update(ApplicationDocument)
            .where(ApplicationDocument.document_id == doc_uuid)
            .values(**values)
        )
        await db.execute(stmt)
        await db.flush()

    # ------------------------------------------------------------------
    # DOCUMENT — CREATE
    # ------------------------------------------------------------------

    async def create_document(
        self,
        db: AsyncSession,
        application_id: str,
        chat_session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ApplicationDocument:
        """
        Create an application document record.

        Args:
            db: Async database session.
            application_id: FK to applications table.
            chat_session_id: Optional FK to a chat session.
            **kwargs: Required fields (original_filename, stored_filename)
                      and optional fields (public_url, mime_type, size_bytes,
                      metadata, etc.).

        Returns:
            The newly created ApplicationDocument instance.

        Raises:
            ValueError: If application_id is not a valid UUID.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            raise ValueError(f"Invalid application_id: {application_id}")

        sess_uuid = safe_uuid(chat_session_id) if chat_session_id else None

        doc = ApplicationDocument(
            application_id=app_uuid,
            chat_session_id=sess_uuid,
            **kwargs,
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc

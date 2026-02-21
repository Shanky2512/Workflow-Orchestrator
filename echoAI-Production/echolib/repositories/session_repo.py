"""
EchoAI Session Repository

Provides data access for chat sessions with Memcached caching.
Sessions are the context for user conversations with agents/workflows.

Cache Strategy:
    - Cache key format: session:{session_id}
    - Check cache first on get_by_id
    - Populate cache on miss
    - Invalidate cache on update/delete
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import ChatSession
from echolib.cache import cache_client
from .base import BaseRepository, safe_uuid


logger = logging.getLogger(__name__)


class SessionRepository(BaseRepository[ChatSession]):
    """
    Repository for ChatSession entity operations.

    Provides standard CRUD with Memcached caching plus:
        - Message management (add_message)
        - Context-based filtering
        - Tool selection updates

    Cache Integration:
        - get_by_id: Check cache first, then DB, populate cache on miss
        - update/delete: Invalidate cache
        - add_message: Update DB and invalidate cache
    """

    model_class = ChatSession
    id_field = "session_id"
    user_id_field = "user_id"
    soft_delete_field = "is_deleted"

    def _cache_key(self, session_id: str) -> str:
        """Generate cache key for a session."""
        return f"session:{session_id}"

    async def get_by_id(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str
    ) -> Optional[ChatSession]:
        """
        Get a session by ID with cache integration.

        Cache flow:
        1. Check cache first
        2. If cached and user_id matches, return from cache
        3. On cache miss, query database
        4. If found, populate cache
        5. Return result

        Args:
            db: Async database session
            session_id: Session primary key
            user_id: Owner user ID for scoping

        Returns:
            ChatSession instance if found and owned by user, None otherwise
        """
        cache_key = self._cache_key(session_id)

        # Try cache first
        try:
            cached = await cache_client.get(cache_key)
            if cached:
                # Verify user ownership
                cached_user_id = cached.get("user_id")
                if cached_user_id == user_id:
                    logger.debug(f"Cache hit for session {session_id}")
                    # Return a reconstructed session (for consistency, query DB)
                    # Cache is used for API response caching, not ORM session management
        except Exception as e:
            logger.warning(f"Cache get failed for session {session_id}: {e}")

        # Cache miss or validation failed - query database
        session = await super().get_by_id(db, session_id, user_id)

        if session:
            # Populate cache
            try:
                session_dict = session.to_dict()
                await cache_client.set(cache_key, session_dict)
                logger.debug(f"Cache populated for session {session_id}")
            except Exception as e:
                logger.warning(f"Cache set failed for session {session_id}: {e}")

        return session

    async def get_by_id_dict(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID as dictionary, with cache integration.

        This method returns the dict representation directly,
        leveraging cache more efficiently for API responses.

        Args:
            db: Async database session
            session_id: Session primary key
            user_id: Owner user ID for scoping

        Returns:
            Session dict if found and owned by user, None otherwise
        """
        cache_key = self._cache_key(session_id)

        # Try cache first
        try:
            cached = await cache_client.get(cache_key)
            if cached:
                # Verify user ownership
                cached_user_id = cached.get("user_id")
                if cached_user_id == user_id:
                    logger.debug(f"Cache hit for session dict {session_id}")
                    return cached
        except Exception as e:
            logger.warning(f"Cache get failed for session {session_id}: {e}")

        # Cache miss - query database
        session = await super().get_by_id(db, session_id, user_id)

        if session:
            session_dict = session.to_dict()
            # Populate cache
            try:
                await cache_client.set(cache_key, session_dict)
                logger.debug(f"Cache populated for session {session_id}")
            except Exception as e:
                logger.warning(f"Cache set failed for session {session_id}: {e}")
            return session_dict

        return None

    async def add_message(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        message: Dict[str, Any]
    ) -> Optional[ChatSession]:
        """
        Append a message to session's messages JSONB array.

        Args:
            db: Async database session
            session_id: Session primary key
            user_id: Owner user ID for scoping
            message: Message dict to append

        Returns:
            Updated ChatSession if found, None otherwise
        """
        # Return None for invalid UUIDs
        session_uuid = safe_uuid(session_id)
        user_uuid = safe_uuid(user_id)
        if session_uuid is None or user_uuid is None:
            return None

        # Get current session
        session = await super().get_by_id(db, session_id, user_id)
        if not session:
            return None

        # Append message to messages array
        messages = list(session.messages) if session.messages else []
        messages.append(message)

        # Update session
        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == session_uuid,
                ChatSession.user_id == user_uuid
            )
            .values(
                messages=messages,
                last_activity=datetime.now(timezone.utc)
            )
            .returning(ChatSession)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
            # Invalidate cache
            await self._invalidate_cache(session_id)

        return updated

    async def list_by_context(
        self,
        db: AsyncSession,
        user_id: str,
        context_type: str,
        context_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatSession]:
        """
        List sessions by context type and optional context ID.

        Args:
            db: Async database session
            user_id: Owner user ID
            context_type: Context type (general/workflow/agent/workflow_design)
            context_id: Optional context entity ID
            limit: Maximum results to return
            offset: Number of records to skip

        Returns:
            List of ChatSession instances matching the context
        """
        # Return empty list for invalid UUIDs (e.g., "anonymous" user)
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            return []

        stmt = (
            select(ChatSession)
            .where(
                ChatSession.user_id == user_uuid,
                ChatSession.context_type == context_type,
                ChatSession.is_deleted == False
            )
        )

        # Add context_id filter if provided
        if context_id:
            context_uuid = safe_uuid(context_id)
            if context_uuid is not None:
                stmt = stmt.where(ChatSession.context_id == context_uuid)

        stmt = (
            stmt
            .order_by(ChatSession.last_activity.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_tools(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        tool_ids: List[str]
    ) -> Optional[ChatSession]:
        """
        Update the selected tool IDs for a session.

        Args:
            db: Async database session
            session_id: Session primary key
            user_id: Owner user ID for scoping
            tool_ids: List of tool UUIDs to select

        Returns:
            Updated ChatSession if found, None otherwise
        """
        # Return None for invalid UUIDs
        session_uuid = safe_uuid(session_id)
        user_uuid = safe_uuid(user_id)
        if session_uuid is None or user_uuid is None:
            return None

        # Convert tool IDs to UUIDs, filtering out invalid ones
        tool_uuids = [safe_uuid(tid) for tid in tool_ids]
        tool_uuids = [u for u in tool_uuids if u is not None]

        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == session_uuid,
                ChatSession.user_id == user_uuid,
                ChatSession.is_deleted == False
            )
            .values(selected_tool_ids=tool_uuids)
            .returning(ChatSession)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
            # Invalidate cache
            await self._invalidate_cache(session_id)

        return updated

    async def update(
        self,
        db: AsyncSession,
        id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ChatSession]:
        """
        Update a session and invalidate cache.

        Override to add cache invalidation.
        """
        result = await super().update(db, id, user_id, updates)
        if result:
            await self._invalidate_cache(id)
        return result

    async def delete(
        self,
        db: AsyncSession,
        id: str,
        user_id: str
    ) -> bool:
        """
        Soft delete a session and invalidate cache.

        Override to add cache invalidation.
        """
        result = await super().delete(db, id, user_id)
        if result:
            await self._invalidate_cache(id)
        return result

    async def _invalidate_cache(self, session_id: str) -> None:
        """
        Remove a session from cache.

        Args:
            session_id: Session ID to invalidate
        """
        cache_key = self._cache_key(session_id)
        try:
            await cache_client.delete(cache_key)
            logger.debug(f"Cache invalidated for session {session_id}")
        except Exception as e:
            logger.warning(f"Cache delete failed for session {session_id}: {e}")

    async def touch_activity(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        Update last_activity timestamp for a session.

        Args:
            db: Async database session
            session_id: Session primary key
            user_id: Owner user ID for scoping

        Returns:
            True if session was updated, False if not found
        """
        # Return False for invalid UUIDs
        session_uuid = safe_uuid(session_id)
        user_uuid = safe_uuid(user_id)
        if session_uuid is None or user_uuid is None:
            return False

        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == session_uuid,
                ChatSession.user_id == user_uuid,
                ChatSession.is_deleted == False
            )
            .values(last_activity=datetime.now(timezone.utc))
        )

        result = await db.execute(stmt)
        await db.flush()

        if result.rowcount > 0:
            await self._invalidate_cache(session_id)
            return True
        return False

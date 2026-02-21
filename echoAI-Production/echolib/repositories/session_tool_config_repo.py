"""
EchoAI Session Tool Config Repository

Provides data access for session-specific tool configuration overrides.
These allow users to customize tool behavior per-session without modifying
the base tool definition.

Usage Flow:
    1. User selects tools for session (stored in chat_sessions.selected_tool_ids)
    2. User optionally customizes tool configs (stored here)
    3. At runtime, merge base tool config with session overrides
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import SessionToolConfig


class SessionToolConfigRepository:
    """
    Repository for SessionToolConfig entity operations.

    SessionToolConfig uses a composite primary key (session_id, tool_id),
    so it doesn't follow the standard BaseRepository pattern.

    Operations:
        - get_configs: Get all tool configs for a session
        - set_config: Create or update a tool config
        - delete_config: Remove a tool config
    """

    async def get_configs(
        self,
        db: AsyncSession,
        session_id: str
    ) -> List[SessionToolConfig]:
        """
        Get all tool configurations for a session.

        Args:
            db: Async database session
            session_id: Session to get configs for

        Returns:
            List of SessionToolConfig instances
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id

        stmt = (
            select(SessionToolConfig)
            .where(SessionToolConfig.session_id == session_uuid)
            .order_by(SessionToolConfig.created_at)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_config(
        self,
        db: AsyncSession,
        session_id: str,
        tool_id: str
    ) -> Optional[SessionToolConfig]:
        """
        Get a specific tool configuration for a session.

        Args:
            db: Async database session
            session_id: Session ID
            tool_id: Tool ID

        Returns:
            SessionToolConfig if found, None otherwise
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id
        tool_uuid = UUID(tool_id) if isinstance(tool_id, str) else tool_id

        stmt = select(SessionToolConfig).where(
            SessionToolConfig.session_id == session_uuid,
            SessionToolConfig.tool_id == tool_uuid
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def set_config(
        self,
        db: AsyncSession,
        session_id: str,
        tool_id: str,
        config_overrides: Dict[str, Any]
    ) -> SessionToolConfig:
        """
        Create or update a tool configuration for a session.

        Uses UPSERT to handle both create and update cases.

        Args:
            db: Async database session
            session_id: Session ID
            tool_id: Tool ID
            config_overrides: Configuration overrides dict

        Returns:
            The created or updated SessionToolConfig instance
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id
        tool_uuid = UUID(tool_id) if isinstance(tool_id, str) else tool_id

        # Use UPSERT pattern
        stmt = insert(SessionToolConfig).values(
            session_id=session_uuid,
            tool_id=tool_uuid,
            config_overrides=config_overrides
        )

        # On conflict, update config_overrides
        stmt = stmt.on_conflict_do_update(
            index_elements=["session_id", "tool_id"],
            set_={"config_overrides": config_overrides}
        ).returning(SessionToolConfig)

        result = await db.execute(stmt)
        await db.flush()

        config = result.scalar_one()
        await db.refresh(config)
        return config

    async def delete_config(
        self,
        db: AsyncSession,
        session_id: str,
        tool_id: str
    ) -> bool:
        """
        Delete a tool configuration for a session.

        Args:
            db: Async database session
            session_id: Session ID
            tool_id: Tool ID

        Returns:
            True if config was deleted, False if not found
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id
        tool_uuid = UUID(tool_id) if isinstance(tool_id, str) else tool_id

        stmt = delete(SessionToolConfig).where(
            SessionToolConfig.session_id == session_uuid,
            SessionToolConfig.tool_id == tool_uuid
        )

        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def delete_all_for_session(
        self,
        db: AsyncSession,
        session_id: str
    ) -> int:
        """
        Delete all tool configurations for a session.

        Useful when clearing session state or deleting a session.

        Args:
            db: Async database session
            session_id: Session ID

        Returns:
            Number of configs deleted
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id

        stmt = delete(SessionToolConfig).where(
            SessionToolConfig.session_id == session_uuid
        )

        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount

    async def bulk_set_configs(
        self,
        db: AsyncSession,
        session_id: str,
        configs: Dict[str, Dict[str, Any]]
    ) -> List[SessionToolConfig]:
        """
        Set multiple tool configurations for a session.

        Args:
            db: Async database session
            session_id: Session ID
            configs: Dict mapping tool_id -> config_overrides

        Returns:
            List of created/updated SessionToolConfig instances
        """
        results = []
        for tool_id, config_overrides in configs.items():
            config = await self.set_config(
                db, session_id, tool_id, config_overrides
            )
            results.append(config)
        return results

    async def get_configs_by_tool_ids(
        self,
        db: AsyncSession,
        session_id: str,
        tool_ids: List[str]
    ) -> List[SessionToolConfig]:
        """
        Get tool configurations for specific tools in a session.

        Args:
            db: Async database session
            session_id: Session ID
            tool_ids: List of tool IDs to get configs for

        Returns:
            List of SessionToolConfig instances for the specified tools
        """
        session_uuid = UUID(session_id) if isinstance(session_id, str) else session_id
        tool_uuids = [UUID(tid) if isinstance(tid, str) else tid for tid in tool_ids]

        stmt = select(SessionToolConfig).where(
            SessionToolConfig.session_id == session_uuid,
            SessionToolConfig.tool_id.in_(tool_uuids)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    # Alias methods for route compatibility
    async def upsert_config(
        self,
        db: AsyncSession,
        session_id: str,
        tool_id: str,
        user_id: str,
        config: Dict[str, Any]
    ) -> SessionToolConfig:
        """
        Create or update a tool configuration (alias for set_config).

        Includes user_id for consistency with route parameters (not used in query
        since session ownership is verified separately).

        Args:
            db: Async database session
            session_id: Session ID
            tool_id: Tool ID
            user_id: User ID (for API consistency, not used in query)
            config: Configuration overrides dict

        Returns:
            The created or updated SessionToolConfig instance
        """
        return await self.set_config(db, session_id, tool_id, config)

    async def get_configs_for_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str
    ) -> List[SessionToolConfig]:
        """
        Get all tool configurations for a session (alias for get_configs).

        Includes user_id for consistency with route parameters.

        Args:
            db: Async database session
            session_id: Session ID
            user_id: User ID (for API consistency)

        Returns:
            List of SessionToolConfig instances
        """
        return await self.get_configs(db, session_id)

"""
EchoAI Tool Repository

Provides data access for user-owned tool definitions.
Tools can be local functions, MCP endpoints, API calls, or CrewAI tools.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update, any_
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import Tool
from .base import BaseRepository


class ToolRepository(BaseRepository[Tool]):
    """
    Repository for Tool entity operations.

    Provides standard CRUD plus:
        - Type-based filtering (local/mcp/api/crewai/custom)
        - Tag-based search using PostgreSQL array operators
        - Status filtering

    Tool types:
        - local: Python function executed locally
        - mcp: MCP connector endpoint
        - api: Direct HTTP API call
        - crewai: CrewAI-native tool
        - custom: User-defined execution type
    """

    model_class = Tool
    id_field = "tool_id"
    user_id_field = "user_id"
    soft_delete_field = "is_deleted"

    async def get_by_type(
        self,
        db: AsyncSession,
        user_id: str,
        tool_type: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tool]:
        """
        Get tools by type for a user.

        Args:
            db: Async database session
            user_id: Owner user ID
            tool_type: Tool type (local/mcp/api/crewai/custom)
            limit: Maximum results to return
            offset: Number of records to skip

        Returns:
            List of Tool instances of the specified type
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            select(Tool)
            .where(
                Tool.user_id == user_uuid,
                Tool.tool_type == tool_type,
                Tool.is_deleted == False
            )
            .order_by(Tool.name)
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def search_by_tags(
        self,
        db: AsyncSession,
        user_id: str,
        tags: List[str],
        match_all: bool = False,
        limit: int = 50
    ) -> List[Tool]:
        """
        Search tools by tags.

        Args:
            db: Async database session
            user_id: Owner user ID
            tags: List of tags to search for
            match_all: If True, tool must have ALL tags. If False, any tag matches.
            limit: Maximum results to return

        Returns:
            List of Tool instances matching the tag criteria
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = select(Tool).where(
            Tool.user_id == user_uuid,
            Tool.is_deleted == False
        )

        if match_all:
            # Tool must contain ALL specified tags
            # Use contains operator (@>) for array containment
            stmt = stmt.where(Tool.tags.contains(tags))
        else:
            # Tool must contain ANY of the specified tags
            # Use overlap operator (&&) for array overlap
            stmt = stmt.where(Tool.tags.overlap(tags))

        stmt = stmt.order_by(Tool.name).limit(limit)

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_name(
        self,
        db: AsyncSession,
        user_id: str,
        name: str
    ) -> Optional[Tool]:
        """
        Get a tool by name for a user.

        Args:
            db: Async database session
            user_id: Owner user ID
            name: Tool name to look up

        Returns:
            Tool instance if found, None otherwise
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = select(Tool).where(
            Tool.user_id == user_uuid,
            Tool.name == name,
            Tool.is_deleted == False
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def search_by_name(
        self,
        db: AsyncSession,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[Tool]:
        """
        Search tools by name using case-insensitive LIKE.

        Args:
            db: Async database session
            user_id: Owner user ID
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching Tool instances
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        search_pattern = f"%{query}%"

        stmt = (
            select(Tool)
            .where(
                Tool.user_id == user_uuid,
                Tool.name.ilike(search_pattern),
                Tool.is_deleted == False
            )
            .order_by(Tool.name)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_active(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tool]:
        """
        Get only active tools for a user.

        Filters by status='active' in addition to not deleted.

        Args:
            db: Async database session
            user_id: Owner user ID
            limit: Maximum results to return
            offset: Number of records to skip

        Returns:
            List of active Tool instances
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            select(Tool)
            .where(
                Tool.user_id == user_uuid,
                Tool.status == "active",
                Tool.is_deleted == False
            )
            .order_by(Tool.name)
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_ids(
        self,
        db: AsyncSession,
        user_id: str,
        tool_ids: List[str]
    ) -> List[Tool]:
        """
        Get multiple tools by IDs.

        Args:
            db: Async database session
            user_id: Owner user ID
            tool_ids: List of tool IDs to retrieve

        Returns:
            List of Tool instances (may be less than requested if some not found)
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        tool_uuids = [UUID(tid) if isinstance(tid, str) else tid for tid in tool_ids]

        stmt = select(Tool).where(
            Tool.user_id == user_uuid,
            Tool.tool_id.in_(tool_uuids),
            Tool.is_deleted == False
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        db: AsyncSession,
        tool_id: str,
        user_id: str,
        new_status: str
    ) -> Optional[Tool]:
        """
        Update tool status.

        Args:
            db: Async database session
            tool_id: Tool ID
            user_id: Owner user ID for scoping
            new_status: New status value (active/deprecated/disabled)

        Returns:
            Updated Tool if found, None otherwise
        """
        return await self.update(
            db,
            tool_id,
            user_id,
            {"status": new_status}
        )

    async def add_tags(
        self,
        db: AsyncSession,
        tool_id: str,
        user_id: str,
        new_tags: List[str]
    ) -> Optional[Tool]:
        """
        Add tags to a tool (preserving existing tags).

        Args:
            db: Async database session
            tool_id: Tool ID
            user_id: Owner user ID for scoping
            new_tags: Tags to add

        Returns:
            Updated Tool if found, None otherwise
        """
        tool = await self.get_by_id(db, tool_id, user_id)
        if not tool:
            return None

        # Merge existing and new tags (dedup)
        existing_tags = set(tool.tags or [])
        all_tags = list(existing_tags.union(set(new_tags)))

        return await self.update(db, tool_id, user_id, {"tags": all_tags})

    async def remove_tags(
        self,
        db: AsyncSession,
        tool_id: str,
        user_id: str,
        tags_to_remove: List[str]
    ) -> Optional[Tool]:
        """
        Remove tags from a tool.

        Args:
            db: Async database session
            tool_id: Tool ID
            user_id: Owner user ID for scoping
            tags_to_remove: Tags to remove

        Returns:
            Updated Tool if found, None otherwise
        """
        tool = await self.get_by_id(db, tool_id, user_id)
        if not tool:
            return None

        # Remove specified tags
        existing_tags = set(tool.tags or [])
        remaining_tags = list(existing_tags - set(tags_to_remove))

        return await self.update(db, tool_id, user_id, {"tags": remaining_tags})

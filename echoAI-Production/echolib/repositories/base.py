"""
EchoAI Base Repository

Provides common CRUD operations with user scoping for multi-tenancy.
All derived repositories inherit these operations and can override as needed.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.database import Base


logger = logging.getLogger(__name__)


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        UUID(value)
        return True
    except (ValueError, TypeError):
        return False


def safe_uuid(value: Union[str, UUID]) -> Optional[UUID]:
    """Convert string to UUID, returning None if invalid."""
    if isinstance(value, UUID):
        return value
    try:
        return UUID(value)
    except (ValueError, TypeError):
        return None


# Default admin user ID for database operations (until proper auth is implemented)
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_EMAIL = "admin@echoai.local"
DEFAULT_USER_DISPLAY_NAME = "System Admin"


async def ensure_default_user_exists(db: AsyncSession) -> bool:
    """
    Ensure the default system user exists in the database.

    This function is idempotent - safe to call multiple times.
    It checks if the default user exists and creates it if not.

    This is necessary because sessions and other entities reference
    user_id via FK constraint. Without a valid user, these operations
    will fail silently due to constraint violations.

    Args:
        db: Async database session (must be within a transaction context)

    Returns:
        True if user exists or was created successfully, False on error

    Note:
        This function does NOT commit the transaction. The caller's
        context manager handles commit/rollback.
    """
    from echolib.models.user import User

    try:
        default_uuid = UUID(DEFAULT_USER_ID)

        # Check if user already exists
        result = await db.execute(
            select(User).where(User.user_id == default_uuid)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            logger.debug(f"Default user already exists: {DEFAULT_USER_ID}")
            return True

        # Create the default user
        default_user = User(
            user_id=default_uuid,
            email=DEFAULT_USER_EMAIL,
            display_name=DEFAULT_USER_DISPLAY_NAME,
            user_metadata={"role": "admin", "system_user": True},
            is_active=True
        )
        db.add(default_user)
        await db.flush()  # Flush to ensure user is created before dependent operations

        logger.info(f"Created default system user: {DEFAULT_USER_ID}")
        return True

    except Exception as e:
        logger.error(f"Failed to ensure default user exists: {e}", exc_info=True)
        return False


# Type variable for generic repository pattern
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with common CRUD operations.

    All queries include user_id filter for multi-tenancy data isolation.
    Soft delete is used instead of hard delete (is_deleted=True).

    Subclasses must set:
        - model_class: The SQLAlchemy model class
        - id_field: Name of the primary key field (e.g., 'agent_id')
        - user_id_field: Name of the user foreign key field (default: 'user_id')
        - soft_delete_field: Name of the soft delete field (default: 'is_deleted')

    Example:
        class AgentRepository(BaseRepository[Agent]):
            model_class = Agent
            id_field = "agent_id"
    """

    model_class: Type[ModelType]
    id_field: str = "id"
    user_id_field: str = "user_id"
    soft_delete_field: str = "is_deleted"

    async def create(
        self,
        db: AsyncSession,
        user_id: str,
        data: Dict[str, Any]
    ) -> ModelType:
        """
        Create a new record owned by the user.

        Args:
            db: Async database session
            user_id: Owner user ID
            data: Dictionary of field values

        Returns:
            The created model instance
        """
        # Ensure user_id is a valid UUID
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            raise ValueError(f"Cannot create record: invalid user_id '{user_id}'")
        data[self.user_id_field] = user_uuid

        instance = self.model_class(**data)
        db.add(instance)
        await db.flush()
        await db.refresh(instance)
        return instance

    async def get_by_id(
        self,
        db: AsyncSession,
        id: str,
        user_id: str
    ) -> Optional[ModelType]:
        """
        Get a record by ID, scoped to user.

        Args:
            db: Async database session
            id: Record primary key
            user_id: Owner user ID for scoping

        Returns:
            Model instance if found and owned by user, None otherwise
        """
        id_column = getattr(self.model_class, self.id_field)
        user_column = getattr(self.model_class, self.user_id_field)
        soft_delete_column = getattr(self.model_class, self.soft_delete_field, None)

        # Convert string IDs to UUID - return None for invalid UUIDs (e.g., "anonymous")
        id_uuid = safe_uuid(id)
        user_uuid = safe_uuid(user_id)

        if id_uuid is None or user_uuid is None:
            return None

        stmt = select(self.model_class).where(
            id_column == id_uuid,
            user_column == user_uuid
        )

        # Add soft delete filter if field exists
        if soft_delete_column is not None:
            stmt = stmt.where(soft_delete_column == False)

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelType]:
        """
        List all records for a user with pagination.

        Args:
            db: Async database session
            user_id: Owner user ID
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of model instances owned by user
        """
        user_column = getattr(self.model_class, self.user_id_field)
        soft_delete_column = getattr(self.model_class, self.soft_delete_field, None)

        # Return empty list for invalid UUIDs (e.g., "anonymous" user)
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            return []

        stmt = select(self.model_class).where(user_column == user_uuid)

        # Add soft delete filter if field exists
        if soft_delete_column is not None:
            stmt = stmt.where(soft_delete_column == False)

        # Apply pagination
        stmt = stmt.limit(limit).offset(offset)

        # Order by created_at descending for most recent first
        if hasattr(self.model_class, "created_at"):
            stmt = stmt.order_by(self.model_class.created_at.desc())

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        db: AsyncSession,
        id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ModelType]:
        """
        Update a record by ID, scoped to user.

        Args:
            db: Async database session
            id: Record primary key
            user_id: Owner user ID for scoping
            updates: Dictionary of field updates

        Returns:
            Updated model instance if found, None otherwise
        """
        id_column = getattr(self.model_class, self.id_field)
        user_column = getattr(self.model_class, self.user_id_field)
        soft_delete_column = getattr(self.model_class, self.soft_delete_field, None)

        # Return None for invalid UUIDs (e.g., "anonymous" user)
        id_uuid = safe_uuid(id)
        user_uuid = safe_uuid(user_id)
        if id_uuid is None or user_uuid is None:
            return None

        # Build where clause
        where_clause = [
            id_column == id_uuid,
            user_column == user_uuid
        ]
        if soft_delete_column is not None:
            where_clause.append(soft_delete_column == False)

        stmt = (
            update(self.model_class)
            .where(*where_clause)
            .values(**updates)
            .returning(self.model_class)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
        return updated

    async def delete(
        self,
        db: AsyncSession,
        id: str,
        user_id: str
    ) -> bool:
        """
        Soft delete a record by ID, scoped to user.

        Sets is_deleted=True instead of removing the record.

        Args:
            db: Async database session
            id: Record primary key
            user_id: Owner user ID for scoping

        Returns:
            True if record was marked deleted, False if not found
        """
        id_column = getattr(self.model_class, self.id_field)
        user_column = getattr(self.model_class, self.user_id_field)
        soft_delete_column = getattr(self.model_class, self.soft_delete_field, None)

        if soft_delete_column is None:
            raise ValueError(
                f"Model {self.model_class.__name__} does not have soft delete field"
            )

        # Return False for invalid UUIDs (e.g., "anonymous" user)
        id_uuid = safe_uuid(id)
        user_uuid = safe_uuid(user_id)
        if id_uuid is None or user_uuid is None:
            return False

        stmt = (
            update(self.model_class)
            .where(
                id_column == id_uuid,
                user_column == user_uuid,
                soft_delete_column == False
            )
            .values(is_deleted=True)
        )

        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def exists(
        self,
        db: AsyncSession,
        id: str,
        user_id: str
    ) -> bool:
        """
        Check if a record exists for the user.

        Args:
            db: Async database session
            id: Record primary key
            user_id: Owner user ID for scoping

        Returns:
            True if record exists and is not soft-deleted
        """
        record = await self.get_by_id(db, id, user_id)
        return record is not None

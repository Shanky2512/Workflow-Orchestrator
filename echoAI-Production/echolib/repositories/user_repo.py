"""
EchoAI User Repository

Provides data access for user accounts.
Users are the root entity for multi-tenancy - all other entities reference users.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import User


class UserRepository:
    """
    Repository for User entity operations.

    Users are special: they don't have a user_id foreign key,
    they ARE the user. Methods here don't follow the BaseRepository
    pattern since users are the root of the ownership hierarchy.
    """

    async def get_by_id(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Optional[User]:
        """
        Get a user by their ID.

        Args:
            db: Async database session
            user_id: User's primary key

        Returns:
            User instance if found and active, None otherwise
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = select(User).where(
            User.user_id == user_uuid,
            User.is_active == True
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        db: AsyncSession,
        email: str
    ) -> Optional[User]:
        """
        Get a user by their email address.

        Args:
            db: Async database session
            email: User's email address (case-insensitive search)

        Returns:
            User instance if found and active, None otherwise
        """
        # Normalize email to lowercase for comparison
        normalized_email = email.lower().strip()

        stmt = select(User).where(
            User.email == normalized_email,
            User.is_active == True
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        db: AsyncSession,
        email: str,
        display_name: Optional[str] = None
    ) -> User:
        """
        Get an existing user by email or create a new one.

        This is the primary method for user authentication flows.
        If user exists, returns existing user. If not, creates new user.

        Args:
            db: Async database session
            email: User's email address
            display_name: Optional display name for new users

        Returns:
            User instance (existing or newly created)
        """
        # Normalize email
        normalized_email = email.lower().strip()

        # Try to find existing user (including inactive for reactivation)
        stmt = select(User).where(User.email == normalized_email)
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            # Reactivate if inactive
            if not existing_user.is_active:
                existing_user.is_active = True
                await db.flush()
                await db.refresh(existing_user)
            return existing_user

        # Create new user
        new_user = User(
            email=normalized_email,
            display_name=display_name or email.split("@")[0],
            is_active=True,
            metadata={}
        )
        db.add(new_user)
        await db.flush()
        await db.refresh(new_user)

        return new_user

    async def update_last_login(
        self,
        db: AsyncSession,
        user_id: str
    ) -> None:
        """
        Update user's last login timestamp.

        Called after successful authentication to track activity.

        Args:
            db: Async database session
            user_id: User's primary key
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            update(User)
            .where(User.user_id == user_uuid)
            .values(last_login_at=datetime.now(timezone.utc))
        )

        await db.execute(stmt)
        await db.flush()

    async def update(
        self,
        db: AsyncSession,
        user_id: str,
        updates: dict
    ) -> Optional[User]:
        """
        Update user profile fields.

        Args:
            db: Async database session
            user_id: User's primary key
            updates: Dictionary of field updates

        Returns:
            Updated user instance if found, None otherwise
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            update(User)
            .where(User.user_id == user_uuid, User.is_active == True)
            .values(**updates)
            .returning(User)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
        return updated

    async def deactivate(
        self,
        db: AsyncSession,
        user_id: str
    ) -> bool:
        """
        Soft delete a user by setting is_active=False.

        This does not delete the user's data - it just marks them as inactive.
        Related entities remain in the database.

        Args:
            db: Async database session
            user_id: User's primary key

        Returns:
            True if user was deactivated, False if not found
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            update(User)
            .where(User.user_id == user_uuid, User.is_active == True)
            .values(is_active=False)
        )

        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

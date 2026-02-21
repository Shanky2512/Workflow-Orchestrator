"""
EchoAI User Model

Stores authenticated user accounts for multi-tenancy support.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class User(BaseModel):
    """
    User account model.

    Stores authenticated users with their profile information.
    All other entities (agents, workflows, sessions) reference users
    for multi-tenancy data isolation.

    Attributes:
        user_id: Primary key UUID
        email: Unique email address
        display_name: User's display name
        avatar_url: Optional profile picture URL
        last_login_at: Last successful login timestamp
        user_metadata: Extensible JSONB for user preferences
        is_active: Soft delete flag (False = deactivated)

    Indexes:
        - idx_users_email: Fast email lookup
        - idx_users_active: Active users only
    """
    __tablename__ = "users"

    # Primary key
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    # Required fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    # Optional profile fields
    display_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )

    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )

    # Activity tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # Extensible metadata (named user_metadata to avoid SQLAlchemy reserved name)
    user_metadata: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",  # Actual column name in DB
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    # Soft delete
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False
    )

    # Table indexes
    __table_args__ = (
        Index("idx_users_active", "is_active", postgresql_where=(is_active == True)),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user to dictionary representation.

        Returns:
            Dict containing user data for API responses.
        """
        return {
            "user_id": str(self.user_id),
            "email": self.email,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.user_metadata or {},
            "is_active": self.is_active
        }

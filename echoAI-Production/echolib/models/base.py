"""
EchoAI Base Model

Provides common functionality for all SQLAlchemy models.
"""

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class BaseModel(Base):
    """
    Abstract base class for all EchoAI models.

    Provides:
    - Common timestamp fields (created_at, updated_at)
    - Default to_dict() method for serialization
    - Consistent datetime handling with timezone awareness

    All models should inherit from this class.
    """
    __abstract__ = True

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary representation.

        Override in subclasses to customize serialization.
        By default, returns all column values with UUID/datetime conversion.

        Returns:
            Dict containing model data suitable for JSON serialization.
        """
        result: Dict[str, Any] = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # Convert UUIDs to strings
            if isinstance(value, UUID):
                value = str(value)
            # Convert datetimes to ISO format
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        class_name = self.__class__.__name__
        # Get primary key column(s)
        pk_cols = [col.name for col in self.__table__.primary_key.columns]
        pk_values = [f"{col}={getattr(self, col, None)}" for col in pk_cols]
        return f"<{class_name}({', '.join(pk_values)})>"

"""
EchoAI Session Tool Configuration Model

Stores tool configuration overrides per session.
Enables session-specific tool customization beyond default tool settings.

Design Principle:
- chat_sessions.selected_tool_ids stores which tools are selected
- session_tool_configs stores configuration overrides for those tools
- This hybrid approach allows flexibility for custom tool configurations
"""

from typing import Any, Dict
from uuid import UUID

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class SessionToolConfig(BaseModel):
    """
    Session-specific tool configuration overrides.

    Stores custom configuration for tools used within a specific session.
    This allows users to customize tool behavior per-session without
    modifying the base tool definition.

    Attributes:
        session_id: Reference to chat session (composite PK)
        tool_id: Reference to tool (composite PK)
        config_overrides: JSONB with configuration overrides

    Primary Key: (session_id, tool_id) composite

    Usage Flow:
        1. User selects tools for session:
           chat_sessions.selected_tool_ids = [tool_1, tool_2]

        2. User customizes tool config:
           INSERT INTO session_tool_configs (session_id, tool_id, config_overrides)
           VALUES (sess_xxx, tool_1, '{"max_results": 10}')

        3. At runtime, merge base config with overrides:
           SELECT t.definition, stc.config_overrides
           FROM tools t
           LEFT JOIN session_tool_configs stc
             ON t.tool_id = stc.tool_id AND stc.session_id = ?
           WHERE t.tool_id = ANY(session.selected_tool_ids)

    config_overrides Example:
        {
            "max_results": 10,
            "timeout_seconds": 30,
            "api_key": "session-specific-key",
            "custom_headers": {"X-Custom": "value"}
        }

    Indexes:
        - idx_session_tool_configs_session: Fast session lookup
        - idx_session_tool_configs_tool: Fast tool lookup
    """
    __tablename__ = "session_tool_configs"

    # Composite primary key
    session_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        primary_key=True
    )

    tool_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tools.tool_id", ondelete="CASCADE"),
        primary_key=True
    )

    # Configuration overrides
    config_overrides: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False
    )

    # Table indexes
    __table_args__ = (
        Index("idx_session_tool_configs_session", "session_id"),
        Index("idx_session_tool_configs_tool", "tool_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dict containing session tool config data.
        """
        return {
            "session_id": str(self.session_id),
            "tool_id": str(self.tool_id),
            "config_overrides": self.config_overrides or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

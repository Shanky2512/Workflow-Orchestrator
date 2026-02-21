"""
EchoAI Tool Model

Stores user-owned tool definitions.
Tools can be local functions, MCP endpoints, API calls, or CrewAI tools.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, ForeignKey, Index, CheckConstraint, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class Tool(BaseModel):
    """
    Tool definition model.

    Stores complete tool definitions including execution configuration.
    Tools are owned by users and can be attached to agents or sessions.

    Attributes:
        tool_id: Primary key UUID
        user_id: Owner user (FK to users)
        name: Tool name for display
        description: Tool functionality description
        tool_type: Execution type (local/mcp/api/crewai/custom)
        definition: Complete tool definition JSONB
        status: Tool status (active/deprecated/disabled)
        version: Tool version string
        tags: Categorization tags for discovery

    Tool Types:
        - local: Python function in tools folder (executed locally)
        - mcp: MCP connector endpoint (executed via MCP protocol)
        - api: Direct HTTP API call (executed via HTTP request)
        - crewai: CrewAI-native tool (executed within CrewAI context)
        - custom: User-defined execution type

    Definition JSONB Structure:
        {
            "tool_id": "tool_xxx",
            "name": "Calculator",
            "description": "...",
            "tool_type": "local",
            "input_schema": {...},  # JSON Schema
            "output_schema": {...},  # JSON Schema
            "execution_config": {
                "module": "tools.calculator",
                "function": "calculate"
            },
            "version": "1.0",
            "tags": ["math", "utility"],
            "metadata": {...}
        }

    Indexes:
        - idx_tools_user: Fast user lookup
        - idx_tools_user_active: Active tools for user
        - idx_tools_type: Filter by tool type
        - idx_tools_tags: GIN index for tag search
    """
    __tablename__ = "tools"

    # Primary key
    tool_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )

    # Owner (FK to users)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Key fields for display/search
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    tool_type: Mapped[str] = mapped_column(
        String(20),
        default="local",
        server_default="local",
        nullable=False
    )

    # Complete tool definition
    definition: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False
    )

    # Status and versioning
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        server_default="active",
        nullable=False
    )

    version: Mapped[str] = mapped_column(
        String(20),
        default="1.0",
        server_default="1.0",
        nullable=False
    )

    # Categorization
    tags: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        default=list,
        server_default="{}",
        nullable=False
    )

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False
    )

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "tool_type IN ('local', 'mcp', 'api', 'crewai', 'custom')",
            name="valid_tool_type"
        ),
        CheckConstraint(
            "status IN ('active', 'deprecated', 'disabled')",
            name="valid_tool_status"
        ),
        Index("idx_tools_user_active", "user_id", postgresql_where=(is_deleted == False)),
        Index("idx_tools_type", "tool_type"),
        Index("idx_tools_tags", "tags", postgresql_using="gin"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the tool definition directly.

        The definition JSONB contains the complete tool configuration.

        Returns:
            The complete tool definition dict.
        """
        if self.definition:
            return self.definition.copy()
        return {
            "tool_id": str(self.tool_id),
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Return tool metadata without full definition.

        Useful for listing tools without loading full definitions.

        Returns:
            Dict with tool metadata fields only.
        """
        return {
            "tool_id": str(self.tool_id),
            "user_id": str(self.user_id),
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type,
            "status": self.status,
            "version": self.version,
            "tags": self.tags or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_deleted": self.is_deleted
        }

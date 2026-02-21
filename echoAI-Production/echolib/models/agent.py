"""
EchoAI Agent Model

Stores AI agent definitions with simplified schema.
Complete agent JSON is stored in the definition JSONB column,
with only key fields extracted for indexing/querying.

Design Principle:
- Store complete agent JSON as-is in `definition`
- When retrieved, return `definition` directly (no reconstruction)
- Agents are synced via UPSERT when workflows are saved
"""

from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class Agent(BaseModel):
    """
    AI Agent model with simplified schema.

    The complete agent definition is stored as-is in the `definition` JSONB column.
    This JSON is directly usable by CrewAI/Agent Factory without transformation.

    Key Fields (for indexing only):
        agent_id: Primary key UUID
        user_id: Owner user (FK to users)
        name: Agent name (for search/display)
        source_workflow_id: Which workflow last updated this agent

    Definition JSONB contains:
        agent_id, name, role, description, prompt, icon, model,
        tools, variables, settings, permissions, metadata,
        input_schema, output_schema

    Sync Behavior:
        When a workflow is saved, each embedded agent is UPSERTed:
        - If agent_id exists -> UPDATE
        - If agent_id is new -> INSERT

    Indexes:
        - idx_agents_user: Fast user lookup
        - idx_agents_user_active: Active agents for user
        - idx_agents_source_workflow: Find agents from workflow
        - idx_agents_name_search: Full-text search on name
    """
    __tablename__ = "agents"

    # Primary key
    agent_id: Mapped[UUID] = mapped_column(
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

    # Key field for display/search
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    # Complete agent JSON (stored as-is, directly usable by CrewAI)
    definition: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False
    )

    # Source tracking (which workflow last updated this agent)
    source_workflow_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True
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
        UniqueConstraint("user_id", "name", name="unique_agent_name_per_user"),
        Index("idx_agents_user_active", "user_id", postgresql_where=(is_deleted == False)),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the agent definition directly.

        The definition JSONB contains the complete agent JSON
        that is directly usable by CrewAI/Agent Factory.
        No reconstruction needed.

        Returns:
            The complete agent definition dict.
        """
        # Return definition directly - it contains the complete agent JSON
        if self.definition:
            return self.definition.copy()
        return {
            "agent_id": str(self.agent_id),
            "name": self.name
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Return agent metadata without full definition.

        Useful for listing agents without loading full definitions.

        Returns:
            Dict with agent metadata fields only.
        """
        return {
            "agent_id": str(self.agent_id),
            "user_id": str(self.user_id),
            "name": self.name,
            "source_workflow_id": str(self.source_workflow_id) if self.source_workflow_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_deleted": self.is_deleted
        }

"""
EchoAI Workflow Model

Stores workflow definitions with simplified schema.
Complete workflow JSON is stored in the definition JSONB column,
with only key fields extracted for indexing/querying.

Design Principle:
- Store complete workflow JSON as-is in `definition`
- When retrieved, return `definition` directly (no reconstruction)
- Workflow JSON includes embedded agents (full copies, not references)
"""

from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, ForeignKey, Index, CheckConstraint, Integer, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class Workflow(BaseModel):
    """
    Workflow model with simplified schema.

    The complete workflow definition is stored as-is in the `definition` JSONB column.
    This JSON is directly usable by the executor/CrewAI without transformation.

    Key Fields (for indexing only):
        workflow_id: Primary key UUID
        user_id: Owner user (FK to users)
        name: Workflow name (for search/display)
        status: Workflow status (draft/validated/final/archived)

    Definition JSONB contains:
        workflow_id, name, description, status, version,
        execution_model, agents (embedded), connections,
        state_schema, human_in_loop, metadata

    Selective Dual-Write Strategy:
        - DRAFT save: Filesystem + Database + Agent sync
        - TEMP save: Filesystem ONLY (no DB write)
        - FINAL save: Filesystem + Database + Versions table + Agent sync

    Valid statuses:
        - draft: Initial editable state
        - validated: Passed validation, ready for testing
        - final: Production-ready, immutable
        - archived: Previous version, kept for history

    Note: 'temp' status is NOT stored in database (filesystem only)

    Indexes:
        - idx_workflows_user: Fast user lookup
        - idx_workflows_user_status: Filter by user and status
        - idx_workflows_user_active: Active workflows for user
        - idx_workflows_name_search: Full-text search on name
    """
    __tablename__ = "workflows"

    # Primary key
    workflow_id: Mapped[UUID] = mapped_column(
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

    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        server_default="draft",
        nullable=False
    )

    # Complete workflow JSON (stored as-is, directly usable by executor)
    definition: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
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
            "status IN ('draft', 'validated', 'final', 'archived')",
            name="valid_status"
        ),
        Index("idx_workflows_user_status", "user_id", "status"),
        Index("idx_workflows_user_active", "user_id", postgresql_where=(is_deleted == False)),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the workflow definition directly.

        The definition JSONB contains the complete workflow JSON
        (including embedded agents) that is directly usable by the executor.
        No reconstruction needed.

        Returns:
            The complete workflow definition dict.
        """
        # Return definition directly - it contains the complete workflow JSON
        if self.definition:
            return self.definition.copy()
        return {
            "workflow_id": str(self.workflow_id),
            "name": self.name,
            "status": self.status
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Return workflow metadata without full definition.

        Useful for listing workflows without loading full definitions.

        Returns:
            Dict with workflow metadata fields only.
        """
        return {
            "workflow_id": str(self.workflow_id),
            "user_id": str(self.user_id),
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_deleted": self.is_deleted
        }


class WorkflowVersion(BaseModel):
    """
    Immutable workflow version snapshot.

    Stores finalized workflow versions for history and rollback.
    Created when a workflow is saved to 'final' status.

    Attributes:
        id: Auto-increment primary key
        workflow_id: Reference to main workflow (FK)
        version: Version string (e.g., "1.0", "2.1")
        definition: Complete workflow JSON at this version
        status: Version status (final/archived)
        created_by: User who created this version
        notes: Optional release notes

    Constraints:
        - Unique (workflow_id, version) combination
        - Status must be 'final' or 'archived'
    """
    __tablename__ = "workflow_versions"

    # Primary key (auto-increment)
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Reference to main workflow
    workflow_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Version identifier
    version: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )

    # Complete workflow JSON at this version
    definition: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False
    )

    # Version status
    status: Mapped[str] = mapped_column(
        String(20),
        default="final",
        server_default="final",
        nullable=False
    )

    # Who created this version
    created_by: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id"),
        nullable=True
    )

    # Optional release notes
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('final', 'archived')",
            name="valid_version_status"
        ),
        Index(
            "unique_workflow_version",
            "workflow_id",
            "version",
            unique=True
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the versioned workflow definition.

        Returns:
            The complete workflow definition dict for this version.
        """
        if self.definition:
            return self.definition.copy()
        return {
            "workflow_id": str(self.workflow_id),
            "version": self.version
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Return version metadata without full definition.

        Returns:
            Dict with version metadata fields only.
        """
        return {
            "id": self.id,
            "workflow_id": str(self.workflow_id),
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": str(self.created_by) if self.created_by else None,
            "notes": self.notes
        }

"""
EchoAI Application Model

Core application model for the AI Application Orchestrator module.
Stores application configuration along with association (link) tables
that bind an application to LLMs, skills, data sources, designations,
business units, tags, and guardrail categories.

Design Principles:
- UUID primary keys consistent with existing models
- Soft delete via is_deleted flag
- user_id FK to users table for multi-tenancy isolation
- Partial indexes filtered on is_deleted = FALSE for performance
- JSONB for flexible starter_questions storage
- Composite primary keys on all association tables (no surrogate keys)

Association / Link Tables:
- ApplicationLlmLink         (application_id, llm_id) + role
- ApplicationSkillLink       (application_id, skill_type, skill_id)
- ApplicationDataSourceLink  (application_id, data_source_id)
- ApplicationDesignationLink (application_id, designation_id)
- ApplicationBusinessUnitLink(application_id, business_unit_id)
- ApplicationTagLink         (application_id, tag_id)
- ApplicationGuardrailLink   (application_id, guardrail_category_id)
"""

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    String,
    Boolean,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel
from ..database import Base


# ---------------------------------------------------------------------------
# Association / Link Tables
# ---------------------------------------------------------------------------
# These are lightweight join tables with composite primary keys.
# They inherit from Base directly (no created_at/updated_at needed).
# ---------------------------------------------------------------------------

class ApplicationLlmLink(Base):
    """
    Links an application to an LLM.

    The llm_id is a string that may be a UUID (from app_llms catalog)
    or a slug-style identifier from the JSON provider config files
    (e.g. "ollama-qwen3-vl-8b", "gpt-4o").

    Each LLM binding has a role indicating how the application uses it:
        - orchestrator: main LLM for skill selection / planning
        - enhancer: dedicated prompt-enhancement LLM
        - general: any other purpose

    Composite PK: (application_id, llm_id)
    """
    __tablename__ = "application_llm_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    llm_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        default="general",
        server_default="general",
        nullable=False,
    )
    name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )


class ApplicationSkillLink(Base):
    """
    Links an application to a skill (workflow or agent).

    skill_type is 'workflow' or 'agent'.
    skill_id references the workflow_id or agent_id from the main system tables.
    skill_name and skill_description are denormalized for quick access
    in orchestrator prompts without extra joins.

    Composite PK: (application_id, skill_type, skill_id)
    """
    __tablename__ = "application_skill_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    skill_type: Mapped[str] = mapped_column(
        String(10),
        primary_key=True,
    )
    skill_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    skill_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    skill_description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )


class ApplicationDataSourceLink(Base):
    """
    Links an application to a data source from the app_data_sources catalog.

    Composite PK: (application_id, data_source_id)
    """
    __tablename__ = "application_data_source_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    data_source_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )


class ApplicationDesignationLink(Base):
    """
    Links an application to a designation for access control.

    Composite PK: (application_id, designation_id)
    """
    __tablename__ = "application_designation_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    designation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("app_designations.designation_id"),
        primary_key=True,
    )


class ApplicationBusinessUnitLink(Base):
    """
    Links an application to a business unit for access control.

    Composite PK: (application_id, business_unit_id)
    """
    __tablename__ = "application_business_unit_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    business_unit_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("app_business_units.business_unit_id"),
        primary_key=True,
    )


class ApplicationTagLink(Base):
    """
    Links an application to a tag.

    Composite PK: (application_id, tag_id)
    """
    __tablename__ = "application_tag_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("app_tags.tag_id"),
        primary_key=True,
    )


class ApplicationGuardrailLink(Base):
    """
    Links an application to a guardrail category.

    Composite PK: (application_id, guardrail_category_id)
    """
    __tablename__ = "application_guardrail_links"

    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("applications.application_id", ondelete="CASCADE"),
        primary_key=True,
    )
    guardrail_category_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("app_guardrail_categories.guardrail_category_id"),
        primary_key=True,
    )


# ---------------------------------------------------------------------------
# Core Application Model
# ---------------------------------------------------------------------------

class Application(BaseModel):
    """
    AI Application model -- the central entity of the Application Orchestrator.

    An application is a configurable AI assistant that orchestrates skills
    (workflows and agents), applies guardrails, enhances prompts, and
    interacts with users through a chat interface.

    Key Fields:
        application_id: Primary key UUID
        user_id: Owner user (FK to users) for multi-tenancy
        name: Application name (only required field for draft creation)
        status: Lifecycle state -- draft / published / error

    Configuration Sections (all optional for draft, some required at publish):
        - Access: available_for_all_users, upload_enabled
        - Context: welcome_prompt, disclaimer, sorry_message, starter_questions
        - Chat Config: persona_id / persona_text, guardrail_text
        - Setup: LLMs, skills, data sources (via link tables)
        - Permissions: designations, business_units, tags (via link tables)

    Status State Machine:
        CREATE -> DRAFT -> (publish + validation) -> PUBLISHED
        PUBLISHED -> (edit + save) -> stays PUBLISHED
        PUBLISHED -> (unpublish) -> DRAFT
        Any -> (soft delete) -> is_deleted = True

    Indexes:
        - idx_applications_user: user_id WHERE is_deleted = FALSE
        - idx_applications_status: (user_id, status) WHERE is_deleted = FALSE
    """
    __tablename__ = "applications"

    # Primary key
    application_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Owner (FK to users)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Application name -- the only required field for draft creation
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Optional description
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Lifecycle status
    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        server_default="draft",
        nullable=False,
    )

    # Error message when status = 'error'
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # --- Access flags ---
    available_for_all_users: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )
    upload_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )

    # --- Context fields ---
    welcome_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    disclaimer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    sorry_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    starter_questions: Mapped[Optional[List]] = mapped_column(
        JSONB,
        server_default="'[]'::jsonb",
        nullable=True,
    )

    # --- Chat config ---
    persona_id: Mapped[Optional[UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("app_personas.persona_id"),
        nullable=True,
    )
    persona_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    guardrail_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # --- Consolidated JSONB configuration ---
    # Primary source for app config; link tables kept for backward compatibility.
    # Populated via dual-write: every link-table sync rebuilds this column.
    config: Mapped[Optional[Dict]] = mapped_column(
        JSONB,
        server_default="'{}'::jsonb",
        nullable=True,
    )

    # --- Misc ---
    logo_url: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )

    # --- Relationships to link tables ---
    llm_links: Mapped[List["ApplicationLlmLink"]] = relationship(
        "ApplicationLlmLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    skill_links: Mapped[List["ApplicationSkillLink"]] = relationship(
        "ApplicationSkillLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    data_source_links: Mapped[List["ApplicationDataSourceLink"]] = relationship(
        "ApplicationDataSourceLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    designation_links: Mapped[List["ApplicationDesignationLink"]] = relationship(
        "ApplicationDesignationLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    business_unit_links: Mapped[List["ApplicationBusinessUnitLink"]] = relationship(
        "ApplicationBusinessUnitLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    tag_links: Mapped[List["ApplicationTagLink"]] = relationship(
        "ApplicationTagLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    guardrail_links: Mapped[List["ApplicationGuardrailLink"]] = relationship(
        "ApplicationGuardrailLink",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    # --- Table constraints and indexes ---
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'error')",
            name="valid_app_status",
        ),
        Index(
            "idx_applications_user",
            "user_id",
            postgresql_where=(is_deleted == False),
        ),
        Index(
            "idx_applications_status",
            "user_id",
            "status",
            postgresql_where=(is_deleted == False),
        ),
    )

    # ------------------------------------------------------------------
    # Config JSONB helper accessors
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return config dict, defaulting to empty dict if None."""
        return self.config or {}

    def get_config_skills(self) -> list:
        """Get skills from config JSONB."""
        return self.get_config().get("skills", [])

    def get_config_llm_bindings(self) -> list:
        """Get LLM bindings from config JSONB."""
        return self.get_config().get("llm_bindings", [])

    def get_config_data_source_ids(self) -> list:
        """Get data source IDs from config JSONB."""
        return self.get_config().get("data_source_ids", [])

    def get_config_access_control(self) -> dict:
        """Get access control config."""
        return self.get_config().get("access_control", {})

    def get_config_guardrails(self) -> dict:
        """Get guardrails config."""
        return self.get_config().get("guardrails", {})

    def get_config_persona(self) -> dict:
        """Get persona config."""
        return self.get_config().get("persona", {})

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the application.

        Includes all scalar fields and the config JSONB column.
        Relationship data (link tables) should be loaded and serialized
        separately by the repository or Pydantic response models.

        Returns:
            Dict containing application data suitable for JSON serialization.
        """
        return {
            "application_id": str(self.application_id),
            "user_id": str(self.user_id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "error_message": self.error_message,
            "available_for_all_users": self.available_for_all_users,
            "upload_enabled": self.upload_enabled,
            "welcome_prompt": self.welcome_prompt,
            "disclaimer": self.disclaimer,
            "sorry_message": self.sorry_message,
            "starter_questions": self.starter_questions or [],
            "persona_id": str(self.persona_id) if self.persona_id else None,
            "persona_text": self.persona_text,
            "guardrail_text": self.guardrail_text,
            "config": self.config or {},
            "logo_url": self.logo_url,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Return application metadata without full configuration details.

        Useful for list/card views where full detail is not needed.

        Returns:
            Dict with summary metadata fields only.
        """
        return {
            "application_id": str(self.application_id),
            "user_id": str(self.user_id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "error_message": self.error_message,
            "logo_url": self.logo_url,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

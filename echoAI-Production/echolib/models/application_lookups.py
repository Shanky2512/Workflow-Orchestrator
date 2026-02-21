"""
EchoAI Application Lookup / Catalog Models

Reference tables that populate dropdown selections in the application wizard UI.
These are seeded via Alembic migration and managed through lookup API endpoints.

Tables:
- AppPersona: Persona archetypes (e.g. "Support Agent", "HR Assistant")
- AppGuardrailCategory: Guardrail categories (e.g. "Compliance", "PII", "Safety")
- AppDesignation: User designations for access control (e.g. "Executive", "Manager")
- AppBusinessUnit: Organizational units for access control (e.g. "Engineering", "Sales")
- AppTag: Free-form tags for application categorization
- AppLlm: Available LLM configurations (loaded from JSON catalog)
- AppDataSource: Available data sources (MCP connectors, APIs)

Design Principles:
- UUID primary keys for all tables except AppDataSource (uses VARCHAR PK)
- Unique name constraints to prevent duplicates
- Minimal schema: id + name + created_at (no updated_at needed for lookups)
- Inherit from Base directly (not BaseModel) since these only need created_at
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class AppPersona(Base):
    """
    Persona archetype for application chat behavior.

    A persona defines the tone and personality of the application's
    responses (e.g. "Support Agent", "HR Assistant", "Sales Assistant").

    Applications reference a persona via persona_id FK on the applications table.
    Custom persona_text overrides the named persona when both are set.

    Attributes:
        persona_id: Primary key UUID
        name: Unique persona name
        created_at: Record creation timestamp
    """
    __tablename__ = "app_personas"

    persona_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return persona data as a dictionary."""
        return {
            "persona_id": str(self.persona_id),
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppGuardrailCategory(Base):
    """
    Guardrail category for application safety enforcement.

    Categories define the types of guardrail checks applied to
    user input and LLM output (e.g. "Compliance", "PII", "Safety").

    Applications link to categories via the application_guardrail_links table.

    Attributes:
        guardrail_category_id: Primary key UUID
        name: Unique category name
        created_at: Record creation timestamp
    """
    __tablename__ = "app_guardrail_categories"

    guardrail_category_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return guardrail category data as a dictionary."""
        return {
            "guardrail_category_id": str(self.guardrail_category_id),
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppDesignation(Base):
    """
    Designation for user access control on applications.

    Designations represent organizational roles (e.g. "Executive", "Manager",
    "Individual Contributor", "Intern") that can be used to restrict which
    users can access a published application.

    Applications link to designations via the application_designation_links table.

    Attributes:
        designation_id: Primary key UUID
        name: Unique designation name
        created_at: Record creation timestamp
    """
    __tablename__ = "app_designations"

    designation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return designation data as a dictionary."""
        return {
            "designation_id": str(self.designation_id),
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppBusinessUnit(Base):
    """
    Business unit for organizational access control on applications.

    Business units represent organizational divisions (e.g. "Engineering",
    "Sales", "Human Resources") that can be used to restrict which
    users can access a published application.

    Applications link to business units via the application_business_unit_links table.

    Attributes:
        business_unit_id: Primary key UUID
        name: Unique business unit name
        created_at: Record creation timestamp
    """
    __tablename__ = "app_business_units"

    business_unit_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return business unit data as a dictionary."""
        return {
            "business_unit_id": str(self.business_unit_id),
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppTag(Base):
    """
    Tag for application categorization and filtering.

    Tags are free-form labels (e.g. "Internal", "External", "Priority",
    "Experimental") that can be attached to applications for organization.

    Applications link to tags via the application_tag_links table.

    Attributes:
        tag_id: Primary key UUID
        name: Unique tag name
        created_at: Record creation timestamp
    """
    __tablename__ = "app_tags"

    tag_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return tag data as a dictionary."""
        return {
            "tag_id": str(self.tag_id),
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppLlm(Base):
    """
    LLM configuration entry in the application LLM catalog.

    Stores available LLM providers and models that can be bound to
    applications for orchestration, prompt enhancement, or general use.
    Loaded from the llm_provider JSON catalog file.

    Attributes:
        llm_id: Primary key UUID
        name: Display name for the LLM
        provider: Provider identifier (e.g. "openai", "anthropic", "ollama")
        model_name: Model identifier within the provider
        base_url: API base URL for the LLM provider
        api_key_env: Environment variable name containing the API key
                     (the actual key is never stored in the database)
        is_default: Whether this LLM is the default selection
        created_at: Record creation timestamp
    """
    __tablename__ = "app_llms"

    llm_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    provider: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    model_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    base_url: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )
    api_key_env: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return LLM data as a dictionary."""
        return {
            "llm_id": str(self.llm_id),
            "name": self.name,
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AppDataSource(Base):
    """
    Data source entry in the application data source catalog.

    Represents an available data source (MCP connector, API, etc.)
    that can be bound to an application for RAG or tool access.

    Uses a VARCHAR primary key (data_source_id) to match connector IDs
    from external systems rather than auto-generated UUIDs.

    Attributes:
        data_source_id: Primary key VARCHAR (matches connector IDs)
        name: Display name for the data source
        kind: Data source type ('mcp', 'api', etc.)
        metadata: JSONB for additional configuration and metadata
        created_at: Record creation timestamp
    """
    __tablename__ = "app_data_sources"

    data_source_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    kind: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    source_metadata: Mapped[Optional[Dict]] = mapped_column(
        "metadata",
        JSONB,
        server_default="'{}'::jsonb",
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return data source data as a dictionary."""
        return {
            "data_source_id": self.data_source_id,
            "name": self.name,
            "kind": self.kind,
            "metadata": self.source_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

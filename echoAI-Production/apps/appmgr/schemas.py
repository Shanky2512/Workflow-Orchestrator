"""
EchoAI Application Orchestrator — Pydantic Schemas

Request and response models for all Application API endpoints.

These schemas are designed for the new PostgreSQL-based backend and
replace the older SQLModel/SQLite Pydantic schemas that lived inside
``apps/appmgr/main.py``.

Design Principles:
    - Request schemas are permissive (most fields optional for draft)
    - Response schemas use ``model_config = ConfigDict(from_attributes=True)``
      for direct SQLAlchemy model serialisation
    - Field names match the PostgreSQL column names exactly
    - UUID fields are typed as ``str`` in schemas for JSON compatibility
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# REQUEST SCHEMAS
# ===========================================================================


class ApplicationCreate(BaseModel):
    """
    Create a new application.

    Only ``name`` is required. All other fields are optional and can be
    filled later via the section-specific PATCH endpoints.
    """
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    logo_url: Optional[str] = None
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    sorry_message: Optional[str] = None
    starter_questions: Optional[List[str]] = None
    persona_id: Optional[str] = None
    persona_text: Optional[str] = None
    guardrail_text: Optional[str] = None
    available_for_all_users: Optional[bool] = None
    upload_enabled: Optional[bool] = None

    # Setup (Step 1) — link lists
    llm_links: Optional[List[Dict[str, Any]]] = None
    skill_links: Optional[List[Dict[str, Any]]] = None
    data_source_ids: Optional[List[str]] = None

    # Access (Step 2) — link lists
    designation_ids: Optional[List[str]] = None
    business_unit_ids: Optional[List[str]] = None
    tag_ids: Optional[List[str]] = None

    # Chat Config (Step 4) — link list
    guardrail_category_ids: Optional[List[str]] = None


class ApplicationSetupUpdate(BaseModel):
    """
    Update the "App Setup" section (Step 1): LLMs, skills, data sources.

    All fields are Optional so that ``null`` values from the frontend
    are treated as "don't change" rather than triggering a 422.
    """
    llm_links: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of LLM bindings. Each dict must contain 'llm_id' (str) "
            "and optionally 'role' ('orchestrator' | 'enhancer' | 'general'). "
            "Pass null to leave unchanged."
        ),
    )
    skill_links: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of skill bindings. Each dict must contain 'skill_id', "
            "'skill_type' ('workflow' | 'agent'), and optionally "
            "'skill_name' and 'skill_description'. "
            "Pass null to leave unchanged."
        ),
    )
    data_source_ids: Optional[List[str]] = Field(
        default=None,
        description="List of data_source_id strings to link. Pass null to leave unchanged.",
    )


class ApplicationAccessUpdate(BaseModel):
    """
    Update the "Access Permission" section (Step 2).
    """
    designation_ids: List[str] = Field(default_factory=list)
    business_unit_ids: List[str] = Field(default_factory=list)
    tag_ids: List[str] = Field(default_factory=list)
    available_for_all_users: Optional[bool] = None
    upload_enabled: Optional[bool] = None


class ApplicationContextUpdate(BaseModel):
    """
    Update the "Context" section (Step 3).
    """
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    sorry_message: Optional[str] = None
    starter_questions: Optional[List[str]] = None


class ApplicationChatConfigUpdate(BaseModel):
    """
    Update the "Chat Configuration" section (Step 4).
    """
    persona_id: Optional[str] = None
    persona_text: Optional[str] = None
    guardrail_text: Optional[str] = None
    guardrail_category_ids: List[str] = Field(default_factory=list)


class ApplicationFullUpdate(BaseModel):
    """
    Full update of an application (all sections combined).

    Used by ``PUT /api/applications/{id}``. All fields are optional;
    only provided fields are written.
    """
    # Basic
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    logo_url: Optional[str] = None

    # Setup (Step 1) — link lists
    llm_links: Optional[List[Dict[str, Any]]] = None
    skill_links: Optional[List[Dict[str, Any]]] = None
    data_source_ids: Optional[List[str]] = None

    # Access (Step 2)
    designation_ids: Optional[List[str]] = None
    business_unit_ids: Optional[List[str]] = None
    tag_ids: Optional[List[str]] = None
    available_for_all_users: Optional[bool] = None
    upload_enabled: Optional[bool] = None

    # Context (Step 3)
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    sorry_message: Optional[str] = None
    starter_questions: Optional[List[str]] = None

    # Chat Config (Step 4)
    persona_id: Optional[str] = None
    persona_text: Optional[str] = None
    guardrail_text: Optional[str] = None
    guardrail_category_ids: Optional[List[str]] = None


class ChatMessageRequest(BaseModel):
    """
    Send a chat message to an application.

    If ``session_id`` is omitted, a new chat session is created automatically.
    """
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(
        None,
        description="Existing chat session UUID. Omit to start a new session.",
    )


# ===========================================================================
# RESPONSE SCHEMAS
# ===========================================================================


class LlmLinkResponse(BaseModel):
    """Single LLM binding in an application detail response."""
    llm_id: str
    role: str
    name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SkillLinkResponse(BaseModel):
    """Single skill binding in an application detail response."""
    skill_id: str
    skill_type: str
    skill_name: Optional[str] = None
    skill_description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ApplicationCard(BaseModel):
    """
    Single card in the dashboard grid.

    Contains only the fields needed for the list/card view.
    Link counts are computed from the loaded relationships.
    """
    application_id: str
    name: str
    status: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    llm_count: int = 0
    data_source_count: int = 0
    skill_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ApplicationDetail(BaseModel):
    """
    Full application data including all link lists.

    Used for the detail / edit view (GET single application).
    """
    application_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    status: str
    error_message: Optional[str] = None

    # Access
    available_for_all_users: bool = False
    upload_enabled: bool = False

    # Context
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    sorry_message: Optional[str] = None
    starter_questions: Optional[List[str]] = None

    # Chat Config
    persona_id: Optional[str] = None
    persona_text: Optional[str] = None
    guardrail_text: Optional[str] = None

    # Misc
    logo_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    # Resolved link lists
    llm_links: List[LlmLinkResponse] = Field(default_factory=list)
    skill_links: List[SkillLinkResponse] = Field(default_factory=list)
    data_source_ids: List[str] = Field(default_factory=list)
    designation_ids: List[str] = Field(default_factory=list)
    business_unit_ids: List[str] = Field(default_factory=list)
    tag_ids: List[str] = Field(default_factory=list)
    guardrail_category_ids: List[str] = Field(default_factory=list)

    # Resolved names (for display)
    designation_names: List[str] = Field(default_factory=list)
    business_unit_names: List[str] = Field(default_factory=list)
    tag_names: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ApplicationStats(BaseModel):
    """Stats row above the dashboard grid."""
    total: int = 0
    published: int = 0
    draft: int = 0
    errors: int = 0


class ApplicationListResponse(BaseModel):
    """Full dashboard response with stats and paginated application cards."""
    page: int
    page_size: int
    total: int
    stats: ApplicationStats
    items: List[ApplicationCard]


class ChatSessionResponse(BaseModel):
    """Chat session summary (for listing sessions)."""
    chat_session_id: str
    application_id: str
    title: Optional[str] = None
    conversation_state: str
    started_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    """Single chat message in a session."""
    message_id: str
    role: str
    content: str
    enhanced_prompt: Optional[str] = None
    execution_trace: Optional[Dict[str, Any]] = None
    guardrail_flags: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSessionMetadata(BaseModel):
    """
    Metadata returned when a new chat session is created.

    Contains all app-level configuration fields the frontend needs
    to render the chat UI: welcome greeting, disclaimer banner,
    suggested starter questions, and basic app identity.

    Only populated on the first message (when no session_id was
    provided and a new session was created).
    """
    app_name: str
    app_description: Optional[str] = None
    app_logo_url: Optional[str] = None
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    starter_questions: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response from the chat endpoint (POST /applications/{id}/chat)."""
    session_id: str
    message: ChatMessageResponse
    conversation_state: str
    session_metadata: Optional[ChatSessionMetadata] = Field(
        None,
        description=(
            "Populated only when a new session is created (first message). "
            "Contains welcome_prompt, disclaimer, starter_questions, and app identity "
            "for the frontend to render the initial chat UI."
        ),
    )


class DocumentUploadResponse(BaseModel):
    """Response from the document upload endpoint."""
    document_id: str
    filename: str
    processing_status: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    chunk_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class LookupResponse(BaseModel):
    """Generic lookup item (for personas, designations, tags, etc.)."""
    id: str
    name: str

    model_config = ConfigDict(from_attributes=True)


class SkillResponse(BaseModel):
    """
    Skill entry returned by GET /api/skills.

    Combines workflows and agents from the main PostgreSQL tables,
    each labeled with its type.
    """
    skill_id: str
    skill_type: str
    name: str
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# HITL (Human-in-the-Loop) DECISION SCHEMAS
# ===========================================================================


class HITLDecisionRequest(BaseModel):
    """
    Request body for the HITL decision endpoint.

    Submitted by a human reviewer to approve, reject, edit, or defer
    a workflow that has paused at a HITL node.
    """
    session_id: str = Field(..., description="Chat session UUID where the HITL interrupt occurred.")
    action: str = Field(
        ...,
        description="Human decision: 'approve', 'reject', 'edit', or 'defer'.",
        pattern="^(approve|reject|edit|defer)$",
    )
    payload: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional data for the decision (e.g. edited content for 'edit' action).",
    )
    rationale: Optional[str] = Field(
        None,
        description="Optional human-provided rationale for the decision.",
    )


class HITLDecisionResponse(BaseModel):
    """
    Response from the HITL decision endpoint.
    """
    status: str = Field(..., description="Result status: 'completed', 'interrupted', 'rejected', 'deferred'.")
    message: str = Field(..., description="Human-readable description of the outcome.")
    session_id: str
    conversation_state: str = Field(..., description="Updated conversation state for the session.")
    execution_trace: Optional[Dict[str, Any]] = Field(
        None,
        description="Execution trace including HITL context when applicable.",
    )


class HITLContextSchema(BaseModel):
    """
    HITL interrupt context returned to the frontend for review.

    Contains the information a human reviewer needs to make a decision
    about a paused workflow.
    """
    run_id: str
    workflow_id: str
    title: Optional[str] = None
    message: Optional[str] = None
    priority: Optional[str] = None
    allowed_decisions: List[str] = Field(default_factory=lambda: ["approve", "reject"])
    node_outputs: Optional[Dict[str, Any]] = None

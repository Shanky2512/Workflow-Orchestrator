"""
EchoAI Application Orchestrator -- API Routes (Phases 2 + 4)

Async CRUD and Chat endpoints backed by PostgreSQL via SQLAlchemy 2.0 + asyncpg.
Replaces the previous SQLite/SQLModel sync implementation.

Route groups:
    /api/applications/...            -- Application CRUD, publish, logo
    /api/applications/{id}/chat/...  -- Chat (pipeline, history, upload)
    /api/llms, /api/data-sources,
    /api/designations, etc.          -- Lookup / catalog endpoints
    /api/skills                      -- Aggregated workflows + agents

Authentication:
    Uses ``get_current_user`` from ``echolib.security`` (optional/required
    mode is controlled by the ``AUTH_ENFORCEMENT`` setting).
    Falls back to ``DEFAULT_USER_ID`` for anonymous callers.

Database:
    Uses ``get_db_session()`` async context manager from ``echolib.database``.
    All mutations flush inside the session; the context manager commits or
    rolls back automatically.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from starlette.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.database import get_db_session
from echolib.models import (
    AppBusinessUnit,
    AppDataSource,
    AppDesignation,
    AppGuardrailCategory,
    AppLlm,
    AppPersona,
    AppTag,
    Application,
)
from echolib.models.workflow import Workflow
from echolib.models.agent import Agent
from echolib.repositories.application_repo import ApplicationRepository
from echolib.repositories.base import DEFAULT_USER_ID
from echolib.security import get_current_user
from echolib.types import UserContext

from echolib.repositories.application_chat_repo import ApplicationChatRepository

from .schemas import (
    ApplicationAccessUpdate,
    ApplicationCard,
    ApplicationChatConfigUpdate,
    ApplicationContextUpdate,
    ApplicationCreate,
    ApplicationDetail,
    ApplicationFullUpdate,
    ApplicationListResponse,
    ApplicationSetupUpdate,
    ApplicationStats,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatResponse,
    ChatSessionMetadata,
    ChatSessionResponse,
    DocumentUploadResponse,
    HITLDecisionRequest,
    HITLDecisionResponse,
    LlmLinkResponse,
    LookupResponse,
    SkillLinkResponse,
    SkillResponse,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories for file uploads
# ---------------------------------------------------------------------------
LOGOS_DIR = os.getenv("APP_LOGOS_DIR", os.path.join("static", "logos"))
Path(LOGOS_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Directories for file uploads -- chat documents
# ---------------------------------------------------------------------------
UPLOADS_DIR = os.getenv("APP_UPLOADS_DIR", os.path.join("apps", "appmgr", "uploads"))
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Repository singletons (stateless, safe to reuse)
# ---------------------------------------------------------------------------
_app_repo = ApplicationRepository()
_chat_repo = ApplicationChatRepository()


def _resolve_user_id(user: UserContext) -> str:
    """Return a usable user_id, falling back to DEFAULT_USER_ID for anonymous."""
    uid = user.user_id
    if not uid or uid == "anonymous":
        return DEFAULT_USER_ID
    return uid


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _app_to_card(app: Application) -> ApplicationCard:
    """Convert an Application ORM instance to an ApplicationCard schema."""
    return ApplicationCard(
        application_id=str(app.application_id),
        name=app.name,
        status=app.status,
        description=app.description,
        logo_url=app.logo_url,
        llm_count=len(app.llm_links) if app.llm_links else 0,
        data_source_count=len(app.data_source_links) if app.data_source_links else 0,
        skill_count=len(app.skill_links) if app.skill_links else 0,
        error_message=app.error_message,
        created_at=app.created_at,
        updated_at=app.updated_at,
    )


async def _build_app_detail(db: AsyncSession, app: Application) -> ApplicationDetail:
    """Convert an Application ORM instance to an ApplicationDetail schema.

    Resolves LLM names via the eager-loaded relationship and resolves
    designation, business unit, and tag names from their lookup tables.
    """
    from echolib.repositories.base import safe_uuid

    # Resolve designation names
    designation_names: List[str] = []
    designation_ids_list = [
        str(link.designation_id) for link in (app.designation_links or [])
    ]
    if designation_ids_list:
        _d_uuids = [safe_uuid(did) for did in designation_ids_list]
        _d_uuids = [u for u in _d_uuids if u is not None]
        if _d_uuids:
            _d_result = await db.execute(
                select(AppDesignation).where(AppDesignation.designation_id.in_(_d_uuids))
            )
            designation_names = [d.name for d in _d_result.scalars().all()]

    # Resolve business unit names
    business_unit_names: List[str] = []
    business_unit_ids_list = [
        str(link.business_unit_id) for link in (app.business_unit_links or [])
    ]
    if business_unit_ids_list:
        _bu_uuids = [safe_uuid(buid) for buid in business_unit_ids_list]
        _bu_uuids = [u for u in _bu_uuids if u is not None]
        if _bu_uuids:
            _bu_result = await db.execute(
                select(AppBusinessUnit).where(AppBusinessUnit.business_unit_id.in_(_bu_uuids))
            )
            business_unit_names = [bu.name for bu in _bu_result.scalars().all()]

    # Resolve tag names
    tag_names: List[str] = []
    tag_ids_list = [str(link.tag_id) for link in (app.tag_links or [])]
    if tag_ids_list:
        _t_uuids = [safe_uuid(tid) for tid in tag_ids_list]
        _t_uuids = [u for u in _t_uuids if u is not None]
        if _t_uuids:
            _t_result = await db.execute(
                select(AppTag).where(AppTag.tag_id.in_(_t_uuids))
            )
            tag_names = [t.name for t in _t_result.scalars().all()]

    return ApplicationDetail(
        application_id=str(app.application_id),
        user_id=str(app.user_id),
        name=app.name,
        description=app.description,
        status=app.status,
        error_message=app.error_message,
        available_for_all_users=app.available_for_all_users,
        upload_enabled=app.upload_enabled,
        welcome_prompt=app.welcome_prompt,
        disclaimer=app.disclaimer,
        sorry_message=app.sorry_message,
        starter_questions=app.starter_questions or [],
        persona_id=str(app.persona_id) if app.persona_id else None,
        persona_text=app.persona_text,
        guardrail_text=app.guardrail_text,
        logo_url=app.logo_url,
        created_at=app.created_at,
        updated_at=app.updated_at,
        llm_links=[
            LlmLinkResponse(
                llm_id=link.llm_id,
                role=link.role,
                name=link.name,
            )
            for link in (app.llm_links or [])
        ],
        skill_links=[
            SkillLinkResponse(
                skill_id=link.skill_id,
                skill_type=link.skill_type,
                skill_name=link.skill_name,
                skill_description=link.skill_description,
            )
            for link in (app.skill_links or [])
        ],
        data_source_ids=[
            link.data_source_id for link in (app.data_source_links or [])
        ],
        designation_ids=designation_ids_list,
        business_unit_ids=business_unit_ids_list,
        tag_ids=tag_ids_list,
        guardrail_category_ids=[
            str(link.guardrail_category_id)
            for link in (app.guardrail_links or [])
        ],
        designation_names=designation_names,
        business_unit_names=business_unit_names,
        tag_names=tag_names,
    )


# ===================================================================
# Application CRUD Router
# ===================================================================

router = APIRouter(prefix="/api/applications", tags=["Applications"])


# -------------------------------------------------------------------
# POST /api/applications -- Create
# -------------------------------------------------------------------
@router.post("", response_model=ApplicationDetail, status_code=201)
async def create_application(
    payload: ApplicationCreate,
    user: UserContext = Depends(get_current_user),
):
    """Create a new application in DRAFT status. Only ``name`` is required."""
    user_id = _resolve_user_id(user)

    # Build optional kwargs from payload
    kwargs = {}
    if payload.description is not None:
        kwargs["description"] = payload.description
    if payload.logo_url is not None:
        kwargs["logo_url"] = payload.logo_url
    if payload.welcome_prompt is not None:
        kwargs["welcome_prompt"] = payload.welcome_prompt
    if payload.disclaimer is not None:
        kwargs["disclaimer"] = payload.disclaimer
    if payload.sorry_message is not None:
        kwargs["sorry_message"] = payload.sorry_message
    if payload.starter_questions is not None:
        kwargs["starter_questions"] = payload.starter_questions
    if payload.persona_id is not None:
        kwargs["persona_id"] = payload.persona_id
    if payload.persona_text is not None:
        kwargs["persona_text"] = payload.persona_text
    if payload.guardrail_text is not None:
        kwargs["guardrail_text"] = payload.guardrail_text
    if payload.available_for_all_users is not None:
        kwargs["available_for_all_users"] = payload.available_for_all_users
    if payload.upload_enabled is not None:
        kwargs["upload_enabled"] = payload.upload_enabled

    async with get_db_session() as db:
        app = await _app_repo.create_application(
            db, user_id=user_id, name=payload.name.strip(), **kwargs
        )

        app_id_str = str(app.application_id)

        # Sync link tables if provided at creation time
        if payload.llm_links is not None:
            await _app_repo.sync_llm_links(db, app_id_str, payload.llm_links)
        if payload.skill_links is not None:
            await _app_repo.sync_skill_links(db, app_id_str, payload.skill_links)
        if payload.data_source_ids is not None:
            await _app_repo.sync_data_source_links(db, app_id_str, payload.data_source_ids)
        if payload.designation_ids is not None:
            await _app_repo.sync_designation_links(db, app_id_str, payload.designation_ids)
        if payload.business_unit_ids is not None:
            await _app_repo.sync_business_unit_links(db, app_id_str, payload.business_unit_ids)
        if payload.tag_ids is not None:
            await _app_repo.sync_tag_links(db, app_id_str, payload.tag_ids)
        if payload.guardrail_category_ids is not None:
            await _app_repo.sync_guardrail_links(db, app_id_str, payload.guardrail_category_ids)

        # Expire cached objects so relationships reload from DB
        db.expire_all()

        # Rebuild config JSONB from fresh link data (dual-write)
        await _app_repo.sync_config_jsonb(
            db, app_id_str, user_id
        )
        app = await _app_repo.get_application(
            db, app_id_str, user_id
        )
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# GET /api/applications -- List (paginated, with stats)
# -------------------------------------------------------------------
@router.get("", response_model=ApplicationListResponse)
async def list_applications(
    status: Optional[str] = Query(None, description="Filter: draft|published|error"),
    q: Optional[str] = Query(None, description="Search by name (ILIKE)"),
    sort: Optional[str] = Query(None, description="Sort mode: recently_updated"),
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
):
    """List applications as paginated cards with dashboard stats."""
    user_id = _resolve_user_id(user)

    if status and status not in {"draft", "published", "error"}:
        raise HTTPException(400, "Invalid status filter. Use: draft, published, error")

    async with get_db_session() as db:
        items, total = await _app_repo.list_applications(
            db,
            user_id=user_id,
            status=status,
            q=q,
            sort=sort,
            page=page,
            page_size=pageSize,
        )
        stats_dict = await _app_repo.get_application_stats(db, user_id=user_id)

        return ApplicationListResponse(
            page=page,
            page_size=pageSize,
            total=total,
            stats=ApplicationStats(**stats_dict),
            items=[_app_to_card(app) for app in items],
        )


# -------------------------------------------------------------------
# GET /api/applications/{application_id} -- Detail
# -------------------------------------------------------------------
@router.get("/{application_id}", response_model=ApplicationDetail)
async def get_application(
    application_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get a single application with all relationships."""
    user_id = _resolve_user_id(user)
    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# PUT /api/applications/{application_id} -- Full Update
# -------------------------------------------------------------------
@router.put("/{application_id}", response_model=ApplicationDetail)
async def update_application(
    application_id: str,
    payload: ApplicationFullUpdate,
    user: UserContext = Depends(get_current_user),
):
    """
    Full update of an application (all sections at once).

    EDIT BEHAVIOR: If the app is PUBLISHED, it stays PUBLISHED after save.
    The status does NOT revert to DRAFT on edit.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        # Collect scalar updates (only provided fields)
        scalar_kwargs = {}
        if payload.name is not None:
            scalar_kwargs["name"] = payload.name.strip()
        if payload.description is not None:
            scalar_kwargs["description"] = payload.description
        if payload.logo_url is not None:
            scalar_kwargs["logo_url"] = payload.logo_url
        if payload.available_for_all_users is not None:
            scalar_kwargs["available_for_all_users"] = payload.available_for_all_users
        if payload.upload_enabled is not None:
            scalar_kwargs["upload_enabled"] = payload.upload_enabled
        if payload.welcome_prompt is not None:
            scalar_kwargs["welcome_prompt"] = payload.welcome_prompt
        if payload.disclaimer is not None:
            scalar_kwargs["disclaimer"] = payload.disclaimer
        if payload.sorry_message is not None:
            scalar_kwargs["sorry_message"] = payload.sorry_message
        if payload.starter_questions is not None:
            scalar_kwargs["starter_questions"] = payload.starter_questions
        if payload.persona_id is not None:
            scalar_kwargs["persona_id"] = payload.persona_id
        if payload.persona_text is not None:
            scalar_kwargs["persona_text"] = payload.persona_text
        if payload.guardrail_text is not None:
            scalar_kwargs["guardrail_text"] = payload.guardrail_text

        # Apply scalar updates
        if scalar_kwargs:
            app = await _app_repo.update_application(
                db, application_id, user_id, **scalar_kwargs
            )
            if app is None:
                raise HTTPException(404, "Application not found after update")

        # Sync link tables (only if provided in payload)
        app_id_str = application_id
        if payload.llm_links is not None:
            await _app_repo.sync_llm_links(db, app_id_str, payload.llm_links)
        if payload.skill_links is not None:
            await _app_repo.sync_skill_links(db, app_id_str, payload.skill_links)
        if payload.data_source_ids is not None:
            await _app_repo.sync_data_source_links(
                db, app_id_str, payload.data_source_ids
            )
        if payload.designation_ids is not None:
            await _app_repo.sync_designation_links(
                db, app_id_str, payload.designation_ids
            )
        if payload.business_unit_ids is not None:
            await _app_repo.sync_business_unit_links(
                db, app_id_str, payload.business_unit_ids
            )
        if payload.tag_ids is not None:
            await _app_repo.sync_tag_links(db, app_id_str, payload.tag_ids)
        if payload.guardrail_category_ids is not None:
            await _app_repo.sync_guardrail_links(
                db, app_id_str, payload.guardrail_category_ids
            )

        # Expire cached objects so relationships reload from DB
        db.expire_all()

        # Rebuild config JSONB after link and scalar changes (dual-write)
        await _app_repo.sync_config_jsonb(db, application_id, user_id)

        # Re-fetch with refreshed relationships
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found after update")
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# PATCH /api/applications/{application_id}/setup
# -------------------------------------------------------------------
@router.patch("/{application_id}/setup", response_model=ApplicationDetail)
async def patch_setup(
    application_id: str,
    payload: ApplicationSetupUpdate,
    user: UserContext = Depends(get_current_user),
):
    """Update the App Setup section (Step 1): LLMs, Skills, Data Sources."""
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        if payload.llm_links is not None:
            await _app_repo.sync_llm_links(db, application_id, payload.llm_links)
        if payload.skill_links is not None:
            await _app_repo.sync_skill_links(db, application_id, payload.skill_links)
        if payload.data_source_ids is not None:
            await _app_repo.sync_data_source_links(
                db, application_id, payload.data_source_ids
            )

        # Expire cached objects so relationships reload from DB
        db.expire_all()

        # Rebuild config JSONB after link changes (dual-write)
        await _app_repo.sync_config_jsonb(db, application_id, user_id)

        app = await _app_repo.get_application(db, application_id, user_id)
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# PATCH /api/applications/{application_id}/access
# -------------------------------------------------------------------
@router.patch("/{application_id}/access", response_model=ApplicationDetail)
async def patch_access(
    application_id: str,
    payload: ApplicationAccessUpdate,
    user: UserContext = Depends(get_current_user),
):
    """Update the Access Permission section (Step 2)."""
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        # Scalar access fields
        scalar_kwargs = {}
        if payload.available_for_all_users is not None:
            scalar_kwargs["available_for_all_users"] = payload.available_for_all_users
        if payload.upload_enabled is not None:
            scalar_kwargs["upload_enabled"] = payload.upload_enabled
        if scalar_kwargs:
            await _app_repo.update_application(
                db, application_id, user_id, **scalar_kwargs
            )

        # Link tables
        await _app_repo.sync_designation_links(
            db, application_id, payload.designation_ids
        )
        await _app_repo.sync_business_unit_links(
            db, application_id, payload.business_unit_ids
        )
        await _app_repo.sync_tag_links(db, application_id, payload.tag_ids)

        # Expire cached objects so relationships reload from DB
        db.expire_all()

        # Rebuild config JSONB after link changes (dual-write)
        await _app_repo.sync_config_jsonb(db, application_id, user_id)

        app = await _app_repo.get_application(db, application_id, user_id)
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# PATCH /api/applications/{application_id}/context
# -------------------------------------------------------------------
@router.patch("/{application_id}/context", response_model=ApplicationDetail)
async def patch_context(
    application_id: str,
    payload: ApplicationContextUpdate,
    user: UserContext = Depends(get_current_user),
):
    """Update the Context section (Step 3)."""
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        kwargs = {}
        if payload.welcome_prompt is not None:
            kwargs["welcome_prompt"] = payload.welcome_prompt
        if payload.disclaimer is not None:
            kwargs["disclaimer"] = payload.disclaimer
        if payload.sorry_message is not None:
            kwargs["sorry_message"] = payload.sorry_message
        if payload.starter_questions is not None:
            kwargs["starter_questions"] = payload.starter_questions

        if kwargs:
            await _app_repo.update_application(
                db, application_id, user_id, **kwargs
            )

        app = await _app_repo.get_application(db, application_id, user_id)
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# PATCH /api/applications/{application_id}/chat-config
# -------------------------------------------------------------------
@router.patch("/{application_id}/chat-config", response_model=ApplicationDetail)
async def patch_chat_config(
    application_id: str,
    payload: ApplicationChatConfigUpdate,
    user: UserContext = Depends(get_current_user),
):
    """Update the Chat Configuration section (Step 4)."""
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        kwargs = {}
        if payload.persona_id is not None:
            kwargs["persona_id"] = payload.persona_id
        if payload.persona_text is not None:
            kwargs["persona_text"] = payload.persona_text
        if payload.guardrail_text is not None:
            kwargs["guardrail_text"] = payload.guardrail_text

        if kwargs:
            await _app_repo.update_application(
                db, application_id, user_id, **kwargs
            )

        # Sync guardrail category links
        await _app_repo.sync_guardrail_links(
            db, application_id, payload.guardrail_category_ids
        )

        # Expire cached objects so relationships reload from DB
        db.expire_all()

        # Rebuild config JSONB after link and scalar changes (dual-write)
        await _app_repo.sync_config_jsonb(db, application_id, user_id)

        app = await _app_repo.get_application(db, application_id, user_id)
        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# DELETE /api/applications/{application_id} -- Soft Delete
# -------------------------------------------------------------------
@router.delete("/{application_id}", status_code=204)
async def delete_application(
    application_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Soft-delete an application (sets is_deleted = TRUE)."""
    user_id = _resolve_user_id(user)
    async with get_db_session() as db:
        deleted = await _app_repo.soft_delete_application(
            db, application_id, user_id
        )
        if not deleted:
            raise HTTPException(404, "Application not found")
    return Response(status_code=204)


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/publish
# -------------------------------------------------------------------
@router.post("/{application_id}/publish", response_model=ApplicationDetail)
async def publish_application(
    application_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Validate required fields and transition DRAFT -> PUBLISHED.

    Returns 400 with a list of missing fields if validation fails.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        if app.status == "published":
            raise HTTPException(400, "Application is already published")

        # Validate required fields for publish
        errors: List[str] = []
        warnings: List[str] = []

        # Required — block publish if missing
        if not app.llm_links:
            errors.append("At least 1 LLM must be linked")
        if not app.skill_links:
            errors.append("At least 1 skill (workflow or agent) must be linked")

        # Optional — warn but don't block
        if not app.welcome_prompt:
            warnings.append("welcome_prompt is not set — a default greeting will be used")
        if not app.sorry_message:
            warnings.append("sorry_message is not set — a default fallback will be used")

        starters = app.starter_questions or []
        if not starters or not any(
            s.strip() for s in starters if isinstance(s, str)
        ):
            warnings.append("No starter questions set — chat UI will show no suggestions")

        has_persona = bool(app.persona_id) or bool(
            app.persona_text and app.persona_text.strip()
        )
        if not has_persona:
            warnings.append("No persona set — a default persona will be used")

        if warnings:
            logger.warning(f"Publish warnings for app {application_id}: {warnings}")

        if errors:
            raise HTTPException(
                status_code=400,
                detail={"message": "Publish validation failed", "errors": errors},
            )

        app = await _app_repo.publish_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(500, "Failed to publish application")

        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/unpublish
# -------------------------------------------------------------------
@router.post("/{application_id}/unpublish", response_model=ApplicationDetail)
async def unpublish_application(
    application_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Transition PUBLISHED -> DRAFT."""
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")
        if app.status != "published":
            raise HTTPException(400, "Application is not currently published")

        app = await _app_repo.unpublish_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(500, "Failed to unpublish application")

        return await _build_app_detail(db, app)


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/logo -- Upload Logo
# -------------------------------------------------------------------
@router.post("/{application_id}/logo", response_model=ApplicationDetail)
async def upload_logo(
    application_id: str,
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user),
):
    """Upload an application logo (PNG/JPEG/WEBP, resized to 250x250 max)."""
    user_id = _resolve_user_id(user)

    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(400, "Only PNG, JPEG, or WEBP images are allowed")

    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        filename = f"{application_id}_{uuid.uuid4().hex}.png"
        path = os.path.join(LOGOS_DIR, filename)

        try:
            from PIL import Image

            img = Image.open(file.file).convert("RGBA")
        except Exception:
            raise HTTPException(400, "Invalid image file")

        max_dim = 250
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        img.save(path, format="PNG")

        logo_url = f"/static/logos/{filename}"
        app = await _app_repo.update_application(
            db, application_id, user_id, logo_url=logo_url
        )
        if app is None:
            raise HTTPException(500, "Failed to update logo")

        return await _build_app_detail(db, app)


# ===================================================================
# Chat Endpoints (Phase 4)
# ===================================================================


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/chat -- Full Pipeline
# -------------------------------------------------------------------
@router.post("/{application_id}/chat", response_model=ChatResponse)
async def chat_with_application(
    application_id: str,
    payload: ChatMessageRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Send a chat message to a published application.

    Runs the full orchestration pipeline: pre-guardrails, prompt enhancement,
    orchestrator LLM, skill execution, post-guardrails, persona formatting.

    If ``session_id`` is omitted from the request, a new chat session is
    created automatically. The response includes the session_id for
    subsequent messages.
    """
    user_id = _resolve_user_id(user)

    # Validate application exists and is published
    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")
        if app.status != "published":
            raise HTTPException(
                400,
                "Application is not published. Only published applications can be used for chat.",
            )

    # Run the full orchestration pipeline in its own DB session
    # (pipeline manages its own session lifecycle internally for
    # multi-step operations that may need partial commits)
    from .orchestrator.pipeline import OrchestrationPipeline

    pipeline = OrchestrationPipeline()

    async with get_db_session() as db:
        result = await pipeline.run(
            db=db,
            application_id=application_id,
            chat_session_id=payload.session_id,
            user_message=payload.message,
            user_id=user_id,
        )

    # Build the ChatMessageResponse for the assistant reply
    assistant_message = ChatMessageResponse(
        message_id=result.get("execution_trace", {}).get("message_id", ""),
        role="assistant",
        content=result.get("response", ""),
        enhanced_prompt=result.get("execution_trace", {}).get("enhanced_prompt"),
        execution_trace=result.get("execution_trace"),
        guardrail_flags=result.get("execution_trace", {}).get("guardrail_output"),
        created_at=datetime.now(timezone.utc),
    )

    # Build session metadata if this was a new session
    session_meta = None
    raw_meta = result.get("session_metadata")
    if raw_meta:
        session_meta = ChatSessionMetadata(
            app_name=raw_meta.get("app_name", ""),
            app_description=raw_meta.get("app_description"),
            app_logo_url=raw_meta.get("app_logo_url"),
            welcome_prompt=raw_meta.get("welcome_prompt"),
            disclaimer=raw_meta.get("disclaimer"),
            starter_questions=raw_meta.get("starter_questions", []),
        )

    return ChatResponse(
        session_id=result.get("session_id", ""),
        message=assistant_message,
        conversation_state=result.get("conversation_state", "awaiting_input"),
        session_metadata=session_meta,
    )


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/chat/stream -- A2UI SSE
# -------------------------------------------------------------------
@router.post("/{application_id}/chat/stream")
async def chat_stream(
    application_id: str,
    payload: ChatMessageRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Streaming version of chat that emits A2UI v0.8 JSONL events via SSE.

    The client receives real-time execution step updates as the pipeline
    runs.  Each SSE event ``data:`` field contains a single A2UI JSONL
    message (surfaceUpdate, dataModelUpdate, or beginRendering).

    The stream terminates after the pipeline completes and the final
    output has been sent.
    """
    import json as _json
    from apps.workflow.runtime.ws_manager import ws_manager

    user_id = _resolve_user_id(user)

    # Validate application exists and is published
    async with get_db_session() as db:
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")
        if app.status != "published":
            raise HTTPException(
                400,
                "Application is not published. Only published applications can be used for chat.",
            )

    run_id = uuid.uuid4().hex

    sse_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def sse_generator():
        try:
            while True:
                item = await asyncio.wait_for(sse_queue.get(), timeout=120.0)
                if item is None:  # sentinel -- stream ended
                    break
                # Ensure item is a string for the SSE data field
                if isinstance(item, dict):
                    data = _json.dumps(item, separators=(",", ":"))
                else:
                    data = str(item)
                yield f"data: {data}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {_json.dumps({'type': 'error', 'message': 'Stream timeout'})}\n\n"
        finally:
            ws_manager.unsubscribe_queue(run_id, sse_queue)

    from .orchestrator.pipeline import OrchestrationPipeline

    pipeline = OrchestrationPipeline()

    async def _run_pipeline():
        async with get_db_session() as db:
            await pipeline.run_stream(
                sse_queue=sse_queue,
                run_id=run_id,
                db=db,
                app_id=application_id,
                user_input=payload.message,
                session_id=payload.session_id,
                user_id=user_id,
                uploaded_file_ids=None,
            )

    asyncio.create_task(_run_pipeline())

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/chat/hitl-decide
# -- Human-in-the-Loop Decision
# -------------------------------------------------------------------
@router.post(
    "/{application_id}/chat/hitl-decide",
    response_model=HITLDecisionResponse,
)
async def hitl_decide(
    application_id: str,
    payload: HITLDecisionRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Submit a human decision for a workflow paused at a HITL node.

    The chat session must be in ``awaiting_clarification`` state with
    a valid ``hitl_state`` stored in ``context_data``.  The action must
    be one of the ``allowed_decisions`` defined by the HITL node.

    Supported actions:
        - **approve**: Resume the workflow from the interrupt point.
        - **reject**: Terminate the workflow and return to normal chat.
        - **edit**: Resume with modified payload.
        - **defer**: Keep the session paused for later review.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # 1. Validate application exists and is published
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")
        if app.status != "published":
            raise HTTPException(
                400,
                "Application is not published.",
            )

        # 2. Validate session exists and is in HITL review state
        session = await _chat_repo.get_session(db, payload.session_id, user_id)
        if session is None:
            raise HTTPException(404, "Chat session not found")

        # HITL uses 'awaiting_clarification' as DB state with hitl_state
        # in context_data to distinguish from regular clarification.
        context_data = session.context_data or {}
        hitl_state = context_data.get("hitl_state")

        if session.conversation_state != "awaiting_clarification" or not hitl_state:
            raise HTTPException(
                400,
                "Session is not awaiting HITL review. "
                "Current state does not have a pending HITL interrupt.",
            )

        # 3. Validate action is allowed
        interrupt_payload = hitl_state.get("interrupt_payload", {})
        allowed_decisions = interrupt_payload.get(
            "allowed_decisions", ["approve", "reject"]
        )

        # Also try to derive from the HITL config if available
        try:
            from apps.workflow.runtime.hitl import derive_allowed_decisions
            hitl_config = interrupt_payload.get("hitl_config")
            if hitl_config:
                allowed_decisions = derive_allowed_decisions(hitl_config)
        except ImportError:
            pass

        if payload.action not in allowed_decisions:
            raise HTTPException(
                400,
                f"Action '{payload.action}' is not allowed. "
                f"Allowed decisions: {allowed_decisions}",
            )

        run_id = hitl_state.get("run_id", "")
        workflow_id = hitl_state.get("workflow_id", "")
        fs_workflow_id = hitl_state.get("fs_workflow_id", "")

        # ----- DEFER -----
        if payload.action == "defer":
            # Session stays in awaiting_clarification with hitl_state
            defer_msg = "Decision deferred. The workflow remains paused for later review."
            if payload.rationale:
                defer_msg += f"\n\nRationale: {payload.rationale}"

            await _chat_repo.add_message(
                db, payload.session_id, "assistant", defer_msg,
                execution_trace={"type": "hitl_defer", "run_id": run_id},
            )

            return HITLDecisionResponse(
                status="deferred",
                message=defer_msg,
                session_id=payload.session_id,
                conversation_state="awaiting_hitl_review",
                execution_trace={"action": "defer", "run_id": run_id},
            )

        # ----- REJECT -----
        if payload.action == "reject":
            reject_msg = "Workflow execution was rejected by human reviewer."
            if payload.rationale:
                reject_msg += f"\n\nRationale: {payload.rationale}"

            await _chat_repo.add_message(
                db, payload.session_id, "assistant", reject_msg,
                execution_trace={"type": "hitl_reject", "run_id": run_id},
            )

            # Clear HITL state and return to normal
            await _chat_repo.update_session_state(
                db,
                payload.session_id,
                conversation_state="awaiting_input",
                context_data={},
            )

            return HITLDecisionResponse(
                status="rejected",
                message=reject_msg,
                session_id=payload.session_id,
                conversation_state="awaiting_input",
                execution_trace={"action": "reject", "run_id": run_id},
            )

        # ----- APPROVE or EDIT -----
        # Both require resuming the workflow via WorkflowExecutor
        from .orchestrator.skill_executor import SkillExecutor

        executor = SkillExecutor._resolve_workflow_executor()

        resume_payload = payload.payload or {}
        if payload.rationale:
            resume_payload["rationale"] = payload.rationale

        # resume_workflow is synchronous (LangGraph invoke)
        resume_result = await asyncio.to_thread(
            executor.resume_workflow,
            workflow_id=fs_workflow_id or workflow_id,
            run_id=run_id,
            action=payload.action,
            payload=resume_payload,
            execution_mode="draft",
        )

        if resume_result.get("status") == "failed":
            error_msg = (
                f"Workflow resume failed: {resume_result.get('error', 'unknown')}"
            )
            await _chat_repo.add_message(
                db, payload.session_id, "assistant", error_msg,
                execution_trace={"type": "hitl_resume_failed", "run_id": run_id},
            )
            await _chat_repo.update_session_state(
                db,
                payload.session_id,
                conversation_state="awaiting_input",
                context_data={},
            )
            return HITLDecisionResponse(
                status="failed",
                message=error_msg,
                session_id=payload.session_id,
                conversation_state="awaiting_input",
                execution_trace={
                    "action": payload.action,
                    "run_id": run_id,
                    "error": resume_result.get("error"),
                },
            )

        # Check if we hit ANOTHER interrupt (chained HITL nodes)
        if resume_result.get("status") == "interrupted":
            new_interrupt = resume_result.get("interrupt", {})
            new_allowed = new_interrupt.get(
                "allowed_decisions", ["approve", "reject"]
            )

            new_hitl_state = {
                "hitl_state": {
                    "run_id": resume_result.get("run_id", run_id),
                    "workflow_id": workflow_id,
                    "fs_workflow_id": fs_workflow_id,
                    "interrupt_payload": new_interrupt,
                    "completed_steps": hitl_state.get("completed_steps", []),
                    "remaining_plan": hitl_state.get("remaining_plan", []),
                    "execution_plan": hitl_state.get("execution_plan", []),
                    "execution_strategy": hitl_state.get("execution_strategy", "single"),
                    "enhanced_prompt": hitl_state.get("enhanced_prompt", ""),
                }
            }

            re_interrupt_msg = (
                "Workflow paused again for human review."
                f"\n\n**{new_interrupt.get('title', 'Review Required')}**: "
                f"{new_interrupt.get('message', 'Please review and decide.')}"
            )

            await _chat_repo.add_message(
                db, payload.session_id, "assistant", re_interrupt_msg,
                execution_trace={
                    "type": "hitl_interrupt",
                    "hitl_context": {
                        "title": new_interrupt.get("title"),
                        "message": new_interrupt.get("message"),
                        "priority": new_interrupt.get("priority"),
                        "allowed_decisions": new_allowed,
                        "run_id": resume_result.get("run_id", run_id),
                        "node_outputs": new_interrupt.get("node_outputs"),
                    },
                },
            )

            await _chat_repo.update_session_state(
                db,
                payload.session_id,
                conversation_state="awaiting_clarification",
                context_data=new_hitl_state,
            )

            return HITLDecisionResponse(
                status="interrupted",
                message=re_interrupt_msg,
                session_id=payload.session_id,
                conversation_state="awaiting_hitl_review",
                execution_trace={
                    "action": payload.action,
                    "hitl_context": {
                        "title": new_interrupt.get("title"),
                        "message": new_interrupt.get("message"),
                        "priority": new_interrupt.get("priority"),
                        "allowed_decisions": new_allowed,
                        "run_id": resume_result.get("run_id", run_id),
                    },
                },
            )

        # ----- COMPLETED -----
        # Workflow resumed and finished successfully.
        # Now we must also execute any remaining steps from the original
        # execution plan (the interrupted workflow was only ONE step).
        output = resume_result.get("output", {})
        resumed_output_text = SkillExecutor._extract_output_text(output)

        # Record the resumed workflow's output alongside prior completed steps
        all_completed_outputs = []
        for cs in hitl_state.get("completed_steps", []):
            all_completed_outputs.append(cs.get("output", ""))
        all_completed_outputs.append(resumed_output_text)

        # Execute remaining plan steps (if any) that were after the
        # interrupted workflow in the original execution plan.
        remaining_plan = hitl_state.get("remaining_plan", [])
        remaining_execution_log: List[Dict[str, Any]] = []

        if remaining_plan:
            skill_executor = SkillExecutor()
            enhanced_prompt = hitl_state.get("enhanced_prompt", "")

            # Feed the resumed workflow output as input to remaining steps
            remaining_result = await skill_executor.execute_plan(
                db=db,
                execution_plan=remaining_plan,
                execution_strategy="sequential",
                user_input=resumed_output_text,
            )

            # Check if remaining steps also hit a HITL interrupt
            if remaining_result.get("hitl_interrupted"):
                remaining_interrupt = remaining_result.get("interrupt_payload", {})
                remaining_int_ctx = remaining_interrupt.get("interrupt", {})

                # Merge all completed so far
                for cs in remaining_result.get("completed_steps", []):
                    all_completed_outputs.append(cs.get("output", ""))

                new_hitl_state = {
                    "hitl_state": {
                        "run_id": remaining_result.get("run_id", ""),
                        "workflow_id": remaining_result.get("workflow_id", ""),
                        "fs_workflow_id": remaining_interrupt.get("fs_workflow_id", ""),
                        "interrupt_payload": remaining_int_ctx,
                        "completed_steps": (
                            hitl_state.get("completed_steps", [])
                            + [{"step": "resumed", "skill_name": "resumed_workflow", "output": resumed_output_text}]
                            + remaining_result.get("completed_steps", [])
                        ),
                        "remaining_plan": remaining_result.get("remaining_plan", []),
                        "execution_plan": hitl_state.get("execution_plan", []),
                        "execution_strategy": hitl_state.get("execution_strategy", "sequential"),
                        "enhanced_prompt": enhanced_prompt,
                    }
                }

                new_allowed = remaining_int_ctx.get(
                    "allowed_decisions", ["approve", "reject"]
                )
                re_interrupt_msg = (
                    "Workflow paused again for human review."
                    f"\n\n**{remaining_int_ctx.get('title', 'Review Required')}**: "
                    f"{remaining_int_ctx.get('message', 'Please review and decide.')}"
                )

                await _chat_repo.add_message(
                    db, payload.session_id, "assistant", re_interrupt_msg,
                    execution_trace={
                        "type": "hitl_interrupt",
                        "hitl_context": {
                            "title": remaining_int_ctx.get("title"),
                            "message": remaining_int_ctx.get("message"),
                            "priority": remaining_int_ctx.get("priority"),
                            "allowed_decisions": new_allowed,
                            "run_id": remaining_result.get("run_id", ""),
                        },
                    },
                )
                await _chat_repo.update_session_state(
                    db, payload.session_id,
                    conversation_state="awaiting_clarification",
                    context_data=new_hitl_state,
                )

                return HITLDecisionResponse(
                    status="interrupted",
                    message=re_interrupt_msg,
                    session_id=payload.session_id,
                    conversation_state="awaiting_hitl_review",
                    execution_trace={
                        "action": payload.action,
                        "hitl_context": {
                            "title": remaining_int_ctx.get("title"),
                            "message": remaining_int_ctx.get("message"),
                            "priority": remaining_int_ctx.get("priority"),
                            "allowed_decisions": new_allowed,
                            "run_id": remaining_result.get("run_id", ""),
                        },
                    },
                )

            # Remaining steps completed normally — append their output
            remaining_final = remaining_result.get("final_output", "")
            if remaining_final:
                all_completed_outputs.append(remaining_final)
            remaining_execution_log = remaining_result.get("execution_log", [])

        # Merge all outputs into final response
        final_output = all_completed_outputs[-1] if all_completed_outputs else resumed_output_text

        # Post-guardrails on the final output
        from .orchestrator.guardrails import GuardrailsEngine

        guardrails_engine = GuardrailsEngine()
        guardrail_categories: List[str] = []
        if app.guardrail_links:
            from echolib.models.application_lookups import AppGuardrailCategory
            from echolib.repositories.base import safe_uuid

            cat_ids = []
            for link in app.guardrail_links:
                uid = safe_uuid(str(link.guardrail_category_id))
                if uid:
                    cat_ids.append(uid)
            if cat_ids:
                cat_result = await db.execute(
                    select(AppGuardrailCategory).where(
                        AppGuardrailCategory.guardrail_category_id.in_(cat_ids)
                    )
                )
                guardrail_categories = [
                    c.name for c in cat_result.scalars().all()
                ]

        post_guard = guardrails_engine.post_process(
            final_output,
            guardrail_categories=guardrail_categories,
            guardrail_text=app.guardrail_text,
        )
        final_output = post_guard.sanitized_text

        # Save the final assistant message
        await _chat_repo.add_message(
            db, payload.session_id, "assistant", final_output,
            execution_trace={
                "type": "hitl_resumed_completion",
                "run_id": run_id,
                "action": payload.action,
                "remaining_steps_executed": len(remaining_plan),
                "remaining_execution_log": remaining_execution_log,
                "guardrail_output": {
                    "is_safe": post_guard.is_safe,
                    "violations": post_guard.violations,
                },
            },
        )

        # Clear HITL state and return to normal
        await _chat_repo.update_session_state(
            db,
            payload.session_id,
            conversation_state="awaiting_input",
            context_data={},
        )

        return HITLDecisionResponse(
            status="completed",
            message=final_output,
            session_id=payload.session_id,
            conversation_state="awaiting_input",
            execution_trace={
                "action": payload.action,
                "run_id": run_id,
                "status": "completed",
                "remaining_steps_executed": len(remaining_plan),
            },
        )


# -------------------------------------------------------------------
# GET /api/applications/{application_id}/chat/history -- List Sessions
# -------------------------------------------------------------------
@router.get(
    "/{application_id}/chat/history",
    response_model=List[ChatSessionResponse],
)
async def list_chat_sessions(
    application_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    List all chat sessions for an application, scoped to the current user.

    Returns sessions ordered by ``updated_at`` descending (most recent first).
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Validate application exists
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        sessions = await _chat_repo.list_sessions(db, application_id, user_id)

        return [
            ChatSessionResponse(
                chat_session_id=str(s.chat_session_id),
                application_id=str(s.application_id),
                title=s.title,
                conversation_state=s.conversation_state,
                started_at=s.started_at,
                updated_at=s.updated_at,
                message_count=len(s.messages) if s.messages else 0,
            )
            for s in sessions
        ]


# -------------------------------------------------------------------
# GET /api/applications/{application_id}/chat/sessions/{session_id}
# -- Get Messages for a Session
# -------------------------------------------------------------------
@router.get(
    "/{application_id}/chat/sessions/{session_id}",
    response_model=List[ChatMessageResponse],
)
async def get_chat_messages(
    application_id: str,
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get all messages for a specific chat session.

    Validates that the session belongs to the current user and the
    specified application before returning messages.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Verify session exists and is owned by user
        session = await _chat_repo.get_session(db, session_id, user_id)
        if session is None:
            raise HTTPException(404, "Chat session not found")

        # Verify session belongs to this application
        if str(session.application_id) != application_id:
            raise HTTPException(404, "Chat session not found for this application")

        messages = await _chat_repo.get_messages(db, session_id)

        return [
            ChatMessageResponse(
                message_id=str(msg.message_id),
                role=msg.role,
                content=msg.content,
                enhanced_prompt=msg.enhanced_prompt,
                execution_trace=msg.execution_trace,
                guardrail_flags=msg.guardrail_flags,
                created_at=msg.created_at,
            )
            for msg in messages
        ]


# -------------------------------------------------------------------
# DELETE /api/applications/{application_id}/chat/sessions/{session_id}
# -- Close a Chat Session
# -------------------------------------------------------------------
@router.delete(
    "/{application_id}/chat/sessions/{session_id}",
    status_code=204,
)
async def close_chat_session(
    application_id: str,
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Close (soft-delete) a chat session.

    The session is marked as closed and will no longer accept new messages.
    Existing messages are preserved for audit purposes.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Verify session exists and is owned by user
        session = await _chat_repo.get_session(db, session_id, user_id)
        if session is None:
            raise HTTPException(404, "Chat session not found")

        # Verify session belongs to this application
        if str(session.application_id) != application_id:
            raise HTTPException(404, "Chat session not found for this application")

        await _chat_repo.close_session(db, session_id)

    return Response(status_code=204)


# -------------------------------------------------------------------
# POST /api/applications/{application_id}/chat/upload -- Document Upload
# -------------------------------------------------------------------
@router.post(
    "/{application_id}/chat/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_chat_document(
    application_id: str,
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(
        None, description="Optional chat session UUID to bind the document to."
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Upload a document for RAG processing within an application.

    The application must have ``upload_enabled == True``. The file is
    saved to disk, an ``ApplicationDocument`` record is created, and the
    document is immediately chunked, embedded, and indexed into a
    session-scoped FAISS index for retrieval at query time.

    An optional ``session_id`` query parameter binds the document to a
    specific chat session. If omitted, a new chat session is created
    automatically.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Validate application exists
        app = await _app_repo.get_application(db, application_id, user_id)
        if app is None:
            raise HTTPException(404, "Application not found")

        # Validate upload is enabled
        if not app.upload_enabled:
            raise HTTPException(
                400,
                "Document upload is not enabled for this application.",
            )

        # If session_id is provided, verify it exists and belongs to user
        if session_id:
            session = await _chat_repo.get_session(db, session_id, user_id)
            if session is None:
                raise HTTPException(404, "Chat session not found")
            if str(session.application_id) != application_id:
                raise HTTPException(
                    404, "Chat session not found for this application"
                )
        else:
            # Auto-create a new session if none provided
            session = await _chat_repo.create_session(
                db, application_id, user_id,
                title="Document Upload Session",
            )
            session_id = str(session.chat_session_id)

        # Generate a unique stored filename
        original_filename = file.filename or "unnamed"
        stored_filename = f"{application_id}_{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(UPLOADS_DIR, stored_filename)

        # Read file content and save to disk
        content = await file.read()
        size_bytes = len(content)

        with open(file_path, "wb") as f:
            f.write(content)

        # Create document record in database (status: processing)
        doc = await _chat_repo.create_document(
            db,
            application_id=application_id,
            chat_session_id=session_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            mime_type=file.content_type,
            size_bytes=size_bytes,
        )
        doc_id = str(doc.document_id)

        # Update status to processing
        await _chat_repo.update_document_status(db, doc_id, "processing")

        # Trigger RAG processing: chunk + embed + index
        chunk_count = 0
        processing_status = "ready"
        try:
            from echolib.di import container as di_container
            rag_manager = di_container.resolve('rag.session_manager')
            result = rag_manager.process_document(
                session_id=session_id,
                file_path=file_path,
                filename=original_filename,
            )
            chunk_count = result.get("chunk_count", 0)
            processing_status = result.get("status", "ready")
        except Exception as exc:
            logger.exception(
                "RAG processing failed for document %s in session %s",
                doc_id, session_id,
            )
            processing_status = "failed"

        # Update document record with final status and chunk count
        await _chat_repo.update_document_status(
            db, doc_id, processing_status, chunk_count=chunk_count
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=original_filename,
            processing_status=processing_status,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            chunk_count=chunk_count,
        )


# ===================================================================
# Lookup / Catalog Router
# ===================================================================

lookup_router = APIRouter(tags=["Application Lookups"])


@lookup_router.get("/api/llms", response_model=List[LookupResponse])
async def list_llms():
    """List all available LLMs for the application wizard."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppLlm).order_by(AppLlm.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=str(row.llm_id), name=row.name)
            for row in rows
        ]


@lookup_router.get("/api/data-sources", response_model=List[LookupResponse])
async def list_data_sources():
    """List all available data sources."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppDataSource).order_by(AppDataSource.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=row.data_source_id, name=row.name)
            for row in rows
        ]


@lookup_router.get("/api/designations", response_model=List[LookupResponse])
async def list_designations():
    """List all available designations."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppDesignation).order_by(AppDesignation.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=str(row.designation_id), name=row.name)
            for row in rows
        ]


@lookup_router.get("/api/business-units", response_model=List[LookupResponse])
async def list_business_units():
    """List all available business units."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppBusinessUnit).order_by(AppBusinessUnit.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=str(row.business_unit_id), name=row.name)
            for row in rows
        ]


@lookup_router.get("/api/tags", response_model=List[LookupResponse])
async def list_tags():
    """List all available tags."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppTag).order_by(AppTag.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=str(row.tag_id), name=row.name)
            for row in rows
        ]


@lookup_router.get("/api/personas", response_model=List[LookupResponse])
async def list_personas():
    """List all available personas."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppPersona).order_by(AppPersona.name.asc())
        )
        rows = result.scalars().all()
        return [
            LookupResponse(id=str(row.persona_id), name=row.name)
            for row in rows
        ]


@lookup_router.get(
    "/api/guardrail-categories", response_model=List[LookupResponse]
)
async def list_guardrail_categories():
    """List all available guardrail categories."""
    async with get_db_session() as db:
        result = await db.execute(
            select(AppGuardrailCategory).order_by(
                AppGuardrailCategory.name.asc()
            )
        )
        rows = result.scalars().all()
        return [
            LookupResponse(
                id=str(row.guardrail_category_id), name=row.name
            )
            for row in rows
        ]


# -------------------------------------------------------------------
# POST /api/llms/seed -- Seed default LLMs if table is empty
# -------------------------------------------------------------------
@lookup_router.post("/api/llms/seed", response_model=List[LookupResponse])
async def seed_llms():
    """Seed the app_llms table with default LLMs if empty."""
    async with get_db_session() as db:
        existing = await db.execute(select(func.count()).select_from(AppLlm))
        count = existing.scalar() or 0
        if count > 0:
            result = await db.execute(select(AppLlm).order_by(AppLlm.name.asc()))
            rows = result.scalars().all()
            return [LookupResponse(id=str(row.llm_id), name=row.name) for row in rows]

        defaults = [
            AppLlm(name="GPT-4o", provider="openai", model_name="gpt-4o", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY", is_default=False),
            AppLlm(name="GPT-4o Mini", provider="openai", model_name="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY", is_default=False),
            AppLlm(name="Claude Sonnet", provider="anthropic", model_name="claude-sonnet-4-20250514", base_url="https://api.anthropic.com", api_key_env="ANTHROPIC_API_KEY", is_default=False),
            AppLlm(name="OpenRouter Free", provider="openrouter", model_name="liquid/lfm-2.5-1.2b-instruct:free", base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY", is_default=True),
            AppLlm(name="Ollama Local", provider="ollama", model_name="llama3", base_url="http://localhost:11434/v1", api_key_env=None, is_default=False),
        ]
        db.add_all(defaults)
        await db.flush()

        result = await db.execute(select(AppLlm).order_by(AppLlm.name.asc()))
        rows = result.scalars().all()
        return [LookupResponse(id=str(row.llm_id), name=row.name) for row in rows]


# -------------------------------------------------------------------
# POST /api/data-sources/seed -- Seed default data sources if empty
# -------------------------------------------------------------------
@lookup_router.post("/api/data-sources/seed", response_model=List[LookupResponse])
async def seed_data_sources():
    """Seed the app_data_sources table with defaults if empty."""
    async with get_db_session() as db:
        existing = await db.execute(select(func.count()).select_from(AppDataSource))
        count = existing.scalar() or 0
        if count > 0:
            result = await db.execute(select(AppDataSource).order_by(AppDataSource.name.asc()))
            rows = result.scalars().all()
            return [LookupResponse(id=row.data_source_id, name=row.name) for row in rows]

        defaults = [
            AppDataSource(data_source_id="mcp_filesystem", name="Local Filesystem", kind="mcp", source_metadata={"description": "Access local files"}),
            AppDataSource(data_source_id="mcp_web_search", name="Web Search", kind="mcp", source_metadata={"description": "Search the web"}),
            AppDataSource(data_source_id="api_rest", name="REST API", kind="api", source_metadata={"description": "Connect to REST APIs"}),
        ]
        db.add_all(defaults)
        await db.flush()

        result = await db.execute(select(AppDataSource).order_by(AppDataSource.name.asc()))
        rows = result.scalars().all()
        return [LookupResponse(id=row.data_source_id, name=row.name) for row in rows]


# ===================================================================
# Skills Router
# ===================================================================

skills_router = APIRouter(tags=["Application Skills"])


@skills_router.get("/api/skills", response_model=List[SkillResponse])
async def list_skills(
    user: UserContext = Depends(get_current_user),
):
    """
    List all workflows and agents for the current user.

    Queries both the ``workflows`` and ``agents`` PostgreSQL tables
    and returns a unified list labeled by skill_type.
    """
    user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        skills: List[SkillResponse] = []

        # ---- Workflows ----
        wf_stmt = (
            select(Workflow)
            .where(
                Workflow.user_id == func.cast(user_id, Workflow.user_id.type),
                Workflow.status != "archived",
                Workflow.is_deleted == False,  # noqa: E712
            )
            .order_by(Workflow.name.asc())
        )
        wf_result = await db.execute(wf_stmt)
        for wf in wf_result.scalars().all():
            description = None
            if wf.definition and isinstance(wf.definition, dict):
                description = wf.definition.get("description")
            skills.append(
                SkillResponse(
                    skill_id=str(wf.workflow_id),
                    skill_type="workflow",
                    name=wf.name,
                    description=description,
                )
            )

        # ---- Agents ----
        agt_stmt = (
            select(Agent)
            .where(
                Agent.user_id == func.cast(user_id, Agent.user_id.type),
                Agent.is_deleted == False,  # noqa: E712
            )
            .order_by(Agent.name.asc())
        )
        agt_result = await db.execute(agt_stmt)
        for agt in agt_result.scalars().all():
            name = agt.name
            description = None
            if agt.definition and isinstance(agt.definition, dict):
                name = agt.definition.get("name", agt.name)
                description = agt.definition.get("role")
            skills.append(
                SkillResponse(
                    skill_id=str(agt.agent_id),
                    skill_type="agent",
                    name=name,
                    description=description,
                )
            )

        return skills

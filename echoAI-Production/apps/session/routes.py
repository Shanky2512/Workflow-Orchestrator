"""
EchoAI Session API Routes

Provides CRUD operations for chat sessions with:
- Embedded messages (JSONB)
- Memcached caching with PostgreSQL fallback
- Tool selection per session
- Tool configuration overrides

Routes:
    POST /sessions/login - Dev login (create token)
    GET /sessions/me - Get current user info
    POST /sessions/logout - Invalidate session

    GET /sessions - List user's chat sessions
    POST /sessions - Create new session
    GET /sessions/{id} - Get session with embedded messages
    DELETE /sessions/{id} - Soft delete session
    PATCH /sessions/{id}/title - Update session title
    POST /sessions/{id}/tools - Set selected tools for session
    PUT /sessions/{id}/tools/{tool_id}/config - Set tool config override
    DELETE /sessions/{id}/tools/{tool_id}/config - Remove tool config override

    POST /sessions/{id}/messages - Add message to session
    GET /sessions/{id}/messages - Get session messages
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from echolib.di import container
from echolib.security import get_current_user, require_user, create_token
from echolib.types import UserContext
from echolib.database import get_db_session
from echolib.repositories.session_repo import SessionRepository
from echolib.repositories.session_tool_config_repo import SessionToolConfigRepository
from echolib.repositories.base import (
    is_valid_uuid,
    DEFAULT_USER_ID,
    ensure_default_user_exists,
)

logger = logging.getLogger(__name__)


def _resolve_user_id(user: UserContext) -> str:
    """
    Resolve the effective user ID for database queries.

    For anonymous or invalid user IDs, returns DEFAULT_USER_ID to ensure
    sessions created by /chat/start (which uses DEFAULT_USER_ID) are visible.

    Args:
        user: User context from authentication

    Returns:
        Valid user ID string (either user's actual ID or DEFAULT_USER_ID)
    """
    user_id = user.user_id
    # Use default user for anonymous or invalid UUIDs
    if not user_id or user_id == "anonymous" or not is_valid_uuid(user_id):
        return DEFAULT_USER_ID
    return user_id

router = APIRouter(prefix="/sessions", tags=["SessionApi"])


# ==============================================================================
# Pydantic Models for Request/Response
# ==============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""

    title: Optional[str] = Field(default="New Chat", max_length=255)
    context_type: str = Field(
        default="general",
        description="Session context type: general, workflow, agent, workflow_design",
    )
    context_id: Optional[str] = Field(
        default=None, description="UUID of workflow or agent for context binding"
    )
    workflow_mode: Optional[str] = Field(
        default=None, description="For workflow contexts: draft, test, final"
    )
    workflow_version: Optional[str] = Field(default=None)
    initial_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    state_schema: Optional[Dict[str, Any]] = Field(default_factory=dict)
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UpdateTitleRequest(BaseModel):
    """Request to update session title."""

    title: str = Field(..., min_length=1, max_length=255)


class SetToolsRequest(BaseModel):
    """Request to set selected tools for session."""

    tool_ids: List[str] = Field(default_factory=list)


class ToolConfigRequest(BaseModel):
    """Request to set tool configuration override."""

    config: Dict[str, Any] = Field(default_factory=dict)


class AddMessageRequest(BaseModel):
    """Request to add a message to session."""

    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., min_length=1)
    agent_id: Optional[str] = Field(default=None)
    run_id: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    """Session summary for list responses."""

    session_id: str
    title: str
    context_type: str
    context_id: Optional[str]
    workflow_mode: Optional[str]
    message_count: int
    created_at: str
    last_activity: str


class SessionDetail(BaseModel):
    """Full session detail with messages."""

    session_id: str
    user_id: str
    title: str
    context_type: str
    context_id: Optional[str]
    workflow_mode: Optional[str]
    workflow_version: Optional[str]
    messages: List[Dict[str, Any]]
    selected_tool_ids: List[str]
    context_data: Dict[str, Any]
    variables: Dict[str, Any]
    state_schema: Dict[str, Any]
    run_ids: List[str]
    created_at: str
    last_activity: str


# ==============================================================================
# Repository Helpers
# ==============================================================================


def get_session_repo() -> SessionRepository:
    """Get session repository instance."""
    return SessionRepository()


def get_tool_config_repo() -> SessionToolConfigRepository:
    """Get session tool config repository instance."""
    return SessionToolConfigRepository()


# ==============================================================================
# Legacy Auth Routes (Backward Compatibility)
# ==============================================================================


@router.post("/login")
async def login(email: str):
    """
    Dev login endpoint - creates a token and session.

    This is for development only. In production, use proper OAuth/OIDC.
    """
    from echolib.adapters import MemcachedSessionStore

    user_id = f"usr_{uuid4().hex[:12]}"
    tok = create_token(user_id, email)

    # Try to use session store if available
    try:
        store = container.resolve("session.store")
        s = store.createSession(user_id, {"email": email})
        return {"token": tok, "session": s.model_dump()}
    except Exception:
        # Session store not configured, just return token
        return {"token": tok, "user_id": user_id, "email": email}


@router.get("/me")
async def me(user: UserContext = Depends(get_current_user)):
    """Get current user info."""
    return user.model_dump()


@router.post("/logout")
async def logout(session_id: str):
    """Invalidate session."""
    try:
        store = container.resolve("session.store")
        store.invalidateSession(session_id)
        return {"ok": True}
    except Exception:
        return {"ok": True, "message": "Session store not available"}


# ==============================================================================
# Session CRUD Routes
# ==============================================================================


@router.get("", response_model=Dict[str, Any])
async def list_sessions(
    context_type: Optional[str] = Query(
        default=None, description="Filter by context type"
    ),
    context_id: Optional[str] = Query(
        default=None, description="Filter by context ID"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: UserContext = Depends(get_current_user),
):
    """
    List user's chat sessions.

    Supports filtering by context_type and context_id for finding
    sessions bound to specific workflows or agents.
    """
    repo = get_session_repo()

    # FIX Bug 5: Use default user for anonymous/invalid users
    # Sessions created by /chat/start use DEFAULT_USER_ID, so queries should too
    effective_user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        if context_type:
            sessions = await repo.list_by_context(
                db=db,
                user_id=effective_user_id,
                context_type=context_type,
                context_id=context_id,
                limit=limit,
                offset=offset,
            )
        else:
            sessions = await repo.list_by_user(
                db=db, user_id=effective_user_id, limit=limit, offset=offset
            )

        return {
            "sessions": [s.to_summary_dict() for s in sessions],
            "count": len(sessions),
            "limit": limit,
            "offset": offset,
        }


@router.post("", response_model=Dict[str, Any])
async def create_session(
    request: CreateSessionRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Create a new chat session.

    Session can be bound to different contexts:
    - general: Free-form chat (default)
    - workflow: Testing a specific workflow
    - agent: Direct chat with an agent
    - workflow_design: Iterating on workflow design
    """
    repo = get_session_repo()

    # Validate context_type
    valid_context_types = {"general", "workflow", "agent", "workflow_design"}
    if request.context_type not in valid_context_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid context_type. Must be one of: {valid_context_types}",
        )

    # Validate workflow_mode if provided
    if request.workflow_mode:
        valid_modes = {"draft", "test", "final"}
        if request.workflow_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow_mode. Must be one of: {valid_modes}",
            )

    # Build session data
    session_data = {
        "title": request.title or "New Chat",
        "context_type": request.context_type,
        "context_id": UUID(request.context_id) if request.context_id else None,
        "workflow_mode": request.workflow_mode,
        "workflow_version": request.workflow_version,
        "context_data": request.initial_context or {},
        "state_schema": request.state_schema or {},
        "variables": request.variables or {},
        "messages": [],
        "selected_tool_ids": [],
        "run_ids": [],
    }

    # Resolve user ID (handles anonymous users)
    effective_user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Ensure default user exists if using DEFAULT_USER_ID (FK constraint)
        if effective_user_id == DEFAULT_USER_ID:
            await ensure_default_user_exists(db)

        session = await repo.create(db=db, user_id=effective_user_id, data=session_data)

        # Add system message for workflow contexts
        if request.context_type == "workflow" and request.context_id:
            system_msg = {
                "id": f"msg_{uuid4().hex[:12]}",
                "role": "system",
                "content": f"Started testing workflow: {request.context_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": None,
                "run_id": None,
                "metadata": {},
            }
            await repo.add_message(
                db=db,
                session_id=str(session.session_id),
                user_id=effective_user_id,
                message=system_msg,
            )
            # Refresh to get updated session
            session = await repo.get_by_id(
                db=db, session_id=str(session.session_id), user_id=effective_user_id
            )

        return {
            "session_id": str(session.session_id),
            "title": session.title,
            "context_type": session.context_type,
            "context_id": str(session.context_id) if session.context_id else None,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "message": "Session created successfully",
        }


@router.get("/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get session with embedded messages.

    Uses Memcached caching with PostgreSQL fallback.
    Cache key format: session:{session_id}
    """
    repo = get_session_repo()

    # FIX Bug 5: Use default user for anonymous/invalid users
    effective_user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        # Use dict version for efficient cache retrieval
        session_dict = await repo.get_by_id_dict(
            db=db, session_id=session_id, user_id=effective_user_id
        )

        if not session_dict:
            raise HTTPException(status_code=404, detail="Session not found")

        return session_dict


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Soft delete a session.

    Sets is_deleted=True and invalidates cache.
    """
    repo = get_session_repo()

    # FIX: Use default user for anonymous/invalid users
    # Sessions created by /chat/start use DEFAULT_USER_ID, so queries should too
    effective_user_id = _resolve_user_id(user)

    async with get_db_session() as db:
        deleted = await repo.delete(db=db, id=session_id, user_id=effective_user_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "success": True,
            "session_id": session_id,
            "message": "Session deleted",
        }


@router.patch("/{session_id}/title")
async def update_session_title(
    session_id: str,
    request: UpdateTitleRequest,
    user: UserContext = Depends(get_current_user),
):
    """Update session title."""
    repo = get_session_repo()

    async with get_db_session() as db:
        session = await repo.update(
            db=db,
            id=session_id,
            user_id=user.user_id,
            updates={"title": request.title},
        )

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": str(session.session_id),
            "title": session.title,
            "message": "Title updated",
        }


# ==============================================================================
# Tool Selection Routes
# ==============================================================================


@router.post("/{session_id}/tools")
async def set_session_tools(
    session_id: str,
    request: SetToolsRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Set selected tools for session.

    Replaces the current tool selection with the provided list.
    """
    repo = get_session_repo()

    async with get_db_session() as db:
        session = await repo.update_tools(
            db=db,
            session_id=session_id,
            user_id=user.user_id,
            tool_ids=request.tool_ids,
        )

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": str(session.session_id),
            "selected_tool_ids": [str(tid) for tid in session.selected_tool_ids]
            if session.selected_tool_ids
            else [],
            "message": f"Selected {len(request.tool_ids)} tools",
        }


@router.put("/{session_id}/tools/{tool_id}/config")
async def set_tool_config(
    session_id: str,
    tool_id: str,
    request: ToolConfigRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Set tool configuration override for this session.

    Allows customizing tool behavior per-session without modifying
    the global tool definition.
    """
    session_repo = get_session_repo()
    config_repo = get_tool_config_repo()

    async with get_db_session() as db:
        # Verify session exists and belongs to user
        session = await session_repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Upsert tool config
        config = await config_repo.upsert_config(
            db=db,
            session_id=session_id,
            tool_id=tool_id,
            user_id=user.user_id,
            config=request.config,
        )

        return {
            "session_id": session_id,
            "tool_id": tool_id,
            "config": config.config_overrides if config else {},
            "message": "Tool config updated",
        }


@router.delete("/{session_id}/tools/{tool_id}/config")
async def delete_tool_config(
    session_id: str,
    tool_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Remove tool configuration override for this session."""
    session_repo = get_session_repo()
    config_repo = get_tool_config_repo()

    async with get_db_session() as db:
        # Verify session exists and belongs to user
        session = await session_repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Delete config
        deleted = await config_repo.delete_config(
            db=db, session_id=session_id, tool_id=tool_id, user_id=user.user_id
        )

        return {
            "session_id": session_id,
            "tool_id": tool_id,
            "deleted": deleted,
            "message": "Tool config removed" if deleted else "Config not found",
        }


@router.get("/{session_id}/tools/configs")
async def get_session_tool_configs(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get all tool configuration overrides for a session."""
    session_repo = get_session_repo()
    config_repo = get_tool_config_repo()

    async with get_db_session() as db:
        # Verify session exists and belongs to user
        session = await session_repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get all configs for session
        configs = await config_repo.get_configs_for_session(
            db=db, session_id=session_id, user_id=user.user_id
        )

        return {
            "session_id": session_id,
            "configs": [
                {
                    "tool_id": str(c.tool_id),
                    "config_overrides": c.config_overrides,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in configs
            ],
            "count": len(configs),
        }


# ==============================================================================
# Message Routes
# ==============================================================================


@router.post("/{session_id}/messages")
async def add_message(
    session_id: str,
    request: AddMessageRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Add a message to session.

    Messages are embedded in the session's messages JSONB array.
    """
    repo = get_session_repo()

    # Validate role
    valid_roles = {"user", "assistant", "system"}
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {valid_roles}",
        )

    # Build message dict
    message = {
        "id": f"msg_{uuid4().hex[:12]}",
        "role": request.role,
        "content": request.content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": request.agent_id,
        "run_id": request.run_id,
        "metadata": request.metadata or {},
    }

    async with get_db_session() as db:
        session = await repo.add_message(
            db=db,
            session_id=session_id,
            user_id=user.user_id,
            message=message,
        )

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": str(session.session_id),
            "message_id": message["id"],
            "message_count": len(session.messages) if session.messages else 0,
        }


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: UserContext = Depends(get_current_user),
):
    """
    Get messages from a session.

    Returns messages in chronological order with optional pagination.
    """
    repo = get_session_repo()

    async with get_db_session() as db:
        session_dict = await repo.get_by_id_dict(
            db=db, session_id=session_id, user_id=user.user_id
        )

        if not session_dict:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = session_dict.get("messages", [])

        # Apply pagination
        paginated = messages[offset : offset + limit]

        return {
            "session_id": session_id,
            "messages": paginated,
            "total": len(messages),
            "limit": limit,
            "offset": offset,
        }


# ==============================================================================
# Context Update Routes
# ==============================================================================


@router.patch("/{session_id}/context")
async def update_session_context(
    session_id: str,
    context_updates: Dict[str, Any],
    user: UserContext = Depends(get_current_user),
):
    """Update session context data (merge with existing)."""
    repo = get_session_repo()

    async with get_db_session() as db:
        # Get current session
        session = await repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Merge context
        current_context = session.context_data or {}
        current_context.update(context_updates)

        # Update session
        updated = await repo.update(
            db=db,
            id=session_id,
            user_id=user.user_id,
            updates={"context_data": current_context},
        )

        return {
            "session_id": session_id,
            "context_data": updated.context_data if updated else {},
            "message": "Context updated",
        }


@router.patch("/{session_id}/variables")
async def update_session_variables(
    session_id: str,
    variable_updates: Dict[str, Any],
    user: UserContext = Depends(get_current_user),
):
    """Update session workflow variables (merge with existing)."""
    repo = get_session_repo()

    async with get_db_session() as db:
        # Get current session
        session = await repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Merge variables
        current_vars = session.variables or {}
        current_vars.update(variable_updates)

        # Update session
        updated = await repo.update(
            db=db,
            id=session_id,
            user_id=user.user_id,
            updates={"variables": current_vars},
        )

        return {
            "session_id": session_id,
            "variables": updated.variables if updated else {},
            "message": "Variables updated",
        }


@router.post("/{session_id}/run")
async def add_run_id(
    session_id: str,
    run_id: str = Query(..., description="Execution run ID to add"),
    user: UserContext = Depends(get_current_user),
):
    """Add an execution run ID to the session."""
    repo = get_session_repo()

    async with get_db_session() as db:
        # Get current session
        session = await repo.get_by_id(
            db=db, session_id=session_id, user_id=user.user_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add run ID to list
        run_ids = list(session.run_ids) if session.run_ids else []
        run_uuid = UUID(run_id)
        if run_uuid not in run_ids:
            run_ids.append(run_uuid)

        # Update session
        updated = await repo.update(
            db=db,
            id=session_id,
            user_id=user.user_id,
            updates={"run_ids": run_ids},
        )

        return {
            "session_id": session_id,
            "run_ids": [str(rid) for rid in updated.run_ids] if updated else [],
            "message": "Run ID added",
        }

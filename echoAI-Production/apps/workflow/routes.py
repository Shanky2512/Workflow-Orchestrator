from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request, BackgroundTasks
from echolib.di import container
from echolib.security import get_current_user, require_user, user_context
from echolib.types import UserContext
from echolib.types import *

from echolib.services import WorkflowService
from echolib.types import (
    Agent,
    Workflow,
    WorkflowValidationRequest,
    SaveFinalRequest,
    CloneWorkflowRequest,
    ExecuteWorkflowRequest
)

# Import default user constants and helper from shared location
from echolib.repositories.base import (
    DEFAULT_USER_ID,
    ensure_default_user_exists,
)

import logging
_db_logger = logging.getLogger(__name__)


def _prefixed_id_to_uuid(prefixed_id: str) -> str:
    """
    Convert a prefixed hex ID (e.g., wf_xxx, agt_xxx) to a proper UUID string.

    The application uses IDs like 'wf_550e8400e29b41d4a716446655440000'
    but the database expects UUIDs like '550e8400-e29b-41d4-a716-446655440000'.

    Args:
        prefixed_id: ID with prefix (wf_, agt_, session_, etc.)

    Returns:
        Properly formatted UUID string with hyphens

    Raises:
        ValueError: If the hex portion is not exactly 32 characters
    """
    # Remove any known prefix
    hex_part = prefixed_id
    for prefix in ("wf_", "agt_", "session_", "run_", "tool_"):
        if prefixed_id.startswith(prefix):
            hex_part = prefixed_id[len(prefix):]
            break

    # Validate hex length
    if len(hex_part) != 32:
        raise ValueError(f"Invalid ID format: {prefixed_id} (hex part must be 32 chars)")

    # Insert hyphens: 8-4-4-4-12
    return f"{hex_part[:8]}-{hex_part[8:12]}-{hex_part[12:16]}-{hex_part[16:20]}-{hex_part[20:]}"


router = APIRouter(prefix='/workflows', tags=['WorkflowApi'])

def svc() -> WorkflowService:
    return container.resolve('workflow.service')


def hitl_manager():
    """Get the singleton HITLManager (or HITLDBManager) from DI container."""
    return container.resolve('workflow.hitl')

# ==================== EXISTING ROUTES (UNCHANGED) ====================

@router.post('/create/prompt')
async def create_prompt(prompt: str, agents: list[Agent]):
    """Simple workflow creation (existing API)."""
    return svc().createFromPrompt(prompt, agents).model_dump()

@router.post('/create/canvas')
async def create_canvas(canvasJSON: dict):
    """Canvas-based workflow creation (existing API)."""
    return svc().createFromCanvas(canvasJSON).model_dump()

@router.post('/validate')
async def validate(workflow: Workflow):
    """Simple workflow validation (existing API)."""
    return svc().validate(workflow).model_dump()

# ==================== NEW ORCHESTRATOR ROUTES ====================

# Workflow Design
@router.post('/design/prompt')
async def design_from_prompt(
    request: Request,
    prompt: str = Query(None, description="Natural language prompt for workflow design")
):
    """
    Design workflow from natural language prompt.
    Returns draft workflow + agent definitions.

    Accepts TWO formats:
    1. Query parameters: POST /workflows/design/prompt?prompt=...
    2. Request body: POST /workflows/design/prompt with {"prompt": "...", "default_llm": {...}}

    Query parameters take precedence over body if both are provided.
    """
    try:
        # Determine prompt and default_llm from either query params or body
        final_prompt = prompt  # from query param
        final_llm = None

        # If no query param, try to parse body
        if not final_prompt:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    final_prompt = body.get("prompt")
                    final_llm = body.get("default_llm")
            except Exception:
                # No valid JSON body, that's okay if we have query params
                pass

        if not final_prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required. Provide via query parameter (?prompt=...) or request body ({\"prompt\": \"...\"})"
            )

        designer = container.resolve('workflow.designer')
        workflow, agents = designer.design_from_prompt(final_prompt, final_llm)
        return {
            "workflow": workflow,
            "agents": agents
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)

@router.post('/design/modify')
async def modify_workflow_from_prompt(request: dict):
    """
    Modify existing workflow via natural language.

    Uses LLM-based intent detection (not keyword matching) to understand the user's
    request and apply appropriate modifications to the workflow.

    Request:
    {
        "prompt": "user's natural language request",
        "existing_workflow": {...current workflow...},
        "existing_agents": {...current agents...},
        "conversation_history": [...] (optional)
    }

    Supported intents (detected by LLM from natural language):
    - add: Add a new agent (e.g., "I think we need someone to check the code")
    - remove: Remove an agent (e.g., "The executor seems redundant")
    - modify: Modify agent config (e.g., "Make the analyzer focus on security")
    - add_tool: Add tool to agent (e.g., "Add web search to the reviewer")
    - remove_tool: Remove tool (e.g., "Remove calculator from the first agent")
    - reorder: Change agent order (e.g., "Switch the order of last two agents")
    - change_topology: Change execution model (e.g., "Run them all at once")
    - create_new: Signal to create fresh workflow (e.g., "Let's start over")
    - execute: Signal execution request, not modification (e.g., "Run this analysis")

    Response:
    {
        "intent": "add|remove|modify|...",
        "confidence": 0.0-1.0,
        "reasoning": "LLM's interpretation explanation",
        "changes": {...specific changes...},
        "updated_workflow": {...} or null,
        "updated_agents": {...} or null,
        "success": true|false,
        "message": "Human-readable status message"
    }
    """
    try:
        prompt = request.get("prompt")
        existing_workflow = request.get("existing_workflow")
        existing_agents = request.get("existing_agents")
        conversation_history = request.get("conversation_history", [])

        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="prompt is required"
            )

        if not existing_workflow:
            raise HTTPException(
                status_code=400,
                detail="existing_workflow is required"
            )

        if existing_agents is None:
            raise HTTPException(
                status_code=400,
                detail="existing_agents is required (can be empty dict)"
            )

        designer = container.resolve('workflow.designer')
        result = designer.modify_from_prompt(
            user_prompt=prompt,
            existing_workflow=existing_workflow,
            existing_agents=existing_agents,
            conversation_history=conversation_history
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)


@router.post('/build')
async def build_workflow_manual(request: dict):
    """
    Build workflow manually with agent I/O schema specification.
    Allows reusing existing agents and defining their I/O schemas for this workflow.

    Request format:
    {
      "workflow": { ... },
      "agent_schemas": {
        "agt_001": {
          "input_schema": ["sales_data"],
          "output_schema": ["analysis"]
        },
        "agt_002": {
          "input_schema": ["analysis"],
          "output_schema": ["report"]
        }
      },
      "update_base_agents": false  // Optional: update base agent definitions
    }
    """
    try:
        workflow = request.get("workflow")
        agent_schemas = request.get("agent_schemas", {})
        update_base = request.get("update_base_agents", False)

        if not workflow:
            raise HTTPException(status_code=400, detail="workflow field required")

        # If update_base_agents is True, update the agent registry
        if update_base and agent_schemas:
            registry = container.resolve('agent.registry')
            for agent_id, schemas in agent_schemas.items():
                try:
                    registry.update_agent(agent_id, schemas)
                except Exception as e:
                    print(f"Warning: Failed to update agent {agent_id}: {e}")

        # Store agent schema overrides in workflow metadata
        if agent_schemas:
            if "metadata" not in workflow:
                workflow["metadata"] = {}
            workflow["metadata"]["agent_schemas"] = agent_schemas

        return {"workflow": workflow, "agent_schemas": agent_schemas}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Validation
@router.post('/validate/draft')
async def validate_draft(req: WorkflowValidationRequest):
    """Validate draft workflow (sync only, before HITL)."""
    try:
        validator = container.resolve('workflow.validator')
        result = validator.validate_draft(req.workflow, req.agents)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/validate/final')
async def validate_final(req: WorkflowValidationRequest):
    """Validate workflow after HITL (full async validation)."""
    try:
        validator = container.resolve('workflow.validator')
        result = await validator.validate_final(req.workflow, req.agents)

        if result.is_valid():
            # Mark as validated
            req.workflow["status"] = "validated"
            req.workflow["validation"] = {
                "validated_at": "placeholder_timestamp",  # TODO: Add real timestamp
                "validation_hash": "placeholder_hash"  # TODO: Add real hash
            }

        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Draft Listing
@router.get('/draft/list')
async def list_draft_workflows():
    """List all workflows in draft folder."""
    try:
        storage = container.resolve('workflow.storage')
        workflows = storage.list_draft_workflows()
        return {"workflows": workflows, "total": len(workflows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Storage
@router.post('/temp/save')
async def save_temp(workflow: dict):
    """Save workflow as temp for testing."""
    try:
        if workflow.get("status") != "validated":
            raise HTTPException(
                status_code=400,
                detail="Workflow must be validated before saving as temp"
            )

        storage = container.resolve('workflow.storage')
        result = storage.save_workflow(workflow, state="temp")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/temp')
async def load_temp(workflow_id: str):
    """Load temp workflow."""
    try:
        storage = container.resolve('workflow.storage')
        workflow = storage.load_workflow(workflow_id=workflow_id, state="temp")
        return workflow
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Temp workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete('/{workflow_id}/temp')
async def delete_temp(workflow_id: str):
    """Delete temp workflow."""
    try:
        storage = container.resolve('workflow.storage')
        storage.delete_workflow(workflow_id, state="temp")
        return {"message": "Temp workflow deleted", "workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/final/save')
async def save_final(req: SaveFinalRequest):
    """Save workflow as final (versioned, immutable)."""
    try:
        storage = container.resolve('workflow.storage')
        result = storage.save_final_workflow(req.workflow)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/final/{version}')
async def load_final(workflow_id: str, version: str):
    """Load specific final version."""
    try:
        storage = container.resolve('workflow.storage')
        workflow = storage.load_workflow(
            workflow_id=workflow_id,
            state="final",
            version=version
        )
        return workflow
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow version not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/{workflow_id}/versions')
async def list_versions(workflow_id: str):
    """List all final versions of a workflow."""
    try:
        storage = container.resolve('workflow.storage')
        versions = storage.list_versions(workflow_id)
        if not versions:
            raise HTTPException(status_code=404, detail="No versions found")
        return {
            "workflow_id": workflow_id,
            "versions": versions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/clone')
async def clone_final(req: CloneWorkflowRequest):
    """Clone final workflow to draft for editing."""
    try:
        storage = container.resolve('workflow.storage')
        cloned = storage.clone_final_to_draft(
            workflow_id=req.workflow_id,
            from_version=req.from_version
        )
        return {
            "message": "Workflow cloned to draft",
            "workflow_id": req.workflow_id,
            "base_version": req.from_version
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Final workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Load workflow by ID (tries draft → temp → final)
@router.get('/{workflow_id}')
async def load_workflow(workflow_id: str):
    """Load a workflow by ID, searching across draft/temp/final states."""
    try:
        storage = container.resolve('workflow.storage')
        agent_registry = container.resolve('agent.registry')

        workflow = None
        for state in ["draft", "temp", "final"]:
            try:
                workflow = storage.load_workflow(workflow_id=workflow_id, state=state)
                if workflow:
                    break
            except (FileNotFoundError, Exception):
                continue

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Handle both embedded agents and ID-referenced agents
        workflow_agents = workflow.get("agents", [])
        agents = {}

        # Check if agents are embedded (list of dicts) or referenced (list of IDs)
        if workflow_agents and isinstance(workflow_agents[0], dict):
            # New format: agents are embedded in workflow
            for agent_obj in workflow_agents:
                agent_id = agent_obj.get("agent_id")
                if agent_id:
                    agents[agent_id] = agent_obj
        else:
            # Old format: agents is list of IDs, need to load from registry
            agents = agent_registry.get_agents_for_workflow(workflow_agents)

        return {
            "workflow": workflow,
            "agents": agents
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Execution
@router.post('/execute')
async def execute_workflow(req: ExecuteWorkflowRequest):
    """Execute workflow (test or final mode)."""
    try:
        executor = container.resolve('workflow.executor')
        result = executor.execute_workflow(
            workflow_id=req.workflow_id,
            execution_mode=req.mode,
            version=req.version,
            input_payload=req.input_payload
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Visualization
@router.get('/{workflow_id}/graph')
async def get_graph(workflow_id: str, state: str = "temp"):
    """Get graph representation of workflow."""
    try:
        graph_mapper = container.resolve('workflow.graph_mapper')
        graph = graph_mapper.get_graph(workflow_id, state)
        return graph
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==================== NODE MAPPER ROUTES ====================

@router.post('/canvas/to-backend')
async def canvas_to_backend(request: dict):
    """
    Convert frontend canvas to backend workflow format.

    Request:
    {
      "canvas_nodes": [...],
      "connections": [...],
      "workflow_name": "Optional explicit name"
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')

        canvas_nodes = request.get("canvas_nodes", [])
        connections = request.get("connections", [])
        workflow_name = request.get("workflow_name")

        if not canvas_nodes:
            raise HTTPException(status_code=400, detail="canvas_nodes required")

        workflow, agents = node_mapper.map_frontend_to_backend(
            canvas_nodes=canvas_nodes,
            connections=connections,
            workflow_name=workflow_name,
            auto_generate_name=True
        )

        return {
            "workflow": workflow,
            "agents": agents
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/backend/to-canvas')
async def backend_to_canvas(request: dict):
    """
    Convert backend workflow to frontend canvas format.

    Request:
    {
      "workflow": {...},
      "agents": {...}
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')

        workflow = request.get("workflow")
        agents = request.get("agents")

        # Check workflow exists (required)
        if not workflow:
            raise HTTPException(
                status_code=400,
                detail="workflow is required"
            )

        # agents can be empty dict {} but must be present
        if agents is None:
            raise HTTPException(
                status_code=400,
                detail="agents is required (can be empty dict)"
            )

        canvas_nodes, connections = node_mapper.map_backend_to_frontend(
            workflow=workflow,
            agents_dict=agents
        )

        return {
            "canvas_nodes": canvas_nodes,
            "connections": connections,
            "workflow_name": workflow.get("name", "Untitled Workflow")
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"ERROR in backend_to_canvas: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@router.post('/canvas/save')
async def save_canvas_workflow(request: dict):
    """
    Save canvas workflow directly (converts + validates + saves).

    Implements dual-write strategy per db_plan.md:
    - DRAFT save: Filesystem + Database
    - TEMP save: Filesystem ONLY (no database)
    - FINAL save: Filesystem + Database (handled by /save-final endpoint)

    Request:
    {
      "canvas_nodes": [...],
      "connections": [...],
      "workflow_name": "Optional",
      "save_as": "draft|temp",
      "workflow_id": "Optional - preserves existing ID if provided"
    }
    """
    try:
        node_mapper = container.resolve('workflow.node_mapper')
        validator = container.resolve('workflow.validator')
        storage = container.resolve('workflow.storage')
        agent_registry = container.resolve('agent.registry')

        # Convert to backend format (preserve existing workflow_id if provided)
        workflow, agents = node_mapper.map_frontend_to_backend(
            canvas_nodes=request.get("canvas_nodes", []),
            connections=request.get("connections", []),
            workflow_name=request.get("workflow_name"),
            execution_model=request.get("execution_model"),
            workflow_id=request.get("workflow_id")
        )

        # Validate
        validation_result = validator.validate_draft(workflow, agents)
        if not validation_result.is_valid():
            return {
                "success": False,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings
            }

        # Register agents
        for agent_id, agent in agents.items():
            agent["agent_id"] = agent_id
            agent_registry.register_agent(agent)

        # Save workflow
        save_as = request.get("save_as", "draft")
        if save_as == "temp":
            workflow["status"] = "validated"

        # Filesystem save (always happens)
        result = storage.save_workflow(workflow, state=save_as)

        # Database save for DRAFT and FINAL states only (not TEMP per db_plan.md)
        db_saved = False
        if save_as in ("draft", "final"):
            try:
                from echolib.database import get_db_session
                from echolib.repositories.workflow_repo import WorkflowRepository

                workflow_repo = WorkflowRepository()

                # Prepare workflow data for database
                # Compute UUID for DB primary key separately so the
                # definition JSONB preserves the original wf_ prefixed ID
                import copy as _copy
                db_workflow_data = _copy.deepcopy(workflow)
                try:
                    db_uuid = _prefixed_id_to_uuid(workflow["workflow_id"])
                except ValueError as e:
                    _db_logger.warning(f"Cannot convert workflow_id for DB save: {e}")
                    raise

                # Convert embedded agent IDs to UUIDs for agent sync
                if db_workflow_data.get("agents"):
                    db_agents = []
                    for agent in db_workflow_data["agents"]:
                        agent_copy = agent.copy()
                        if agent_copy.get("agent_id"):
                            try:
                                agent_copy["agent_id"] = _prefixed_id_to_uuid(agent_copy["agent_id"])
                            except ValueError:
                                pass  # Keep original if conversion fails
                        db_agents.append(agent_copy)
                    db_workflow_data["agents"] = db_agents

                async with get_db_session() as db:
                    await workflow_repo.save_with_agents(
                        db=db,
                        user_id=DEFAULT_USER_ID,
                        workflow_data=db_workflow_data,
                        db_workflow_id=db_uuid
                    )
                db_saved = True
                _db_logger.info(f"Workflow {workflow['workflow_id']} saved to database")
            except Exception as e:
                # Log but don't fail - filesystem save already succeeded
                _db_logger.warning(f"Database save failed for workflow {workflow['workflow_id']}: {e}")
                # Continue without database save - graceful degradation

        return {
            "success": True,
            "workflow_id": workflow["workflow_id"],
            "workflow_name": workflow["name"],
            "state": save_as,
            "db_saved": db_saved,
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WORKFLOW CHAT (RUNTIME TESTING) ====================

@router.post('/chat/start')
async def start_chat_session(request: dict):
    """
    Start a new chat session for workflow testing.

    Creates both a filesystem session (for chat operations) and a database
    session (for history listing). Dual-write ensures history is visible
    while maintaining filesystem-based chat functionality.

    Request:
    {
      "workflow_id": "wf_xxx",
      "mode": "test",  // or "workflow_mode" for backward compatibility
      "version": null,
      "initial_context": {}  // or "context" for backward compatibility
    }
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()

        workflow_id = request.get("workflow_id")
        # Accept both 'mode' and 'workflow_mode' for backward compatibility
        mode = request.get("mode") or request.get("workflow_mode", "test")
        version = request.get("version")
        # Accept both 'initial_context' and 'context' for backward compatibility
        initial_context = request.get("initial_context") or request.get("context", {})

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")

        # Create filesystem session (for chat operations)
        session = session_manager.create_session(
            workflow_id=workflow_id,
            workflow_mode=mode,
            workflow_version=version,
            initial_context=initial_context
        )

        # Also create database session for history/listing
        # This enables the History button to show workflow sessions
        db_session_created = False
        try:
            from echolib.database import get_db_session
            from echolib.repositories.session_repo import SessionRepository

            session_repo = SessionRepository()

            # Convert workflow_id to UUID for context_id if possible
            context_id_uuid = None
            try:
                context_id_uuid = _prefixed_id_to_uuid(workflow_id)
            except ValueError:
                # If conversion fails, leave context_id as None
                _db_logger.warning(f"Could not convert workflow_id to UUID: {workflow_id}")

            # Prepare session data for database
            db_session_data = {
                "session_id": session.session_id,  # Already a proper UUID from ChatSessionManager
                "title": f"Workflow Test: {workflow_id}",
                "context_type": "workflow",
                "context_id": context_id_uuid,
                "workflow_mode": mode,
                "workflow_version": version,
                "messages": [],  # Will be populated by chat operations
                "context_data": initial_context or {},
                "variables": session.variables,
                "state_schema": session.state_schema,
            }

            async with get_db_session() as db:
                # Ensure default user exists before creating session (FK constraint)
                user_created = await ensure_default_user_exists(db)
                if not user_created:
                    _db_logger.error("Could not ensure default user exists, session creation may fail")

                await session_repo.create(
                    db=db,
                    user_id=DEFAULT_USER_ID,
                    data=db_session_data
                )
            db_session_created = True
            _db_logger.info(f"Database session created for workflow chat: {session.session_id}")
        except Exception as e:
            # Log with full traceback for debugging FK constraint issues
            _db_logger.error(f"Database session creation failed: {e}", exc_info=True)
            # Chat will still work via filesystem session

        return {
            "session_id": session.session_id,
            "workflow_id": session.workflow_id,
            "mode": session.workflow_mode,
            "created_at": session.created_at.isoformat(),
            "message": "Chat session started. Send messages to test workflow.",
            "db_session_created": db_session_created
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/chat/send')
async def send_chat_message(request: dict, background_tasks: BackgroundTasks):
    """
    Send message and execute workflow with real-time transparency.

    This endpoint implements the "dual endpoint" pattern:
    1. Returns run_id immediately (before execution)
    2. Client connects to WebSocket for real-time updates
    3. Execution runs in background

    Request:
    {
      "session_id": "session_xxx",
      "message": "Analyze this feedback: Great product!",
      "execute_workflow": true
    }

    Response (immediate):
    {
      "session_id": "session_xxx",
      "run_id": "run_xxx",
      "status": "executing",
      "message": "Connect to WebSocket for real-time updates."
    }

    WebSocket: ws://host/ws/execution/{run_id}
    """
    from .runtime.chat_session import ChatSessionManager
    from echolib.utils import new_id
    from echolib.config import settings
    from datetime import datetime, timezone

    try:
        session_manager = ChatSessionManager()
        executor = container.resolve('workflow.executor')

        session_id = request.get("session_id")
        message = request.get("message")
        execute_workflow = request.get("execute_workflow", True)

        if not session_id or not message:
            raise HTTPException(
                status_code=400,
                detail="session_id and message required"
            )

        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add user message to filesystem session
        session_manager.add_user_message(session_id, message)

        # Sync user message to database session and update title if first user message
        try:
            from echolib.database import get_db_session
            from echolib.repositories.session_repo import SessionRepository

            session_repo = SessionRepository()

            # Build message dict for database
            db_message = {
                "id": f"msg_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}",
                "role": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": None,
                "run_id": None,
                "metadata": {}
            }

            async with get_db_session() as db:
                # Ensure default user exists (handles edge cases where session creation failed)
                await ensure_default_user_exists(db)

                # Add user message to database session
                await session_repo.add_message(
                    db=db,
                    session_id=session_id,
                    user_id=DEFAULT_USER_ID,
                    message=db_message
                )

                # Check if this is the first user message by counting user messages
                # in the filesystem session (excluding system messages)
                user_messages = [m for m in session.messages if m.role == "user"]
                is_first_user_message = len(user_messages) == 1  # Just added one

                if is_first_user_message:
                    # Truncate message to 50 chars for title
                    truncated_title = message[:50] + "..." if len(message) > 50 else message
                    await session_repo.update(
                        db=db,
                        id=session_id,
                        user_id=DEFAULT_USER_ID,
                        updates={"title": truncated_title}
                    )
                    _db_logger.info(f"Updated session title to first message: {truncated_title}")

            _db_logger.debug(f"Synced user message to database for session {session_id}")
        except Exception as e:
            # Log with traceback for debugging
            _db_logger.warning(f"Failed to sync user message to database: {e}", exc_info=True)

        # Execute workflow if requested
        if execute_workflow:
            # Generate run_id BEFORE execution (dual endpoint pattern)
            run_id = new_id("run_")

            # Initialize step tracker if transparency is enabled
            if settings.transparency_enabled:
                from .runtime.transparency import step_tracker
                step_tracker.create_run(run_id, session.workflow_id)

            # Schedule background execution
            background_tasks.add_task(
                _execute_workflow_with_transparency,
                run_id=run_id,
                session=session,
                message=message,
                session_id=session_id,
                session_manager=session_manager,
                executor=executor
            )

            # Return immediately with run_id
            return {
                "session_id": session_id,
                "run_id": run_id,
                "status": "executing",
                "message": "Connect to WebSocket for real-time updates.",
                "ws_url": f"/ws/execution/{run_id}"
            }
        else:
            # Just acknowledge message
            return {
                "session_id": session_id,
                "status": "message_added",
                "message": "Message added to session"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_workflow_with_transparency(
    run_id: str,
    session,
    message: str,
    session_id: str,
    session_manager,
    executor
):
    """
    Execute workflow with transparency event emission.

    This runs in a background task after /chat/send returns.
    Emits run_started, run_completed, or run_failed events.
    """
    import asyncio
    import logging
    from echolib.config import settings

    logger = logging.getLogger(__name__)

    try:
        # Give the frontend time to connect to WebSocket before starting execution
        # This ensures real-time streaming instead of replaying all events at once
        await asyncio.sleep(0.5)  # 500ms delay for WebSocket connection

        # Emit run_started event
        if settings.transparency_enabled:
            from .runtime.transparency import step_tracker
            from .runtime.event_publisher import publish_run_started, publish_run_completed, publish_run_failed
            await publish_run_started(run_id, session.workflow_id)

        # Detect if workflow has HITL nodes — use HITL-aware executor if so
        _has_hitl_nodes = False
        try:
            _wf_storage = container.resolve('workflow.storage')
            try:
                _wf_def = _wf_storage.load_workflow(
                    workflow_id=session.workflow_id, state="draft"
                )
            except FileNotFoundError:
                _wf_def = _wf_storage.load_workflow(
                    workflow_id=session.workflow_id, state="temp"
                )
            _has_hitl_nodes = any(
                (a.get("metadata", {}).get("node_type") == "HITL")
                for a in _wf_def.get("agents", [])
                if isinstance(a, dict)
            )
        except Exception:
            pass  # If detection fails, use standard executor

        # Execute workflow in a SEPARATE THREAD to keep event loop free
        # This is CRITICAL for real-time WebSocket streaming - without this,
        # the sync workflow blocks the event loop and WebSocket sends timeout
        _exec_method = (
            executor.execute_workflow_hitl if _has_hitl_nodes
            else executor.execute_workflow
        )
        result = await asyncio.to_thread(
            _exec_method,
            workflow_id=session.workflow_id,
            execution_mode=session.workflow_mode,
            version=session.workflow_version,
            input_payload={
                "message": message,
                "context": session.context,
                "run_id": run_id  # Pass run_id for transparency
            }
        )

        # Handle HITL interrupt — workflow paused, notify frontend
        if result.get("status") == "interrupted":
            interrupt_payload = result.get("interrupt", {})
            logger.info(
                f"Workflow paused at HITL node for run {run_id}"
            )
            if settings.transparency_enabled:
                from .runtime.transparency import step_tracker
                from .runtime.event_publisher import publish_run_completed
                # Emit a completion event with interrupted status so frontend
                # can switch to HITL review UI
                await publish_run_completed(run_id, {
                    "status": "interrupted",
                    "interrupt": interrupt_payload,
                })

            # Add a message to the session indicating HITL pause
            session_manager.add_assistant_message(
                session_id=session_id,
                message="Workflow paused for human review. Please check the HITL review panel.",
                run_id=run_id,
                metadata={
                    "execution_status": "interrupted",
                    "interrupt": interrupt_payload,
                }
            )
            return  # Do NOT continue — workflow is paused

        # Check if execution failed BEFORE processing output
        if result.get("status") == "failed":
            error_msg = result.get("error", "Unknown workflow execution error")
            logger.error(f"Workflow execution failed for run {run_id}: {error_msg}")

            if settings.transparency_enabled:
                from .runtime.transparency import step_tracker
                from .runtime.event_publisher import publish_run_failed
                step_tracker.fail_run(run_id)
                await publish_run_failed(run_id, error_msg)

            # Still add a message to the session with the error
            session_manager.add_assistant_message(
                session_id=session_id,
                message=f"⚠️ Workflow execution failed: {error_msg}",
                run_id=run_id,
                metadata={"execution_status": "failed", "error": error_msg}
            )
            return

        # Extract response from workflow output
        output = result.get("output", {})

        # Try to extract meaningful response from output
        if "crew_result" in output and output["crew_result"]:
            response_text = output["crew_result"]
        elif "result" in output and output["result"]:
            response_text = output["result"]
        elif "hierarchical_output" in output and output["hierarchical_output"]:
            response_text = output["hierarchical_output"]
        elif "parallel_output" in output and output["parallel_output"]:
            response_text = output["parallel_output"]
        elif "message" in output:
            response_text = output["message"]
        elif "messages" in output and output["messages"]:
            last_msg = output["messages"][-1]
            if isinstance(last_msg, dict):
                response_text = last_msg.get("content", str(last_msg))
            else:
                response_text = str(last_msg)
        else:
            response_text = f"Workflow executed: {str(output)}"

        # Add assistant response to filesystem session
        session_manager.add_assistant_message(
            session_id=session_id,
            message=response_text,
            run_id=run_id,
            metadata={"execution_status": result.get("status")}
        )

        # Sync assistant response to database session
        try:
            from echolib.database import get_db_session
            from echolib.repositories.session_repo import SessionRepository
            from echolib.repositories.base import DEFAULT_USER_ID, ensure_default_user_exists
            from datetime import datetime, timezone

            session_repo = SessionRepository()

            # Build assistant message dict for database
            db_assistant_message = {
                "id": f"msg_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}",
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": None,
                "run_id": run_id,
                "metadata": {"execution_status": result.get("status")}
            }

            async with get_db_session() as db:
                # Ensure default user exists (handles edge cases)
                await ensure_default_user_exists(db)

                await session_repo.add_message(
                    db=db,
                    session_id=session_id,
                    user_id=DEFAULT_USER_ID,
                    message=db_assistant_message
                )

            logger.debug(f"Synced assistant message to database for session {session_id}")
        except Exception as e:
            # Log with traceback for debugging
            logger.warning(f"Failed to sync assistant message to database: {e}", exc_info=True)

        # Emit run_completed event
        if settings.transparency_enabled:
            step_tracker.complete_run(run_id, response_text)
            await publish_run_completed(run_id, output)

    except Exception as e:
        # Emit run_failed event
        if settings.transparency_enabled:
            from .runtime.transparency import step_tracker
            from .runtime.event_publisher import publish_run_failed
            step_tracker.fail_run(run_id)
            await publish_run_failed(run_id, str(e))

        # Log the error (background task can't raise to client)
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Background workflow execution failed for run {run_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


@router.get('/chat/history/{session_id}')
async def get_chat_history(session_id: str):
    """
    Get chat history for a session.

    Returns:
    {
      "session_id": "...",
      "workflow_id": "...",
      "messages": [...],
      "context": {...}
    }
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXECUTION TRANSPARENCY ====================

@router.get('/execution/{run_id}/steps')
async def get_execution_steps(run_id: str):
    """
    Get current step state for a running or recently completed execution.

    This endpoint is for late-joining clients who connect to a run already
    in progress. It returns the current state of all steps.

    Response:
    {
      "run_id": "run_xxx",
      "workflow_id": "wf_xxx",
      "started_at": "ISO 8601",
      "completed_at": "ISO 8601" | null,
      "steps": [
        {
          "step_id": "step_xxx",
          "step_name": "Agent Name",
          "step_type": "agent",
          "status": "completed",
          "input_summary": {...},
          "output_summary": {...},
          "started_at": "ISO 8601",
          "completed_at": "ISO 8601",
          "error": null
        }
      ]
    }
    """
    from .runtime.transparency import step_tracker

    context = step_tracker.get_run(run_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail="Run not found or expired. Runs are kept in memory for 60 seconds after completion."
        )

    return {
        "run_id": run_id,
        "workflow_id": context.workflow_id,
        "started_at": context.started_at.isoformat(),
        "completed_at": context.completed_at.isoformat() if context.completed_at else None,
        "steps": [context.steps[sid].to_dict() for sid in context.step_order]
    }


@router.get('/debug/buffer/{run_id}')
async def debug_buffer_status(run_id: str):
    """
    Debug endpoint to check WebSocket event buffer status.

    Use this to verify that events are being buffered correctly
    and to check the state of the buffer for a specific run.

    NOTE: This endpoint is for debugging only. Remove in production.
    """
    from .runtime.ws_manager import ws_manager
    from .runtime.transparency import step_tracker

    # Get buffer info
    has_buffered = ws_manager.has_buffered_events(run_id)
    buffer_size = ws_manager.get_buffer_size(run_id)
    connection_count = ws_manager.get_connection_count(run_id)

    # Get run context info
    context = step_tracker.get_run(run_id)
    run_status = "not_found"
    step_count = 0
    if context:
        run_status = "completed" if context.completed_at else "running"
        step_count = len(context.step_order)

    return {
        "run_id": run_id,
        "buffer": {
            "has_events": has_buffered,
            "event_count": buffer_size
        },
        "websocket": {
            "connection_count": connection_count,
            "has_connections": connection_count > 0
        },
        "run_context": {
            "status": run_status,
            "step_count": step_count
        }
    }


@router.get('/chat/sessions')
async def list_chat_sessions(workflow_id: str = None, limit: int = 50):
    """
    List chat sessions.

    Query params:
    - workflow_id: Filter by workflow
    - limit: Max sessions to return
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        sessions = session_manager.list_sessions(
            workflow_id=workflow_id,
            limit=limit
        )

        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "workflow_id": s.workflow_id,
                    "mode": s.workflow_mode,
                    "message_count": len(s.messages),
                    "created_at": s.created_at.isoformat(),
                    "last_activity": s.last_activity.isoformat()
                }
                for s in sessions
            ],
            "count": len(sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{workflow_id}/chat/history')
async def get_workflow_chat_history(
    workflow_id: str,
    limit: int = Query(default=20, le=100)
):
    """
    Get chat history for a specific workflow.

    Returns sessions that are bound to this workflow (context_type='workflow').
    This enables workflow-scoped chat history in the UI.

    Args:
        workflow_id: Workflow ID (e.g., wf_xxx)
        limit: Maximum sessions to return (default 20, max 100)

    Returns:
        {
            "sessions": [...],  # List of session summaries
            "count": int,       # Number of sessions returned
            "workflow_id": str  # The workflow ID queried
        }
    """
    try:
        from echolib.repositories.session_repo import SessionRepository
        from echolib.database import get_db_session

        session_repo = SessionRepository()

        # Convert workflow_id to UUID format for database query
        try:
            context_uuid = _prefixed_id_to_uuid(workflow_id)
        except ValueError as e:
            _db_logger.warning(f"Invalid workflow_id format: {workflow_id} - {e}")
            # Return empty result for invalid IDs instead of error
            return {
                "sessions": [],
                "count": 0,
                "workflow_id": workflow_id
            }

        async with get_db_session() as db:
            await ensure_default_user_exists(db)
            sessions = await session_repo.list_by_context(
                db=db,
                user_id=DEFAULT_USER_ID,
                context_type="workflow",
                context_id=context_uuid,
                limit=limit
            )

        return {
            "sessions": [s.to_summary_dict() for s in sessions],
            "count": len(sessions),
            "workflow_id": workflow_id
        }

    except Exception as e:
        _db_logger.error(f"Failed to get workflow chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/chat/{session_id}')
async def delete_chat_session(session_id: str):
    """
    Delete a chat session.
    """
    try:
        from .runtime.chat_session import ChatSessionManager

        session_manager = ChatSessionManager()
        session_manager.delete_session(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "message": "Session deleted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WORKFLOW SAVE (AFTER TESTING) ====================

@router.post('/save-final')
async def save_tested_workflow(request: dict):
    """
    Save workflow as final after testing.

    Request:
    {
      "workflow_id": "wf_xxx",
      "version": "1.0",
      "notes": "Tested and approved"
    }
    """
    try:
        storage = container.resolve('workflow.storage')

        workflow_id = request.get("workflow_id")
        version = request.get("version", "1.0")
        notes = request.get("notes", "")

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")

        # Load temp workflow
        workflow = storage.load_workflow(
            workflow_id=workflow_id,
            state="temp"
        )

        # Update version and notes
        workflow["version"] = version
        workflow["metadata"]["save_notes"] = notes
        workflow["status"] = "validated"

        # Save as final
        result = storage.save_final_workflow(workflow)

        return {
            "success": True,
            "workflow_id": workflow_id,
            "version": version,
            "path": result["path"],
            "message": "Workflow saved as final"
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Temp workflow not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HITL (HUMAN-IN-THE-LOOP) ====================

@router.get('/hitl/status/{run_id}')
async def get_hitl_status(run_id: str):
    """
    Get HITL status for a run.

    Returns:
    {
      "run_id": "run_xxx",
      "state": "waiting_for_human",
      "blocked_at": "agent_id",
      "allowed_actions": ["approve", "reject", "modify", "defer"],
      "has_pending_review": true
    }
    """
    try:
        hitl = hitl_manager()
        status = hitl.get_status(run_id)

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/context/{run_id}')
async def get_hitl_context(run_id: str):
    """
    Get full HITL context for decision-making.

    Returns:
    {
      "workflow_id": "wf_xxx",
      "run_id": "run_xxx",
      "blocked_at": "agent_id",
      "agent_output": {...},
      "tools_used": [...],
      "execution_metrics": {...},
      "state_snapshot": {...},
      "previous_decisions": [...]
    }
    """
    try:
        hitl = hitl_manager()
        context = hitl.get_context(run_id)

        if not context:
            raise HTTPException(status_code=404, detail="HITL context not found")

        return context.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/approve')
async def approve_hitl(request: dict):
    """
    Approve workflow execution at HITL checkpoint.

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "rationale": "Looks good, proceed"
    }

    Response:
    {
      "action": "approve",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "approved",
      "can_resume": true
    }
    """
    try:
        hitl = hitl_manager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        result = hitl.approve(run_id=run_id, actor=actor, rationale=rationale)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/reject')
async def reject_hitl(request: dict):
    """
    Reject workflow execution at HITL checkpoint.

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "rationale": "Output doesn't meet requirements"
    }

    Response:
    {
      "action": "reject",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "rejected",
      "can_resume": false,
      "terminated": true
    }
    """
    try:
        hitl = hitl_manager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        result = hitl.reject(run_id=run_id, actor=actor, rationale=rationale)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/modify')
async def modify_hitl(request: dict):
    """
    Modify workflow/agent configuration at HITL checkpoint (TEMP workflows only).

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "changes": {
        "agent_id": "agt_001",
        "llm": {"temperature": 0.5},
        "prompt": "Updated prompt"
      },
      "rationale": "Need higher temperature for creativity"
    }

    Response:
    {
      "action": "modify",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "modified",
      "changes": {...},
      "validation_required": true,
      "can_resume": false
    }
    """
    try:
        hitl = hitl_manager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        changes = request.get("changes", {})
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        if not changes:
            raise HTTPException(status_code=400, detail="changes required")

        result = hitl.modify(
            run_id=run_id,
            actor=actor,
            changes=changes,
            rationale=rationale
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/hitl/defer')
async def defer_hitl(request: dict):
    """
    Defer HITL decision (postpone approval).

    Request:
    {
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "defer_until": "2026-01-20T10:00:00Z",
      "rationale": "Need to review with team"
    }

    Response:
    {
      "action": "defer",
      "run_id": "run_xxx",
      "actor": "user@example.com",
      "status": "deferred",
      "defer_until": "2026-01-20T10:00:00Z",
      "can_resume": false
    }
    """
    try:
        from datetime import datetime

        hitl = hitl_manager()

        run_id = request.get("run_id")
        actor = request.get("actor", "unknown")
        defer_until_str = request.get("defer_until")
        rationale = request.get("rationale")

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")

        defer_until = None
        if defer_until_str:
            defer_until = datetime.fromisoformat(defer_until_str.replace('Z', '+00:00'))

        result = hitl.defer(
            run_id=run_id,
            actor=actor,
            defer_until=defer_until,
            rationale=rationale
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/pending')
async def list_pending_hitl_reviews():
    """
    List all workflows waiting for HITL review.

    Response:
    [
      {
        "run_id": "run_xxx",
        "workflow_id": "wf_xxx",
        "blocked_at": "agent_id",
        "created_at": "2026-01-17T10:00:00Z"
      }
    ]
    """
    try:
        hitl = hitl_manager()
        pending = hitl.list_pending_reviews()

        return {"pending_reviews": pending, "count": len(pending)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/hitl/decisions/{run_id}')
async def get_hitl_decisions(run_id: str):
    """
    Get HITL decision audit trail for a run.

    Response:
    [
      {
        "decision_id": "hitl_dec_xxx",
        "run_id": "run_xxx",
        "action": "approve",
        "actor": "user@example.com",
        "timestamp": "2026-01-17T10:05:00Z",
        "rationale": "Looks good"
      }
    ]
    """
    try:
        hitl = hitl_manager()
        decisions = hitl.get_decisions(run_id)

        return {
            "run_id": run_id,
            "decisions": [d.to_dict() for d in decisions],
            "count": len(decisions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HITL RESUME (Direct, no Celery) ====================

async def _resume_workflow_direct(
    run_id: str,
    workflow_id: str,
    action: str,
    payload: dict = None,
):
    """
    Resume a HITL-paused workflow directly (without Celery).

    Runs executor.resume_workflow in a background thread so the event loop
    stays free for WebSocket streaming of transparency events.

    After resume completes, publishes the result (completed or another
    interrupt) via the event publisher.
    """
    import asyncio
    import logging
    from echolib.config import settings

    logger = logging.getLogger(__name__)

    if not workflow_id:
        logger.error(f"[HITL-RESUME] Cannot resume run '{run_id}': no workflow_id")
        return

    logger.info(
        f"[HITL-RESUME] Directly resuming workflow '{workflow_id}' "
        f"(run_id={run_id}, action={action})"
    )

    try:
        executor = container.resolve('workflow.executor')

        # Run in background thread to keep event loop free
        result = await asyncio.to_thread(
            executor.resume_workflow,
            workflow_id=workflow_id,
            run_id=run_id,
            action=action,
            payload=payload or {},
            execution_mode="draft",
        )

        logger.info(
            f"[HITL-RESUME] Resume result for run '{run_id}': "
            f"status={result.get('status')}"
        )

        # Publish result via transparency events
        if settings.transparency_enabled:
            from .runtime.event_publisher import publish_run_completed, publish_run_failed

            if result.get("status") == "failed":
                await publish_run_failed(run_id, result.get("error", "Unknown error"))
            elif result.get("status") == "interrupted":
                # Chained HITL: resume hit another interrupt node.
                # Pass status + interrupt payload so publish_run_completed
                # emits the correct HITL event to the frontend.
                await publish_run_completed(run_id, {
                    "status": "interrupted",
                    "interrupt": result.get("interrupt", {}),
                })
            else:
                await publish_run_completed(run_id, result.get("output", result))

    except Exception as e:
        logger.error(
            f"[HITL-RESUME] Direct resume failed for run '{run_id}': {e}",
            exc_info=True,
        )


# ==================== HITL UNIFIED DECIDE ROUTE ====================

@router.post('/hitl/decide')
async def hitl_decide(request: dict):
    """
    Unified HITL decision endpoint.

    Dispatches to approve/reject/edit/defer based on the action field.
    Validates that the action is allowed per the workflow's hitl_config
    (allowed_decisions enforcement).

    Request:
    {
      "run_id": "run_xxx",
      "action": "approve" | "reject" | "edit" | "defer",
      "actor": "user@example.com",
      "rationale": "Optional comment",
      "payload": {}  // For edit: changes dict. For defer: defer_until.
    }

    Response varies by action (same as individual endpoints).

    Raises:
        403: If action is not in allowed_decisions for this HITL node.
        400: If run_id or action is missing.
    """
    try:
        from .runtime.hitl import derive_allowed_decisions

        hitl = hitl_manager()

        run_id = request.get("run_id")
        action = request.get("action")
        actor = request.get("actor", "unknown")
        rationale = request.get("rationale")
        payload = request.get("payload", {})

        if not run_id:
            raise HTTPException(status_code=400, detail="run_id required")
        if not action:
            raise HTTPException(status_code=400, detail="action required")

        # Normalize action
        action = action.strip().lower()

        # Validate action is allowed
        # Try to get hitl_config from the HITL context
        context = hitl.get_context(run_id)
        if context and context.state_snapshot:
            hitl_config = context.state_snapshot.get("hitl_config", {})
            allowed = derive_allowed_decisions(hitl_config)
            if action not in allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Action '{action}' is not allowed. Permitted: {allowed}"
                )

        # Dispatch to appropriate handler
        if action == "approve":
            result = hitl.approve(run_id=run_id, actor=actor, rationale=rationale)

            # Trigger resume via Celery if HITLDBManager
            if hasattr(hitl, 'trigger_resume'):
                resume_info = hitl.trigger_resume(
                    run_id=run_id, action="approve", payload=payload
                )
                result["resume_info"] = resume_info

                # If Celery not available, resume directly via executor
                if resume_info.get("status") == "resume_pending":
                    import asyncio
                    asyncio.ensure_future(
                        _resume_workflow_direct(
                            run_id=run_id,
                            workflow_id=resume_info.get("workflow_id"),
                            action="approve",
                            payload=payload,
                        )
                    )

            return result

        elif action == "reject":
            result = hitl.reject(run_id=run_id, actor=actor, rationale=rationale)

            # Trigger resume (will record termination, no actual resume)
            if hasattr(hitl, 'trigger_resume'):
                resume_info = hitl.trigger_resume(
                    run_id=run_id, action="reject", payload=payload
                )
                result["resume_info"] = resume_info

            return result

        elif action in ("edit", "modify"):
            changes = payload.get("changes", payload)
            if not changes:
                raise HTTPException(status_code=400, detail="payload.changes required for edit action")

            result = hitl.modify(
                run_id=run_id, actor=actor, changes=changes, rationale=rationale
            )

            # Trigger resume with edit payload
            if hasattr(hitl, 'trigger_resume'):
                resume_info = hitl.trigger_resume(
                    run_id=run_id, action="edit", payload=payload
                )
                result["resume_info"] = resume_info

                # If Celery not available, resume directly via executor
                if resume_info.get("status") == "resume_pending":
                    import asyncio
                    asyncio.ensure_future(
                        _resume_workflow_direct(
                            run_id=run_id,
                            workflow_id=resume_info.get("workflow_id"),
                            action="edit",
                            payload=payload,
                        )
                    )

            return result

        elif action == "defer":
            from datetime import datetime

            defer_until = None
            defer_until_str = payload.get("defer_until")
            if defer_until_str:
                defer_until = datetime.fromisoformat(
                    defer_until_str.replace('Z', '+00:00')
                )

            result = hitl.defer(
                run_id=run_id, actor=actor,
                defer_until=defer_until, rationale=rationale
            )

            # Trigger resume (will record deferral, no actual resume)
            if hasattr(hitl, 'trigger_resume'):
                resume_info = hitl.trigger_resume(
                    run_id=run_id, action="defer", payload=payload
                )
                result["resume_info"] = resume_info

            return result

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: '{action}'. Valid: approve, reject, edit, defer"
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

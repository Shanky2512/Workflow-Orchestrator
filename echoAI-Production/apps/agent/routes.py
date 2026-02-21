from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from echolib.di import container
from echolib.security import user_context
from echolib.types import *
from echolib.services import AgentService
from echolib.types import AgentTemplate, Agent
from typing import AsyncGenerator, Dict, Any, List, Optional
import copy
import json
import inspect
import logging

_db_logger = logging.getLogger(__name__)


def _prefixed_id_to_uuid(prefixed_id: str) -> str:
    """
    Convert a prefixed hex ID (e.g., agt_xxx) to a proper UUID string.

    The application uses IDs like 'agt_550e8400e29b41d4a716446655440000'
    but the database expects UUIDs like '550e8400-e29b-41d4-a716-446655440000'.
    """
    hex_part = prefixed_id
    for prefix in ("wf_", "agt_", "session_", "run_", "tool_"):
        if prefixed_id.startswith(prefix):
            hex_part = prefixed_id[len(prefix):]
            break
    if len(hex_part) != 32:
        raise ValueError(f"Invalid ID format: {prefixed_id} (hex part must be 32 chars)")
    return f"{hex_part[:8]}-{hex_part[8:12]}-{hex_part[12:16]}-{hex_part[16:20]}-{hex_part[20:]}"


router = APIRouter(prefix='/agents', tags=['AgentApi'])

def svc() -> AgentService:
    return container.resolve('agent.service')


# ==================== EXISTING ROUTES (UNCHANGED) ====================

@router.post('/create/prompt')
async def create_prompt(request: dict):
    """
    Create or update agent from prompt with template matching.
    Accepts a JSON body with:
    - prompt (str, required): Natural language description of the desired agent.
    - agent_id (str, optional): If provided, treats this as an UPDATE request.
    - name (str, optional): Override agent name.
    - icon (str, optional): Override icon.
    - role (str, optional): Override role.
    - description (str, optional): Override description.
    - tools (list, optional): Override tools list.
    - variables (list, optional): Override variables.
    - settings (dict, optional): Override settings.

    Response includes "action" field indicating what happened:
    - "CREATE_AGENT": New agent was created
    - "UPDATE_AGENT": Existing agent was updated (when agent_id provided)
    - "AGENT_EXISTS": Similar agent already exists (can be configured/modified)

    The service will:
    1. If agent_id provided: Update existing agent, preserving name/ID
    2. Analyze intent from the prompt.
    3. Check for existing similar agents (returns AGENT_EXISTS if found).
    4. Match against predefined templates.
    5. Build from template if matched, else use LLM generation.
    6. Register the agent in the registry.
    7. Return the full agent definition with action type.
    """
    try:
        prompt = request.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        agent_id = request.get("agent_id")
        service = svc()

        # UPDATE MODE: If agent_id is provided, treat as update request
        if agent_id:
            try:
                result = service.updateFromPrompt(agent_id, prompt)
                return result
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # CREATE MODE: Check for existing similar agents first
        # TODO: Implement _analyze_intent and _check_existing_agents on AgentService
        try:
            intent = service._analyze_intent(prompt)
            existing_match = service._check_existing_agents(intent, prompt)
            if existing_match:
                # Return existing agent info without creating new one
                return existing_match
        except AttributeError:
            # Methods not yet implemented on AgentService; skip similarity check
            pass

        # Build AgentTemplate from request overrides
        template = AgentTemplate(
            name=request.get("name", ""),
            icon=request.get("icon"),
            role=request.get("role"),
            description=request.get("description"),
            prompt=request.get("system_prompt"),
            tools=request.get("tools"),
            variables=request.get("variables"),
            settings=request.get("settings"),
        )
        agent = service.createFromPrompt(prompt, template)

        # Persist agent to filesystem registry and database
        agent_dict = agent.model_dump()
        agent_dict["agent_id"] = agent_dict.pop("id")

        # 1) Save to filesystem registry
        try:
            registry = container.resolve('agent.registry')
            registry.register_agent(agent_dict)
        except Exception as e:
            _db_logger.warning(f"Filesystem registry save failed for agent: {e}")

        # 2) Save to PostgreSQL database (graceful degradation)
        db_saved = False
        try:
            from echolib.database import get_db_session
            from echolib.repositories.agent_repo import AgentRepository
            from echolib.repositories.base import DEFAULT_USER_ID

            agent_repo = AgentRepository()

            db_agent_data = copy.deepcopy(agent_dict)
            db_agent_data["agent_id"] = _prefixed_id_to_uuid(db_agent_data["agent_id"])

            async with get_db_session() as db:
                await agent_repo.upsert(db, DEFAULT_USER_ID, db_agent_data, source_workflow_id=None)
            db_saved = True
            _db_logger.info(f"Agent {agent_dict['agent_id']} saved to database")
        except Exception as e:
            _db_logger.warning(f"Database save failed for agent {agent_dict.get('agent_id')}: {e}")

        # Return with action field
        response = agent.model_dump()
        response["action"] = "CREATE_AGENT"
        response["db_saved"] = db_saved
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/create/card')
async def create_card(cardJSON: dict, template: AgentTemplate):
    """Canvas-based agent creation (existing API)."""
    return svc().createFromCanvasCard(cardJSON, template).model_dump()


@router.post('/validate')
async def validate(agent: Agent):
    """Simple agent validation (existing API)."""
    return svc().validateA2A(agent).model_dump()


@router.get('/list')
async def list_agents():
    """List all agents (existing API)."""
    return [a.model_dump() for a in svc().listAgents()]


# ==================== NEW ORCHESTRATOR ROUTES ====================

# Agent Design
@router.post('/design/prompt')
async def design_agent_from_prompt(request: dict):
    """
    Design agent from natural language prompt.

    Accepts a JSON body with:
    - prompt (str, required): Natural language description
    - agent_id (str, optional): If provided, updates existing agent
    - model (str, optional): LLM model to use
    - icon (str, optional): Agent icon
    - tools (list, optional): Tools list
    - variables (list, optional): Variables list

    Response includes "action" field:
    - "CREATE_AGENT": New agent was designed
    - "UPDATE_AGENT": Existing agent was updated
    - "AGENT_EXISTS": Similar agent/template already exists (can be configured/modified)
    """
    try:
        designer = container.resolve('agent.designer')
        registry = container.resolve('agent.registry')

        # Extract request fields
        user_prompt = request.get("prompt", "")
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        agent_id = request.get("agent_id")

        # UPDATE MODE: If agent_id provided, update existing agent
        if agent_id:
            existing_agent = registry.get_agent(agent_id)
            if not existing_agent:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

            # Use update method that preserves name/ID
            updated_agent = designer.update_from_prompt(
                existing_agent=existing_agent,
                user_prompt=user_prompt
            )

            # Save updates to registry
            registry.update_agent(agent_id, updated_agent)
            return {
                "action": "UPDATE_AGENT",
                "agent_id": agent_id,
                "agent_name": updated_agent.get("name"),
                "agent": updated_agent
            }

        # CREATE MODE: Check for existing similar agents/templates first
        service = container.resolve('agent.service')
        intent = service._analyze_intent(user_prompt)
        existing_match = service._check_existing_agents(intent, user_prompt)
        if existing_match:
            # Return existing agent/template info without creating new one
            return existing_match

        # No existing match found - proceed to design new agent
        default_model = request.get("model", "mistral-nemo-12b")
        icon = request.get("icon", "")
        tools = request.get("tools", [])
        variables = request.get("variables", [])

        # Design agent
        agent = designer.design_from_prompt(
            user_prompt=user_prompt,
            default_model=default_model,
            icon=icon,
            tools=tools,
            variables=variables
        )

        # Register agent automatically
        registry.register_agent(agent)
        return {
            "action": "CREATE_AGENT",
            "agent": agent
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent Registry
@router.post('/register')
async def register_agent(agent: dict):
    """Register a new agent in the registry."""
    try:
        registry = container.resolve('agent.registry')
        result = registry.register_agent(agent)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== BACKEND-DRIVEN CONVERSATION ==================
# NOTE: These routes MUST be defined before the /{agent_id} catch-all
# route so FastAPI matches the specific path first.

@router.post('/design/chat')
async def design_chat(request: dict):
    """
    Single endpoint that drives the entire agent-builder conversation.

    The frontend sends the user message, current step, and agent state.
    The backend classifies intent, performs the action, and returns a
    structured response.  The frontend is a thin renderer — no business
    logic required.

    Request body:
    {
      "message": "user's message",
      "step": "initial" | "name" | "refine" | "done",
      "agent_state": { ... },       // current agent definition (empty on initial)
      "model": "openrouter-devstral" // optional LLM model override
    }

    Response:
    {
      "reply": "chat bubble text",
      "step": "name" | "refine" | "done",
      "action": "AGENT_DESIGNED" | "NAME_CONFIRMED" | ... ,
      "agent_state": { ... },
      "tools_available": [ ... ]     // present on tool-related actions
    }
    """
    try:
        designer = container.resolve('agent.designer')

        message = request.get("message", "").strip()
        step = request.get("step", "initial")
        agent_state = request.get("agent_state", {})
        model = request.get("model", "openrouter-devstral")

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        result = designer.handle_chat_step(
            message=message,
            step=step,
            agent_state=agent_state,
            model=model,
        )

        # On AGENT_DESIGNED, also persist to registries
        if result.get("action") == "AGENT_DESIGNED":
            agent = result.get("agent_state", {})
            try:
                registry = container.resolve('agent.registry')
                registry.register_agent(agent)
            except Exception as e:
                _db_logger.warning(f"Agent registry save failed: {e}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        _db_logger.error(f"design/chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DYNAMIC TOOL SELECTION ====================

@router.get('/available-tools')
async def get_available_tools():
    """
    Return all available tools (built-in + connector-synced) for the
    frontend tool picker.

    Response:
    {
      "tools": [
        {
          "tool_id": "tool_web_search",
          "name": "Web Search",
          "description": "Search the web for information",
          "tool_type": "local",
          "tags": [],
          "source": "builtin",
          "status": "active",
          "version": "1.0"
        },
        ...
      ],
      "builtin_count": 5,
      "connector_count": 3,
      "total": 8
    }
    """
    try:
        designer = container.resolve('agent.designer')
        result = designer.get_available_tools()
        return result
    except Exception as e:
        _db_logger.error(f"Failed to get available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/create-tool-from-connector')
async def create_tool_from_connector(request: dict):
    """
    Create a new tool from an existing connector and register it.

    Use this endpoint when the user asks to create a tool from a
    connector they have already registered (e.g. "create me a Teams tool").

    Request body:
    {
      "connector_id": "conn_abc123",
      "connector_type": "api" | "mcp",
      "tool_name": "My Teams Tool",          // optional override
      "tool_description": "Send messages..." // optional override
    }

    Response:
    {
      "tool_id": "tool_api_my_teams_tool",
      "name": "My Teams Tool",
      "description": "...",
      "tool_type": "api",
      "tags": ["api", "connector", "synced"],
      "source": "connector",
      "status": "active",
      "already_existed": false,
      "message": "Tool 'My Teams Tool' registered successfully"
    }
    """
    try:
        connector_id = request.get("connector_id")
        connector_type = request.get("connector_type")

        if not connector_id:
            raise HTTPException(status_code=400, detail="connector_id is required")
        if not connector_type:
            raise HTTPException(status_code=400, detail="connector_type is required")
        if connector_type not in ("api", "mcp"):
            raise HTTPException(
                status_code=400,
                detail="connector_type must be 'api' or 'mcp'"
            )

        tool_name = request.get("tool_name")
        tool_description = request.get("tool_description")

        designer = container.resolve('agent.designer')
        result = designer.create_tool_from_connector(
            connector_id=connector_id,
            connector_type=connector_type,
            tool_name=tool_name,
            tool_description=tool_description,
        )
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _db_logger.error(f"Failed to create tool from connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{agent_id}')
async def get_agent(agent_id: str):
    """Get agent by ID from registry."""
    try:
        registry = container.resolve('agent.registry')
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/registry/list')
async def list_registered_agents():
    """List all agents in registry."""
    try:
        registry = container.resolve('agent.registry')
        agents = registry.list_agents()
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/registry/master-list')
async def get_master_agent_list():
    """Get master agent list for workflow builder display."""
    try:
        registry = container.resolve('agent.registry')
        return registry.get_master_list()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put('/{agent_id}')
async def update_agent(agent_id: str, updates: dict):
    """Update an existing agent."""
    try:
        registry = container.resolve('agent.registry')
        updated = registry.update_agent(agent_id, updates)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch('/{agent_id}/schema')
async def update_agent_io_schema(agent_id: str, request: dict):
    """Update agent's input/output schema (for workflow integration)."""
    try:
        registry = container.resolve('agent.registry')
        input_schema = request.get("input_schema")
        output_schema = request.get("output_schema")

        updates = {}
        if input_schema is not None:
            updates["input_schema"] = input_schema
        if output_schema is not None:
            updates["output_schema"] = output_schema
        if not updates:
            raise HTTPException(status_code=400, detail="Provide input_schema or output_schema")

        updated = registry.update_agent(agent_id, updates)
        return {
            "agent_id": agent_id,
            "input_schema": updated.get("input_schema", []),
            "output_schema": updated.get("output_schema", [])
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete('/{agent_id}')
async def delete_agent(agent_id: str):
    """Delete an agent from registry."""
    try:
        registry = container.resolve('agent.registry')
        registry.delete_agent(agent_id)
        return {"message": "Agent deleted", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/role/{role}')
async def get_agents_by_role(role: str):
    """Get agents by role."""
    try:
        registry = container.resolve('agent.registry')
        agents = registry.get_agents_by_role(role)
        return {"role": role, "agents": agents, "count": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Agent Factory
@router.post('/instantiate/{agent_id}')
async def instantiate_agent(agent_id: str, bind_tools: bool = True):
    """Create runtime agent instance from definition."""
    try:
        registry = container.resolve('agent.registry')
        factory = container.resolve('agent.factory')
        agent_def = registry.get_agent(agent_id)
        if not agent_def:
            raise HTTPException(status_code=404, detail="Agent not found")
        instance = factory.create_agent(agent_def, bind_tools=bind_tools)
        return instance
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/instantiate/batch')
async def instantiate_agents_batch(agent_ids: list):
    """Create multiple runtime agent instances."""
    try:
        registry = container.resolve('agent.registry')
        factory = container.resolve('agent.factory')
        agent_defs = registry.get_agents_for_workflow(agent_ids)
        instances = factory.create_agents_for_workflow(agent_defs)
        return {"instances": instances, "count": len(instances)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Agent Permissions
@router.post('/permissions/check')
async def check_permission(
    caller_id: str,
    target_id: str,
    workflow: dict,
    agents: dict
):
    """Check if caller can communicate with target agent."""
    try:
        permissions = container.resolve('agent.permissions')
        allowed = permissions.can_call_agent(
            caller_id=caller_id,
            target_id=target_id,
            workflow=workflow,
            agent_registry=agents
        )
        return {
            "caller_id": caller_id,
            "target_id": target_id,
            "allowed": allowed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/permissions/validate')
async def validate_permissions(workflow: dict, agents: dict):
    """Validate all permissions in a workflow."""
    try:
        permissions = container.resolve('agent.permissions')
        errors = permissions.validate_workflow_permissions(workflow, agents)
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/permissions/targets/{agent_id}')
async def get_allowed_targets(agent_id: str, workflow: dict, agents: dict):
    """Get list of agents that the given agent can call."""
    try:
        permissions = container.resolve('agent.permissions')
        targets = permissions.get_allowed_targets(
            agent_id=agent_id,
            workflow=workflow,
            agent_registry=agents
        )
        return {
            "agent_id": agent_id,
            "allowed_targets": targets
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== AGENT TEMPLATES ====================

@router.get('/templates/all')
async def get_all_agent_templates():
    """
    Get all agent templates (static + created agents).
    Combines static templates from JSON file with agents from registry.
    """
    try:
        from pathlib import Path
        registry = container.resolve('agent.registry')

        # Load static templates
        templates_path = Path(__file__).parent.parent / "storage" / "agent_templates.json"
        static_templates = []
        if templates_path.exists():
            with open(templates_path, encoding='utf-8') as f:
                data = json.load(f)
                static_templates = data.get("templates", [])

        # Get created agents from registry
        master_list = registry.get_master_list()
        created_agents = master_list.get("agents", [])

        # Format created agents as templates
        created_templates = []
        for agent in created_agents:
            created_templates.append({
                "name": agent.get("name", "Unnamed Agent"),
                "icon": agent.get("icon", "🤖"),
                "description": agent.get("description", ""),
                "role": agent.get("role", ""),
                "agent_id": agent.get("agent_id"),
                "source": "created"
            })

        return {
            "templates": static_templates,
            "created": created_templates,
            "total_templates": len(static_templates),
            "total_created": len(created_templates),
            "total": len(static_templates) + len(created_templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/templates/static')
async def get_static_templates():
    """Get only static agent templates from JSON file."""
    try:
        from pathlib import Path
        templates_path = Path(__file__).parent.parent / "storage" / "agent_templates.json"
        if not templates_path.exists():
            return {"templates": [], "count": 0}
        with open(templates_path, encoding='utf-8') as f:
            data = json.load(f)
            templates = data.get("templates", [])
        return {
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== INTENT CLASSIFICATION (LLM-based) ====================

@router.post('/classify-intent')
async def classify_user_intent(request: dict):
    """
    Classify user intent using LLM reasoning.
    This endpoint enables natural language understanding for conversational flows.
    Instead of pattern matching, it uses LLM to understand user intent.

    Request body:
    {
      "context": "name_confirmation" | "refinement" | "tool_selection" | "general",
      "suggested_value": "FinGenius Pro",
      "user_message": "oh yes this name is great",
      "conversation_history": [...] // Optional: previous messages for context
    }

    Response:
    {
      "intent": "CONFIRMATION" | "MODIFICATION" | "REJECTION" | "CLARIFICATION",
      "confidence": 0.95,
      "reasoning": "...",
      "extracted_value": null | "new value if modification"
    }
    """
    try:
        service = container.resolve('agent.service')
        context = request.get("context", "general")
        suggested_value = request.get("suggested_value", "")
        user_message = request.get("user_message", "")
        conversation_history = request.get("conversation_history", [])
        if not user_message:
            raise HTTPException(status_code=400, detail="user_message is required")

        result = service.classify_user_intent(
            context=context,
            suggested_value=suggested_value,
            user_message=user_message,
            conversation_history=conversation_history
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AGENT CHAT (ADDED) ====================

def _get_registry():
    return container.resolve('agent.registry')

def _get_factory():
    return container.resolve('agent.factory')

def _get_chat_store():
    """
    Optional conversation store registered as 'agent.chat_store' in the container.
    Expected interface:
      - new_session(agent_id: str) -> str
      - append(session_id: str, role: str, content: str) -> None
      - history(session_id: str) -> List[Dict[str, str]]
    """
    try:
        return container.resolve('agent.chat_store')
    except Exception:
        return None

async def _maybe_await(result):
    if inspect.isawaitable(result):
        return await result
    return result

@router.post('/{agent_id}/session')
async def create_session(agent_id: str):
    """
    Create a new chat session id for multi-turn conversations.
    Falls back to a random UUID if no chat_store is available.
    """
    try:
        chat_store = _get_chat_store()
        if chat_store:
            session_id = chat_store.new_session(agent_id)
        else:
            import uuid
            session_id = str(uuid.uuid4())
        return {"agent_id": agent_id, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/{agent_id}/chat')
async def chat_with_agent(agent_id: str, payload: dict = Body(...)):
    """
    Non-streaming chat with an agent.
    Body:
      {
        "message": "text",                # required
        "session_id": "uuid",             # optional
        "context": {...},                 # optional
        "metadata": {...},                # optional
        "tools": [ { "name": "...", "args": {...} } ]  # optional
      }
    """
    try:
        registry = _get_registry()
        factory = _get_factory()
        agent_def = registry.get_agent(agent_id)
        if not agent_def:
            raise HTTPException(status_code=404, detail="Agent not found")

        runtime = factory.create_agent(agent_def, bind_tools=True)

        user_message = (payload.get("message") or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="`message` is required")

        session_id: Optional[str] = payload.get("session_id")
        context: Dict[str, Any] = payload.get("context") or {}
        metadata: Dict[str, Any] = payload.get("metadata") or {}
        tool_overrides: List[Dict[str, Any]] = payload.get("tools") or []

        chat_store = _get_chat_store()
        history = []
        if session_id and chat_store:
            history = chat_store.history(session_id)
        if chat_store:
            if not session_id:
                session_id = chat_store.new_session(agent_id)
            chat_store.append(session_id, "user", user_message)

        # Call into runtime; support both sync and async runtimes.
        reply = await _maybe_await(
            getattr(runtime, "chat")(
                message=user_message,
                history=history,
                context=context,
                tools=tool_overrides,
                metadata=metadata,
            )
        )

        # Persist assistant turn if we have text content
        if chat_store:
            content = reply.get("content") if isinstance(reply, dict) else str(reply)
            if content:
                chat_store.append(session_id, "assistant", content)

        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "reply": reply
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/{agent_id}/chat/stream')
async def chat_with_agent_stream(agent_id: str, payload: dict = Body(...)):
    """
    Streaming chat via chunked JSON lines. Each line is a JSON object.
    Client can read incrementally; not SSE, but simple line-delimited JSON.
    """
    async def token_stream() -> AsyncGenerator[bytes, None]:
        try:
            registry = _get_registry()
            factory = _get_factory()
            agent_def = registry.get_agent(agent_id)
            if not agent_def:
                yield json.dumps({"event": "error", "detail": "Agent not found"}).encode() + b"\n"
                return

            runtime = factory.create_agent(agent_def, bind_tools=True)

            user_message = (payload.get("message") or "").strip()
            if not user_message:
                yield json.dumps({"event": "error", "detail": "`message` is required"}).encode() + b"\n"
                return

            session_id: Optional[str] = payload.get("session_id")
            context: Dict[str, Any] = payload.get("context") or {}
            metadata: Dict[str, Any] = payload.get("metadata") or {}
            tool_overrides: List[Dict[str, Any]] = payload.get("tools") or []

            chat_store = _get_chat_store()
            history = []
            if session_id and chat_store:
                history = chat_store.history(session_id)
            if chat_store:
                if not session_id:
                    session_id = chat_store.new_session(agent_id)
                chat_store.append(session_id, "user", user_message)

            yield json.dumps({"event": "start", "session_id": session_id}).encode() + b"\n"

            # Prefer async generator if available; otherwise fall back to non-streaming
            stream_fn = getattr(runtime, "stream_chat", None)
            if stream_fn is None:
                reply = await _maybe_await(
                    getattr(runtime, "chat")(
                        message=user_message,
                        history=history,
                        context=context,
                        tools=tool_overrides,
                        metadata=metadata,
                    )
                )
                text = reply if isinstance(reply, str) else reply.get("content", str(reply))
                yield json.dumps({"event": "token", "text": text}).encode() + b"\n"
                if chat_store and text:
                    chat_store.append(session_id, "assistant", text)
                yield json.dumps({"event": "end"}).encode() + b"\n"
                return

            # If stream_chat exists, support async or sync iterables.
            collected = []
            result = stream_fn(
                message=user_message,
                history=history,
                context=context,
                tools=tool_overrides,
                metadata=metadata,
            )
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    if isinstance(chunk, str):
                        collected.append(chunk)
                        yield json.dumps({"event": "token", "text": chunk}).encode() + b"\n"
                    else:
                        if "text" in chunk:
                            collected.append(chunk["text"])
                        yield (json.dumps({"event": "token", **chunk}) + "\n").encode()
            else:
                for chunk in result:
                    if isinstance(chunk, str):
                        collected.append(chunk)
                        yield json.dumps({"event": "token", "text": chunk}).encode() + b"\n"
                    else:
                        if "text" in chunk:
                            collected.append(chunk["text"])
                        yield (json.dumps({"event": "token", **chunk}) + "\n").encode()

            final_text = "".join(collected)
            if chat_store and final_text:
                chat_store.append(session_id, "assistant", final_text)
            yield json.dumps({"event": "end"}).encode() + b"\n"
        except Exception as e:
            yield json.dumps({"event": "error", "detail": str(e)}).encode() + b"\n"

    return StreamingResponse(token_stream(), media_type="application/json")

@router.get('/{agent_id}/history/{session_id}')
async def get_history(agent_id: str, session_id: str):
    """
    Return full conversation history for a session (requires agent.chat_store).
    """
    chat_store = _get_chat_store()
    if not chat_store:
        raise HTTPException(status_code=404, detail="No chat history backend configured")
    try:
        history = chat_store.history(session_id)
        return {"agent_id": agent_id, "session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
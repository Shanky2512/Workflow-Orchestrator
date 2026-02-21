"""
Workbench Mode -- API Routes

All Workbench endpoints. These are completely isolated from
the orchestrator pipeline (appmgr/orchestrator).

Endpoints:
    GET  /workbench/health    -- Health check
    GET  /workbench/status    -- Workbench mode availability
    POST /workbench/chat      -- Workbench chat passthrough
    POST /workbench/translate  -- Detect + translate to English
    POST /workbench/detect     -- Language detection only
"""

from fastapi import APIRouter, HTTPException

from echolib.di import container
from .services.translation_service import TextInput, TranslationResponse
from .schema import WorkbenchChatRequest, WorkbenchChatResponse, WorkbenchStatusResponse

router = APIRouter(prefix='/workbench', tags=['Workbench API'])


def _translation_svc():
    return container.resolve('workbench.translation')


def _chat_svc():
    return container.resolve('workbench.chat')


def _manager():
    return container.resolve('workbench.manager')


# ── Health & Status ────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Health check for Workbench service."""
    return {"status": "healthy", "mode": "workbench"}


@router.get("/status", response_model=WorkbenchStatusResponse)
def status():
    """Check if Workbench mode is available and list services."""
    mgr = _manager()
    return WorkbenchStatusResponse(
        enabled=mgr.is_workbench_enabled(),
        available_services=mgr.available_services(),
    )


# ── Chat ───────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=WorkbenchChatResponse)
def workbench_chat(payload: WorkbenchChatRequest):
    """
    Workbench chat endpoint.

    Accepts a message and optional session_id. Returns a response envelope.
    Frontend handles LLM calls via its own 3rd-party Workbench endpoints.
    Backend acts as a passthrough -- NO orchestrator, NO LLM invocation.
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        result = _chat_svc().chat(message, payload.session_id)
        return WorkbenchChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workbench chat error: {str(e)}")


# ── Translation ────────────────────────────────────────────────────────────

@router.post("/translate", response_model=TranslationResponse)
def detect_and_translate(input_data: TextInput):
    """Detect language and translate to English."""
    text = input_data.Text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        return _translation_svc().translate(text)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/detect")
def detect_language(input_data: TextInput):
    """Detect language only (no translation)."""
    text = input_data.Text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        return _translation_svc().detect(text)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

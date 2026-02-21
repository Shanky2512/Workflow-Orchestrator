"""
LLM Routes - FastAPI endpoints that expose LLM functionality.

Uses LLMService which delegates to LLMFactory.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json
import logging

# Import LLMService (which wraps LLMFactory)
from echolib.services import LLMService
from echolib.LLM_Details.Exceptions.custom_exceptions import (
    ModelNotFoundError,
    MissingAPIKeyError,
    LLMExecutionError,
    ValidationError,
    EmbeddingError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIRouter
router = APIRouter(
    prefix="/LLMs",
    tags=["LLMs"]
)

# Global LLMService instance (will be initialized with tool_service)
llm_service = LLMService(None)

# ==================== REQUEST/RESPONSE MODELS ====================

class InvokeRequest(BaseModel):
    model_id: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatRequest(BaseModel):
    model_id: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class StreamRequest(BaseModel):
    model_id: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class JsonModeRequest(BaseModel):
    model_id: str
    prompt: str
    schema: Optional[Dict] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class FunctionCallRequest(BaseModel):
    model_id: str
    tools: List[Dict]
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class EmbedRequest(BaseModel):
    model_id: str
    text: Union[str, List[str]]


class CustomModelRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    provider: str = Field(..., description="Provider (openrouter, custom, etc.)")
    base_url: str = Field(..., description="API endpoint URL")
    api_key_env: str = Field(..., description="Environment variable for API key")
    model_name: str = Field(..., description="Model identifier at provider")
    description: Optional[str] = ""
    is_default: Optional[bool] = False


# ==================== ENDPOINTS ====================

@router.get("/")
def root():
    """Root endpoint - API information."""
    return {
        "name": "LLM Factory API",
        "version": "2.0.0",
        "status": "running",
        "core_methods": ["invoke", "chat", "stream", "json_mode", "function_call", "embed"],
        "architecture": "routes → LLMService → LLMFactory"
    }


@router.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "llm-factory",
            "factory_initialized": llm_service
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ==================== MODEL MANAGEMENT ====================

@router.get("/models/list")
def get_models():
    """
    Get list of all available models (pre-configured + custom).
    
    Returns:
        List of model information dictionaries
    """
    try:
        return llm_service.list_models()
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load models"
        )


@router.get("/models")
def get_model_info(model_id: str):
    """Get information about a specific model."""
    try:
        model = llm_service.get_model(model_id)
        return {
            "id": model.get_model_id(),
            "provider": model.get_provider(),
            "config": model.get_config()
        }
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/add-custom-model")
def add_custom_model(request: CustomModelRequest):
    """
    Add a user-defined custom model.
    
    This endpoint allows users to add their own LLMs via a form.
    """
    try:
        result = llm_service.register_custom_model(
            model_id=request.model_id,
            name=request.name,
            provider=request.provider,
            base_url=request.base_url,
            api_key_env=request.api_key_env,
            model_name=request.model_name,
            description=request.description,
            is_default=request.is_default
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
            
    except Exception as e:
        logger.error(f"Error adding custom model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/models")
def remove_custom_model(model_id: str):
    """Remove a custom model."""
    try:
        result = llm_service.remove_custom_model(model_id)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 1: INVOKE ====================

@router.post("/invoke")
def invoke(request: InvokeRequest):
    """
    Simple text generation - single turn.
    
    Example:
        POST /api/llms/invoke
        {
            "model_id": "claude-opus-4",
            "prompt": "What is Python?",
            "temperature": 0.7
        }
    """
    try:
        model = llm_service.get_model(
            request.model_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response = model.invoke(request.prompt)
        
        return {
            "response": response,
            "model": request.model_id,
            "method": "invoke"
        }
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except MissingAPIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )
    except LLMExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 2: CHAT ====================

@router.post("/chat")
def chat(request: ChatRequest):
    """
    Conversation with history - multi-turn.
    
    Example:
        POST /api/llms/chat
        {
            "model_id": "claude-opus-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
        }
    """
    try:
        model = llm_service.get_model(
            request.model_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response = model.chat(request.messages)
        
        return {
            "response": response,
            "model": request.model_id,
            "method": "chat"
        }
        
    except (ModelNotFoundError, MissingAPIKeyError, LLMExecutionError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 3: STREAM ====================

@router.post("/stream")
def stream(request: StreamRequest):
    """
    Streaming text generation - real-time chunks.
    
    Returns Server-Sent Events (SSE).
    """
    try:
        model = llm_service.get_model(
            request.model_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        def generate():
            try:
                for chunk in model.stream(request.prompt):
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                
                # Final message
                yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 4: JSON MODE ====================

@router.post("/json-mode")
def json_mode(request: JsonModeRequest):
    """
    Structured JSON output - guaranteed valid JSON.
    
    Example:
        POST /api/llms/json-mode
        {
            "model_id": "claude-opus-4",
            "prompt": "Extract person info: John is 25 years old",
            "schema": {"name": "string", "age": "integer"}
        }
    """
    try:
        model = llm_service.get_model(
            request.model_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response = model.json_mode(request.prompt, schema=request.schema)
        
        return {
            "result": response,
            "model": request.model_id,
            "method": "json_mode"
        }
        
    except (ModelNotFoundError, MissingAPIKeyError, LLMExecutionError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 5: FUNCTION CALL ====================

@router.post("/function-call")
def function_call(request: FunctionCallRequest):
    """
    Native function/tool calling.
    
    Example:
        POST /api/llms/function-call
        {
            "model_id": "claude-opus-4",
            "tools": [
                {
                    "name": "search_database",
                    "description": "Search company database",
                    "parameters": {
                        "query": {"type": "string"}
                    }
                }
            ],
            "prompt": "Find sales data for Q4"
        }
    """
    try:
        model = llm_service.get_model(
            request.model_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response = model.function_call(request.tools, request.prompt)
        
        return {
            "result": response,
            "model": request.model_id,
            "method": "function_call"
        }
        
    except (ModelNotFoundError, MissingAPIKeyError, LLMExecutionError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== METHOD 6: EMBED ====================

@router.post("/embed")
def embed(request: EmbedRequest):
    """
    Generate text embeddings - vector representations.
    
    Example:
        POST /api/llms/embed
        {
            "model_id": "claude-opus-4",
            "text": "Hello world"
        }
        
        Or for multiple texts:
        {
            "model_id": "claude-opus-4",
            "text": ["Hello", "World"]
        }
    """
    try:
        model = llm_service.get_model(request.model_id)
        
        embeddings = model.embed(request.text)
        
        return {
            "embeddings": embeddings,
            "model": request.model_id,
            "method": "embed",
            "dimensions": len(embeddings) if isinstance(request.text, str) else len(embeddings[0])
        }
        
    except EmbeddingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except (ModelNotFoundError, MissingAPIKeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== UTILITY ENDPOINTS ====================

@router.post("/refresh-config")
def refresh_config():
    """Reload configuration from file."""
    try:
        llm_service.refresh_configuration()
        return {"message": "Configuration refreshed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/providers")
def get_providers():
    """Get list of available providers."""
    try:
        providers = llm_service.get_providers()
        return {"providers": providers}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

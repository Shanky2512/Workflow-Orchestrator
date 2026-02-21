"""
Unified Model Wrapper - Implements all 6 core methods for any LLM.

This wrapper provides a consistent interface regardless of the underlying provider.
"""

import json
import logging
from typing import List, Dict, Any, Generator, Optional, Union
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echolib.LLM_Details.base_model import BaseModel
from echolib.LLM_Details.Exceptions.custom_exceptions import LLMExecutionError, ValidationError, EmbeddingError

logger = logging.getLogger(__name__)


class UnifiedModelWrapper(BaseModel):
    """
    Unified wrapper that provides consistent interface for all LLM models.
    
    Wraps LangChain clients and provides:
    - Normalized responses
    - Error handling
    - The 6 core methods
    - Access to raw client for advanced use
    """
    
    def __init__(
        self,
        client,
        embeddings_client,
        model_id: str,
        model_config: Dict
    ):
        """
        Initialize the wrapper.
        
        Args:
            client: LangChain ChatOpenAI instance
            embeddings_client: LangChain OpenAIEmbeddings instance
            model_id: Model identifier
            model_config: Full model configuration dict
        """
        self._client = client
        self._embeddings_client = embeddings_client
        self._model_id = model_id
        self._config = model_config
        self._provider = model_config.get("provider", "unknown")
    
    @property
    def raw_client(self):
        """Access to raw LangChain client for advanced use."""
        return self._client
    
    @property
    def raw_embeddings_client(self):
        """Access to raw embeddings client."""
        return self._embeddings_client
    
    # ==================== METHOD 1: INVOKE ====================
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Simple text generation - single turn.
        
        Args:
            prompt: Text prompt
            **kwargs: temperature, max_tokens, etc.
            
        Returns:
            Generated text
        """
        try:
            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValidationError("Prompt must be a non-empty string")
            
            # Update client parameters if provided
            if kwargs:
                self._update_client_params(**kwargs)
            
            # Call LangChain client
            response = self._client.invoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            raise LLMExecutionError(f"Failed to execute invoke: {str(e)}")
    
    # ==================== METHOD 2: CHAT ====================
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Conversation with history - multi-turn.
        
        Args:
            messages: List of message dicts
                     Example: [
                         {"role": "system", "content": "You are helpful"},
                         {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi!"},
                         {"role": "user", "content": "How are you?"}
                     ]
            **kwargs: temperature, max_tokens, etc.
            
        Returns:
            Assistant's response text
        """
        try:
            # Validate input
            if not isinstance(messages, list):
                raise ValidationError("Messages must be a list")
            
            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValidationError("Each message must be a dict")
                if "role" not in msg or "content" not in msg:
                    raise ValidationError("Each message must have 'role' and 'content'")
                if msg["role"] not in ["system", "user", "assistant"]:
                    raise ValidationError(f"Invalid role: {msg['role']}")
            
            # Update client parameters if provided
            if kwargs:
                self._update_client_params(**kwargs)
            
            # Call LangChain client
            response = self._client.invoke(messages)
            
            # Extract text
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise LLMExecutionError(f"Failed to execute chat: {str(e)}")
    
    # ==================== METHOD 3: STREAM ====================
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValidationError("Prompt must be a non-empty string")

            if kwargs:
                self._update_client_params(**kwargs)

            # Toggle .streaming only if the client has it
            had_streaming_attr = hasattr(self._client, "streaming")
            original_streaming = getattr(self._client, "streaming", None)
            if had_streaming_attr:
                self._client.streaming = True

            try:
                if hasattr(self._client, "stream"):
                    for chunk in self._client.stream(prompt):
                        if hasattr(chunk, "content"):
                            yield chunk.content
                        else:
                            yield str(chunk)
                else:
                    # Fallback: not a streaming client—return once
                    response = self._client.invoke(prompt)
                    yield response.content if hasattr(response, "content") else str(response)
            finally:
                if had_streaming_attr:
                    self._client.streaming = original_streaming

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            raise LLMExecutionError(f"Failed to execute stream: {str(e)}")

    
    # ==================== METHOD 4: JSON MODE ====================
    
    def json_mode(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Structured JSON output - guaranteed valid JSON when possible.

        Tries native JSON mode for OpenAI-compatible clients via
        model_kwargs["response_format"] = {"type": "json_object"}.
        Falls back to a strict instruction prompt and robust JSON parsing.
        """
        try:
            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValidationError("Prompt must be a non-empty string")

            # Build enhanced prompt (adds clear JSON-only constraints)
            enhanced_prompt = self._build_json_prompt(prompt, schema)

            # Update runtime params if provided
            if kwargs:
                self._update_client_params(**kwargs)

            # Use native JSON mode when available
            had_kwargs = hasattr(self._client, "model_kwargs") and isinstance(self._client.model_kwargs, dict)
            original_kwargs = self._client.model_kwargs.copy() if had_kwargs else None

            try:
                if had_kwargs:
                    # Prefer OpenAI-style json_object. If you later support JSON schema natively,
                    # you can switch to {"type": "json_schema", "json_schema": {...}}.
                    self._client.model_kwargs["response_format"] = {"type": "json_object"}

                # Invoke
                response = self._client.invoke(enhanced_prompt)
                text = response.content if hasattr(response, "content") else str(response)

                # Parse JSON robustly
                result = self._extract_and_parse_json(text)

                # Optional primitive schema check
                if schema:
                    self._validate_against_schema(result, schema)

                return result

            finally:
                if had_kwargs:
                    self._client.model_kwargs = original_kwargs

        except ValidationError:
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise LLMExecutionError(f"LLM did not return valid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error in json_mode: {e}")
            raise LLMExecutionError(f"Failed to execute json_mode: {str(e)}")
    
    # ==================== METHOD 5: FUNCTION CALL ====================
    
    def function_call(
        self,
        tools: List[Dict],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Native function/tool calling when the client supports it (OpenAI-compatible).
        Falls back to instruction-driven JSON output for clients without native tools.

        Returns:
            {
            "function": <str or None>,
            "arguments": <dict>  (present when function != None)
            # or "text_response": <str> if no function selected
            }
        """
        try:
            # Validate inputs
            if not isinstance(tools, list) or not tools:
                raise ValidationError("Tools must be a non-empty list")

            if not prompt or not isinstance(prompt, str):
                raise ValidationError("Prompt must be a non-empty string")

            # Update runtime params if provided
            if kwargs:
                self._update_client_params(**kwargs)

            # Check whether the client supports OpenAI-style tool calling
            had_kwargs = hasattr(self._client, "model_kwargs") and isinstance(self._client.model_kwargs, dict)
            original_kwargs = self._client.model_kwargs.copy() if had_kwargs else None

            try:
                if had_kwargs:
                    # Native tools path (OpenAI-compatible)
                    formatted_tools = self._format_tools_for_openai(tools)
                    self._client.model_kwargs["tools"] = formatted_tools
                    self._client.model_kwargs["tool_choice"] = "auto"

                    response = self._client.invoke(prompt)

                    # Try structured tool_calls extraction first
                    result = self._extract_function_call(response)
                    if result.get("function"):
                        # We got a function name + arguments from tool_calls
                        return result

                    # If no tool_call was emitted, attempt to parse JSON fallback in content
                    text = response.content if hasattr(response, "content") else str(response)
                    try:
                        data = self._extract_and_parse_json(text)
                        fn = data.get("function")
                        args = data.get("arguments", {})
                        if fn and isinstance(args, dict):
                            return {"function": fn, "arguments": args}
                    except Exception:
                        pass  # fall through

                    # No function selected—return plain text
                    return {
                        "function": None,
                        "text_response": response.content if hasattr(response, "content") else str(response)
                    }

                else:
                    # Fallback: non-native (e.g., Ollama). Ask the model to return JSON that we can parse.
                    # Build a strict instruction with the tools list.
                    tool_specs = []
                    for t in tools:
                        tool_specs.append({
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {})
                        })

                    fallback_instruction = (
                        "You have the following callable functions available. "
                        "Choose exactly one that best accomplishes the user's request, or choose none.\n\n"
                        "TOOLS:\n"
                        f"{json.dumps(tool_specs, indent=2)}\n\n"
                        "Return ONLY valid JSON with this shape:\n"
                        "{\n"
                        '  "function": "<function_name_or_null>",\n'
                        '  "arguments": { /* key/value arguments for the selected function, or empty object if none */ }\n'
                        "}\n\n"
                        "IMPORTANT:\n"
                        "- If no tool is appropriate, set function to null and arguments to {}.\n"
                        "- Do not include any commentary or prose; return JSON only.\n"
                    )

                    full_prompt = f"{fallback_instruction}\nUser task:\n{prompt}"
                    response = self._client.invoke(full_prompt)
                    text = response.content if hasattr(response, "content") else str(response)

                    data = self._extract_and_parse_json(text)
                    fn = data.get("function")
                    args = data.get("arguments", {})

                    # Validate the function name (if provided) against our tools list
                    valid_names = {t.get("name") for t in tools}
                    if fn is not None:
                        if not isinstance(fn, str):
                            # e.g., null (None) returned -> treated as no function
                            return {"function": None, "text_response": ""}
                        if fn not in valid_names:
                            # Model hallucinated a function name: treat as no function
                            return {
                                "function": None,
                                "text_response": f"(No valid tool selected; model suggested '{fn}')"
                            }

                    if not isinstance(args, dict):
                        args = {}

                    if fn:
                        return {"function": fn, "arguments": args}
                    return {"function": None, "text_response": ""}

            finally:
                if had_kwargs:
                    self._client.model_kwargs = original_kwargs

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in function_call: {e}")
            raise LLMExecutionError(f"Failed to execute function_call: {str(e)}")
    # ==================== METHOD 6: EMBED ====================
    
    def embed(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate text embeddings - vector representations.
        
        Args:
            text: Single text string or list of texts
            **kwargs: Additional parameters
            
        Returns:
            Single embedding vector or list of vectors
            Each vector is a list of floats (typically 1536 dimensions)
        """
        try:
            # Validate input
            if isinstance(text, str):
                if not text:
                    raise ValidationError("Text must not be empty")
                single_text = True
            elif isinstance(text, list):
                if not text:
                    raise ValidationError("Text list must not be empty")
                if not all(isinstance(t, str) for t in text):
                    raise ValidationError("All items in text list must be strings")
                single_text = False
            else:
                raise ValidationError("Text must be string or list of strings")
            
            # Generate embeddings
            if single_text:
                embedding = self._embeddings_client.embed_query(text)
                return embedding
            else:
                embeddings = self._embeddings_client.embed_documents(text)
                return embeddings
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in embed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    # ==================== HELPER METHODS ====================
    
    def _update_client_params(self, **kwargs):
        """Update client parameters like temperature, max_tokens (if supported)."""
        if "temperature" in kwargs and hasattr(self._client, "temperature"):
            self._client.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            # Some clients call it 'max_tokens', Ollama uses num_predict internally,
            # but we already mapped that in the builder so this is fine.
            if hasattr(self._client, "max_tokens"):
                self._client.max_tokens = kwargs["max_tokens"]

    def _build_json_prompt(self, prompt: str, schema: Optional[Dict]) -> str:
        """Build enhanced prompt for JSON mode."""
        enhanced = f"{prompt}\n\nRespond ONLY with valid JSON."
        
        if schema:
            enhanced += f"\n\nUse this schema:\n{json.dumps(schema, indent=2)}"
        
        enhanced += "\n\nDo not include any text before or after the JSON."
        
        return enhanced
    
    def _extract_and_parse_json(self, text: str) -> Dict:
        """Extract and parse JSON from text response."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Parse JSON
        return json.loads(text)
    
    def _validate_against_schema(self, data: Dict, schema: Dict):
        """Basic schema validation."""
        for key in schema.keys():
            if key not in data:
                raise ValidationError(f"Missing required field: {key}")
    
    def _format_tools_for_openai(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to OpenAI function calling format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": tool.get("parameters", {}),
                        "required": list(tool.get("parameters", {}).keys())
                    }
                }
            })
        return formatted
    
    def _extract_function_call(self, response) -> Dict[str, Any]:
        """Extract function call from response."""
        # Check if response has tool_calls
        if hasattr(response, 'additional_kwargs'):
            tool_calls = response.additional_kwargs.get('tool_calls', [])
            if tool_calls:
                call = tool_calls[0]
                return {
                    "function": call['function']['name'],
                    "arguments": json.loads(call['function']['arguments'])
                }
        
        # If no function call, return text response
        if hasattr(response, 'content'):
            return {
                "function": None,
                "text_response": response.content
            }
        
        return {"function": None, "text_response": str(response)}
"""
LLM Client Builder - Builds LangChain LLM client instances from configuration.
"""

from typing import Dict, Optional
import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI

# Load .env file
from dotenv import load_dotenv

# Find .env next to llm_provider.json
env_path = Path(__file__).parent.parent / "Providers" / ".env"
load_dotenv(dotenv_path = env_path)

# If not defined elsewhere:
class MissingAPIKeyError(Exception):
    pass

class InvalidProviderError(Exception):
    pass

# You’ll need these imports available:
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class LLMClientBuilder:
    """
    Builds LangChain LLM clients from model configuration.

    Supports OpenAI-compatible APIs (OpenRouter, OpenAI, Azure) and Ollama.
    """

    def __init__(self):
        pass

    def build_client(
        self,
        model_config: Dict,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = False
    ):
        """
        Build a LangChain LLM client from model configuration.
        """
        # Extract configuration
        base_url = model_config.get("base_url")
        api_key_env = model_config.get("api_key_env")  # may be None
        model_name = model_config.get("model_name")
        provider = (model_config.get("provider") or "").strip().lower()

        # Validate required fields (API key intentionally not required here)
        if not all([base_url, model_name, provider]):
            raise ValueError(
                "Model configuration must include: base_url, model_name, and provider"
            )

        # Sanitize params
        temp = 0.7 if temperature is None else float(temperature)
        if temp < 0.0: temp = 0.0
        if temp > 2.0: temp = 2.0  # some providers allow up to 2.0

        tokens = 1000 if max_tokens is None else int(max_tokens)
        if tokens <= 0: tokens = 1

        # ---- Provider routing
        if provider in ["openrouter", "openai", "azure"]:
            # These providers REQUIRE an API key. Do not allow None.
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise ValueError(
                    f"'api_key_env' is required for provider '{provider}'. "
                    f"Set it to the environment variable that holds your API key."
                )
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise MissingAPIKeyError(
                    f"API key not found in environment variable: {api_key_env}. "
                    f"Please set this environment variable with your API key."
                )

            return self._build_openai_compatible_client(
                base_url=base_url,
                api_key=api_key,         # guaranteed non-empty
                model_name=model_name,
                temperature=temp,
                max_tokens=tokens,
                streaming=streaming
            )

        elif provider in ["ollama"]:
            # No API key required
            return self._build_ollama_client(
                base_url=base_url,
                model_name=model_name,
                temperature=temp,        # use sanitized values
                max_tokens=tokens,
                streaming=streaming
            )

        else:
            raise InvalidProviderError(
                f"Provider '{provider}' not supported. Supported: openrouter, openai, azure, ollama"
            )

    def _build_openai_compatible_client(self, base_url, api_key, model_name, temperature, max_tokens, streaming):
        """Build OpenAI-compatible client."""
        
        # Check if it's Azure (Azure URLs contain 'azure.com')
        if 'azure.com' in base_url:
            from langchain_openai import AzureChatOpenAI
            
            llm = AzureChatOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                deployment_name=model_name,  # For Azure, this is deployment name
                api_version="2024-02-01",
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming
            )
        else:
            # Standard OpenAI/OpenRouter
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming
            )
        
        return llm

    def build_embeddings_client(
        self,
        model_config: Dict,
        dimensions: Optional[int] = None
    ):
        """
        Build embeddings client for .embed() functionality.
        For OpenAI-compatible providers, returns OpenAIEmbeddings.
        For Ollama, build a simple native client (no API key).
        """
        provider = (model_config.get("provider") or "").strip().lower()
        base_url = model_config.get("base_url")
        if not base_url:
            raise ValueError("model_config.base_url is required")

        # ---- Ollama: no API key
        if provider == "ollama":
            model_name = model_config.get("embedding_model") or model_config.get("model_name") or "nomic-embed-text"
            return self._build_ollama_embeddings_client(
                base_url=base_url,
                model_name=model_name,
                dimensions=dimensions
            )

        # ---- OpenAI-compatible: require API key
        if provider in ["openrouter", "openai", "azure"]:
            api_key_env = model_config.get("api_key_env")
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise ValueError(
                    f"'api_key_env' is required for provider '{provider}' when building embeddings."
                )
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise MissingAPIKeyError(f"API key not found: {api_key_env}")

            # Use a current embeddings model name (avoid 'text-embedding-ada-002' which is deprecated).
            embed_model = model_config.get("embedding_model") or "text-embedding-3-small"

            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url=base_url,
                model=embed_model,
                dimensions=dimensions
            )
            return embeddings

        raise InvalidProviderError(
            f"Provider '{provider}' not supported for embeddings. Supported: openrouter, openai, azure, ollama"
        )

    def validate_api_key(self, api_key_env: Optional[str]) -> bool:
        """Check if API key exists in environment (safe for None)."""
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            return False
        return bool(os.getenv(api_key_env))

    def _build_ollama_client(self, base_url, model_name, temperature, max_tokens, streaming):
        """
        Simple native Ollama client using requests.
        Exposes .invoke(messages) and .stream(messages) to match your LLMService usage.
        """
        import requests

        num_predict = max_tokens if max_tokens is not None else 1000

        class OllamaClient:
            def _to_prompt(self, messages):
                if isinstance(messages, str):
                    return messages
                return "\n".join([m.get("content", "") for m in messages])

            def invoke(self, messages):
                prompt = self._to_prompt(messages)
                resp = requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": num_predict,
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                class Result:
                    content = data.get("response", "")
                    response_metadata = {"token_usage": {}}
                return Result()

            def stream(self, messages):
                prompt = self._to_prompt(messages)
                with requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": num_predict,
                        "stream": True,
                    },
                    stream=True,
                    timeout=None,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        text = obj.get("response", "")
                        if text:
                            class Chunk:
                                content = text
                            yield Chunk()
                        if obj.get("done"):
                            break

        return OllamaClient()

    def _build_ollama_embeddings_client(self, base_url: str, model_name: str, dimensions: Optional[int] = None):
        """
        Minimal Ollama embeddings client using POST /api/embeddings.
        """
        import requests

        class OllamaEmbeddingsClient:
            def embed(self, texts):
                if isinstance(texts, str):
                    texts_list = [texts]
                else:
                    texts_list = list(texts or [])
                vectors = []
                for t in texts_list:
                    payload = {"model": model_name, "prompt": t}
                    resp = requests.post(f"{base_url}/api/embeddings", json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    vec = data.get("embedding")
                    if not isinstance(vec, list):
                        vec = []
                    vectors.append(vec)
                return vectors

        return OllamaEmbeddingsClient()



# Example usage
if __name__ == "__main__":
    print("=== LLM Client Builder Test ===\n")
    
    mock_config = {
        "id": "claude-opus-4",
        "name": "Claude Opus 4",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_name": "anthropic/claude-opus-4"
    }
    
    try:
        builder = LLMClientBuilder()
        
        # Check API key
        has_key = builder.validate_api_key("OPENROUTER_API_KEY")
        print(f"API key exists: {has_key}")
        
        if has_key:
            # Build client
            client = builder.build_client(mock_config)
            print(f"✅ Client created: {type(client).__name__}")
            print(f"   Model: {client.model_name}")
            print(f"   Temperature: {client.temperature}")
        else:
            print("⚠️ Set OPENROUTER_API_KEY to test client building")
        
    except Exception as e:
        print(f"❌ Error: {e}")
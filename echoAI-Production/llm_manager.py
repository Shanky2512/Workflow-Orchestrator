"""
Centralized LLM Manager - Single Source of Truth for ALL LLM Operations

This module is the ONLY place in the entire codebase that:
1. Knows about LLM providers (OpenAI, Anthropic, OpenRouter, etc.)
2. Stores API keys
3. Creates LLM client instances
4. Configures models, temperature, tokens, etc.

ALL other modules must import and use this manager. NO other file should:
- Import ChatOpenAI, ChatAnthropic, etc. directly
- Know about provider names or API keys
- Create LLM instances

USAGE:
    from llm_manager import LLMManager

    # Use default LLM
    llm = LLMManager.get_llm()

    # Override provider/model
    llm = LLMManager.get_llm(provider="openai", model="gpt-4")

    # Use for LangChain
    response = llm.invoke("Your prompt here")

TO SWITCH PROVIDERS:
    Edit DEFAULT_PROVIDER and DEFAULT_MODEL below. That's it.
    All workflows, agents, and components will use the new provider.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
import os
from dotenv import load_dotenv
load_dotenv()

class LLMManager:
    """
    Centralized LLM Manager - Controls ALL LLM operations in the system.

    This is a static/singleton class. All methods are class methods.
    Configuration is hardcoded in class variables for maximum control.
    """

    # ========================================================================
    # CONFIGURATION - EDIT THESE TO CONTROL ALL LLM USAGE
    # ========================================================================

    # DEFAULT PROVIDER: Change this to switch the entire system's LLM provider
    # Options: "openrouter", "openai", "anthropic", "ollama", "azure"
    DEFAULT_PROVIDER = "openrouter"

    # DEFAULT MODEL: The model used when no override is specified
    DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL")

    # DEFAULT PARAMETERS
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000

    # API KEYS - Hardcoded here (add to .gitignore to prevent commits)
    # IMPORTANT: Keep this file in .gitignore!
    API_KEYS = {
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        #, "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"
        "openai": "sk-YOUR_OPENAI_KEY_HERE",
        "anthropic": "sk-ant-YOUR_ANTHROPIC_KEY_HERE",
        "azure": "YOUR_AZURE_KEY_HERE"
        # Ollama doesn't need API key
    }

    # BASE URLs for providers
    BASE_URLS = {
        "openrouter": os.getenv("OPENROUTER_BASE_URL"),
        "ollama": "http://localhost:11434/v1",
        "azure": "YOUR_AZURE_ENDPOINT_HERE"  # e.g., https://your-resource.openai.azure.com/
    }

    # Azure-specific config
    AZURE_CONFIG = {
        "deployment_name": "YOUR_DEPLOYMENT_NAME",
        "api_version": "2024-02-15-preview"
    }

    # LLM instance cache (prevents recreating same LLM multiple times)
    _llm_cache: Dict[str, Any] = {}

    # ========================================================================
    # PUBLIC API - Use these methods from other components
    # ========================================================================

    @classmethod
    def get_llm(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Get an LLM instance with specified or default configuration.

        This is the MAIN method that all other components should use.

        Args:
            provider: LLM provider ("openrouter", "openai", "anthropic", "ollama", "azure")
                     If None, uses DEFAULT_PROVIDER
            model: Model name (e.g., "gpt-4", "claude-3-opus")
                   If None, uses DEFAULT_MODEL
            temperature: Sampling temperature (0.0 to 1.0)
                        If None, uses DEFAULT_TEMPERATURE
            max_tokens: Maximum tokens to generate
                       If None, uses DEFAULT_MAX_TOKENS
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM instance compatible with LangChain (has .invoke() method)

        Raises:
            ValueError: If provider is unsupported or configuration is invalid

        Examples:
            # Use default configuration
            llm = LLMManager.get_llm()

            # Override provider
            llm = LLMManager.get_llm(provider="openai", model="gpt-4")

            # Override temperature
            llm = LLMManager.get_llm(temperature=0.7)

            # Use in LangChain
            response = llm.invoke("Write a haiku")
        """
        # Use defaults if not specified
        provider = provider or cls.DEFAULT_PROVIDER
        model = model or cls.DEFAULT_MODEL
        temperature = temperature if temperature is not None else cls.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or cls.DEFAULT_MAX_TOKENS

        # Create cache key
        cache_key = f"{provider}:{model}:{temperature}:{max_tokens}"

        # Return cached instance if exists
        if cache_key in cls._llm_cache:
            logger.debug(f"Using cached LLM: {cache_key}")
            return cls._llm_cache[cache_key]

        # Create new LLM instance
        logger.info(f"Creating new LLM: provider={provider}, model={model}")

        if provider == "openrouter":
            llm = cls._create_openrouter_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            llm = cls._create_openai_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            llm = cls._create_anthropic_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            llm = cls._create_ollama_llm(model, temperature, max_tokens, **kwargs)
        elif provider == "azure":
            llm = cls._create_azure_llm(model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: openrouter, openai, anthropic, ollama, azure"
            )

        # Cache and return
        cls._llm_cache[cache_key] = llm
        return llm

    @classmethod
    def get_crewai_llm(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Get a CrewAI-compatible LLM instance.

        CrewAI uses its own LLM class (not LangChain's ChatOpenAI).
        This method returns a CrewAI LLM object.

        Args:
            provider: LLM provider (defaults to DEFAULT_PROVIDER)
            model: Model name (defaults to DEFAULT_MODEL)
            temperature: Temperature (defaults to DEFAULT_TEMPERATURE)
            max_tokens: Max tokens (defaults to DEFAULT_MAX_TOKENS)

        Returns:
            CrewAI LLM instance
        """
        from crewai import LLM

        provider = provider or cls.DEFAULT_PROVIDER
        model = model or cls.DEFAULT_MODEL
        temperature = temperature if temperature is not None else cls.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or cls.DEFAULT_MAX_TOKENS

        cache_key = f"crewai:{provider}:{model}:{temperature}:{max_tokens}"

        if cache_key in cls._llm_cache:
            logger.debug(f"Using cached CrewAI LLM: {cache_key}")
            return cls._llm_cache[cache_key]

        logger.info(f"Creating CrewAI LLM: provider={provider}, model={model}")

        # Build model string for CrewAI LLM class
        if provider == "openrouter":
            # OpenRouter is OpenAI-compatible; use openai/ prefix with custom base_url
            api_key = cls.API_KEYS.get("openrouter")
            llm = LLM(
                model=f"openai/{model}",
                api_key=api_key,
                base_url=cls.BASE_URLS["openrouter"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "openai":
            litellm_model = model  # OpenAI models don't need prefix
            api_key = cls.API_KEYS.get("openai")
            llm = LLM(
                model=litellm_model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "anthropic":
            litellm_model = f"anthropic/{model}"
            api_key = cls.API_KEYS.get("anthropic")
            llm = LLM(
                model=litellm_model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "ollama":
            litellm_model = f"ollama/{model}"
            llm = LLM(
                model=litellm_model,
                base_url=cls.BASE_URLS["ollama"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "azure":
            litellm_model = f"azure/{model}"
            api_key = cls.API_KEYS.get("azure")
            llm = LLM(
                model=litellm_model,
                api_key=api_key,
                base_url=cls.BASE_URLS["azure"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        else:
            raise ValueError(f"Unsupported provider for CrewAI: {provider}")

        cls._llm_cache[cache_key] = llm
        return llm

    @classmethod
    def get_default_llm(cls):
        """
        Get LLM with default configuration.

        Convenience method equivalent to: LLMManager.get_llm()
        """
        return cls.get_llm()

    @classmethod
    def clear_cache(cls):
        """Clear the LLM instance cache."""
        cls._llm_cache.clear()
        logger.info("LLM cache cleared")

    # ========================================================================
    # PRIVATE METHODS - LLM Creation Logic
    # ========================================================================

    @classmethod
    def _create_openrouter_llm(cls, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create OpenRouter LLM instance."""
        from langchain_openai import ChatOpenAI

        api_key = cls.API_KEYS.get("openrouter")
        if not api_key or api_key == "sk-or-v1-YOUR_OPENROUTER_KEY_HERE":
            raise ValueError(
                "OpenRouter API key not configured. "
                "Edit llm_manager.py and set API_KEYS['openrouter']"
            )

        return ChatOpenAI(
            base_url=cls.BASE_URLS["openrouter"],
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @classmethod
    def _create_openai_llm(cls, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create OpenAI LLM instance."""
        from langchain_openai import ChatOpenAI

        api_key = cls.API_KEYS.get("openai")
        if not api_key or api_key == "sk-YOUR_OPENAI_KEY_HERE":
            raise ValueError(
                "OpenAI API key not configured. "
                "Edit llm_manager.py and set API_KEYS['openai']"
            )

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @classmethod
    def _create_anthropic_llm(cls, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Anthropic (Claude) LLM instance."""
        from langchain_anthropic import ChatAnthropic

        api_key = cls.API_KEYS.get("anthropic")
        if not api_key or api_key == "sk-ant-YOUR_ANTHROPIC_KEY_HERE":
            raise ValueError(
                "Anthropic API key not configured. "
                "Edit llm_manager.py and set API_KEYS['anthropic']"
            )

        return ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @classmethod
    def _create_ollama_llm(cls, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Ollama LLM instance (local)."""
        from langchain_openai import ChatOpenAI

        # Ollama uses OpenAI-compatible API locally
        return ChatOpenAI(
            base_url=cls.BASE_URLS["ollama"],
            api_key="ollama",  # Dummy key, not used by Ollama
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @classmethod
    def _create_azure_llm(cls, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Azure OpenAI LLM instance."""
        from langchain_openai import AzureChatOpenAI

        api_key = cls.API_KEYS.get("azure")
        if not api_key or api_key == "YOUR_AZURE_KEY_HERE":
            raise ValueError(
                "Azure API key not configured. "
                "Edit llm_manager.py and set API_KEYS['azure']"
            )

        return AzureChatOpenAI(
            azure_deployment=cls.AZURE_CONFIG["deployment_name"],
            api_version=cls.AZURE_CONFIG["api_version"],
            azure_endpoint=cls.BASE_URLS["azure"],
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @classmethod
    def get_supported_providers(cls):
        """Get list of supported LLM providers."""
        return ["openrouter", "openai", "anthropic", "ollama", "azure"]

    @classmethod
    def validate_configuration(cls):
        """
        Validate that the current configuration is valid.

        Checks:
        - Default provider is supported
        - API key is configured for default provider
        - Base URLs are set correctly

        Raises:
            ValueError: If configuration is invalid
        """
        # Check default provider
        if cls.DEFAULT_PROVIDER not in cls.get_supported_providers():
            raise ValueError(
                f"Invalid DEFAULT_PROVIDER: {cls.DEFAULT_PROVIDER}. "
                f"Must be one of: {cls.get_supported_providers()}"
            )

        # Check API key for default provider (except Ollama)
        if cls.DEFAULT_PROVIDER != "ollama":
            api_key = cls.API_KEYS.get(cls.DEFAULT_PROVIDER)
            if not api_key or "YOUR_" in api_key or "_HERE" in api_key:
                raise ValueError(
                    f"API key not configured for {cls.DEFAULT_PROVIDER}. "
                    f"Edit llm_manager.py and set API_KEYS['{cls.DEFAULT_PROVIDER}']"
                )

        logger.info(f"LLM configuration valid: {cls.DEFAULT_PROVIDER} / {cls.DEFAULT_MODEL}")
        return True


# ============================================================================
# CONVENIENCE FUNCTIONS (Optional - for even simpler usage)
# ============================================================================

def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Convenience function to get LLM.

    Equivalent to LLMManager.get_llm(...) but shorter.
    """
    return LLMManager.get_llm(provider=provider, model=model, **kwargs)


def get_default_llm():
    """Get LLM with default configuration."""
    return LLMManager.get_default_llm()


# ============================================================================
# INITIALIZATION - Validate config on module import
# ============================================================================

try:
    # Validate configuration when module is imported
    # This will fail fast if config is invalid
    LLMManager.validate_configuration()
    logger.info("LLM Manager initialized successfully")
except Exception as e:
    logger.error(f"LLM Manager configuration error: {e}")
    logger.error("Edit llm_manager.py to fix configuration")
    # Don't raise - allow import to succeed for development

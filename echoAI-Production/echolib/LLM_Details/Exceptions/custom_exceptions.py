"""
Custom exceptions for LLM Factory system.
"""


class LLMBackendError(Exception):
    """Base exception for all LLM backend errors."""
    pass


class ConfigError(LLMBackendError):
    """Raised when there's an issue with configuration."""
    pass


class ModelNotFoundError(LLMBackendError):
    """Raised when requested model ID doesn't exist in config."""
    pass


class MissingAPIKeyError(LLMBackendError):
    """Raised when required API key is not found in environment."""
    pass


class InvalidProviderError(LLMBackendError):
    """Raised when provider is not supported."""
    pass


class LLMExecutionError(LLMBackendError):
    """Raised when LLM execution fails."""
    pass


class ValidationError(LLMBackendError):
    """Raised when validation fails."""
    pass


class EmbeddingError(LLMBackendError):
    """Raised when embedding generation fails."""
    pass
"""This will be used for Adapters"""

from typing import Dict, Any, Protocol, Optional
from openai import OpenAI
import logging
from pathlib import Path
import requests
import sys
logger = logging.getLogger(__name__)

# LLM Details
from echolib.LLM_Details.Core.llm_client_builder import LLMClientBuilder
from echolib.LLM_Details.Core.config_loader import ConfigLoader
from echolib.LLM_Details.Core.unified_model_wrapper import UnifiedModelWrapper
from echolib.LLM_Details.Exceptions import ModelNotFoundError, ValidationError, MissingAPIKeyError, ConfigError
from echolib.LLM_Details import fetch_models

class LLMFactory:
    """
    Factory for creating LLM model instances.
    
    Responsibilities:
    - Load configuration
    - Build LangChain clients
    - Wrap clients in UnifiedModelWrapper
    - Return model objects to users
    - Manage custom model registration
    """
    
    
    def __init__(self, tool_service: Optional[str] = None):
        """
        Initialize the LLM Factory.
        
        Args:
            tool_service: Optional tool service instance
        """
        
        print("✅ Initializing LLMFactory core components...")
        
        self.config_loader = ConfigLoader()
        self.tool_service = tool_service
        self.client_builder = LLMClientBuilder()
        self._model_cache = {}  # Optional: cache model instances
        
        logger.info("LLMFactory initialized successfully")
        print("✅ LLMFactory ready!")

    
    def get_model(
        self,
        model_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache: bool = False
    ) -> UnifiedModelWrapper:
        """
        Get a model instance by ID.
        
        This is the main method users call to get a model object.
        
        Args:
            model_id: Model identifier (e.g., "claude-opus-4")
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            cache: Whether to cache and reuse model instance
            
        Returns:
            UnifiedModelWrapper instance with all 6 core methods
            
        Example:
            factory = LLMFactory()
            model = factory.get_model("claude-opus-4")
            
            # Now use the model
            response = model.invoke("Hello")
            chat_response = model.chat([{"role": "user", "content": "Hi"}])
            embeddings = model.embed("Some text")
        """
        # Check cache first (if caching enabled)
        cache_key = f"{model_id}_{temperature}_{max_tokens}"
        if cache and cache_key in self._model_cache:
            logger.info(f"Returning cached model: {model_id}")
            return self._model_cache[cache_key]
        
        # Get model configuration
        logger.info(f"Creating model instance: {model_id}")
        model_config = self.config_loader.get_model_by_id(model_id)
        
        # Build LangChain clients
        llm_client = self.client_builder.build_client(
            model_config=model_config,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False
        )
        
        embeddings_client = self.client_builder.build_embeddings_client(
            model_config=model_config
        )
        
        # Wrap in UnifiedModelWrapper
        model = UnifiedModelWrapper(
            client=llm_client,
            embeddings_client=embeddings_client,
            model_id=model_id,
            model_config=model_config
        )
        
        # Cache if requested
        if cache:
            self._model_cache[cache_key] = model
        
        logger.info(f"Model created successfully: {model_id}")
        return model
    
    def list_models(self) -> Dict:
        """
        List all available models (pre-configured + custom).
        
        Returns:
            Dictionary with model information
        """
        models = self.config_loader.get_all_models()
        
        # Format for frontend
        formatted = []
        for model in models:
            formatted.append({
                "id": model.get("id"),
                "name": model.get("name"),
                "provider": model.get("provider"),
                "is_default": model.get("is_default", False),
                "description": model.get("description", "")
            })
        
        return {"models": formatted}
    
    def get_default_model(self) -> UnifiedModelWrapper:
        """Get the default model instance."""
        default_id = self.config_loader.get_default_model_id()
        if not default_id:
            raise ModelNotFoundError("No default model configured")
        
        return self.get_model(default_id)
    
    def register_custom_model(
        self,
        model_id: str,
        name: str,
        provider: str,
        base_url: str,
        api_key_env: str,
        model_name: str,
        description: str = "",
        is_default: bool = False
    ) -> Dict:
        """
        Register a user-defined custom model.
        
        This allows users to add their own LLMs via API/form.
        
        Args:
            model_id: Unique identifier
            name: Display name
            provider: Provider name (openrouter, custom, etc.)
            base_url: API endpoint URL
            api_key_env: Environment variable name for API key
            model_name: Model identifier at the provider
            description: Optional description
            is_default: Whether this should be the default model
            
        Returns:
            Dictionary with registration status
        """
        try:
            # Build configuration
            model_config = {
                "id": model_id,
                "name": name,
                "provider": provider,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "model_name": model_name,
                "description": description,
                "is_default": is_default
            }
            
            # Add to config loader
            self.config_loader.add_custom_model(model_config)
            
            logger.info(f"Custom model registered: {model_id}")
            
            return {
                "success": True,
                "message": f"Model '{name}' registered successfully",
                "model_id": model_id
            }
            
        except ValidationError as e:
            logger.error(f"Failed to register model: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def remove_custom_model(self, model_id: str) -> Dict:
        """Remove a custom model."""
        try:
            self.config_loader.remove_custom_model(model_id)
            
            # Clear from cache if present
            self._model_cache = {
                k: v for k, v in self._model_cache.items()
                if not k.startswith(model_id)
            }
            
            logger.info(f"Custom model removed: {model_id}")
            
            return {
                "success": True,
                "message": f"Model '{model_id}' removed successfully"
            }
            
        except ModelNotFoundError as e:
            return {
                "success": False,
                "message": str(e)
            }
    
    def clear_cache(self):
        """Clear the model instance cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
    
    def refresh_configuration(self):
        """Reload configuration from file."""
        self.config_loader.refresh_config()
        self.clear_cache()
        logger.info("Configuration refreshed")
    
    def get_providers(self) -> list:
        """Get list of available providers."""
        return self.config_loader.get_providers_list()
    
    def validate_model_config(self, model_config: Dict) -> Dict:
        """
        Validate a model configuration before adding.
        
        Args:
            model_config: Model configuration to validate
            
        Returns:
            Dictionary with validation result
        """
        try:
            # Check required fields
            required = ["id", "name", "provider", "base_url", "api_key_env", "model_name"]
            missing = [f for f in required if f not in model_config]
            
            if missing:
                return {
                    "valid": False,
                    "errors": [f"Missing required field: {f}" for f in missing]
                }
            
            # Check if model ID already exists
            if self.config_loader.validate_model_exists(model_config["id"]):
                return {
                    "valid": False,
                    "errors": [f"Model ID '{model_config['id']}' already exists"]
                }
            
            # Check if API key exists in environment
            if not self.client_builder.validate_api_key(model_config["api_key_env"]):
                return {
                    "valid": False,
                    "warnings": [
                        f"API key '{model_config['api_key_env']}' not found in environment. "
                        "Model will fail until key is set."
                    ]
                }
            
            return {"valid": True, "errors": [], "warnings": []}
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }



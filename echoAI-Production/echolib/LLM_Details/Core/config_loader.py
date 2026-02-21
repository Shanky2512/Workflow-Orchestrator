"""
Configuration Loader - Handles reading and parsing llm_provider.json and custom models.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from echolib.LLM_Details.Exceptions.custom_exceptions import ConfigError, ModelNotFoundError, ValidationError


class ConfigLoader:
    """
    Handles all operations related to LLM provider configuration.
    
    Loads models from:
    1. llm_provider.json (pre-configured models)
    2. In-memory storage for user-added custom models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ConfigLoader."""
        
        if config_path is None:
            # Find config at project root
            self.config_path = Path(__file__).parent.parent / "Providers" / "llm_provider.json"
        elif isinstance(config_path, str):
            self.config_path = Path(config_path)
        elif isinstance(config_path, Path):
            self.config_path = config_path
        else:
            # Something weird was passed
            raise TypeError(f"config_path must be str or Path, got: {type(config_path)}")
        
        self._config = None
        self._custom_models = {}
        
        # Validate file exists
        if not self.config_path.exists():
            raise ConfigError(f"Configuration file not found: {self.config_path}")

    
    def load_config(self) -> Dict:
        """Load and cache the configuration from JSON file."""
        if self._config is not None:
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            # Validate structure
            if "models" not in self._config:
                raise ConfigError("Configuration must contain 'models' key")
            
            if not isinstance(self._config["models"], list):
                raise ConfigError("'models' must be a list")
            
            return self._config
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    def get_all_models(self) -> List[Dict]:
        """Get list of all available models (pre-configured + custom)."""
        config = self.load_config()
        pre_configured = config.get("models", [])
        custom = list(self._custom_models.values())
        return pre_configured + custom
    
    def get_model_by_id(self, model_id: str) -> Dict:
        """Retrieve a specific model configuration by its ID."""
        # Check custom models first
        if model_id in self._custom_models:
            return self._custom_models[model_id]
        
        # Check pre-configured models
        models = self.load_config().get("models", [])
        for model in models:
            if model.get("id") == model_id:
                return model
        
        # Not found
        available_ids = [m.get("id") for m in self.get_all_models()]
        raise ModelNotFoundError(
            f"Model '{model_id}' not found. Available models: {available_ids}"
        )
    
    def add_custom_model(self, model_config: Dict) -> None:
        """Add a user-defined custom model."""
        # Validate required fields
        required = ["id", "name", "provider", "base_url", "api_key_env", "model_name"]
        for field in required:
            if field not in model_config:
                raise ValidationError(f"Missing required field: {field}")
        
        model_id = model_config["id"]
        
        # Check for duplicates
        try:
            self.get_model_by_id(model_id)
            raise ValidationError(f"Model '{model_id}' already exists")
        except ModelNotFoundError:
            # Good, doesn't exist yet
            pass
        
        # Add to custom models
        self._custom_models[model_id] = model_config
    
    def remove_custom_model(self, model_id: str) -> None:
        """Remove a custom model."""
        if model_id in self._custom_models:
            del self._custom_models[model_id]
        else:
            raise ModelNotFoundError(f"Custom model '{model_id}' not found")
    
    def get_default_model(self) -> Optional[Dict]:
        """Get the model marked as default."""
        models = self.get_all_models()
        
        if not models:
            return None
        
        # Look for is_default=true
        for model in models:
            if model.get("is_default", False):
                return model
        
        # Return first model if no default
        return models[0]
    
    def get_default_model_id(self) -> Optional[str]:
        """Get the ID of the default model."""
        default_model = self.get_default_model()
        return default_model.get("id") if default_model else None
    
    def validate_model_exists(self, model_id: str) -> bool:
        """Check if a model ID exists."""
        try:
            self.get_model_by_id(model_id)
            return True
        except ModelNotFoundError:
            return False
    
    def get_models_by_provider(self, provider: str) -> List[Dict]:
        """Get all models for a specific provider."""
        models = self.get_all_models()
        return [m for m in models if m.get("provider") == provider]
    
    def get_providers_list(self) -> List[str]:
        """Get list of unique providers."""
        models = self.get_all_models()
        providers = set(m.get("provider") for m in models if m.get("provider"))
        return sorted(list(providers))
    
    def refresh_config(self):
        """Force reload of configuration from file."""
        self._config = None
        self.load_config()


# Example usage
if __name__ == "__main__":
    try:
        loader = ConfigLoader()
        print("✅ ConfigLoader initialized")
        
        models = loader.get_all_models()
        print(f"✅ Found {len(models)} models")
        
        for model in models:
            print(f"   - {model['id']}: {model['name']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
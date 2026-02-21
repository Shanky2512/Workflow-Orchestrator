"""
Base Model - Abstract base class that defines the contract for all LLM models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional, Union


class BaseModel(ABC):
    """
    Abstract base class for all LLM models.
    
    All models MUST implement these 6 core methods:
    1. invoke() - Simple text generation
    2. chat() - Conversation with history
    3. stream() - Streaming generation
    4. json_mode() - Structured JSON output
    5. function_call() - Tool/function calling
    6. embed() - Text embeddings
    """
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Simple text generation - single turn.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Conversation with history - multi-turn.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            **kwargs: Additional parameters
            
        Returns:
            Assistant's response text
        """
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Streaming text generation - yields chunks.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they're generated
        """
        pass
    
    @abstractmethod
    def json_mode(
        self, 
        prompt: str, 
        schema: Optional[Dict] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Structured JSON output - guaranteed valid JSON.
        
        Args:
            prompt: Text prompt
            schema: Optional JSON schema to enforce structure
            **kwargs: Additional parameters
            
        Returns:
            Dictionary (parsed JSON)
        """
        pass
    
    @abstractmethod
    def function_call(
        self,
        tools: List[Dict],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Native function/tool calling.
        
        Args:
            tools: List of function definitions
            prompt: User prompt/task
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with function name and arguments
            Format: {"function": "name", "arguments": {...}}
        """
        pass
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate text embeddings - vector representations.
        
        Args:
            text: Single text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            Single embedding vector or list of vectors
        """
        pass
    
    # Optional helper methods that subclasses can override
    def get_config(self) -> Dict:
        """Get model configuration."""
        return getattr(self, '_config', {})
    
    def get_model_id(self) -> str:
        """Get model identifier."""
        return getattr(self, '_model_id', 'unknown')
    
    def get_provider(self) -> str:
        """Get provider name."""
        return getattr(self, '_provider', 'unknown')
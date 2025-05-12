"""
Base LLM configuration for the Agent Framework.
Provides common functionality and interfaces for LLM interactions.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

class BaseLLMConfig(ABC):
    """
    Base configuration for Large Language Model interactions.
    Provides common functionality and interfaces.
    """
    
    def __init__(
        self, 
        model_name: str,
        temperature: float = 0.7, 
        max_tokens: int = 32768,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        api_key: Optional[str] = None,
        timeout: int = 60,
        retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the base LLM configuration.
        
        Args:
            model_name: Name of the model to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalizes repeated tokens
            presence_penalty: Penalizes repeated topics
            api_key: API key for the provider (defaults to env variable)
            timeout: Timeout for API calls in seconds
            retry_attempts: Number of retry attempts for failed calls
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_key = api_key or self._get_api_key_from_env()
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variables."""
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning("No API key found in environment variables (LLM_API_KEY)")
        return api_key
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for LLM calls.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Main prompt text
            system_prompt: Optional system instructions
            params: Optional parameters to override defaults
            
        Returns:
            Response from the LLM
        """
        pass
    
    @abstractmethod
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with conversation context.
        
        Args:
            prompt: Main prompt text
            context: List of previous messages
            system_prompt: Optional system instructions
            params: Optional parameters to override defaults
            
        Returns:
            Response from the LLM
        """
        pass
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
        """
        attempts = 0
        last_error = None
        
        while attempts < self.retry_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempts+1} failed: {str(e)}")
                attempts += 1
                if attempts < self.retry_attempts:
                    time.sleep(self.retry_delay)
        
        logger.error(f"All {self.retry_attempts} attempts failed. Last error: {last_error}")
        raise last_error
    
    def format_prompt(self, prompt: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a prompt with variables.
        
        Args:
            prompt: Prompt template
            variables: Variables to insert
            
        Returns:
            Formatted prompt
        """
        if not variables:
            return prompt
            
        return prompt.format(**variables)
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a rough estimation method; providers often have their own tokenizers.
        
        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation - about 4 characters per token for English text
        return len(text) // 4
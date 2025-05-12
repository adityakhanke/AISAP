"""
Groq-specific LLM configuration for the Agent Framework.
Implements the Groq Cloud API integration.
"""

import os
import json
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Union

from ...core.llm import BaseLLMConfig

logger = logging.getLogger(__name__)

class GroqLLMConfig(BaseLLMConfig):
    """
    Groq-specific implementation of the LLM configuration.
    Handles communication with the Groq Cloud API.
    """
    
    def __init__(
        self,
        model_name: str = "llama2-70b-4096",
        api_key: Optional[str] = None,
        api_base: str = "https://api.groq.com/openai/v1",
        **kwargs
    ):
        """
        Initialize the Groq LLM configuration.
        
        Args:
            model_name: Name of the model to use on Groq
            api_key: Groq API key (defaults to env variable)
            api_base: Base URL for the Groq API
            **kwargs: Additional parameters for the base class
        """
        # Get API key from GROQ_API_KEY env var as fallback
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.api_base = api_base
        
        # Verify we have an API key
        if not self.api_key:
            logger.error("No Groq API key provided or found in environment variables")
            raise ValueError("Groq API key is required")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables specific to Groq."""
        return os.environ.get("GROQ_API_KEY")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the Groq API.
        
        Args:
            prompt: Main prompt text
            system_prompt: Optional system instructions
            params: Optional parameters to override defaults
            
        Returns:
            Response from the LLM
        """
        # Prepare parameters by combining defaults with overrides
        request_params = self.get_default_params()
        if params:
            request_params.update(params)
        
        # Prepare messages format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request data
        request_data = {
            "model": request_params.pop("model"),
            "messages": messages,
            **request_params
        }
        
        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API returned status {response.status}: {error_text}")

                    result = await response.json()

                    # Extract and process the response
                    processed_response = {
                        "text": result["choices"][0]["message"]["content"],
                        "model": result["model"],
                        "id": result["id"],
                        "finish_reason": result["choices"][0].get("finish_reason"),
                        "raw_response": result
                    }

                    return processed_response
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise

    async def generate_with_context(
    self,
    prompt: str,
    context: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
        """
        Generate text with conversation context using the Groq API.

        Args:
            prompt: Main prompt text
            context: List of previous messages (each with 'role' and 'content')
            system_prompt: Optional system instructions
            params: Optional parameters to override defaults

        Returns:
            Response from the LLM
        """
        # Prepare parameters by combining defaults with overrides
        request_params = self.get_default_params()
        if params:
            request_params.update(params)

        # Prepare messages format with context
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation context with cleaned messages - only include role and content
        for msg in context:
            # Clean the message to only include supported fields
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            messages.append(clean_msg)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Prepare request data
        request_data = {
            "model": request_params.pop("model"),
            "messages": messages,
            **request_params
        }

        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API returned status {response.status}: {error_text}")

                    result = await response.json()
                    
                    # Extract and process the response
                    processed_response = {
                        "text": result["choices"][0]["message"]["content"],
                        "model": result["model"],
                        "id": result["id"],
                        "finish_reason": result["choices"][0].get("finish_reason"),
                        "raw_response": result
                    }

                    return processed_response
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise
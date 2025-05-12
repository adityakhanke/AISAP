"""
Mistral Vertex AI-specific LLM configuration for the Agent Framework.
Implements the Mistral AI integration via Google Vertex AI.
"""

import os
import json
import logging
import aiohttp
import httpx
import google.auth
from google.auth.transport.requests import Request
from typing import Dict, Any, List, Optional, Union

from ...core.llm import BaseLLMConfig

logger = logging.getLogger(__name__)

class MistralLLMConfig(BaseLLMConfig):
    """
    Mistral Vertex AI-specific implementation of the LLM configuration.
    Handles communication with Mistral AI models hosted on Google Vertex AI.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Mistral Vertex AI LLM configuration.
        
        Args:
            model_name: Name of the Mistral model to use on Vertex AI
            project_id: Google Cloud Project ID
            region: Google Cloud region for Vertex AI
            **kwargs: Additional parameters for the base class
        """
        # Remove settings to avoid duplicate parameters
        if 'settings' in kwargs:
            kwargs.pop('settings')
            
        # Hardcode project_id and region - matching your test.py
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.region = region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.model_name = model_name or "codestral-2501"
        
        # Pass remaining kwargs to base class
        super().__init__(model_name=self.model_name, **kwargs)
        
        logger.info(f"Initialized Mistral Vertex AI LLM config with model: {self.model_name} in {self.region}")
    
    def _build_endpoint_url(self, streaming: bool = False) -> str:
        """
        Build the Vertex AI endpoint URL for the model.
        
        Args:
            streaming: Whether to use streaming endpoint
            
        Returns:
            Complete endpoint URL
        """
        base_url = f"https://{self.region}-aiplatform.googleapis.com/v1/"
        project_fragment = f"projects/{self.project_id}"
        location_fragment = f"locations/{self.region}"
        specifier = "streamRawPredict" if streaming else "rawPredict"
        model_fragment = f"publishers/mistralai/models/{self.model_name}"
        url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
        return url
    
    def _get_auth_token(self) -> str:
        """
        Get Google Cloud authentication token.
        
        Returns:
            Authentication token
        """
        try:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            logger.error(f"Error getting Google Cloud authentication token: {e}")
            raise
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Authorization": f"Bearer {self._get_auth_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the Mistral model on Vertex AI.
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
            "model": self.model_name,
            "messages": messages,
            "temperature": request_params.get("temperature", 0.7),
            "max_tokens": request_params.get("max_tokens", 32768),
            "top_p": request_params.get("top_p", 0.9),
            "stream": False
        }

        # Build URL
        url = self._build_endpoint_url(streaming=False)

        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Vertex AI error: {response.status} - {error_text}")
                        raise Exception(f"Vertex AI returned status {response.status}: {error_text}")

                    result = await response.json()
                    logger.debug(f"Response structure: {json.dumps(result)[:500]}")

                    # Extract content from the choices field (not candidates)
                    if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                        content = result["choices"][0]["message"]["content"]
                        finish_reason = result["choices"][0].get("finish_reason", "")
                    else:
                        logger.error(f"Unexpected response structure: {json.dumps(result)}")
                        content = f"Error: Could not parse API response. Raw response: {json.dumps(result)[:500]}"
                        finish_reason = "error"

                    # Extract and process the response
                    processed_response = {
                        "text": content,
                        "model": self.model_name,
                        "id": result.get("id", ""),
                        "finish_reason": finish_reason,
                        "raw_response": result
                    }

                    return processed_response
        except Exception as e:
            logger.error(f"Error calling Vertex AI: {str(e)}")
            # Return a graceful error response
            return {
                "text": f"Error calling Vertex AI: {str(e)}",
                "model": self.model_name,
                "id": "",
                "finish_reason": "error",
                "raw_response": None
            }

    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with conversation context using the Mistral model on Vertex AI.

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
            "model": self.model_name,
            "messages": messages,
            "temperature": request_params.get("temperature", 0.7),
            "max_tokens": request_params.get("max_tokens", 32768),
            "top_p": request_params.get("top_p", 0.9),
            "stream": False
        }

        # Build URL
        url = self._build_endpoint_url(streaming=False)
        
        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Vertex AI error: {response.status} - {error_text}")
                        raise Exception(f"Vertex AI returned status {response.status}: {error_text}")

                    result = await response.json()
                    
                    # Extract and process the response
                    processed_response = {
                        "text": result["candidates"][0]["content"],
                        "model": self.model_name,
                        "id": result.get("id", ""),
                        "finish_reason": result["candidates"][0].get("finish_reason", ""),
                        "raw_response": result
                    }

                    return processed_response
        except Exception as e:
            logger.error(f"Error calling Vertex AI: {str(e)}")
            raise

    def sync_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text synchronously (non-async) for testing or simple use cases.
        
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
            "model": self.model_name,
            "messages": messages,
            "temperature": request_params.get("temperature", 0.7),
            "max_tokens": request_params.get("max_tokens", 32768),
            "top_p": request_params.get("top_p", 0.9),
            "stream": False
        }
        
        # Build URL
        url = self._build_endpoint_url(streaming=False)
        
        # Make API call
        try:
            with httpx.Client() as client:
                response = client.post(
                    url,
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"Vertex AI error: {response.status_code} - {response.text}")
                    raise Exception(f"Vertex AI returned status {response.status_code}: {response.text}")

                result = response.json()
                
                # Extract and process the response
                processed_response = {
                    "text": result["candidates"][0]["content"],
                    "model": self.model_name,
                    "id": result.get("id", ""),
                    "finish_reason": result["candidates"][0].get("finish_reason", ""),
                    "raw_response": result
                }

                return processed_response
        except Exception as e:
            logger.error(f"Error calling Vertex AI: {str(e)}")
            raise
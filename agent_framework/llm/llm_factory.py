"""
Factory for creating LLM configurations.
Simplifies the instantiation of appropriate LLM configs.
"""

import os
import logging
from typing import Dict, Any, Optional, Type

from ..core.llm import BaseLLMConfig
from .providers.groq_llm import GroqLLMConfig
from .providers.mistral_llm import MistralLLMConfig
from .agent_llm.pm_agent_llm import PMAgentLLMConfig
from .agent_llm.dev_agent_llm import DevAgentLLMConfig

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory for creating LLM configurations.
    Simplifies the instantiation of appropriate LLM configs.
    """
    
    # Registry of available LLM configurations
    _registry: Dict[str, Type[BaseLLMConfig]] = {
        "groq": GroqLLMConfig,
        "mistral": MistralLLMConfig,
        "pm_agent": PMAgentLLMConfig,
        "dev_agent": DevAgentLLMConfig,
    }

    @classmethod
    def register_config(cls, name: str, config_class: Type[BaseLLMConfig]) -> None:
        """
        Register a new LLM configuration class.
        
        Args:
            name: Name for the configuration
            config_class: Configuration class to register
        """
        cls._registry[name] = config_class
        logger.info(f"Registered LLM configuration: {name}")
    
    @classmethod
    def create(
        cls, 
        config_type: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLLMConfig:
        """
        Create an LLM configuration instance.
        
        Args:
            config_type: Type of configuration to create
            api_key: Optional API key
            model_name: Optional model name
            project_id: Optional Google Cloud Project ID for Vertex AI
            region: Optional Google Cloud region for Vertex AI
            settings: Optional settings dictionary from config file
            **kwargs: Additional parameters for the configuration
            
        Returns:
            LLM configuration instance
            
        Raises:
            ValueError: If the configuration type is not registered
        """
        if config_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown LLM configuration type: {config_type}. Available types: {available_types}")
        
        config_class = cls._registry[config_type]
        
        # Prepare initialization parameters
        init_params = {}
        
        # Add settings if provided
        if settings:
            init_params["settings"] = settings
            
            # Get appropriate settings section for this LLM type
            llm_settings = None
            
            if "vertex" in config_type:
                llm_settings = settings.get("vertex_ai", {})
            elif "groq" in config_type:
                llm_settings = settings.get("groq", {})
            elif "mistral" in config_type:
                llm_settings = settings.get("mistral", {})
            
            # Get provider-specific settings if available
            if llm_settings:
                # Use model name from settings if not provided explicitly
                if not model_name and "model_name" in llm_settings:
                    model_name = llm_settings["model_name"]
                
                # Use API key from settings if not provided explicitly
                if not api_key and "api_key" in llm_settings:
                    api_key = llm_settings["api_key"]
                
                # For Vertex AI, get project_id and region from settings if available
                if "vertex" in config_type:
                    if not project_id and "project_id" in llm_settings:
                        project_id = llm_settings["project_id"]
                    if not region and "region" in llm_settings:
                        region = llm_settings["region"]
            
            # For agent types, also look in agent-specific settings
            if "agent" in config_type:
                agent_type = config_type.split("_")[0]  # e.g., "dev" from "dev_agent_vertex"
                agent_settings = settings.get("agents", {}).get(f"{agent_type}_agent", {})
                
                # Use agent-specific model if available
                if not model_name:
                    if "vertex" in config_type and "vertex_model_name" in agent_settings:
                        model_name = agent_settings["vertex_model_name"]
                    elif "groq" in config_type and "groq_model_name" in agent_settings:
                        model_name = agent_settings["groq_model_name"]
                    elif "mistral" in config_type and "mistral_model_name" in agent_settings:
                        model_name = agent_settings["mistral_model_name"]
                    elif "model_name" in agent_settings:
                        model_name = agent_settings["model_name"]
                        
                # Use agent-specific API key if available
                if not api_key:
                    if "vertex" in config_type and "vertex_api_key" in agent_settings:
                        api_key = agent_settings["vertex_api_key"]
                    elif "groq" in config_type and "groq_api_key" in agent_settings:
                        api_key = agent_settings["groq_api_key"]
                    elif "mistral" in config_type and "mistral_api_key" in agent_settings:
                        api_key = agent_settings["mistral_api_key"]
                    elif "api_key" in agent_settings:
                        api_key = agent_settings["api_key"]
        
        # Add explicit parameters with highest priority
        if api_key:
            init_params["api_key"] = api_key
        if model_name:
            init_params["model_name"] = model_name
        
        # Add Vertex AI specific parameters if needed
        if "vertex" in config_type:
            if project_id:
                init_params["project_id"] = project_id
            if region:
                init_params["region"] = region
        
        # Add any additional parameters
        init_params.update(kwargs)
        
        # Create and return the instance
        try:
            instance = config_class(**init_params)
            logger.info(f"Created LLM configuration of type: {config_type}")
            return instance
        except Exception as e:
            logger.error(f"Error creating LLM configuration of type {config_type}: {str(e)}")
            raise
    
    @classmethod
    def create_from_env(cls, config_type: str, settings: Optional[Dict[str, Any]] = None, **kwargs) -> BaseLLMConfig:
        """
        Create an LLM configuration instance using API key from environment variables.
        
        Args:
            config_type: Type of configuration to create
            settings: Optional settings dictionary from config file
            **kwargs: Additional parameters for the configuration

        Returns:
            LLM configuration instance
        """
        # For Vertex AI integration, get project_id from environment
        if "vertex" in config_type:
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
            kwargs["project_id"] = project_id
            kwargs["region"] = region
            
        # For other types, get API key from environment
        elif config_type == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            kwargs["api_key"] = api_key
        elif config_type == "mistral":
            api_key = os.environ.get("MISTRAL_API_KEY")
            kwargs["api_key"] = api_key
        else:
            # Try generic API key environment variable
            api_key = os.environ.get("LLM_API_KEY")
            kwargs["api_key"] = api_key
            
        # Pass settings from config if provided
        if settings:
            kwargs["settings"] = settings
            
        return cls.create(config_type, **kwargs)
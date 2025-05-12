"""
Framework initialization and setup utilities.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

from .context.manager import ContextManager
from .core.registry import AgentRegistry
from .tools.registry import ToolRegistry
from .rag.adapter import RAGAdapter
from .rag.contextual_rag import ContextualRAG
from .config.settings import load_config, default_settings

logger = logging.getLogger(__name__)

def initialize_framework(config: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Initialize the agent framework with the given configuration.

    Args:
        config: Path to configuration file, configuration dictionary, or None for defaults

    Returns:
        Dictionary of initialized framework components
    """
    # Handle configuration
    if isinstance(config, str):
        # Load from file path
        settings = load_config(config)
    elif isinstance(config, dict):
        # Use provided dictionary
        settings = config
    else:
        # Use default settings
        settings = default_settings
    
    logger.info("Initializing Agent Framework with settings")
    
    # Initialize context manager
    context_manager = ContextManager(settings)
    logger.info("Context Manager initialized")
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    logger.info("Tool Registry initialized")
    
    # Initialize agent registry
    agent_registry = AgentRegistry()
    logger.info("Agent Registry initialized")
    
    # Initialize RAG if settings are provided
    rag_adapter = None
    contextual_rag = None
    if 'rag' in settings:
        rag_settings = settings['rag']
        try:
            # Try to initialize RAG adapter
            rag_adapter = RAGAdapter(**rag_settings)
            logger.info("RAG Adapter initialized")
            
            # Initialize contextual RAG
            contextual_rag = ContextualRAG(context_manager, rag_adapter)
            logger.info("Contextual RAG initialized")
        except Exception as e:
            logger.warning(f"Could not initialize RAG: {e}")
    
    # Initialize workflow engine if needed (added in future)
    workflow_engine = None  # Placeholder for workflow engine
    
    # Return all initialized components
    return {
        "context_manager": context_manager,
        "agent_registry": agent_registry,
        "tool_registry": tool_registry,
        "rag_adapter": rag_adapter,
        "contextual_rag": contextual_rag,
        "workflow_engine": workflow_engine,
        "settings": settings
    }

def setup_logging(config: Dict[str, Any] = None):
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary with logging settings
    """
    if not config:
        config = {}
    
    log_config = config.get('logging', {})
    log_level_name = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')
    
    # Convert string log level to numeric
    log_level = getattr(logging, log_level_name.upper())
    
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format=log_format
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging initialized at level: {log_level_name}")
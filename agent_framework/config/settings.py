"""
Configuration settings for the Agent Framework.
"""

import os
import yaml
from typing import Dict, Any

# Default settings
default_settings = {
    "rag": {
        "persist_directory": "./chroma_db",
        "embedding_model": "nomic-ai/nomic-embed-code",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunking_strategy": "semantic",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "context": {
        "store_type": "memory",  # Options: memory, sqlite, mongodb
        "store_path": "./context_store",
        "versioning": True,
        "schema_validation": True,
    },
    "workflow": {
        "state_storage": "memory",  # Options: memory, sqlite, redis
        "max_workflow_steps": 50,
        "event_history_size": 100,
    },
    "agents": {
        "default_llm": "gpt-4",
        "memory_limit": 10,  # Number of interactions to keep in memory
    },
    "security": {
        "context_access_control": False,  # Enable/disable access control
        "audit_logging": True,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with defaults.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the merged configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load user configuration
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    # Deep merge with default settings
    merged_config = deep_merge(default_settings, user_config)
    
    return merged_config

def deep_merge(default: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dictionaries, with override values taking precedence.
    
    Args:
        default: Default dictionary
        override: Override dictionary with values to merge
        
    Returns:
        Merged dictionary
    """
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def get_setting(settings: Dict[str, Any], path: str, default=None) -> Any:
    """
    Get a setting using dot notation path.
    
    Args:
        settings: Settings dictionary
        path: Dot notation path (e.g., "rag.embedding_model")
        default: Default value if path not found
        
    Returns:
        Setting value or default
    """
    parts = path.split('.')
    current = settings
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default
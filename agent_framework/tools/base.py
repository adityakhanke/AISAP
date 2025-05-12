"""
Base classes for tools that agents can use.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

from ..core.interfaces import ContextInterface

logger = logging.getLogger(__name__)

class Tool(ABC):
    """
    Abstract base class for tools that agents can use.
    A tool is a capability that an agent can invoke to perform a specific task.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **params) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            **params: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get the parameter schema for the tool.
        
        Returns:
            JSON Schema describing the tool's parameters
        """
        # Inspect the execute method signature
        sig = inspect.signature(self.execute)
        
        # Create a schema for each parameter
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            # Skip self parameter
            if name == 'self':
                continue
            
            # Get parameter type annotation if available
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                param_type = Any
            
            # Determine JSON Schema type based on Python type
            if param_type in (str, Optional[str]):
                schema_type = "string"
            elif param_type in (int, Optional[int]):
                schema_type = "integer"
            elif param_type in (float, Optional[float]):
                schema_type = "number"
            elif param_type in (bool, Optional[bool]):
                schema_type = "boolean"
            elif param_type in (list, List, Optional[list], Optional[List]):
                schema_type = "array"
            elif param_type in (dict, Dict, Optional[dict], Optional[Dict]):
                schema_type = "object"
            else:
                schema_type = "string"  # Default to string for unknown types
            
            # Create parameter schema
            param_schema = {"type": schema_type}
            
            # Add description if available in docstring
            param_schema["description"] = f"Parameter: {name}"
            
            # Add to properties
            properties[name] = param_schema
            
            # If parameter has no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(name)
        
        # Create overall schema
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the complete schema for the tool.
        
        Returns:
            Schema describing the tool, its parameters, and return type
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameter_schema()
        }


class FunctionTool(Tool):
    """
    Tool implementation that wraps a function.
    """
    
    def __init__(self, name: str, description: str, func: Callable):
        """
        Initialize the function tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            func: Function to execute
        """
        super().__init__(name, description)
        self.func = func
    
    def execute(self, **params) -> Any:
        """
        Execute the wrapped function with the given parameters.
        
        Args:
            **params: Function parameters
            
        Returns:
            Function result
        """
        try:
            return self.func(**params)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            raise


class ContextTool(Tool):
    """
    Tool for interacting with the context manager.
    """
    
    def __init__(self, name: str, description: str, context_manager: ContextInterface):
        """
        Initialize the context tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            context_manager: Context manager instance
        """
        super().__init__(name, description)
        self.context_manager = context_manager
    
    def execute(self, **params) -> Any:
        """
        Execute a context operation.
        
        Args:
            **params: Operation parameters
            
        Returns:
            Operation result
        """
        operation = params.pop("operation", None)
        if operation is None:
            raise ValueError("Operation parameter is required")
        
        if operation == "query":
            entity_type = params.pop("entity_type", None)
            filters = params.pop("filters", {})
            limit = params.pop("limit", None)
            
            entities = self.context_manager.query_entities(entity_type, filters, limit)
            return [entity.to_dict() for entity in entities]
        
        elif operation == "get":
            entity_id = params.pop("entity_id", None)
            if entity_id is None:
                raise ValueError("entity_id parameter is required")
            
            entity = self.context_manager.get_entity(entity_id)
            return entity.to_dict() if entity else None
        
        elif operation == "create":
            entity_type = params.pop("entity_type", None)
            data = params.pop("data", {})
            
            if entity_type is None:
                raise ValueError("entity_type parameter is required")
            
            entity = self.context_manager.create_entity(entity_type, data)
            return entity.to_dict()
        
        elif operation == "update":
            entity_id = params.pop("entity_id", None)
            data = params.pop("data", {})
            
            if entity_id is None:
                raise ValueError("entity_id parameter is required")
            
            entity = self.context_manager.update_entity(entity_id, data)
            return entity.to_dict() if entity else None
        
        elif operation == "delete":
            entity_id = params.pop("entity_id", None)
            
            if entity_id is None:
                raise ValueError("entity_id parameter is required")
            
            success = self.context_manager.delete_entity(entity_id)
            return {"success": success}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
"""
Registry for tools that agents can use.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Type

from ..core.interfaces import ToolInterface
from .base import Tool, FunctionTool
import asyncio

logger = logging.getLogger(__name__)

class ToolRegistry(ToolInterface):
    """
    Registry for tools that agents can use.
    Provides functionality for registering and executing tools.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Tool] = {}
        logger.info("Initialized tool registry")
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_function(self, name: str, description: str, func: Callable) -> None:
        """
        Register a function as a tool.
        
        Args:
            name: Name for the tool
            description: Description of what the tool does
            func: Function to execute
        """
        tool = FunctionTool(name, description, func)
        self.register_tool(tool)
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self.tools
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name}")
        
        try:
            result = tool.execute(**params)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was found and unregistered, False otherwise
        """
        if tool_name not in self.tools:
            return False
        
        del self.tools[tool_name]
        logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self.tools.values())
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            List of tool descriptions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.get_schema()
            }
            for tool in self.tools.values()
        ]

    async def execute_tool_async(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool asynchronously with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool asynchronously: {tool_name}")
        
        try:
            # Check if the tool's execute method is a coroutine function
            if asyncio.iscoroutinefunction(tool.execute):
                result = await tool.execute(**params)
            else:
                # For non-async tools, run in executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool.execute(**params))
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
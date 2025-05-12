"""
Base agent implementation - foundation for all specialized agents.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable

from ..core.interfaces import AgentInterface, ContextInterface, ToolInterface
from ..core.memory import WorkingMemory

logger = logging.getLogger(__name__)

class BaseAgent(AgentInterface, ABC):
    """
    Abstract base class for all agents in the framework.
    Provides the common foundation that all specialized agents build upon.
    """
    
    def __init__(self, agent_id: str, name: str, context_manager: ContextInterface, 
               tool_registry: Optional[ToolInterface] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_name: Human-readable name for this agent
            context_manager: Context manager for accessing shared context
            tool_registry: Registry of tools available to this agent
        """
        self._agent_id = agent_id
        self._name = name
        self.context_manager = context_manager
        self.tool_registry = tool_registry
        self.working_memory = WorkingMemory()
        
        # Initialize session state
        self.session_id = None
        self.conversation_history = []
        
        logger.info(f"Initialized agent: {name} (ID: {agent_id})")
    
    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self._agent_id
    
    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new agent session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_history = []
        self.working_memory.clear()
        
        logger.info(f"Agent {self.name} started session: {self.session_id}")
        return self.session_id
    
    def end_session(self) -> None:
        """End the current session."""
        logger.info(f"Agent {self.name} ended session: {self.session_id}")
        self.session_id = None
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an input and generate a response.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Response data
        """
        if not self.session_id:
            self.start_session()
        
        # Add input to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": input_data.get("content", ""),
            "timestamp": time.time()
        })
        
        # Analyze the input (implemented by subclasses)
        analysis_result = self.analyze_input(input_data)
        
        # Determine intent and plan (implemented by subclasses)
        intent, plan = self.determine_intent(analysis_result)
        
        # Execute the plan (implemented by subclasses)
        execution_result = self.execute_plan(plan)
        
        # Generate the response (implemented by subclasses)
        response = self.generate_response(execution_result)
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.get("content", ""),
            "timestamp": time.time()
        })
        
        return response
    
    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use a tool from the tool registry.

        Args:
            tool_name: Name of the tool to use
            params: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.tool_registry or not self.tool_registry.has_tool(tool_name):
            logger.warning(f"Tool not found: {tool_name}")
            return {"success": False, "error": f"Tool not found: {tool_name}"}

        try:
            logger.info(f"Agent {self.name} using tool: {tool_name}")
            result = await self.tool_registry.execute_tool_async(tool_name, params)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def update_context(self, updates: Dict[str, Any]) -> bool:
        """
        Update the shared context with new information.
        
        Args:
            updates: Dictionary of context updates
            
        Returns:
            True if successful
        """
        try:
            for entity_type, entity_data in updates.items():
                if isinstance(entity_data, list):
                    # Process multiple entities of this type
                    for data in entity_data:
                        entity_id = data.pop("id", None)
                        if entity_id:
                            # Update existing entity
                            self.context_manager.update_entity(entity_id, data)
                        else:
                            # Create new entity
                            self.context_manager.create_entity(entity_type, data)
                else:
                    # Process single entity
                    entity_id = entity_data.pop("id", None)
                    if entity_id:
                        # Update existing entity
                        self.context_manager.update_entity(entity_id, entity_data)
                    else:
                        # Create new entity
                        self.context_manager.create_entity(entity_type, entity_data)
            
            return True
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return False
    
    def query_context(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the shared context for information.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching entities
        """
        try:
            entity_type = query.get("entity_type")
            filters = query.get("filters", {})
            limit = query.get("limit")
            
            entities = self.context_manager.query_entities(entity_type, filters, limit)
            return [entity.to_dict() for entity in entities]
        except Exception as e:
            logger.error(f"Error querying context: {e}")
            return []
    
    @abstractmethod
    def analyze_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the input to extract key information.
        
        Args:
            input_data: Input data to analyze
            
        Returns:
            Analysis result
        """
        pass
    
    @abstractmethod
    def determine_intent(self, analysis_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the user's intent and create an execution plan.
        
        Args:
            analysis_result: Result of input analysis
            
        Returns:
            Tuple of (intent, plan)
        """
        pass
    
    @abstractmethod
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan based on the determined intent.
        
        Args:
            plan: Execution plan
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def generate_response(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response based on the execution result.
        
        Args:
            execution_result: Result of plan execution
            
        Returns:
            Response data
        """
        pass
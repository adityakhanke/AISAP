"""
Registry for managing agents in the framework.
"""

import logging
from typing import Dict, Any, List, Optional, Type

from .interfaces import AgentInterface

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Registry for managing agents in the framework.
    Provides functionality for registering, retrieving, and managing agents.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        # Map of agent ID to agent instance
        self.agents: Dict[str, AgentInterface] = {}
        
        # Map of agent type to agent class
        self.agent_types: Dict[str, Type[AgentInterface]] = {}
        
        logger.info("Initialized agent registry")
    
    def register_agent(self, agent: AgentInterface) -> None:
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} (ID: {agent.agent_id})")
    
    def register_agent_type(self, agent_type: str, agent_class: Type[AgentInterface]) -> None:
        """
        Register an agent type.
        
        Args:
            agent_type: Type name for the agent
            agent_class: Agent class
        """
        self.agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    def get_agent(self, agent_id: str) -> Optional[AgentInterface]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[AgentInterface]:
        """
        Get an agent by name.
        
        Args:
            name: Name of the agent to retrieve
            
        Returns:
            Agent instance if found, None otherwise
        """
        for agent in self.agents.values():
            if agent.name == name:
                return agent
        return None
    
    def get_agent_by_type(self, agent_type: str) -> Optional[AgentInterface]:
        """
        Get the first agent of a specific type.
        
        Args:
            agent_type: Type of agent to retrieve
            
        Returns:
            Agent instance if found, None otherwise
        """
        for agent in self.agents.values():
            if type(agent).__name__ == agent_type or agent.__class__.__name__ == agent_type:
                return agent
        return None
    
    def create_agent(self, agent_type: str, agent_id: str, name: str, context_manager, tool_registry=None, **kwargs) -> Optional[AgentInterface]:
        """
        Create and register a new agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            agent_id: ID for the new agent
            name: Name for the new agent
            context_manager: Context manager to use
            tool_registry: Optional tool registry to use
            **kwargs: Additional arguments for the agent constructor
            
        Returns:
            Created agent instance if successful, None otherwise
        """
        if agent_type not in self.agent_types:
            logger.error(f"Unknown agent type: {agent_type}")
            return None
        
        try:
            agent_class = self.agent_types[agent_type]
            agent = agent_class(
                agent_id=agent_id, 
                name=name, 
                context_manager=context_manager,
                tool_registry=tool_registry,
                **kwargs
            )
            self.register_agent(agent)
            return agent
        except Exception as e:
            logger.error(f"Error creating agent of type {agent_type}: {e}")
            return None
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if successful, False if agent not found
        """
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        logger.info(f"Unregistered agent with ID: {agent_id}")
        return True
    
    def get_all_agents(self) -> List[AgentInterface]:
        """
        Get all registered agents.
        
        Returns:
            List of all registered agents
        """
        return list(self.agents.values())
    
    def get_all_agent_types(self) -> List[str]:
        """
        Get all registered agent types.
        
        Returns:
            List of all registered agent types
        """
        return list(self.agent_types.keys())
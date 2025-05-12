"""
Core interfaces and abstract base classes for the Agent Framework.
These interfaces help break circular dependencies between modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable


class ContextInterface(ABC):
    """Interface for context operations to avoid circular dependencies."""
    
    @abstractmethod
    def create_entity(self, entity_type: str, data: Dict[str, Any], 
                     entity_id: Optional[str] = None) -> Any:
        """Create a new entity in the context."""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str, version: Optional[int] = None) -> Optional[Any]:
        """Get an entity by ID and optional version."""
        pass
    
    @abstractmethod
    def update_entity(self, entity_id: str, data: Dict[str, Any], 
                     merge: bool = True) -> Optional[Any]:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        pass
    
    @abstractmethod
    def create_relationship(self, from_entity_id: str, relation_type: str, 
                          to_entity_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a relationship between two entities."""
        pass
    
    @abstractmethod
    def query_entities(self, entity_type: Optional[str] = None, 
                     filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None) -> List[Any]:
        """Query entities based on type and filters."""
        pass


class ToolInterface(ABC):
    """Interface for tools to avoid circular dependencies."""
    
    @abstractmethod
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with the given parameters."""
        pass


class AgentInterface(ABC):
    """Interface for agent operations to avoid circular dependencies."""
    
    @abstractmethod
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an input and generate a response."""
        pass
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's name."""
        pass


class RAGInterface(ABC):
    """Interface for RAG operations to avoid circular dependencies."""
    
    @abstractmethod
    def extract_document_for_llm(self, query_text: str, agent_id: Optional[str] = None,
                               workflow_state: Optional[Dict[str, Any]] = None,
                               k: int = 5) -> str:
        """Extract document context formatted for LLM consumption."""
        pass


class WorkflowInterface(ABC):
    """Interface for workflow operations to avoid circular dependencies."""
    
    @abstractmethod
    def start_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None,
                     instance_id: Optional[str] = None) -> str:
        """Start a new workflow instance."""
        pass
    
    @abstractmethod
    def send_event(self, instance_id: str, event_type: str, 
                 event_data: Dict[str, Any] = None) -> bool:
        """Send an event to a workflow instance."""
        pass
    
    @abstractmethod
    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow instance."""
        pass
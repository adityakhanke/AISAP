"""
Workflow definition models for the workflow engine.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

@dataclass
class WorkflowState:
    """
    Represents a state in a workflow.
    
    Attributes:
        id: Unique identifier for the state
        name: Human-readable name for the state
        description: Optional description of the state
        is_initial: Whether this is an initial state
        is_final: Whether this is a final state
        agent_id: Optional ID of the agent responsible for this state
        instructions: Optional instructions for the agent
        auto_transition: Whether to automatically transition based on conditions
        metadata: Additional metadata for the state
    """
    id: str
    name: str
    description: Optional[str] = None
    is_initial: bool = False
    is_final: bool = False
    agent_id: Optional[str] = None
    instructions: Optional[str] = None
    auto_transition: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_initial": self.is_initial,
            "is_final": self.is_final,
            "agent_id": self.agent_id,
            "instructions": self.instructions,
            "auto_transition": self.auto_transition,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """
        Create a state from a dictionary.
        
        Args:
            data: Dictionary containing state data
            
        Returns:
            WorkflowState object
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            is_initial=data.get("is_initial", False),
            is_final=data.get("is_final", False),
            agent_id=data.get("agent_id"),
            instructions=data.get("instructions"),
            auto_transition=data.get("auto_transition", False),
            metadata=data.get("metadata", {})
        )


@dataclass
class WorkflowTransition:
    """
    Represents a transition between states in a workflow.
    
    Attributes:
        id: Unique identifier for the transition
        from_state: ID of the source state
        to_state: ID of the target state
        name: Optional human-readable name for the transition
        description: Optional description of the transition
        condition: Optional condition for the transition
        event: Optional event that triggers the transition
        actions: Optional actions to execute during the transition
    """
    id: str
    from_state: str
    to_state: str
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    event: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transition to a dictionary.
        
        Returns:
            Dictionary representation of the transition
        """
        return {
            "id": self.id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "event": self.event,
            "actions": self.actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTransition':
        """
        Create a transition from a dictionary.
        
        Args:
            data: Dictionary containing transition data
            
        Returns:
            WorkflowTransition object
        """
        return cls(
            id=data["id"],
            from_state=data["from_state"],
            to_state=data["to_state"],
            name=data.get("name"),
            description=data.get("description"),
            condition=data.get("condition"),
            event=data.get("event"),
            actions=data.get("actions", [])
        )


@dataclass
class Workflow:
    """
    Represents a workflow definition.
    
    Attributes:
        id: Unique identifier for the workflow
        name: Human-readable name for the workflow
        description: Optional description of the workflow
        states: List of states in the workflow
        transitions: List of transitions in the workflow
        metadata: Additional metadata for the workflow
    """
    id: str
    name: str
    description: Optional[str] = None
    states: List[WorkflowState] = field(default_factory=list)
    transitions: List[WorkflowTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the workflow after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the workflow.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If workflow is invalid
        """
        # Check for at least one state
        if not self.states:
            raise ValueError(f"Workflow {self.name} has no states")
        
        # Check for initial state
        initial_states = [state for state in self.states if state.is_initial]
        if not initial_states:
            raise ValueError(f"Workflow {self.name} has no initial state")
        if len(initial_states) > 1:
            raise ValueError(f"Workflow {self.name} has multiple initial states")
        
        # Check for final states
        if not any(state.is_final for state in self.states):
            raise ValueError(f"Workflow {self.name} has no final states")
        
        # Check for valid transitions
        state_ids = {state.id for state in self.states}
        for transition in self.transitions:
            if transition.from_state not in state_ids:
                raise ValueError(f"Transition {transition.id} references unknown source state: {transition.from_state}")
            if transition.to_state not in state_ids:
                raise ValueError(f"Transition {transition.id} references unknown target state: {transition.to_state}")
        
        return True
    
    def get_state(self, state_id: str) -> Optional[WorkflowState]:
        """
        Get a state by ID.
        
        Args:
            state_id: ID of the state to retrieve
            
        Returns:
            State if found, None otherwise
        """
        for state in self.states:
            if state.id == state_id:
                return state
        return None
    
    def get_initial_state(self) -> Optional[WorkflowState]:
        """
        Get the initial state.
        
        Returns:
            Initial state if found, None otherwise
        """
        for state in self.states:
            if state.is_initial:
                return state
        return None
    
    def get_transitions_from(self, state_id: str) -> List[WorkflowTransition]:
        """
        Get transitions from a state.
        
        Args:
            state_id: ID of the source state
            
        Returns:
            List of transitions from the state
        """
        return [t for t in self.transitions if t.from_state == state_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow to a dictionary.
        
        Returns:
            Dictionary representation of the workflow
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """
        Create a workflow from a dictionary.
        
        Args:
            data: Dictionary containing workflow data
            
        Returns:
            Workflow object
        """
        states = [WorkflowState.from_dict(state_data) for state_data in data.get("states", [])]
        transitions = [WorkflowTransition.from_dict(transition_data) for transition_data in data.get("transitions", [])]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            states=states,
            transitions=transitions,
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def create(cls, name: str, description: Optional[str] = None) -> 'Workflow':
        """
        Create a new workflow with a generated ID.
        
        Args:
            name: Name for the workflow
            description: Optional description
            
        Returns:
            New workflow
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
    
    def add_state(self, name: str, is_initial: bool = False, is_final: bool = False,
                agent_id: Optional[str] = None, instructions: Optional[str] = None,
                auto_transition: bool = False, metadata: Dict[str, Any] = None) -> WorkflowState:
        """
        Add a state to the workflow.
        
        Args:
            name: Name for the state
            is_initial: Whether this is an initial state
            is_final: Whether this is a final state
            agent_id: Optional ID of the agent responsible for this state
            instructions: Optional instructions for the agent
            auto_transition: Whether to automatically transition based on conditions
            metadata: Additional metadata for the state
            
        Returns:
            Created state
        """
        state = WorkflowState(
            id=str(uuid.uuid4()),
            name=name,
            is_initial=is_initial,
            is_final=is_final,
            agent_id=agent_id,
            instructions=instructions,
            auto_transition=auto_transition,
            metadata=metadata or {}
        )
        
        # Check for initial state conflicts
        if is_initial and any(s.is_initial for s in self.states):
            # Set existing initial states to non-initial
            for s in self.states:
                if s.is_initial:
                    s.is_initial = False
        
        self.states.append(state)
        return state
    
    def add_transition(self, from_state: str, to_state: str, name: Optional[str] = None,
                     condition: Optional[str] = None, event: Optional[str] = None,
                     actions: List[Dict[str, Any]] = None) -> WorkflowTransition:
        """
        Add a transition to the workflow.
        
        Args:
            from_state: ID of the source state
            to_state: ID of the target state
            name: Optional name for the transition
            condition: Optional condition for the transition
            event: Optional event that triggers the transition
            actions: Optional actions to execute during the transition
            
        Returns:
            Created transition
        """
        # Verify states exist
        if not self.get_state(from_state):
            raise ValueError(f"Source state not found: {from_state}")
        if not self.get_state(to_state):
            raise ValueError(f"Target state not found: {to_state}")
        
        transition = WorkflowTransition(
            id=str(uuid.uuid4()),
            from_state=from_state,
            to_state=to_state,
            name=name,
            condition=condition,
            event=event,
            actions=actions or []
        )
        
        self.transitions.append(transition)
        return transition
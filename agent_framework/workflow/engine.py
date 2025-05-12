"""
Workflow engine for orchestrating multi-agent processes.
"""

import time
import uuid
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
from ..context.manager import ContextManager
from ..core.registry import AgentRegistry
from .events import EventSystem
from .definitions import Workflow, WorkflowState

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """
    Engine for executing and managing workflows across agents.
    Orchestrates complex multi-agent processes.
    """

    def __init__(self, context_manager: ContextManager, agent_registry: AgentRegistry):
        """
        Initialize the workflow engine.

        Args:
            context_manager: Context manager for accessing shared context
            agent_registry: Registry of available agents
        """
        self.context_manager = context_manager
        self.agent_registry = agent_registry
        self.event_system = EventSystem()

        # Workflows by ID
        self.workflows: Dict[str, Workflow] = {}

        # Active workflow instances
        self.active_instances: Dict[str, Dict[str, Any]] = {}

        # Workflow history
        self.workflow_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("Initialized workflow engine")
    
    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register a workflow definition.
        
        Args:
            workflow: Workflow to register
        """
        self.workflows[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} (ID: {workflow.id})")
        
        # Register workflow in context if not already present
        existing = self.context_manager.query_entities(
            entity_type="workflow",
            filters={"id": workflow.id},
            limit=1
        )
        
        if not existing:
            self.context_manager.create_entity(
                entity_type="workflow",
                data={
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "states": [state.to_dict() for state in workflow.states],
                    "transitions": [transition.to_dict() for transition in workflow.transitions]
                }
            )
    
    def start_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None,
                     instance_id: Optional[str] = None) -> str:
        """
        Start a new workflow instance.
        
        Args:
            workflow_id: ID of the workflow to start
            input_data: Input data for the workflow
            instance_id: Optional ID for the instance (generated if not provided)
            
        Returns:
            Instance ID
            
        Raises:
            ValueError: If workflow not found
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        
        # Generate instance ID if not provided
        instance_id = instance_id or str(uuid.uuid4())
        
        # Find the initial state
        initial_state = workflow.get_initial_state()
        if not initial_state:
            raise ValueError(f"Workflow {workflow_id} has no initial state")
        
        # Set up workflow context
        workflow_context = {
            "instance_id": instance_id,
            "workflow_id": workflow_id,
            "current_state": initial_state.id,
            "input": input_data or {},
            "output": {},
            "variables": {},
            "history": [],
            "start_time": time.time(),
            "last_update_time": time.time(),
            "status": "running"
        }
        
        # Store in active instances
        self.active_instances[instance_id] = workflow_context
        
        # Initialize history
        self.workflow_history[instance_id] = []
        
        # Log state entry
        self._log_workflow_event(instance_id, "state_entered", {
            "state_id": initial_state.id,
            "state_name": initial_state.name
        })
        
        # Create workflow instance in context
        self.context_manager.create_entity(
            entity_type="workflow_instance",
            data={
                "instance_id": instance_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "current_state": initial_state.id,
                "status": "running",
                "start_time": workflow_context["start_time"],
                "input": input_data or {}
            }
        )
        
        logger.info(f"Started workflow instance: {instance_id} (Workflow: {workflow.name})")
        
        # Trigger the initial state
        self._execute_state(instance_id, initial_state)
        
        return instance_id
    
    def _execute_state(self, instance_id: str, state: WorkflowState) -> None:
        """
        Execute a workflow state.
        
        Args:
            instance_id: Workflow instance ID
            state: State to execute
        """
        if instance_id not in self.active_instances:
            logger.error(f"Workflow instance not found: {instance_id}")
            return
        
        instance = self.active_instances[instance_id]
        workflow_id = instance["workflow_id"]
        workflow = self.workflows[workflow_id]
        
        logger.info(f"Executing state: {state.name} (Instance: {instance_id})")
        
        # Set the current state
        instance["current_state"] = state.id
        instance["last_update_time"] = time.time()
        
        # Update instance in context
        self._update_instance_context(instance_id)
        
        # Check if there's an agent assigned to this state
        if state.agent_id:
            agent = self.agent_registry.get_agent(state.agent_id)
            
            if agent:
                # Prepare input for the agent
                agent_input = {
                    "content": state.instructions or f"Execute state '{state.name}'",
                    "workflow_instance": instance_id,
                    "workflow_state": state.id,
                    "context": {
                        "workflow_input": instance["input"],
                        "workflow_variables": instance["variables"],
                        "workflow_output": instance["output"]
                    }
                }
                
                # Execute the agent
                try:
                    logger.info(f"Delegating state execution to agent: {agent.name}")
                    agent_output = agent.process_input(agent_input)
                    
                    # Store agent output in workflow variables
                    if "result" in agent_output:
                        instance["variables"][f"agent_{state.id}_result"] = agent_output["result"]
                    
                    # Check if the agent specified a next state
                    next_state_id = agent_output.get("next_state")
                    if next_state_id:
                        self._transition_to_state(instance_id, next_state_id)
                        return
                except Exception as e:
                    logger.error(f"Error executing agent for state {state.id}: {e}")
                    # Don't transition - stay in current state
                    return
        
        # Check if there's an automatic transition
        if state.auto_transition:
            for transition in workflow.transitions:
                if transition.from_state == state.id:
                    # Check transition conditions
                    if self._evaluate_condition(transition.condition, instance):
                        logger.info(f"Auto-transitioning to state: {transition.to_state}")
                        self._transition_to_state(instance_id, transition.to_state)
                        return
        
        # Check if this is a final state
        if state.is_final:
            logger.info(f"Reached final state: {state.name}")
            self._complete_workflow(instance_id)
    
    def _transition_to_state(self, instance_id: str, state_id: str) -> bool:
        """
        Transition a workflow instance to a new state.
        
        Args:
            instance_id: Workflow instance ID
            state_id: ID of the state to transition to
            
        Returns:
            True if transition successful, False otherwise
        """
        if instance_id not in self.active_instances:
            logger.error(f"Workflow instance not found: {instance_id}")
            return False
        
        instance = self.active_instances[instance_id]
        workflow_id = instance["workflow_id"]
        
        if workflow_id not in self.workflows:
            logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.workflows[workflow_id]
        
        # Find the target state
        target_state = workflow.get_state(state_id)
        if not target_state:
            logger.error(f"Target state not found: {state_id}")
            return False
        
        # Check if transition is valid
        current_state_id = instance["current_state"]
        valid_transition = False
        
        for transition in workflow.transitions:
            if transition.from_state == current_state_id and transition.to_state == state_id:
                # Check condition if present
                if not transition.condition or self._evaluate_condition(transition.condition, instance):
                    valid_transition = True
                    
                    # Execute transition actions if any
                    if transition.actions:
                        self._execute_actions(transition.actions, instance)
                    
                    break
        
        if not valid_transition:
            logger.warning(f"Invalid transition from {current_state_id} to {state_id}")
            return False
        
        # Log state exit
        current_state = workflow.get_state(current_state_id)
        if current_state:
            self._log_workflow_event(instance_id, "state_exited", {
                "state_id": current_state.id,
                "state_name": current_state.name
            })
        
        # Log transition
        self._log_workflow_event(instance_id, "transition", {
            "from_state": current_state_id,
            "to_state": state_id
        })
        
        # Log state entry
        self._log_workflow_event(instance_id, "state_entered", {
            "state_id": target_state.id,
            "state_name": target_state.name
        })
        
        # Execute the new state
        self._execute_state(instance_id, target_state)
        
        return True
    
    def _evaluate_condition(self, condition: Optional[str], instance: Dict[str, Any]) -> bool:
        """
        Evaluate a transition condition.
        
        Args:
            condition: Condition expression or None
            instance: Workflow instance data
            
        Returns:
            True if condition evaluates to True or is None, False otherwise
        """
        if not condition:
            return True
        
        try:
            # Create a context for condition evaluation
            context = {
                "input": instance["input"],
                "output": instance["output"],
                "variables": instance["variables"]
            }
            
            # Simple condition evaluation (this is a limited approach - in a real system,
            # you'd want to use a proper expression evaluator with security measures)
            # For a real implementation, consider using a library like simpleeval
            
            # Replace variable references
            for var_name, var_value in context["variables"].items():
                condition = condition.replace(f"${var_name}", json.dumps(var_value))
            
            for input_name, input_value in context["input"].items():
                condition = condition.replace(f"$input.{input_name}", json.dumps(input_value))
            
            for output_name, output_value in context["output"].items():
                condition = condition.replace(f"$output.{output_name}", json.dumps(output_value))
            
            # WARNING: eval is unsafe for production use!
            # This is just for demonstration - use a secure alternative in real code
            result = eval(condition)
            return bool(result)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _execute_actions(self, actions: List[Dict[str, Any]], instance: Dict[str, Any]) -> None:
        """
        Execute transition actions.
        
        Args:
            actions: List of action definitions
            instance: Workflow instance data
        """
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "set_variable":
                var_name = action.get("variable")
                var_value = action.get("value")
                
                if var_name:
                    instance["variables"][var_name] = var_value
                    logger.debug(f"Set variable {var_name} = {var_value}")
            
            elif action_type == "set_output":
                output_name = action.get("name")
                output_value = action.get("value")
                
                if output_name:
                    instance["output"][output_name] = output_value
                    logger.debug(f"Set output {output_name} = {output_value}")
            
            elif action_type == "emit_event":
                event_name = action.get("event")
                event_data = action.get("data", {})
                
                if event_name:
                    self.event_system.emit(event_name, {
                        "workflow_instance": instance["instance_id"],
                        "workflow_id": instance["workflow_id"],
                        "data": event_data
                    })
                    logger.debug(f"Emitted event: {event_name}")
    
    def _complete_workflow(self, instance_id: str) -> None:
        """
        Complete a workflow instance.
        
        Args:
            instance_id: Workflow instance ID
        """
        if instance_id not in self.active_instances:
            logger.error(f"Workflow instance not found: {instance_id}")
            return
        
        instance = self.active_instances[instance_id]
        
        # Update instance status
        instance["status"] = "completed"
        instance["end_time"] = time.time()
        instance["last_update_time"] = instance["end_time"]
        
        # Log completion
        self._log_workflow_event(instance_id, "workflow_completed", {
            "duration": instance["end_time"] - instance["start_time"],
            "output": instance["output"]
        })
        
        # Update instance in context
        self._update_instance_context(instance_id)
        
        # Emit completion event
        self.event_system.emit("workflow_completed", {
            "workflow_instance": instance_id,
            "workflow_id": instance["workflow_id"],
            "duration": instance["end_time"] - instance["start_time"],
            "output": instance["output"]
        })
        
        logger.info(f"Completed workflow instance: {instance_id}")
    
    def _log_workflow_event(self, instance_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Log a workflow event.
        
        Args:
            instance_id: Workflow instance ID
            event_type: Type of event
            event_data: Event data
        """
        if instance_id not in self.workflow_history:
            self.workflow_history[instance_id] = []
        
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": event_data
        }
        
        self.workflow_history[instance_id].append(event)
        
        # Add to instance history
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            instance["history"].append(event)
            instance["last_update_time"] = event["timestamp"]
    
    def _update_instance_context(self, instance_id: str) -> None:
        """
        Update workflow instance in context.
        
        Args:
            instance_id: Workflow instance ID
        """
        if instance_id not in self.active_instances:
            return
        
        instance = self.active_instances[instance_id]
        
        # Query for existing entity
        entities = self.context_manager.query_entities(
            entity_type="workflow_instance",
            filters={"instance_id": instance_id},
            limit=1
        )
        
        if entities:
            # Update existing entity
            entity_id = entities[0].id
            self.context_manager.update_entity(
                entity_id=entity_id,
                data={
                    "current_state": instance["current_state"],
                    "status": instance["status"],
                    "last_update_time": instance["last_update_time"],
                    "variables": instance["variables"],
                    "output": instance["output"],
                    "history": instance["history"][-5:]  # Store only the most recent events
                }
            )
        else:
            # Should not happen, but create if missing
            self.context_manager.create_entity(
                entity_type="workflow_instance",
                data={
                    "instance_id": instance_id,
                    "workflow_id": instance["workflow_id"],
                    "current_state": instance["current_state"],
                    "status": instance["status"],
                    "start_time": instance["start_time"],
                    "last_update_time": instance["last_update_time"],
                    "input": instance["input"],
                    "variables": instance["variables"],
                    "output": instance["output"]
                }
            )
    
    def send_event(self, instance_id: str, event_type: str, event_data: Dict[str, Any] = None) -> bool:
        """
        Send an event to a workflow instance.
        
        Args:
            instance_id: Workflow instance ID
            event_type: Type of event
            event_data: Event data
            
        Returns:
            True if event was processed, False otherwise
        """
        if instance_id not in self.active_instances:
            logger.error(f"Workflow instance not found: {instance_id}")
            return False
        
        instance = self.active_instances[instance_id]
        workflow_id = instance["workflow_id"]
        
        if workflow_id not in self.workflows:
            logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.workflows[workflow_id]
        current_state_id = instance["current_state"]
        
        # Log the event
        self._log_workflow_event(instance_id, f"external_event_{event_type}", event_data or {})
        
        # Check if this event triggers any transitions
        for transition in workflow.transitions:
            if transition.from_state == current_state_id and transition.event == event_type:
                # Check condition if present
                if not transition.condition or self._evaluate_condition(transition.condition, instance):
                    logger.info(f"Event {event_type} triggering transition to {transition.to_state}")
                    return self._transition_to_state(instance_id, transition.to_state)
        
        logger.info(f"Event {event_type} did not trigger any transitions")
        return False
    
    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a workflow instance.
        
        Args:
            instance_id: Workflow instance ID
            
        Returns:
            Instance status data if found, None otherwise
        """
        if instance_id not in self.active_instances:
            # Check if it's in the context store
            entities = self.context_manager.query_entities(
                entity_type="workflow_instance",
                filters={"instance_id": instance_id},
                limit=1
            )
            
            if entities:
                return entities[0].data
            
            return None
        
        instance = self.active_instances[instance_id]
        workflow = self.workflows.get(instance["workflow_id"])
        
        if not workflow:
            return None
        
        current_state = workflow.get_state(instance["current_state"])
        
        return {
            "instance_id": instance_id,
            "workflow_id": instance["workflow_id"],
            "workflow_name": workflow.name,
            "current_state": instance["current_state"],
            "current_state_name": current_state.name if current_state else "Unknown",
            "status": instance["status"],
            "start_time": instance["start_time"],
            "last_update_time": instance["last_update_time"],
            "end_time": instance.get("end_time"),
            "duration": (instance.get("end_time", time.time()) - instance["start_time"]),
            "output": instance["output"]
        }
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a workflow definition.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow definition if found, None otherwise
        """
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "states": [state.to_dict() for state in workflow.states],
            "transitions": [transition.to_dict() for transition in workflow.transitions]
        }
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """
        Get all registered workflow definitions.
        
        Returns:
            List of workflow definitions
        """
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "state_count": len(workflow.states),
                "transition_count": len(workflow.transitions)
            }
            for workflow in self.workflows.values()
        ]
    
    def get_all_instances(self) -> List[Dict[str, Any]]:
        """
        Get all active workflow instances.
        
        Returns:
            List of instance status data
        """
        return [self.get_instance_status(instance_id) for instance_id in self.active_instances]
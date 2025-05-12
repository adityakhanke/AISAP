"""
Event system for the workflow engine.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Set

logger = logging.getLogger(__name__)

class EventSystem:
    """
    Event system for workflow events and inter-agent communication.
    Provides publish-subscribe functionality.
    """
    
    def __init__(self):
        """Initialize the event system."""
        # Map of event name to set of handler functions
        self.handlers: Dict[str, Set[Callable]] = {}
        
        # Event history for debugging
        self.event_history: List[Dict[str, Any]] = []
        
        # Maximum event history size
        self.max_history_size = 100
        
        logger.info("Initialized event system")
    
    def subscribe(self, event_name: str, handler: Callable) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to subscribe to
            handler: Function to call when the event occurs
        """
        if event_name not in self.handlers:
            self.handlers[event_name] = set()
        
        self.handlers[event_name].add(handler)
        logger.debug(f"Subscribed handler to event: {event_name}")
    
    def unsubscribe(self, event_name: str, handler: Callable) -> bool:
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of the event to unsubscribe from
            handler: Handler function to remove
            
        Returns:
            True if handler was removed, False otherwise
        """
        if event_name not in self.handlers:
            return False
        
        if handler not in self.handlers[event_name]:
            return False
        
        self.handlers[event_name].remove(handler)
        
        # Clean up empty handler sets
        if not self.handlers[event_name]:
            del self.handlers[event_name]
        
        logger.debug(f"Unsubscribed handler from event: {event_name}")
        return True
    
    def emit(self, event_name: str, event_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event.
        
        Args:
            event_name: Name of the event to emit
            event_data: Data to pass to handlers
            
        Returns:
            Number of handlers called
        """
        if event_data is None:
            event_data = {}
        
        # Add event to history
        self._add_to_history(event_name, event_data)
        
        # Call handlers
        if event_name not in self.handlers:
            logger.debug(f"No handlers for event: {event_name}")
            return 0
        
        logger.debug(f"Emitting event: {event_name} with {len(self.handlers[event_name])} handlers")
        
        # Make a copy of handlers to avoid issues if handlers modify the set
        handlers = list(self.handlers[event_name])
        
        for handler in handlers:
            try:
                handler(event_name, event_data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
        
        return len(handlers)
    
    def _add_to_history(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        Add an event to the history.
        
        Args:
            event_name: Name of the event
            event_data: Event data
        """
        import time
        
        event_record = {
            "timestamp": time.time(),
            "event": event_name,
            "data": event_data
        }
        
        self.event_history.append(event_record)
        
        # Trim history if it gets too long
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the event history.
        
        Args:
            limit: Optional maximum number of events to return
            
        Returns:
            List of event records
        """
        if limit is None:
            return list(self.event_history)
        
        return list(self.event_history[-limit:])
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self.event_history = []
    
    def get_handlers(self, event_name: Optional[str] = None) -> Dict[str, int]:
        """
        Get the number of handlers for each event.
        
        Args:
            event_name: Optional specific event to get handlers for
            
        Returns:
            Dictionary mapping event names to handler counts
        """
        if event_name is not None:
            if event_name not in self.handlers:
                return {event_name: 0}
            return {event_name: len(self.handlers[event_name])}
        
        return {name: len(handlers) for name, handlers in self.handlers.items()}
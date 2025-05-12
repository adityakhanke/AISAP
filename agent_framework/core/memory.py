"""
Memory system for agents to maintain state during interactions.
"""

import time
from typing import Dict, Any, List, Optional, Deque
from collections import deque

class WorkingMemory:
    """
    Working memory for agents to maintain state during interactions.
    Provides short-term memory with a limited capacity.
    """
    
    def __init__(self, capacity: int = 50):
        """
        Initialize the working memory.
        
        Args:
            capacity: Maximum number of items to store in memory
        """
        self.capacity = capacity
        self.memory: Dict[str, Any] = {}
        self.recent_keys: Deque[str] = deque(maxlen=capacity)
        self.timestamps: Dict[str, float] = {}
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value in memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
        """
        # If key already exists, remove it from the recent keys list
        if key in self.memory:
            try:
                self.recent_keys.remove(key)
            except ValueError:
                pass
        # If at capacity and adding a new key, remove the oldest key
        elif len(self.memory) >= self.capacity and key not in self.memory:
            oldest_key = self.recent_keys.popleft()
            del self.memory[oldest_key]
            del self.timestamps[oldest_key]
        
        # Store the value and update timestamps
        self.memory[key] = value
        self.recent_keys.append(key)
        self.timestamps[key] = time.time()
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from memory.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Retrieved value or default
        """
        return self.memory.get(key, default)
    
    def forget(self, key: str) -> bool:
        """
        Remove a value from memory.
        
        Args:
            key: Key to remove
            
        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self.memory:
            del self.memory[key]
            try:
                self.recent_keys.remove(key)
            except ValueError:
                pass
            del self.timestamps[key]
            return True
        return False
    
    def get_recent(self, n: int = 10) -> List[tuple]:
        """
        Get the most recently accessed items.
        
        Args:
            n: Number of items to return
            
        Returns:
            List of (key, value, timestamp) tuples
        """
        # Get the n most recent keys
        recent_keys = list(self.recent_keys)[-n:]
        
        # Return (key, value, timestamp) tuples
        return [(key, self.memory[key], self.timestamps[key]) for key in recent_keys if key in self.memory]
    
    def clear(self) -> None:
        """Clear all items from memory."""
        self.memory.clear()
        self.recent_keys.clear()
        self.timestamps.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all items in memory.
        
        Returns:
            Dictionary of all items
        """
        return self.memory.copy()
    
    def get_keys(self) -> List[str]:
        """
        Get all keys in memory.
        
        Returns:
            List of keys
        """
        return list(self.memory.keys())
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key is in memory.
        
        Args:
            key: Key to check
            
        Returns:
            True if key is in memory, False otherwise
        """
        return key in self.memory
    
    def __len__(self) -> int:
        """
        Get the number of items in memory.
        
        Returns:
            Number of items
        """
        return len(self.memory)
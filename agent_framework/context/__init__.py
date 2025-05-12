"""
Context management module for the Agent Framework.
"""

from .manager import ContextManager
from .store import ContextStore, MemoryContextStore
from .entity import Entity, EntityReference, EntityType, RelationType
from .schema import EntitySchema, validate_entity
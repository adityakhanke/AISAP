"""
Context Store - Storage implementation for the context management system.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

from .entity import Entity

logger = logging.getLogger(__name__)

class ContextStore(ABC):
    """Abstract base class for context storage implementations."""
    
    @abstractmethod
    def save_entity(self, entity: Entity) -> bool:
        """
        Save an entity to storage.
        
        Args:
            entity: Entity to save
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str, version: Optional[int] = None) -> Optional[Entity]:
        """
        Get an entity by ID and optional version.
        
        Args:
            entity_id: ID of the entity to retrieve
            version: Optional specific version to retrieve
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def create_relationship(self, from_entity_id: str, relation_type: str, 
                          to_entity_id: str, metadata: Dict[str, Any]) -> str:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity_id: Source entity ID
            relation_type: Type of relationship
            to_entity_id: Target entity ID
            metadata: Metadata about the relationship
            
        Returns:
            ID of the created relationship
        """
        pass
    
    @abstractmethod
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None, 
                          direction: str = "outgoing") -> List[Tuple[str, Entity]]:
        """
        Get entities related to the given entity.
        
        Args:
            entity_id: Entity ID to find relationships for
            relation_type: Optional type of relationship to filter
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of (relation_type, entity) tuples
        """
        pass
    
    @abstractmethod
    def query_entities(self, entity_type: Optional[str] = None, 
                     filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None) -> List[Entity]:
        """
        Query entities based on type and filters.
        
        Args:
            entity_type: Optional entity type to filter by
            filters: Optional dictionary of property filters
            limit: Optional maximum number of results
            
        Returns:
            List of matching entities
        """
        pass
    
    @abstractmethod
    def get_all_entities(self) -> List[Entity]:
        """
        Get all entities in the store.
        
        Returns:
            List of all entities
        """
        pass
    
    @abstractmethod
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all relationships in the store.
        
        Returns:
            List of all relationships
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data from the store.
        
        Returns:
            True if successful
        """
        pass


class MemoryContextStore(ContextStore):
    """In-memory implementation of the context store."""
    
    def __init__(self):
        """Initialize the in-memory store."""
        # Entity storage: {entity_id -> {version -> Entity}}
        self.entities: Dict[str, Dict[int, Entity]] = {}
        
        # Relationship storage: {relationship_id -> relationship_data}
        self.relationships: Dict[str, Dict[str, Any]] = {}
        
        # For faster lookups: {entity_id -> {direction -> {relation_type -> [relationship_ids]}}}
        self.relationship_index: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        
        logger.info("Initialized in-memory context store")
    
    def save_entity(self, entity: Entity) -> bool:
        """
        Save an entity to storage.
        
        Args:
            entity: Entity to save
            
        Returns:
            True if successful
        """
        if entity.id not in self.entities:
            self.entities[entity.id] = {}
        
        self.entities[entity.id][entity.version] = entity
        return True
    
    def get_entity(self, entity_id: str, version: Optional[int] = None) -> Optional[Entity]:
        """
        Get an entity by ID and optional version.
        
        Args:
            entity_id: ID of the entity to retrieve
            version: Optional specific version to retrieve
            
        Returns:
            Entity if found, None otherwise
        """
        if entity_id not in self.entities:
            return None
        
        if version is None:
            # Get the latest version
            latest_version = max(self.entities[entity_id].keys())
            return self.entities[entity_id][latest_version]
        
        if version in self.entities[entity_id]:
            return self.entities[entity_id][version]
        
        return None
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        if entity_id not in self.entities:
            return False
        
        # Delete all versions of the entity
        del self.entities[entity_id]
        
        # Clean up relationship index
        if entity_id in self.relationship_index:
            del self.relationship_index[entity_id]
        
        # Remove relationships involving this entity
        relationships_to_delete = []
        for rel_id, rel_data in self.relationships.items():
            if rel_data["from_id"] == entity_id or rel_data["to_id"] == entity_id:
                relationships_to_delete.append(rel_id)
        
        for rel_id in relationships_to_delete:
            del self.relationships[rel_id]
        
        return True
    
    def create_relationship(self, from_entity_id: str, relation_type: str, 
                          to_entity_id: str, metadata: Dict[str, Any]) -> str:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity_id: Source entity ID
            relation_type: Type of relationship
            to_entity_id: Target entity ID
            metadata: Metadata about the relationship
            
        Returns:
            ID of the created relationship
        """
        # Check that entities exist
        if from_entity_id not in self.entities:
            raise ValueError(f"Source entity not found: {from_entity_id}")
        
        if to_entity_id not in self.entities:
            raise ValueError(f"Target entity not found: {to_entity_id}")
        
        # Create relationship ID
        relationship_id = str(uuid.uuid4())
        
        # Store relationship
        self.relationships[relationship_id] = {
            "id": relationship_id,
            "from_id": from_entity_id,
            "relation_type": relation_type,
            "to_id": to_entity_id,
            "metadata": metadata
        }
        
        # Update index for faster lookups
        self._index_relationship(relationship_id, from_entity_id, relation_type, to_entity_id)
        
        return relationship_id
    
    def _index_relationship(self, relationship_id: str, from_id: str, relation_type: str, to_id: str):
        """
        Index a relationship for faster lookups.
        
        Args:
            relationship_id: Relationship ID
            from_id: Source entity ID
            relation_type: Type of relationship
            to_id: Target entity ID
        """
        # Index outgoing relationship from source
        if from_id not in self.relationship_index:
            self.relationship_index[from_id] = {"outgoing": {}, "incoming": {}}
        
        if relation_type not in self.relationship_index[from_id]["outgoing"]:
            self.relationship_index[from_id]["outgoing"][relation_type] = []
        
        self.relationship_index[from_id]["outgoing"][relation_type].append(relationship_id)
        
        # Index incoming relationship to target
        if to_id not in self.relationship_index:
            self.relationship_index[to_id] = {"outgoing": {}, "incoming": {}}
        
        if relation_type not in self.relationship_index[to_id]["incoming"]:
            self.relationship_index[to_id]["incoming"][relation_type] = []
        
        self.relationship_index[to_id]["incoming"][relation_type].append(relationship_id)
    
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None, 
                          direction: str = "outgoing") -> List[Tuple[str, Entity]]:
        """
        Get entities related to the given entity.
        
        Args:
            entity_id: Entity ID to find relationships for
            relation_type: Optional type of relationship to filter
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of (relation_type, entity) tuples
        """
        result = []
        
        if entity_id not in self.relationship_index:
            return result
        
        # Determine which directions to consider
        directions = []
        if direction == "outgoing" or direction == "both":
            directions.append("outgoing")
        if direction == "incoming" or direction == "both":
            directions.append("incoming")
        
        # Collect relationships in each direction
        for dir_type in directions:
            # Get all relation types if none specified
            relation_types = [relation_type] if relation_type else list(self.relationship_index[entity_id][dir_type].keys())
            
            for rel_type in relation_types:
                if rel_type not in self.relationship_index[entity_id][dir_type]:
                    continue
                
                # Get relationships of this type
                rel_ids = self.relationship_index[entity_id][dir_type][rel_type]
                
                for rel_id in rel_ids:
                    rel_data = self.relationships[rel_id]
                    
                    # Determine the ID of the other entity
                    other_id = rel_data["to_id"] if dir_type == "outgoing" else rel_data["from_id"]
                    other_entity = self.get_entity(other_id)
                    
                    if other_entity:
                        result.append((rel_type, other_entity))
        
        return result
    
    def query_entities(self, entity_type: Optional[str] = None, 
                     filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None) -> List[Entity]:
        """
        Query entities based on type and filters.
        
        Args:
            entity_type: Optional entity type to filter by
            filters: Optional dictionary of property filters
            limit: Optional maximum number of results
            
        Returns:
            List of matching entities
        """
        result = []
        
        # Check each entity
        for entity_dict in self.entities.values():
            # Get the latest version of the entity
            latest_version = max(entity_dict.keys())
            entity = entity_dict[latest_version]
            
            # Check entity type
            if entity_type and entity.type != entity_type:
                continue
            
            # Check filters
            if filters:
                match = True
                for key, value in filters.items():
                    # Handle nested keys with dot notation
                    if "." in key:
                        parts = key.split(".")
                        obj = entity.data
                        for part in parts[:-1]:
                            if part not in obj:
                                match = False
                                break
                            obj = obj[part]
                        
                        if match and (parts[-1] not in obj or obj[parts[-1]] != value):
                            match = False
                    elif key not in entity.data or entity.data[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            result.append(entity)
            
            # Check limit
            if limit and len(result) >= limit:
                break
        
        return result
    
    def get_all_entities(self) -> List[Entity]:
        """
        Get all entities in the store.
        
        Returns:
            List of all entities
        """
        result = []
        
        for entity_dict in self.entities.values():
            # Get the latest version of each entity
            latest_version = max(entity_dict.keys())
            result.append(entity_dict[latest_version])
        
        return result
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all relationships in the store.
        
        Returns:
            List of all relationships
        """
        return list(self.relationships.values())
    
    def clear(self) -> bool:
        """
        Clear all data from the store.
        
        Returns:
            True if successful
        """
        self.entities = {}
        self.relationships = {}
        self.relationship_index = {}
        return True
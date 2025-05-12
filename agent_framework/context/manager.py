"""
Context Manager - Core component for managing shared context across agents.
"""

import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..core.interfaces import ContextInterface
from .store import MemoryContextStore
from .schema import validate_entity, EntitySchema
from .entity import Entity, EntityType

logger = logging.getLogger(__name__)

class ContextManager(ContextInterface):
    """
    Core context management system that maintains shared context across agents.
    Provides entity management, versioning, and access control.
    """
    
    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize the context manager with the given settings.
        
        Args:
            settings: Configuration settings for the context manager
        """
        self.settings = settings
        self.context_settings = settings.get('context', {})
        
        # Initialize context store
        store_type = self.context_settings.get('store_type', 'memory')
        if store_type == 'memory':
            self.store = MemoryContextStore()
        else:
            # In the future, add other store types (SQLite, MongoDB, etc.)
            logger.warning(f"Unsupported store type: {store_type}. Using memory store.")
            self.store = MemoryContextStore()
        
        # Settings
        self.enable_versioning = self.context_settings.get('versioning', True)
        self.enable_schema_validation = self.context_settings.get('schema_validation', True)
        
        # Entity type registry - maps entity types to their schemas
        self.entity_schemas: Dict[str, EntitySchema] = {}
        self._register_default_schemas()
        
        logger.info(f"Context Manager initialized with {store_type} store")
    
    def _register_default_schemas(self):
        """Register default entity schemas."""
        # Project schema
        self.register_entity_schema(EntityType.PROJECT, {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "goals": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"}
            },
            "required": ["name"]
        })
        
        # Requirement schema
        self.register_entity_schema(EntityType.REQUIREMENT, {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "status": {"type": "string", "enum": ["draft", "review", "approved", "implemented", "verified"]},
                "metadata": {"type": "object"}
            },
            "required": ["title", "description"]
        })
        
        # Code entity schema
        self.register_entity_schema(EntityType.CODE_ENTITY, {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["function", "class", "module", "file"]},
                "language": {"type": "string"},
                "content": {"type": "string"},
                "path": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["name", "type"]
        })

    def register_entity_schema(self, entity_type: str, schema: Dict[str, Any]):
        """
        Register a schema for an entity type.
        
        Args:
            entity_type: Type of entity
            schema: JSON Schema for validating entities of this type
        """
        self.entity_schemas[entity_type] = EntitySchema(entity_type, schema)
        logger.debug(f"Registered schema for entity type: {entity_type}")
    
    def create_entity(self, entity_type: str, data: Dict[str, Any], 
                     entity_id: Optional[str] = None) -> Entity:
        """
        Create a new entity in the context.
        
        Args:
            entity_type: Type of entity to create
            data: Entity data
            entity_id: Optional ID for the entity (generated if not provided)
            
        Returns:
            Created entity
        """
        # Generate ID if not provided
        if entity_id is None:
            entity_id = str(uuid.uuid4())
        
        # Validate against schema if enabled
        if self.enable_schema_validation and entity_type in self.entity_schemas:
            schema = self.entity_schemas[entity_type]
            validate_entity(data, schema)
        
        # Create entity
        entity = Entity(
            id=entity_id,
            type=entity_type,
            data=data,
            created_at=time.time(),
            updated_at=time.time(),
            version=1
        )
        
        # Store entity
        self.store.save_entity(entity)
        logger.info(f"Created entity: {entity_type} (ID: {entity_id})")
        
        return entity
    
    def get_entity(self, entity_id: str, version: Optional[int] = None) -> Optional[Entity]:
        """
        Get an entity by ID and optional version.
        
        Args:
            entity_id: ID of the entity to retrieve
            version: Optional specific version to retrieve
            
        Returns:
            Entity if found, None otherwise
        """
        return self.store.get_entity(entity_id, version)
    
    def update_entity(self, entity_id: str, data: Dict[str, Any], 
                     merge: bool = True) -> Optional[Entity]:
        """
        Update an existing entity.
        
        Args:
            entity_id: ID of the entity to update
            data: New or updated entity data
            merge: If True, merge with existing data; if False, replace
            
        Returns:
            Updated entity if successful, None if entity not found
        """
        # Get existing entity
        entity = self.store.get_entity(entity_id)
        if entity is None:
            logger.warning(f"Entity not found for update: {entity_id}")
            return None
        
        # Prepare updated data
        if merge:
            updated_data = entity.data.copy()
            self._deep_update(updated_data, data)
        else:
            updated_data = data
        
        # Validate against schema if enabled
        if self.enable_schema_validation and entity.type in self.entity_schemas:
            schema = self.entity_schemas[entity.type]
            validate_entity(updated_data, schema)
        
        # Create new version if versioning is enabled
        if self.enable_versioning:
            # Create new entity with incremented version
            updated_entity = Entity(
                id=entity.id,
                type=entity.type,
                data=updated_data,
                created_at=entity.created_at,
                updated_at=time.time(),
                version=entity.version + 1
            )
        else:
            # Update in place
            entity.data = updated_data
            entity.updated_at = time.time()
            updated_entity = entity
        
        # Store updated entity
        self.store.save_entity(updated_entity)
        logger.info(f"Updated entity: {entity.type} (ID: {entity_id}, Version: {updated_entity.version})")
        
        return updated_entity
    
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]):
        """
        Deep update a dictionary.
        
        Args:
            original: Original dictionary to update
            update: Update dictionary with values to apply
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self.store.delete_entity(entity_id)
    
    def create_relationship(self, from_entity_id: str, relation_type: str, 
                          to_entity_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two entities.
        
        Args:
            from_entity_id: Source entity ID
            relation_type: Type of relationship
            to_entity_id: Target entity ID
            metadata: Optional metadata about the relationship
            
        Returns:
            ID of the created relationship
        """
        if metadata is None:
            metadata = {}
        
        return self.store.create_relationship(from_entity_id, relation_type, to_entity_id, metadata)
    
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
        return self.store.get_related_entities(entity_id, relation_type, direction)
    
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
        return self.store.query_entities(entity_type, filters, limit)
    
    def export_context(self, format_type: str = "json") -> str:
        """
        Export the entire context to the specified format.
        
        Args:
            format_type: Format type (currently only 'json' is supported)
            
        Returns:
            String representation of the context
        """
        if format_type != "json":
            raise ValueError(f"Unsupported export format: {format_type}")
        
        # Get all entities and relationships
        entities = self.store.get_all_entities()
        relationships = self.store.get_all_relationships()
        
        # Create export dictionary
        export_data = {
            "entities": [entity.to_dict() for entity in entities],
            "relationships": relationships
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_context(self, data: str, format_type: str = "json", merge: bool = False) -> bool:
        """
        Import context from the specified format.
        
        Args:
            data: Context data to import
            format_type: Format type (currently only 'json' is supported)
            merge: If True, merge with existing context; if False, replace
            
        Returns:
            True if successful
        """
        if format_type != "json":
            raise ValueError(f"Unsupported import format: {format_type}")
        
        try:
            import_data = json.loads(data)
            
            # Clear existing context if not merging
            if not merge:
                self.store.clear()
            
            # Import entities
            for entity_data in import_data.get("entities", []):
                entity = Entity.from_dict(entity_data)
                self.store.save_entity(entity)
            
            # Import relationships
            for rel_data in import_data.get("relationships", []):
                self.store.create_relationship(
                    rel_data["from_id"],
                    rel_data["relation_type"],
                    rel_data["to_id"],
                    rel_data.get("metadata", {})
                )
            
            logger.info(f"Imported context with {len(import_data.get('entities', []))} entities and "
                      f"{len(import_data.get('relationships', []))} relationships")
            return True
        
        except Exception as e:
            logger.error(f"Error importing context: {e}")
            return False
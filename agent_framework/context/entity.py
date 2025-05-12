"""
Entity models for the context management system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


class EntityType:
    """Constants for common entity types."""
    PROJECT = "project"
    REQUIREMENT = "requirement"
    USER_STORY = "user_story"
    CODE_ENTITY = "code_entity"
    DOCUMENTATION = "documentation"
    TEST = "test"
    WORKFLOW = "workflow"
    TASK = "task"
    USER = "user"
    ARTIFACT = "artifact"


class RelationType:
    """Constants for common relationship types."""
    CONTAINS = "contains"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    RELATES_TO = "relates_to"
    CREATED_BY = "created_by"
    ASSIGNED_TO = "assigned_to"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    REFERENCES = "references"
    DERIVES_FROM = "derives_from"


@dataclass
class Entity:
    """
    Entity model representing an object in the context.
    
    Attributes:
        id: Unique identifier for the entity
        type: Type of the entity
        data: Dictionary containing entity data
        created_at: Timestamp when the entity was created
        updated_at: Timestamp when the entity was last updated
        version: Version number of the entity
    """
    id: str
    type: str
    data: Dict[str, Any]
    created_at: float
    updated_at: float
    version: int
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary containing entity data
            
        Returns:
            Entity object
        """
        return cls(
            id=data["id"],
            type=data["type"],
            data=data["data"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version=data["version"]
        )


@dataclass
class EntityReference:
    """
    Reference to an entity, used in relationships.
    
    Attributes:
        id: ID of the referenced entity
        type: Type of the referenced entity (optional)
    """
    id: str
    type: Optional[str] = None
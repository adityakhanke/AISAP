"""
Schema validation for the context management system.
"""

import logging
from typing import Dict, Any

# Use basic validation to reduce dependencies
# In a production version, we would use a proper JSON schema validator

logger = logging.getLogger(__name__)

class EntitySchema:
    """
    Schema for validating entity data.
    
    Attributes:
        entity_type: Type of entity this schema applies to
        schema: JSON Schema definition
    """
    
    def __init__(self, entity_type: str, schema: Dict[str, Any]):
        """
        Initialize with a JSON Schema for an entity type.
        
        Args:
            entity_type: Type of entity this schema applies to
            schema: JSON Schema definition
        """
        self.entity_type = entity_type
        self.schema = schema
        
        # For a full implementation, validate the schema itself
        # For simplicity in the MVP, we'll skip complex validation


def validate_entity(data: Dict[str, Any], schema: EntitySchema) -> bool:
    """
    Validate entity data against a schema.
    
    Args:
        data: Entity data to validate
        schema: Schema to validate against
        
    Returns:
        True if validation succeeds
        
    Raises:
        ValueError: If validation fails
    """
    schema_def = schema.schema
    
    # Check required fields
    required_fields = schema_def.get("required", [])
    for field in required_fields:
        if field not in data:
            error_msg = f"Missing required field: {field}"
            logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
            raise ValueError(error_msg)
    
    # Check property types (simplified validation for MVP)
    properties = schema_def.get("properties", {})
    for field_name, field_def in properties.items():
        if field_name in data:
            field_value = data[field_name]
            field_type = field_def.get("type")
            
            # Basic type validation
            if field_type == "string" and not isinstance(field_value, str):
                error_msg = f"Field {field_name} should be a string"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            elif field_type == "number" and not isinstance(field_value, (int, float)):
                error_msg = f"Field {field_name} should be a number"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            elif field_type == "integer" and not isinstance(field_value, int):
                error_msg = f"Field {field_name} should be an integer"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            elif field_type == "boolean" and not isinstance(field_value, bool):
                error_msg = f"Field {field_name} should be a boolean"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            elif field_type == "array" and not isinstance(field_value, list):
                error_msg = f"Field {field_name} should be an array"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            elif field_type == "object" and not isinstance(field_value, dict):
                error_msg = f"Field {field_name} should be an object"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
            
            # Check enum values
            if "enum" in field_def and field_value not in field_def["enum"]:
                error_msg = f"Field {field_name} should be one of: {', '.join(field_def['enum'])}"
                logger.error(f"Validation error for entity type {schema.entity_type}: {error_msg}")
                raise ValueError(error_msg)
    
    return True
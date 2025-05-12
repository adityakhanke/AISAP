"""
Tools module for the PM Agent.
Provides implementations for PM-specific tools.
"""

import logging
import time
import re
from typing import Dict, Any, List, Optional

from ...context.entity import EntityType, RelationType

logger = logging.getLogger(__name__)

def register_pm_tools(agent):
    """
    Register PM-specific tools with the agent.
    
    Args:
        agent: PM Agent instance
    """
    if not agent.tool_registry:
        logger.warning("No tool registry available, skipping tool registration")
        return
        
    # Project management tools
    agent.tool_registry.register_function(
        name="create_project",
        description="Create or set the current project",
        func=lambda **kwargs: _create_project(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="get_project",
        description="Get information about the current project",
        func=lambda **kwargs: _get_project(agent, **kwargs)
    )
    
    # Requirements management tools
    agent.tool_registry.register_function(
        name="create_requirement",
        description="Create a new product requirement",
        func=lambda **kwargs: _create_requirement(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="update_requirement",
        description="Update an existing product requirement",
        func=lambda **kwargs: _update_requirement(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="get_requirement",
        description="Get details about a specific requirement",
        func=lambda **kwargs: _get_requirement(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="list_requirements",
        description="List all product requirements with filtering options",
        func=lambda **kwargs: _list_requirements(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="delete_requirement",
        description="Delete a requirement",
        func=lambda **kwargs: _delete_requirement(agent, **kwargs)
    )
    
    # User story tools
    agent.tool_registry.register_function(
        name="create_user_story",
        description="Create a new user story",
        func=lambda **kwargs: _create_user_story(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="update_user_story",
        description="Update an existing user story",
        func=lambda **kwargs: _update_user_story(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="list_user_stories",
        description="List all user stories with filtering options",
        func=lambda **kwargs: _list_user_stories(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="create_user_stories_from_requirement",
        description="Generate user stories from a requirement",
        func=lambda **kwargs: _create_user_stories_from_requirement(agent, **kwargs)
    )
    
    # Prioritization and planning tools
    agent.tool_registry.register_function(
        name="prioritize_requirements",
        description="Prioritize a list of requirements",
        func=lambda **kwargs: _prioritize_requirements(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="create_roadmap",
        description="Create a product roadmap",
        func=lambda **kwargs: _create_roadmap(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="plan_sprint",
        description="Plan a development sprint",
        func=lambda **kwargs: _plan_sprint(agent, **kwargs)
    )
    
    # PRD and documentation tools
    agent.tool_registry.register_function(
        name="generate_prd",
        description="Generate a complete PRD document",
        func=lambda **kwargs: _generate_prd(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="generate_prd_section",
        description="Generate a specific PRD section",
        func=lambda **kwargs: _generate_prd_section(agent, **kwargs)
    )
    
    # Analysis tools
    agent.tool_registry.register_function(
        name="analyze_requirements",
        description="Analyze requirements for completeness, clarity, etc.",
        func=lambda **kwargs: _analyze_requirements(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="calculate_metrics",
        description="Calculate product management metrics",
        func=lambda **kwargs: _calculate_metrics(agent, **kwargs)
    )
    
    logger.info(f"Registered PM-specific tools for agent {agent.name}")

# Tool implementation functions

def _create_project(agent, name: str, description: str = "", goals: List[str] = None) -> Dict[str, Any]:
    """
    Create a new project or set the current project.
    
    Args:
        agent: PM Agent instance
        name: Project name
        description: Project description
        goals: List of project goals
        
    Returns:
        Project data
    """
    # Check if a project with this name already exists
    existing_projects = agent.context_manager.query_entities(
        entity_type=EntityType.PROJECT,
        filters={"name": name}
    )
    
    if existing_projects:
        # Use existing project
        project = existing_projects[0]
        logger.info(f"Using existing project: {name}")
    else:
        # Create new project
        if goals is None:
            goals = []
        
        project = agent.context_manager.create_entity(
            entity_type=EntityType.PROJECT,
            data={
                "name": name,
                "description": description,
                "goals": goals,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        logger.info(f"Created new project: {name}")
    
    # Store as current project
    agent.working_memory.store("current_project", project.id)
    
    # Add entity_type for response formatting
    result = project.to_dict()
    result["entity_type"] = "project"
    
    return result

def _get_project(agent) -> Dict[str, Any]:
    """
    Get information about the current project.
    
    Args:
        agent: PM Agent instance
        
    Returns:
        Project data or error
    """
    project_id = agent.working_memory.retrieve("current_project")
    if not project_id:
        # Try to find any project
        projects = agent.context_manager.query_entities(
            entity_type=EntityType.PROJECT,
            limit=1
        )
        
        if not projects:
            return {
                "success": False,
                "error": "No project found. Please create a project first."
            }
        
        project = projects[0]
        agent.working_memory.store("current_project", project.id)
    else:
        project = agent.context_manager.get_entity(project_id)
        if not project:
            return {
                "success": False,
                "error": f"Project with ID {project_id} not found."
            }
    
    # Get related entities
    requirements = agent.context_manager.query_entities(
        entity_type=EntityType.REQUIREMENT
    )
    
    user_stories = agent.context_manager.query_entities(
        entity_type="user_story"
    )
    
    # Add entity_type and counts for response formatting
    result = project.to_dict()
    result["entity_type"] = "project"
    result["requirement_count"] = len(requirements)
    result["user_story_count"] = len(user_stories)
    
    return result

def _create_requirement(agent, title: str, description: str, 
                   priority: str = "medium", status: str = "draft",
                   tags: List[str] = None) -> Dict[str, Any]:
    """
    Create a new product requirement.
    
    Args:
        agent: PM Agent instance
        title: Requirement title
        description: Requirement description
        priority: Requirement priority (low, medium, high, critical)
        status: Requirement status (draft, review, approved, implemented, verified)
        tags: Optional list of tags
        
    Returns:
        Created requirement
    """
    # Validate inputs
    if not title:
        return {"success": False, "error": "Title is required"}
    
    if not description:
        return {"success": False, "error": "Description is required"}
    
    valid_priorities = ["low", "medium", "high", "critical"]
    if priority not in valid_priorities:
        return {"success": False, "error": f"Priority must be one of: {', '.join(valid_priorities)}"}
    
    valid_statuses = ["draft", "review", "approved", "implemented", "verified"]
    if status not in valid_statuses:
        return {"success": False, "error": f"Status must be one of: {', '.join(valid_statuses)}"}
    
    # Get current project
    project_id = agent.working_memory.retrieve("current_project")
    
    # Prepare data
    if tags is None:
        tags = []
    
    data = {
        "title": title,
        "description": description,
        "priority": priority,
        "status": status,
        "tags": tags,
        "created_by": agent.agent_id,
        "created_at": time.time()
    }
    
    # Create the requirement
    entity = agent.context_manager.create_entity(
        entity_type=EntityType.REQUIREMENT,
        data=data
    )
    
    # Create relationship with project if available
    if project_id:
        agent.context_manager.create_relationship(
            from_entity_id=project_id,
            relation_type=RelationType.CONTAINS,
            to_entity_id=entity.id,
            metadata={"created_by": agent.agent_id}
        )
    
    # Add entity_type for response formatting
    result = entity.to_dict()
    result["entity_type"] = "requirement"
    
    return result

def _update_requirement(agent, requirement_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing product requirement.
    
    Args:
        agent: PM Agent instance
        requirement_id: ID of the requirement to update
        updates: Updates to apply
        
    Returns:
        Updated requirement or error
    """
    # Check that the requirement exists
    entity = agent.context_manager.get_entity(requirement_id)
    if not entity:
        return {"success": False, "error": f"Requirement not found: {requirement_id}"}
    
    if entity.type != EntityType.REQUIREMENT:
        return {"success": False, "error": f"Entity is not a requirement: {requirement_id}"}
    
    # Validate updates
    if "priority" in updates:
        valid_priorities = ["low", "medium", "high", "critical"]
        if updates["priority"] not in valid_priorities:
            return {"success": False, "error": f"Priority must be one of: {', '.join(valid_priorities)}"}
    
    if "status" in updates:
        valid_statuses = ["draft", "review", "approved", "implemented", "verified"]
        if updates["status"] not in valid_statuses:
            return {"success": False, "error": f"Status must be one of: {', '.join(valid_statuses)}"}
    
    # Add updated_at timestamp
    updates["updated_at"] = time.time()
    
    # Update the entity
    updated_entity = agent.context_manager.update_entity(
        entity_id=requirement_id,
        data=updates
    )
    
    if not updated_entity:
        return {"success": False, "error": "Failed to update requirement"}
    
    # Add entity_type for response formatting
    result = updated_entity.to_dict()
    result["entity_type"] = "requirement"
    
    return result

def _get_requirement(agent, requirement_id: str) -> Dict[str, Any]:
    """
    Get details about a specific requirement.
    
    Args:
        agent: PM Agent instance
        requirement_id: ID of the requirement
        
    Returns:
        Requirement details or error
    """
    entity = agent.context_manager.get_entity(requirement_id)
    if not entity:
        return {"success": False, "error": f"Requirement not found: {requirement_id}"}
    
    if entity.type != EntityType.REQUIREMENT:
        return {"success": False, "error": f"Entity is not a requirement: {requirement_id}"}
    
    # Get related user stories
    related_stories = []
    relationships = agent.context_manager.get_related_entities(
        entity_id=requirement_id,
        relation_type=RelationType.CONTAINS,
        direction="outgoing"
    )
    
    for rel_type, rel_entity in relationships:
        if rel_entity.type == "user_story":
            related_stories.append(rel_entity.to_dict())
    
    # Add entity_type and related entities for response formatting
    result = entity.to_dict()
    result["entity_type"] = "requirement"
    result["related_stories"] = related_stories
    
    return result

def _list_requirements(agent, filter: str = "", status: str = None, 
                      priority: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
    """
    List product requirements with filtering options.
    
    Args:
        agent: PM Agent instance
        filter: Text filter
        status: Filter by status
        priority: Filter by priority
        tags: Filter by tags
        
    Returns:
        List of requirements
    """
    # Prepare filters
    filters = {}
    
    if status:
        filters["status"] = status
    
    if priority:
        filters["priority"] = priority
    
    # Query requirements from the context
    entities = agent.context_manager.query_entities(
        entity_type=EntityType.REQUIREMENT,
        filters=filters
    )
    
    # Apply additional filtering
    results = []
    for entity in entities:
        include = True
        
        # Filter by tags if specified
        if tags and entity.data.get("tags"):
            if not any(tag in entity.data.get("tags", []) for tag in tags):
                include = False
        
        # Text filter
        if filter and include:
            filter_lower = filter.lower()
            title = entity.data.get("title", "").lower()
            description = entity.data.get("description", "").lower()
            
            if filter_lower not in title and filter_lower not in description:
                include = False
        
        if include:
            # Add entity_type for response formatting
            result = entity.to_dict()
            result["entity_type"] = "requirement"
            results.append(result)
    
    return results

def _delete_requirement(agent, requirement_id: str) -> Dict[str, Any]:
    """
    Delete a requirement.
    
    Args:
        agent: PM Agent instance
        requirement_id: ID of the requirement to delete
        
    Returns:
        Result of the operation
    """
    # Check that the requirement exists
    entity = agent.context_manager.get_entity(requirement_id)
    if not entity:
        return {"success": False, "error": f"Requirement not found: {requirement_id}"}
    
    if entity.type != EntityType.REQUIREMENT:
        return {"success": False, "error": f"Entity is not a requirement: {requirement_id}"}
    
    # Delete the entity
    success = agent.context_manager.delete_entity(requirement_id)
    
    if not success:
        return {"success": False, "error": "Failed to delete requirement"}
    
    return {
        "success": True,
        "message": f"Requirement '{entity.data.get('title', 'Unnamed')}' deleted successfully",
        "id": requirement_id
    }

def _create_user_story(agent, title: str, description: str, priority: str = "medium",
                      status: str = "draft", acceptance_criteria: List[str] = None,
                      story_points: str = None, requirement_id: str = None) -> Dict[str, Any]:
    """
    Create a new user story.
    
    Args:
        agent: PM Agent instance
        title: User story title
        description: User story description
        priority: User story priority
        status: User story status
        acceptance_criteria: List of acceptance criteria
        story_points: Story point estimate
        requirement_id: Optional ID of the related requirement
        
    Returns:
        Created user story
    """
    # Validate inputs
    if not title:
        return {"success": False, "error": "Title is required"}
    
    # Format as user story if it doesn't follow the pattern
    if "as a" not in description.lower() and "i want" not in description.lower():
        # Extract who, what, why if possible
        who_match = re.search(r'(?:for|by)\s+(\w+)', description, re.IGNORECASE)
        who = who_match.group(1) if who_match else "user"
        
        # Use original description as what
        what = description
        
        # Default why
        why = "to achieve my goals"
        
        # Combine into standard format
        description = f"As a {who}, I want {what}, so that {why}"
    
    # Prepare data
    if acceptance_criteria is None:
        acceptance_criteria = []
    
    data = {
        "title": title,
        "description": description,
        "priority": priority,
        "status": status,
        "acceptance_criteria": acceptance_criteria,
        "story_points": story_points,
        "created_by": agent.agent_id,
        "created_at": time.time()
    }
    
    # Create the user story
    entity = agent.context_manager.create_entity(
        entity_type="user_story",
        data=data
    )
    
    # Create relationship with requirement if provided
    if requirement_id:
        requirement = agent.context_manager.get_entity(requirement_id)
        if requirement:
            agent.context_manager.create_relationship(
                from_entity_id=requirement_id,
                relation_type=RelationType.CONTAINS,
                to_entity_id=entity.id,
                metadata={"created_by": agent.agent_id}
            )
    
    # Add entity_type for response formatting
    result = entity.to_dict()
    result["entity_type"] = "user_story"
    
    return result

def _update_user_story(agent, story_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing user story.
    
    Args:
        agent: PM Agent instance
        story_id: ID of the user story to update
        updates: Updates to apply
        
    Returns:
        Updated user story
    """
    # Check that the user story exists
    entity = agent.context_manager.get_entity(story_id)
    if not entity:
        return {"success": False, "error": f"User story not found: {story_id}"}
    
    if entity.type != "user_story":
        return {"success": False, "error": f"Entity is not a user story: {story_id}"}
    
    # Add updated_at timestamp
    updates["updated_at"] = time.time()
    
    # Update the entity
    updated_entity = agent.context_manager.update_entity(
        entity_id=story_id,
        data=updates
    )
    
    if not updated_entity:
        return {"success": False, "error": "Failed to update user story"}
    
    # Add entity_type for response formatting
    result = updated_entity.to_dict()
    result["entity_type"] = "user_story"
    
    return result

def _list_user_stories(agent, filter: str = "", status: str = None, 
                     requirement_id: str = None) -> List[Dict[str, Any]]:
    """
    List user stories with filtering options.
    
    Args:
        agent: PM Agent instance
        filter: Text filter
        status: Filter by status
        requirement_id: Filter by related requirement
        
    Returns:
        List of user stories
    """
    # Prepare filters
    filters = {}
    
    if status:
        filters["status"] = status
    
    # Query user stories from the context
    entities = agent.context_manager.query_entities(
        entity_type="user_story",
        filters=filters
    )
    
    # Filter by requirement if specified
    if requirement_id:
        # We need to check relationships
        filtered_entities = []
        for entity in entities:
            # Check if this user story is related to the requirement
            relationships = agent.context_manager.get_related_entities(
                entity_id=entity.id,
                direction="incoming"
            )
            
            for rel_type, rel_entity in relationships:
                if rel_entity.id == requirement_id:
                    filtered_entities.append(entity)
                    break
        
        entities = filtered_entities
    
    # Apply text filter
    results = []
    for entity in entities:
        include = True
        
        # Text filter
        if filter:
            filter_lower = filter.lower()
            title = entity.data.get("title", "").lower()
            description = entity.data.get("description", "").lower()
            
            if filter_lower not in title and filter_lower not in description:
                include = False
        
        if include:
            # Add entity_type for response formatting
            result = entity.to_dict()
            result["entity_type"] = "user_story"
            results.append(result)
    
    return results

def _create_user_stories_from_requirement(agent, requirement_id: str) -> List[Dict[str, Any]]:
    """
    Generate user stories from a requirement.
    
    Args:
        agent: PM Agent instance
        requirement_id: ID of the requirement

    Returns:
        List of created user stories
    """
    # Get the requirement
    requirement = agent.context_manager.get_entity(requirement_id)
    if not requirement:
        return {"success": False, "error": f"Requirement not found: {requirement_id}"}
    
    # Extract key information from the requirement
    title = requirement.data.get("title", "")
    description = requirement.data.get("description", "")
    priority = requirement.data.get("priority", "medium")
    
    # Create a primary user story
    primary_story = _create_user_story(
        agent=agent,
        title=f"User story for {title}",
        description=f"As a user, I want to {description.lower() if not description.startswith('I want') else description}, so that I can accomplish my goals more effectively",
        priority=priority,
        status="draft",
        acceptance_criteria=[
            f"The feature described in '{title}' works as expected",
            "The user interface is intuitive and easy to use",
            "The feature performs well under normal usage conditions"
        ],
        requirement_id=requirement_id
    )

    # Create an admin/management story if applicable
    if "manage" in description.lower() or "admin" in description.lower() or "dashboard" in description.lower():
        admin_story = _create_user_story(
            agent=agent,
            title=f"Admin management for {title}",
            description=f"As an administrator, I want to manage and configure {title.lower()}, so that I can ensure it meets organizational requirements",
            priority=priority,
            status="draft",
            acceptance_criteria=[
                "Admin controls are secure and accessible only to authorized users",
                "All management functions work correctly",
                "Changes made by admins are properly logged"
            ],
            requirement_id=requirement_id
        )
        
        stories = [primary_story, admin_story]
    else:
        stories = [primary_story]
    
    return stories

def _prioritize_requirements(agent, filter: str = "", method: str = "value_effort") -> Dict[str, Any]:
    """
    Prioritize a list of requirements.
    
    Args:
        agent: PM Agent instance
        filter: Optional filter string
        method: Prioritization method (simple, value_effort, moscow, rice, kano)

    Returns:
        Prioritized requirements and explanation
    """
    # Get requirements
    requirements = _list_requirements(agent, filter)
    
    if not requirements:
        return {
            "success": False,
            "error": "No requirements found to prioritize"
        }
    
    # Sort based on method
    if method == "simple":
        # Simple sorting by priority
        priority_values = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        sorted_reqs = sorted(
            requirements,
            key=lambda x: priority_values.get(x.get("priority", "medium"), 2),
            reverse=True
        )

        explanation = "Requirements sorted by priority level (critical, high, medium, low)"
        
    elif method == "value_effort":
        # Simulate value/effort prioritization
        # In a real implementation, we would ask for actual values
        for req in requirements:
            # Simulate value based on priority
            priority_value = {"critical": 5, "high": 4, "medium": 3, "low": 2}.get(req.get("priority"), 3)
            
            # Simulate effort based on description length
            desc_length = len(req.get("description", ""))
            effort = 1 if desc_length < 100 else (2 if desc_length < 300 else 3)
            
            # Calculate value/effort ratio
            req["value_effort_ratio"] = priority_value / effort
        
        sorted_reqs = sorted(
            requirements,
            key=lambda x: x.get("value_effort_ratio", 0),
            reverse=True
        )
        
        explanation = "Requirements prioritized using Value/Effort ratio. Higher ratios indicate better return on investment."
        
    elif method == "moscow":
        # MoSCoW prioritization (Must, Should, Could, Won't)
        # Simulate MoSCoW categorization
        moscow_map = {
            "critical": "Must have",
            "high": "Should have",
            "medium": "Could have",
            "low": "Won't have (this time)"
        }
        
        for req in requirements:
            req["moscow"] = moscow_map.get(req.get("priority"), "Could have")

        # Sort by MoSCoW category
        moscow_values = {
            "Must have": 4,
            "Should have": 3,
            "Could have": 2,
            "Won't have (this time)": 1
        }
        
        sorted_reqs = sorted(
            requirements,
            key=lambda x: moscow_values.get(x.get("moscow"), 2),
            reverse=True
        )
        
        explanation = "Requirements categorized using MoSCoW method (Must have, Should have, Could have, Won't have)"
        
    else:
        # Default to simple prioritization
        priority_values = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }

        sorted_reqs = sorted(
            requirements,
            key=lambda x: priority_values.get(x.get("priority", "medium"), 2),
            reverse=True
        )

        explanation = f"Requirements sorted by priority level using method: {method}"
    
    return {
        "prioritized_requirements": sorted_reqs,
        "explanation": explanation,
        "method": method,
        "entity_type": "analysis"
    }

def _create_roadmap(agent, timeframe: str = "6 months", content: str = "") -> Dict[str, Any]:
    """
    Create a product roadmap.
    
    Args:
        agent: PM Agent instance
        timeframe: Timeframe for the roadmap
        content: Additional context or requirements
        
    Returns:
        Generated roadmap
    """
    # Get requirements to include in the roadmap
    requirements = _list_requirements(agent)
    
    if not requirements:
        return {
            "success": False,
            "error": "No requirements found to build roadmap"
        }

    # Get current project
    project_id = agent.working_memory.retrieve("current_project")
    project_name = "Project"
    
    if project_id:
        project = agent.context_manager.get_entity(project_id)
        if project:
            project_name = project.data.get("name", "Project")

    # Determine timeframe components
    timeframe_parts = timeframe.split()
    if len(timeframe_parts) >= 2:
        try:
            duration = int(timeframe_parts[0])
            unit = timeframe_parts[1].lower()
        except ValueError:
            duration = 6
            unit = "months"
    else:
        duration = 6
        unit = "months"
    
    # Create phases based on timeframe
    if unit.startswith("month"):
        # Monthly phases
        phases = []
        for i in range(1, min(duration + 1, 13)):
            phases.append(f"Month {i}")
    elif unit.startswith("quarter"):
        # Quarterly phases
        phases = []
        for i in range(1, min(duration + 1, 5)):
            phases.append(f"Q{i}")
    elif unit.startswith("week"):
        # Weekly phases
        phases = []
        for i in range(1, min(duration + 1, 27)):
            phases.append(f"Week {i}")
    else:
        # Default to quarters if unit is years or unknown
        phases = ["Q1", "Q2", "Q3", "Q4"]
    
    # Sort requirements by priority for allocation
    priority_values = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }

    sorted_reqs = sorted(
        requirements,
        key=lambda x: priority_values.get(x.get("priority", "medium"), 2),
        reverse=True
    )
    
    # Allocate requirements to phases
    roadmap_items = {}
    for i, phase in enumerate(phases):
        roadmap_items[phase] = []
    
    # Distribute requirements across phases (simplified algorithm)
    current_phase_index = 0
    for req in sorted_reqs:
        phase = phases[current_phase_index]
        roadmap_items[phase].append(req)
        
        # Move to next phase after adding a few items
        if len(roadmap_items[phase]) >= 3:
            current_phase_index = (current_phase_index + 1) % len(phases)
    
    # Generate roadmap content
    roadmap_content = f"# Product Roadmap: {project_name}\n\n"
    roadmap_content += f"## Timeframe: {timeframe}\n\n"
    
    if content:
        roadmap_content += f"## Context\n{content}\n\n"
    
    roadmap_content += "## Roadmap Overview\n\n"
    
    for phase in phases:
        roadmap_content += f"### {phase}\n\n"
        
        if phase in roadmap_items and roadmap_items[phase]:
            for req in roadmap_items[phase]:
                title = req.get("title", "Unnamed requirement")
                priority = req.get("priority", "medium")
                roadmap_content += f"- **{title}** ({priority})\n"
        else:
            roadmap_content += "- No items scheduled for this phase\n"
        
        roadmap_content += "\n"
    
    # Create roadmap entity in context
    entity = agent.context_manager.create_entity(
        entity_type="roadmap",
        data={
            "title": f"{project_name} Roadmap",
            "timeframe": timeframe,
            "content": roadmap_content,
            "phases": phases,
            "items": roadmap_items,
            "created_by": agent.agent_id,
            "created_at": time.time()
        }
    )

    # Add to current project if exists
    if project_id:
        agent.context_manager.create_relationship(
            from_entity_id=project_id,
            relation_type=RelationType.CONTAINS,
            to_entity_id=entity.id,
            metadata={"type": "roadmap"}
        )
    
    return {
        "title": f"{project_name} Roadmap",
        "content": roadmap_content,
        "entity_id": entity.id,
        "entity_type": "roadmap"
    }

def _plan_sprint(agent, sprint_name: str = "Next Sprint", capacity: str = "", 
                content: str = "") -> Dict[str, Any]:
    """
    Plan a development sprint.
    
    Args:
        agent: PM Agent instance
        sprint_name: Name of the sprint
        capacity: Team capacity (story points)
        content: Additional context
        
    Returns:
        Sprint plan
    """
    # Get user stories to include in the sprint
    stories = agent.context_manager.query_entities(
        entity_type="user_story",
        filters={"status": "draft"}
    )

    if not stories:
        return {
            "success": False,
            "error": "No draft user stories found for sprint planning"
        }

    # Parse capacity if provided
    try:
        capacity_points = int(capacity) if capacity else 20  # Default capacity
    except ValueError:
        capacity_points = 20

    # Sort stories by priority
    priority_values = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1
    }
    
    sorted_stories = sorted(
        stories,
        key=lambda x: priority_values.get(x.data.get("priority", "medium"), 2),
        reverse=True
    )
    
    # Select stories for the sprint
    sprint_stories = []
    remaining_capacity = capacity_points
    
    for story in sorted_stories:
        # Extract or assign story points
        story_points = story.data.get("story_points")
        
        if story_points is None:
            # Estimate points based on description length if not provided
            desc_length = len(story.data.get("description", ""))
            story_points = 1 if desc_length < 100 else (3 if desc_length < 300 else 5)

            # Update the story with estimated points
            agent.context_manager.update_entity(
                entity_id=story.id,
                data={"story_points": story_points}
            )
        else:
            # Convert to int if it's a string
            try:
                story_points = int(story_points)
            except (ValueError, TypeError):
                story_points = 3  # Default if conversion fails
        
        # Add to sprint if capacity allows
        if story_points <= remaining_capacity:
            sprint_stories.append(story)
            remaining_capacity -= story_points
    
    # Generate sprint plan content
    sprint_content = f"# Sprint Plan: {sprint_name}\n\n"
    sprint_content += f"## Capacity: {capacity_points} story points\n\n"
    
    if content:
        sprint_content += f"## Context\n{content}\n\n"

    sprint_content += f"## Sprint Backlog ({len(sprint_stories)} user stories, {capacity_points - remaining_capacity} story points)\n\n"
    
    for story in sprint_stories:
        title = story.data.get("title", "Unnamed story")
        points = story.data.get("story_points", "?")
        priority = story.data.get("priority", "medium")

        sprint_content += f"### {title}\n"
        sprint_content += f"- **Story Points**: {points}\n"
        sprint_content += f"- **Priority**: {priority}\n"
        sprint_content += f"- **Description**: {story.data.get('description', '')}\n"

        if "acceptance_criteria" in story.data and story.data["acceptance_criteria"]:
            sprint_content += "- **Acceptance Criteria**:\n"
            for ac in story.data["acceptance_criteria"]:
                sprint_content += f"  - {ac}\n"
        
        sprint_content += "\n"
    
    # Create sprint plan entity in context
    entity = agent.context_manager.create_entity(
        entity_type="sprint_plan",
        data={
            "title": sprint_name,
            "capacity": capacity_points,
            "stories": [s.id for s in sprint_stories],
            "content": sprint_content,
            "story_points_planned": capacity_points - remaining_capacity,
            "created_by": agent.agent_id,
            "created_at": time.time()
        }
    )
    
    # Get current project
    project_id = agent.working_memory.retrieve("current_project")
    
    # Add to current project if exists
    if project_id:
        agent.context_manager.create_relationship(
            from_entity_id=project_id,
            relation_type=RelationType.CONTAINS,
            to_entity_id=entity.id,
            metadata={"type": "sprint_plan"}
        )
    
    return {
        "title": sprint_name,
        "content": sprint_content,
        "entity_id": entity.id,
        "entity_type": "sprint_plan"
    }

def _generate_prd(agent, title: str = "Product Requirements Document", 
                 content: str = "") -> Dict[str, Any]:
    """
    Generate a full PRD document.

    Args:
        agent: PM Agent instance
        title: Document title
        content: Additional context or instructions
        
    Returns:
        Generated PRD document
    """
    # Get current project
    project_id = agent.working_memory.retrieve("current_project")
    project_name = "Product"
    project_description = ""

    if project_id:
        project = agent.context_manager.get_entity(project_id)
        if project:
            project_name = project.data.get("name", "Product")
            project_description = project.data.get("description", "")
    
    # Get requirements for the PRD
    requirements = _list_requirements(agent)
    
    # Generate PRD content
    prd_content = f"# {title}\n\n"
    
    # Add document metadata
    prd_content += f"**Document Version**: 1.0\n"
    prd_content += f"**Last Updated**: {time.strftime('%Y-%m-%d')}\n"
    prd_content += f"**Author**: Product Manager\n\n"
    
    # Table of Contents
    prd_content += "## Table of Contents\n\n"
    prd_content += "1. [Introduction](#introduction)\n"
    prd_content += "2. [Product Overview](#product-overview)\n"
    prd_content += "3. [Requirements](#requirements)\n"
    prd_content += "4. [User Stories](#user-stories)\n"
    prd_content += "5. [Non-Functional Requirements](#non-functional-requirements)\n"
    
    # Introduction
    prd_content += "\n## 1. Introduction\n\n"
    prd_content += f"This document outlines the requirements for {project_name}. "
    if project_description:
        prd_content += f"{project_description}\n\n"
    else:
        prd_content += "It provides a comprehensive description of the product's functionality, features, and constraints.\n\n"
    
    # Product Overview
    prd_content += "## 2. Product Overview\n\n"
    prd_content += f"### 2.1 Purpose\n\n"
    prd_content += f"The purpose of {project_name} is to address user needs for [specifics based on requirements].\n\n"
    
    prd_content += f"### 2.2 Target Users\n\n"
    prd_content += "The product is designed for the following user personas:\n\n"
    prd_content += "- [Primary user persona]\n"
    prd_content += "- [Secondary user persona]\n\n"

    # Requirements
    prd_content += "## 3. Requirements\n\n"
    
    # Group requirements by priority
    priority_groups = {"critical": [], "high": [], "medium": [], "low": []}

    for req in requirements:
        priority = req.get("priority", "medium")
        priority_groups[priority].append(req)
    
    # Add requirements by priority
    for priority in ["critical", "high", "medium", "low"]:
        if priority_groups[priority]:
            prd_content += f"### 3.{priority.title()} Priority Requirements\n\n"
            
            for i, req in enumerate(priority_groups[priority], 1):
                req_id = req.get("id", "")
                title = req.get("title", "Unnamed requirement")
                description = req.get("description", "")
                
                prd_content += f"#### 3.{priority[0].upper()}{i}. {title}\n\n"
                prd_content += f"{description}\n\n"
                prd_content += f"*Status: {req.get('status', 'draft')}*\n\n"
    
    # User Stories
    prd_content += "## 4. User Stories\n\n"

    # Get user stories
    user_stories = agent.context_manager.query_entities(
        entity_type="user_story"
    )
    
    if user_stories:
        for i, story in enumerate(user_stories, 1):
            story_data = story.data
            title = story_data.get("title", "Unnamed story")
            description = story_data.get("description", "")
            
            prd_content += f"### 4.{i}. {title}\n\n"
            prd_content += f"{description}\n\n"
            
            if "acceptance_criteria" in story_data and story_data["acceptance_criteria"]:
                prd_content += "**Acceptance Criteria:**\n\n"
                for ac in story_data["acceptance_criteria"]:
                    prd_content += f"- {ac}\n"
                prd_content += "\n"
    else:
        prd_content += "User stories will be developed based on the requirements.\n\n"
    
    # Non-Functional Requirements
    prd_content += "## 5. Non-Functional Requirements\n\n"
    
    prd_content += "### 5.1 Performance\n\n"
    prd_content += "- Response times should be under 2 seconds for all primary operations\n"
    prd_content += "- The system should support X concurrent users\n\n"

    prd_content += "### 5.2 Security\n\n"
    prd_content += "- All user data must be encrypted at rest and in transit\n"
    prd_content += "- Authentication must use industry-standard protocols\n\n"
    
    prd_content += "### 5.3 Usability\n\n"
    prd_content += "- The interface should follow established UX patterns\n"
    prd_content += "- The product should meet WCAG 2.1 AA accessibility standards\n\n"

    # Create PRD entity in context
    entity = agent.context_manager.create_entity(
        entity_type="prd",
        data={
            "title": title,
            "content": prd_content,
            "created_by": agent.agent_id,
            "created_at": time.time()
        }
    )
    
    # Add to current project if exists
    if project_id:
        agent.context_manager.create_relationship(
            from_entity_id=project_id,
            relation_type=RelationType.CONTAINS,
            to_entity_id=entity.id,
            metadata={"type": "prd"}
        )
    
    return {
        "title": title,
        "content": prd_content,
        "entity_id": entity.id,
        "entity_type": "prd"
    }

def _generate_prd_section(agent, section_name: str, content: str = "") -> Dict[str, Any]:
    """
    Generate a specific PRD section.
    
    Args:
        agent: PM Agent instance
        section_name: Name of the section
        content: Additional context or instructions
        
    Returns:
        Generated PRD section
    """
    # Get requirements to include in the section
    requirements = _list_requirements(agent)

    # Get current project
    project_id = agent.working_memory.retrieve("current_project")
    project_name = "Product"
    
    if project_id:
        project = agent.context_manager.get_entity(project_id)
        if project:
            project_name = project.data.get("name", "Product")

    # Generate section content based on section name
    if section_name.lower() in ["introduction", "intro"]:
        section_content = f"# {section_name}\n\n"
        section_content += f"This document outlines the requirements for {project_name}. "
        section_content += "It provides a comprehensive description of the product's functionality, features, and constraints.\n\n"

        section_content += "## Document Purpose\n\n"
        section_content += "This Product Requirements Document (PRD) serves as the definitive source of truth for the product development team. It establishes:\n\n"
        section_content += "- What features and functionality will be developed\n"
        section_content += "- Who the target users are and what their needs are\n"
        section_content += "- The expected behavior and quality attributes of the system\n"
        section_content += "- The scope and constraints of the project\n\n"
        
        section_content += "## Intended Audience\n\n"
        section_content += "This document is intended for:\n\n"
        section_content += "- Product management team\n"
        section_content += "- Development team\n"
        section_content += "- Quality assurance team\n"
        section_content += "- Stakeholders and business owners\n"

        return {
            "section_name": section_name,
            "content": section_content,
            "entity_type": "prd_section"
        }
        
    elif section_name.lower() in ["product overview", "overview"]:
        section_content = f"# {section_name}\n\n"
        section_content += f"{project_name} is designed to [main value proposition]. "
        section_content += "It addresses the following key user needs and market opportunities:\n\n"
        section_content += "- [Need/opportunity 1]\n"
        section_content += "- [Need/opportunity 2]\n"
        section_content += "- [Need/opportunity 3]\n\n"
        
        section_content += "## Target Users\n\n"
        section_content += "The product is designed for the following user personas:\n\n"
        section_content += "### Primary Users\n"
        section_content += "- [Description of primary user persona]\n\n"
        section_content += "### Secondary Users\n"
        section_content += "- [Description of secondary user persona]\n\n"
        
        section_content += "## Product Context\n\n"
        section_content += "This product fits into the overall ecosystem by [describe strategic fit].\n"

        return {
            "section_name": section_name,
            "content": section_content,
            "entity_type": "prd_section"
        }
        
    elif section_name.lower() in ["requirements", "functional requirements"]:
        section_content = f"# {section_name}\n\n"
        section_content += "This section outlines the functional requirements for the product.\n\n"
        
        # Group requirements by priority
        priority_groups = {"critical": [], "high": [], "medium": [], "low": []}
        
        for req in requirements:
            priority = req.get("priority", "medium")
            priority_groups[priority].append(req)
        
        # Add requirements by priority
        for priority in ["critical", "high", "medium", "low"]:
            if priority_groups[priority]:
                section_content += f"## {priority.title()} Priority Requirements\n\n"

                for i, req in enumerate(priority_groups[priority], 1):
                    title = req.get("title", "Unnamed requirement")
                    description = req.get("description", "")
                    
                    section_content += f"### {i}. {title}\n\n"
                    section_content += f"{description}\n\n"
                    section_content += f"*Status: {req.get('status', 'draft')}*\n\n"
            else:
                section_content += f"## {priority.title()} Priority Requirements\n\n"
                section_content += f"No {priority} priority requirements defined yet.\n\n"

        return {
            "section_name": section_name,
            "content": section_content,
            "entity_type": "prd_section"
        }
        
    # Generic section if not recognized
    section_content = f"# {section_name}\n\n"
    section_content += f"This section provides details about {section_name.lower()} for {project_name}.\n\n"
    
    # Include some requirements if available
    if requirements:
        section_content += "## Related Requirements\n\n"
        for i, req in enumerate(requirements[:5], 1):
            title = req.get("title", "Unnamed requirement")
            section_content += f"{i}. {title}\n"
    
    return {
        "section_name": section_name,
        "content": section_content,
        "entity_type": "prd_section"
    }

def _analyze_requirements(agent, filter: str = "") -> Dict[str, Any]:
    """
    Analyze requirements for completeness, clarity, etc.
    
    Args:
        agent: PM Agent instance
        filter: Optional filter string to select requirements

    Returns:
        Analysis results
    """
    # Get requirements to analyze
    requirements = _list_requirements(agent, filter)
    
    if not requirements:
        return {
            "success": False,
            "error": "No requirements found to analyze"
        }
    
    # Initialize analysis variables
    total_count = len(requirements)
    status_counts = {"draft": 0, "review": 0, "approved": 0, "implemented": 0, "verified": 0}
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    avg_description_length = 0
    issues_found = []
    
    # Requirements with issues
    incomplete_reqs = []
    vague_reqs = []
    overlapping_reqs = []
    
    # Analyze each requirement
    for req in requirements:
        # Count by status and priority
        status = req.get("status", "draft")
        status_counts[status] = status_counts.get(status, 0) + 1

        priority = req.get("priority", "medium")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Analyze description
        description = req.get("description", "")
        avg_description_length += len(description)

        # Check for common issues (simplified analysis for MVP)
        
        # Check for incomplete descriptions
        if len(description) < 50:
            incomplete_reqs.append(req)
            issues_found.append(f"Short description in '{req.get('title', 'Unnamed')}'")

        # Check for vague language
        vague_terms = ["etc", "and so on", "and more", "appropriate", "adequate", "sufficient", "suitable"]
        if any(term in description.lower() for term in vague_terms):
            vague_reqs.append(req)
            issues_found.append(f"Vague terms in '{req.get('title', 'Unnamed')}'")

        # Check for potentially overlapping requirements
        for other_req in requirements:
            if req.get('id') != other_req.get('id'):
                req_title = req.get('title', '').lower()
                other_title = other_req.get('title', '').lower()
                req_desc = req.get('description', '').lower()
                other_desc = other_req.get('description', '').lower()
                
                # Very simple overlap detection (would be more sophisticated in a real system)
                words_in_common = set(req_title.split() + req_desc.split()) & set(other_title.split() + other_desc.split())
                if len(words_in_common) > 10 and req.get('id') not in [r.get('id') for r in overlapping_reqs]:
                    overlapping_reqs.append(req)
                    issues_found.append(f"Possible overlap between '{req.get('title', 'Unnamed')}' and '{other_req.get('title', 'Unnamed')}'")
                    break
    
    # Calculate averages and percentages
    if total_count > 0:
        avg_description_length /= total_count
    
    # Prepare recommendations
    recommendations = []
    
    if incomplete_reqs:
        recommendations.append(f"Improve {len(incomplete_reqs)} requirements with short descriptions")

    if vague_reqs:
        recommendations.append(f"Remove vague terms from {len(vague_reqs)} requirements")
    
    if overlapping_reqs:
        recommendations.append(f"Review {len(overlapping_reqs)} potentially overlapping requirements")
    
    if status_counts["draft"] > total_count * 0.7:
        recommendations.append("Move more requirements from draft to review or approval")
    
    if status_counts["approved"] == 0:
        recommendations.append("No requirements have been approved yet - review and approve high priority items")
    
    # Generate analysis summary
    analysis_content = f"# Requirements Analysis\n\n"
    
    analysis_content += f"## Overview\n\n"
    analysis_content += f"- Total requirements: {total_count}\n"
    analysis_content += f"- Average description length: {int(avg_description_length)} characters\n\n"
    
    analysis_content += "## Status Distribution\n\n"
    for status, count in status_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        analysis_content += f"- {status.title()}: {count} ({percentage:.1f}%)\n"
    analysis_content += "\n"
    
    analysis_content += "## Priority Distribution\n\n"
    for priority, count in priority_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        analysis_content += f"- {priority.title()}: {count} ({percentage:.1f}%)\n"
    analysis_content += "\n"

    if issues_found:
        analysis_content += "## Issues Identified\n\n"
        for issue in issues_found[:10]:  # Limit to 10 issues to keep it manageable
            analysis_content += f"- {issue}\n"
        
        if len(issues_found) > 10:
            analysis_content += f"- ... and {len(issues_found) - 10} more issues\n"
        analysis_content += "\n"

    if recommendations:
        analysis_content += "## Recommendations\n\n"
        for recommendation in recommendations:
            analysis_content += f"- {recommendation}\n"
        analysis_content += "\n"
    
    return {
        "analysis": analysis_content,
        "total_count": total_count,
        "status_counts": status_counts,
        "priority_counts": priority_counts,
        "issues_count": len(issues_found),
        "recommendations": recommendations,
        "entity_type": "analysis"
    }

def _calculate_metrics(agent, metric_type: str = "requirements") -> Dict[str, Any]:
    """
    Calculate product management metrics.
    
    Args:
        agent: PM Agent instance
        metric_type: Type of metrics to calculate
        
    Returns:
        Calculated metrics
    """
    if metric_type == "requirements":
        # Get all requirements
        requirements = agent.context_manager.query_entities(
            entity_type=EntityType.REQUIREMENT
        )
        
        # Calculate requirements metrics
        total_count = len(requirements)
        
        if total_count == 0:
            return {
                "success": False,
                "error": "No requirements found"
            }
                
        # Count by status and priority
        status_counts = {"draft": 0, "review": 0, "approved": 0, "implemented": 0, "verified": 0}
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # Calculate creation dates and completion rate
        creation_dates = []
        completed_count = 0  # implemented or verified
        
        for req in requirements:
            # Status counts
            status = req.get("status", "draft")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Priority counts
            priority = req.get("priority", "medium")
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Creation date
            if "created_at" in req:
                creation_dates.append(req["created_at"])
            
            # Completion status
            if status in ["implemented", "verified"]:
                completed_count += 1

        # Calculate metrics
        completion_rate = (completed_count / total_count) if total_count > 0 else 0
        
        # Calculate creation rate (requirements per day)
        creation_rate = 0
        if creation_dates:
            earliest_date = min(creation_dates)
            latest_date = max(creation_dates)
            days_span = (latest_date - earliest_date) / (60 * 60 * 24)  # Convert seconds to days
            if days_span > 0:
                creation_rate = total_count / days_span
        
        # Format metrics as a report
        metrics_content = f"# Requirements Metrics\n\n"
        
        metrics_content += "## Key Metrics\n\n"
        metrics_content += f"- Total Requirements: {total_count}\n"
        metrics_content += f"- Completion Rate: {completion_rate:.1%}\n"
        
        if creation_rate > 0:
            metrics_content += f"- Creation Rate: {creation_rate:.2f} requirements per day\n"

        metrics_content += "\n## Status Distribution\n\n"
        for status, count in status_counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            metrics_content += f"- {status.title()}: {count} ({percentage:.1f}%)\n"
        
        metrics_content += "\n## Priority Distribution\n\n"
        for priority, count in priority_counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            metrics_content += f"- {priority.title()}: {count} ({percentage:.1f}%)\n"
        
        return {
            "metrics": metrics_content,
            "total_count": total_count,
            "completion_rate": completion_rate,
            "creation_rate": creation_rate,
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "entity_type": "metrics"
        }
            
    elif metric_type == "user_stories":
        # Get all user stories
        user_stories = agent.context_manager.query_entities(
            entity_type="user_story"
        )

        # Calculate user story metrics
        total_count = len(user_stories)
        
        if total_count == 0:
            return {
                "success": False,
                "error": "No user stories found"
            }

        # Count by status
        status_counts = {"draft": 0, "in_progress": 0, "testing": 0, "completed": 0}
        
        # Calculate story points
        total_points = 0
        stories_with_points = 0
        
        for story in user_stories:
            story_data = story.data
            
            # Status counts
            status = story_data.get("status", "draft")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Story points
            if "story_points" in story_data and story_data["story_points"] is not None:
                try:
                    points = int(story_data["story_points"])
                    total_points += points
                    stories_with_points += 1
                except (ValueError, TypeError):
                    pass

        # Calculate metrics
        average_points = (total_points / stories_with_points) if stories_with_points > 0 else 0
        
        # Format metrics as a report
        metrics_content = f"# User Story Metrics\n\n"
        
        metrics_content += "## Key Metrics\n\n"
        metrics_content += f"- Total User Stories: {total_count}\n"
        
        if stories_with_points > 0:
            metrics_content += f"- Total Story Points: {total_points}\n"
            metrics_content += f"- Average Points per Story: {average_points:.1f}\n"
        else:
            metrics_content += "- No story points have been assigned yet\n"
        
        metrics_content += "\n## Status Distribution\n\n"
        for status, count in status_counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            metrics_content += f"- {status.title().replace('_', ' ')}: {count} ({percentage:.1f}%)\n"
        
        return {
            "metrics": metrics_content,
            "total_count": total_count,
            "total_points": total_points,
            "average_points": average_points,
            "status_counts": status_counts,
            "entity_type": "metrics"
        }
            
    elif metric_type == "velocity":
        # Get completed user stories
        user_stories = agent.context_manager.query_entities(
            entity_type="user_story",
            filters={"status": "completed"}
        )

        # Get sprint plans
        sprint_plans = agent.context_manager.query_entities(
            entity_type="sprint_plan"
        )
        
        total_completed = len(user_stories)
        total_sprints = len(sprint_plans)
        
        if total_completed == 0 or total_sprints == 0:
            return {
                "success": False,
                "error": "Insufficient data to calculate velocity (need completed stories and sprints)"
            }

        # Calculate total points completed
        total_points_completed = 0
        for story in user_stories:
            story_data = story.data
            if "story_points" in story_data and story_data["story_points"] is not None:
                try:
                    points = int(story_data["story_points"])
                    total_points_completed += points
                except (ValueError, TypeError):
                    pass
        
        # Calculate velocity metrics
        velocity = total_points_completed / total_sprints

        # Format metrics as a report
        metrics_content = f"# Team Velocity Metrics\n\n"

        metrics_content += "## Key Metrics\n\n"
        metrics_content += f"- Completed Stories: {total_completed}\n"
        metrics_content += f"- Total Sprints: {total_sprints}\n"
        metrics_content += f"- Total Story Points Completed: {total_points_completed}\n"
        metrics_content += f"- Average Velocity: {velocity:.1f} points per sprint\n"

        return {
            "metrics": metrics_content,
            "total_completed": total_completed,
            "total_sprints": total_sprints,
            "total_points_completed": total_points_completed,
            "velocity": velocity,
            "entity_type": "metrics"
        }

    # Default to requirements metrics if type not recognized
    return _calculate_metrics(agent, "requirements")
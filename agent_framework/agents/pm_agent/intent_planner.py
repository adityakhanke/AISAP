"""
Intent determination and planning module for the PM Agent.
Provides both LLM-enhanced and traditional intent determination methods.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

async def determine_intent_with_llm(analysis_result: Dict[str, Any], llm: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Determine user intent and create execution plan using LLM.
    
    Args:
        analysis_result: Result from input analysis
        llm: LLM configuration to use
        
    Returns:
        Tuple of (intent, plan)
    """
    content = analysis_result.get("content", "")
    llm_analysis = analysis_result.get("llm_analysis", {})
    
    # Use the LLM's analysis to determine intent
    intent = llm_analysis.get("primary_intent", "general_inquiry").lower()
    is_command = llm_analysis.get("is_command", False)
    
    # Standardize intent names
    intent_mapping = {
        "create requirement": "create_requirement",
        "create user story": "create_user_story",
        "generate prd": "generate_prd",
        "prioritize requirements": "prioritize_requirements",
        "create roadmap": "create_roadmap",
        "plan sprint": "plan_sprint",
        "analyze requirements": "analyze_requirements",
        "general inquiry": "general_inquiry",
        "list requirements": "list_requirements",
        "list user stories": "list_user_stories",
    }
    
    # Try to match to a known intent
    for key, value in intent_mapping.items():
        if key in intent:
            intent = value
            break
    
    # Initialize the plan with analysis data
    plan = {
        "content": content,
        "rag_context": analysis_result.get("rag_context"),
        "workflow_context": analysis_result.get("workflow_context"),
        "entities": llm_analysis.get("entities", {}),
        "themes": llm_analysis.get("themes", []),
        "intent": intent
    }
    
    # If it's a command-style request, add command info
    if is_command:
        command_info = {
            "is_command": True,
            "command_type": llm_analysis.get("command", {}).get("type"),
            "command_target": llm_analysis.get("command", {}).get("target"),
            "command_args": {}
        }
        
        # Extract arguments if available
        if "args" in llm_analysis.get("command", {}):
            command_info["command_args"] = llm_analysis["command"]["args"]
        
        plan["command_info"] = command_info
    
    # Enhance the plan with LLM-generated task-specific parameters
    plan_enhancement_prompt = f"""
    Based on the following user request to a Product Manager agent:
    
    USER REQUEST: {content}
    
    I need to create an execution plan for intent: {intent}
    
    What specific parameters or information should I include in my execution plan?
    Return a JSON object with recommended parameters for this task.
    """
    
    enhancement_response = await llm.generate(
        prompt=plan_enhancement_prompt,
        system_prompt="You are a helpful assistant that recommends parameters for product management tasks. Respond in JSON format.",
        params={"temperature": 0.4, "max_tokens": 32768}
    )
    
    # Parse the enhanced plan parameters
    plan_text = enhancement_response["text"]
    if "```json" in plan_text:
        json_match = re.search(r"```json\n(.*?)\n```", plan_text, re.DOTALL)
        if json_match:
            plan_text = json_match.group(1)
    
    try:
        plan_params = json.loads(plan_text)
        # Merge with our plan
        plan.update(plan_params)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM plan parameters: {e}")
    
    return intent, plan

def determine_intent_traditional(analysis_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Determine user intent and create execution plan using traditional methods.
    
    Args:
        analysis_result: Result from input analysis
        
    Returns:
        Tuple of (intent, plan)
    """
    content = analysis_result.get("content", "")
    command_info = analysis_result.get("command_info", {})
    entities = analysis_result.get("entities", {})
    themes = analysis_result.get("themes", [])
    
    # Initialize the plan with analysis data
    plan = {
        "content": content,
        "rag_context": analysis_result.get("rag_context"),
        "workflow_context": analysis_result.get("workflow_context"),
        "entities": entities,
        "themes": themes
    }
    
    # If it's a command-style request, use command parsing result
    if command_info.get("is_command", False):
        return _determine_intent_from_command(command_info, plan)
    
    # Otherwise, use natural language understanding
    return _determine_intent_from_natural_language(content, plan)

def _determine_intent_from_command(command_info: Dict[str, Any], 
                               plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Determine intent from a command-style request.
    
    Args:
        command_info: Command parsing result
        plan: Base execution plan
        
    Returns:
        Tuple of (intent, plan)
    """
    command_type = command_info.get("command_type")
    command_target = command_info.get("command_target")
    command_args = command_info.get("command_args", {})
    entity_ids = command_info.get("entity_ids", [])
    
    intent = f"{command_type}_{command_target}"
    
    # Standardize some intent names
    intent_mapping = {
        "create_requirement": "create_requirement",
        "create_requirements": "create_requirement",
        "create_story": "create_user_story",
        "create_stories": "create_user_story",
        "create_userstory": "create_user_story",
        "list_story": "list_user_stories",
        "list_stories": "list_user_stories",
        "list_userstory": "list_user_stories",
        "list_userstories": "list_user_stories",
        "update_story": "update_user_story",
        "update_userstory": "update_user_story",
        "generate_document": "generate_prd",
        "generate_prd": "generate_prd",
        "generate_spec": "generate_prd",
        "create_doc": "generate_prd",
        "create_prd": "generate_prd"
    }
    
    if intent in intent_mapping:
        intent = intent_mapping[intent]
    
    # Add intent-specific data to the plan
    if command_type == "create":
        if command_target in ["requirement", "requirements"]:
            plan.update({
                "intent": "create_requirement",
                "title": command_args.get("title", "New Requirement"),
                "description": command_args.get("content", ""),
                "priority": command_args.get("priority", "medium"),
                "status": command_args.get("status", "draft")
            })
            return "create_requirement", plan
        
        elif command_target in ["story", "stories", "userstory", "userstories"]:
            plan.update({
                "intent": "create_user_story",
                "title": command_args.get("title", "New User Story"),
                "description": command_args.get("content", ""),
                "priority": command_args.get("priority", "medium"),
                "status": command_args.get("status", "draft")
            })
            return "create_user_story", plan
        
        elif command_target in ["project"]:
            plan.update({
                "intent": "create_project",
                "name": command_args.get("name", command_args.get("content", "New Project")),
                "description": command_args.get("description", "")
            })
            return "create_project", plan
        
        elif command_target in ["roadmap"]:
            plan.update({
                "intent": "create_roadmap",
                "timeframe": command_args.get("timeframe", "6 months"),
                "content": command_args.get("content", "")
            })
            return "create_roadmap", plan
    
    elif command_type == "list":
        if command_target in ["requirement", "requirements"]:
            plan.update({
                "intent": "list_requirements",
                "filter": command_args.get("filter", command_args.get("content", "")),
                "status": command_args.get("status", ""),
                "priority": command_args.get("priority", "")
            })
            return "list_requirements", plan
        
        elif command_target in ["story", "stories", "userstory", "userstories"]:
            plan.update({
                "intent": "list_user_stories",
                "filter": command_args.get("filter", command_args.get("content", "")),
                "status": command_args.get("status", "")
            })
            return "list_user_stories", plan
    
    elif command_type == "update":
        if command_target in ["requirement", "requirements"] and entity_ids:
            plan.update({
                "intent": "update_requirement",
                "requirement_id": entity_ids[0],
                "updates": command_args
            })
            return "update_requirement", plan
        
        elif command_target in ["story", "userstory"] and entity_ids:
            plan.update({
                "intent": "update_user_story",
                "story_id": entity_ids[0],
                "updates": command_args
            })
            return "update_user_story", plan
    
    elif command_type == "generate":
        if command_target in ["prd", "document", "spec", "documentation"]:
            plan.update({
                "intent": "generate_prd",
                "title": command_args.get("title", "Product Requirements Document"),
                "content": command_args.get("content", "")
            })
            return "generate_prd", plan
    
    elif command_type == "prioritize":
        if command_target in ["requirement", "requirements"]:
            plan.update({
                "intent": "prioritize_requirements",
                "method": command_args.get("method", "value_effort"),
                "filter": command_args.get("filter", "")
            })
            return "prioritize_requirements", plan
    
    elif command_type == "plan":
        if command_target in ["sprint"]:
            plan.update({
                "intent": "plan_sprint",
                "capacity": command_args.get("capacity", ""),
                "sprint_name": command_args.get("name", "Next Sprint")
            })
            return "plan_sprint", plan
    
    elif command_type == "analyze":
        if command_target in ["requirement", "requirements"]:
            plan.update({
                "intent": "analyze_requirements",
                "filter": command_args.get("filter", "")
            })
            return "analyze_requirements", plan
    
    # If no specific command pattern was matched
    plan.update({
        "intent": "unknown_command",
        "command_type": command_type,
        "command_target": command_target
    })
    return "unknown_command", plan

def _determine_intent_from_natural_language(content: str, 
                                        plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Determine intent from natural language.
    
    Args:
        content: The user's request content
        plan: Base execution plan
        
    Returns:
        Tuple of (intent, plan)
    """
    lower_content = content.lower()
    
    # Check for project creation/management
    if re.search(r'create\s+(?:a\s+)?(?:new\s+)?project', lower_content):
        project_name = None
        match = re.search(r'project\s+(?:called|named)\s+["\']?([^"\'\.,;]+)["\']?', lower_content)
        if match:
            project_name = match.group(1).strip()
        
        plan.update({
            "intent": "create_project",
            "name": project_name or "New Project",
            "description": content
        })
        return "create_project", plan
    
    # Check for requirement creation
    if re.search(r'create\s+(?:a\s+)?(?:new\s+)?requirement', lower_content):
        plan.update({
            "intent": "create_requirement",
            "title": _extract_title(content) or "New Requirement",
            "description": content,
            "priority": _extract_priority(content) or "medium",
            "status": "draft"
        })
        return "create_requirement", plan
    
    # Check for user story creation
    if re.search(r'create\s+(?:a\s+)?(?:new\s+)?(?:user\s+)?stor(?:y|ies)', lower_content):
        plan.update({
            "intent": "create_user_story",
            "title": _extract_title(content) or "New User Story",
            "description": content,
            "priority": _extract_priority(content) or "medium",
            "status": "draft"
        })
        return "create_user_story", plan
    
    # Check for requirement listing
    if re.search(r'list\s+(?:all\s+)?requirement', lower_content) or \
       re.search(r'show\s+(?:all\s+)?requirement', lower_content) or \
       re.search(r'what\s+(?:are\s+)?(?:the\s+)?requirement', lower_content):
        plan.update({
            "intent": "list_requirements",
            "filter": content,
            "status": _extract_status(content),
            "priority": _extract_priority(content)
        })
        return "list_requirements", plan
    
    # Check for user story listing
    if re.search(r'list\s+(?:all\s+)?(?:user\s+)?stor(?:y|ies)', lower_content) or \
       re.search(r'show\s+(?:all\s+)?(?:user\s+)?stor(?:y|ies)', lower_content):
        plan.update({
            "intent": "list_user_stories",
            "filter": content,
            "status": _extract_status(content)
        })
        return "list_user_stories", plan
    
    # Check for PRD generation
    if re.search(r'(?:generate|create)\s+(?:a\s+)?(?:PRD|product\s+requirements?\s+document)', lower_content):
        plan.update({
            "intent": "generate_prd",
            "title": _extract_title(content) or "Product Requirements Document",
            "content": content
        })
        return "generate_prd", plan
    
    # Check for roadmap creation
    if re.search(r'(?:create|generate)\s+(?:a\s+)?roadmap', lower_content):
        plan.update({
            "intent": "create_roadmap",
            "timeframe": _extract_timeframe(content) or "6 months",
            "content": content
        })
        return "create_roadmap", plan
    
    # Check for requirement prioritization
    if re.search(r'prioritize\s+(?:the\s+)?requirements', lower_content):
        plan.update({
            "intent": "prioritize_requirements",
            "method": _extract_prioritization_method(content) or "value_effort",
            "filter": content
        })
        return "prioritize_requirements", plan
    
    # Check for sprint planning
    if re.search(r'plan\s+(?:a\s+)?(?:new\s+)?sprint', lower_content):
        plan.update({
            "intent": "plan_sprint",
            "sprint_name": _extract_sprint_name(content) or "Next Sprint",
            "capacity": _extract_capacity(content) or "",
            "content": content
        })
        return "plan_sprint", plan
    
    # Check for requirement analysis
    if re.search(r'analyze\s+(?:the\s+)?requirements', lower_content):
        plan.update({
            "intent": "analyze_requirements",
            "filter": content
        })
        return "analyze_requirements", plan
    
    # Default to general inquiry
    plan.update({
        "intent": "general_inquiry",
        "content": content
    })
    return "general_inquiry", plan

# Helper functions for extracting parameters from natural language

def _extract_title(content: str) -> Optional[str]:
    """Extract a title from content."""
    # Look for explicit title
    title_match = re.search(r'title[:\s]+["\'"]?([^"\'\n.,;]+)["\'"]?', content, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()
    
    # Look for "for X" pattern
    for_match = re.search(r'for\s+["\'"]?([^"\'\n.,;]{3,50})["\'"]?', content)
    if for_match:
        return for_match.group(1).strip()
    
    return None

def _extract_priority(content: str) -> Optional[str]:
    """Extract priority from content."""
    lower_content = content.lower()
    
    if "critical" in lower_content or "highest" in lower_content:
        return "critical"
    elif "high" in lower_content:
        return "high"
    elif "low" in lower_content:
        return "low"
    elif "medium" in lower_content or "normal" in lower_content:
        return "medium"
    
    return None

def _extract_status(content: str) -> Optional[str]:
    """Extract status from content."""
    lower_content = content.lower()
    
    status_patterns = {
        "draft": r'\b(?:draft|new|initial)\b',
        "review": r'\b(?:review|reviewing|in\s+review)\b',
        "approved": r'\b(?:approved|accepted|confirmed)\b',
        "implemented": r'\b(?:implemented|completed|done)\b',
        "verified": r'\b(?:verified|tested|validated)\b'
    }
    
    for status, pattern in status_patterns.items():
        if re.search(pattern, lower_content):
            return status
    
    return None

def _extract_timeframe(content: str) -> Optional[str]:
    """Extract timeframe from content."""
    timeframe_match = re.search(r'(?:for|next|coming)\s+(\d+)\s*(?:month|week|quarter|year)s?', content, re.IGNORECASE)
    if timeframe_match:
        number = timeframe_match.group(1)
        unit = re.search(r'(month|week|quarter|year)', timeframe_match.group(0), re.IGNORECASE)
        if unit:
            return f"{number} {unit.group(1)}s"
    
    return None

def _extract_prioritization_method(content: str) -> Optional[str]:
    """Extract prioritization method from content."""
    lower_content = content.lower()
    
    if "value" in lower_content and "effort" in lower_content:
        return "value_effort"
    elif "mscw" in lower_content or "must should could" in lower_content:
        return "moscow"
    elif "rice" in lower_content:
        return "rice"
    elif "kano" in lower_content:
        return "kano"
    elif "simple" in lower_content:
        return "simple"
    
    return None

def _extract_sprint_name(content: str) -> Optional[str]:
    """Extract sprint name from content."""
    name_match = re.search(r'sprint\s+(?:called|named)\s+["\']?([^"\'\n\.,;]+)["\']?', content, re.IGNORECASE)
    if name_match:
        return name_match.group(1).strip()
    
    return None

def _extract_capacity(content: str) -> Optional[str]:
    """Extract team capacity from content."""
    capacity_match = re.search(r'capacity\s+(?:of\s+)?(\d+)', content, re.IGNORECASE)
    if capacity_match:
        return capacity_match.group(1)
    
    return None
"""
Response generation module for the PM Agent.
Handles formatting and enhancing responses from execution results.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

async def generate_response_with_llm(execution_result: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Generate a formatted response from execution results using LLM.
    
    Args:
        execution_result: Result from plan execution
        llm: LLM configuration to use
        
    Returns:
        Formatted response
    """
    result = execution_result.get("result")
    
    if result is None:
        return {
            "content": "I've processed your request, but no specific result was returned."
        }
    
    # Format using LLM if it's complex
    if isinstance(result, (dict, list)) and "text" not in result:
        # Convert result to a readable format
        result_json = json.dumps(result, indent=2)
        
        formatting_prompt = f"""
        I need to present the following data to a user in a friendly, readable format:
        
        {result_json}
        
        Please format this as a clear, well-structured response. Use markdown formatting.
        The response should be helpful and informative rather than just dumping the raw data.
        """
        
        llm_response = await llm.generate(
            prompt=formatting_prompt,
            system_prompt="You are a helpful assistant that formats data into readable, user-friendly responses.",
            params={"temperature": 0.7, "max_tokens": 32768}
        )
        
        return {
            "content": llm_response["text"]
        }
    
    # Handle string result
    if isinstance(result, str):
        return {
            "content": result
        }
    
    # Return fallback formatting if LLM fails
    return format_response_traditional(execution_result)

def format_response_traditional(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a response from execution results using traditional methods.
    
    Args:
        execution_result: Result from plan execution
        
    Returns:
        Formatted response
    """
    result = execution_result.get("result")
    
    # Handle dictionary result
    if isinstance(result, dict):
        if "entity_type" in result:
            # Format entity-based result
            entity_type = result["entity_type"]
            
            if entity_type == "requirement":
                content = f"# Requirement: {result.get('title', 'New Requirement')}\n\n"
                content += result.get('description', '')
                return {"content": content}
                
            elif entity_type == "user_story":
                content = f"# User Story: {result.get('title', 'New User Story')}\n\n"
                content += result.get('description', '')
                return {"content": content}
                
            elif entity_type == "project":
                content = f"# Project: {result.get('name', 'New Project')}\n\n"
                content += result.get('description', '')
                return {"content": content}
        
        # Generic dictionary result
        content = ""
        for key, value in result.items():
            if key not in ["raw_response", "entity_type"]:
                content += f"**{key.replace('_', ' ').title()}**: {value}\n"
        
        return {"content": content}
    
    # Handle list result
    if isinstance(result, list):
        if not result:
            return {"content": "No items found."}
            
        content = f"# Found {len(result)} items\n\n"
        for i, item in enumerate(result, 1):
            if isinstance(item, dict):
                title = item.get('title', item.get('name', f'Item {i}'))
                content += f"## {i}. {title}\n\n"
                
                for key, value in item.items():
                    if key not in ['title', 'name', 'id']:
                        content += f"**{key.replace('_', ' ').title()}**: {value}\n"
            else:
                content += f"{i}. {item}\n"
                
            content += "\n"
            
        return {"content": content}
    
    # Default response
    return {
        "content": f"Request completed successfully. Result: {result}"
    }
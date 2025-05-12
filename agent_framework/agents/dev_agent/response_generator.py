"""
Response generation module for the Dev Agent.
Handles formatting and enhancing responses from execution results.
"""

import logging
import json
import re
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
    
    # If result is already a string (LLM response), use it directly
    if isinstance(result, str):
        return {
            "content": result
        }
    
    # Format using LLM if it's complex
    if isinstance(result, (dict, list)):
        # Handle code entity result
        if isinstance(result, dict) and result.get("entity_type") == "code_entity":
            code = result.get("code", "")
            language = result.get("language", "")
            
            # Format code with proper markdown
            content = f"Here's the generated code in {language}:\n\n"
            content += f"```{language}\n{code}\n```\n\n"
            
            # Add explanatory notes if available
            if "explanation" in result:
                content += f"## Explanation\n\n{result['explanation']}"
            
            return {"content": content}
        
        # For other complex results, convert to JSON and format
        result_json = json.dumps(result, indent=2)
        
        formatting_prompt = f"""
        I need to present the following technical data to a user in a friendly, readable format:
        
        {result_json}
        
        Please format this as a clear, well-structured response. Use markdown formatting.
        The response should be helpful and informative rather than just dumping the raw data.
        Include code snippets in appropriate markdown code blocks if present in the data.
        """
        
        llm_response = await llm.generate(
            prompt=formatting_prompt,
            system_prompt="You are a helpful assistant that formats technical data into readable, developer-friendly responses.",
            params={"temperature": 0.7, "max_tokens": 32768}
        )
        
        return {
            "content": llm_response["text"]
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
    
    # Handle string result
    if isinstance(result, str):
        return {"content": result}
    
    # Handle dictionary result
    if isinstance(result, dict):
        # Handle code entity result
        if result.get("entity_type") == "code_entity":
            code = result.get("code", "")
            language = result.get("language", "")
            
            # Format code with proper markdown
            content = f"Here's the generated code in {language}:\n\n"
            content += f"```{language}\n{code}\n```\n\n"
            
            # Add explanatory notes if available
            if "explanation" in result:
                content += f"## Explanation\n\n{result['explanation']}"
            
            return {"content": content}
        
        # Handle debug result
        if result.get("entity_type") == "debug_result":
            original_code = result.get("original_code", "")
            fixed_code = result.get("fixed_code", "")
            issue = result.get("issue", "")
            explanation = result.get("explanation", "")
            language = result.get("language", "")
            
            content = f"# Debug Analysis\n\n"
            
            if issue:
                content += f"## Issue Identified\n\n{issue}\n\n"
            
            if explanation:
                content += f"## Explanation\n\n{explanation}\n\n"
            
            if fixed_code:
                content += f"## Fixed Code\n\n```{language}\n{fixed_code}\n```\n\n"
            
            return {"content": content}
        
        # Handle code analysis result
        if result.get("entity_type") == "code_analysis":
            code = result.get("code", "")
            language = result.get("language", "")
            analysis = result.get("analysis", "")
            recommendations = result.get("recommendations", [])
            
            content = f"# Code Analysis\n\n"
            content += f"{analysis}\n\n"
            
            if recommendations:
                content += f"## Recommendations\n\n"
                for i, rec in enumerate(recommendations, 1):
                    content += f"{i}. {rec}\n"
            
            return {"content": content}
        
        # Generic dictionary result
        content = ""
        for key, value in result.items():
            if key not in ["raw_response", "entity_type"]:
                # Format key name nicely
                formatted_key = key.replace("_", " ").title()
                
                # Handle different value types
                if isinstance(value, str):
                    # Check if value is code
                    if re.match(r'^[\s\n]*(function|class|def|import|const|let|var|public|private)', value):
                        # Guess language based on syntax
                        language = "python" if "def " in value or "import " in value else "javascript"
                        content += f"**{formatted_key}**:\n\n```{language}\n{value}\n```\n\n"
                    else:
                        content += f"**{formatted_key}**:\n\n{value}\n\n"
                elif isinstance(value, list):
                    # Format list items
                    content += f"**{formatted_key}**:\n\n"
                    for item in value:
                        content += f"- {item}\n"
                    content += "\n"
                else:
                    content += f"**{formatted_key}**: {value}\n\n"
        
        return {"content": content}
    
    # Handle list result
    if isinstance(result, list):
        if not result:
            return {"content": "No items found."}
            
        content = ""
        for i, item in enumerate(result, 1):
            if isinstance(item, dict):
                # Format each dictionary item
                title = item.get('name', item.get('title', f'Item {i}'))
                content += f"## {i}. {title}\n\n"
                
                for key, value in item.items():
                    if key not in ['name', 'title', 'id']:
                        formatted_key = key.replace("_", " ").title()
                        content += f"**{formatted_key}**: {value}\n"
                        
                content += "\n"
            else:
                content += f"{i}. {item}\n"
                
        return {"content": content}
    
    # Default response
    return {
        "content": f"Request completed successfully. Result: {result}"
    }
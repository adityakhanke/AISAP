"""
Plan execution module for the Dev Agent.
Handles the execution of plans using tools and LLM.
"""

import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

async def execute_plan_with_llm(plan: Dict[str, Any], agent: Any, llm: Any,
                           tool_registry: Any, context_manager: Any,
                           conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute a plan based on the determined intent using tools and LLM.
    """
    intent = plan.get("intent", "general_inquiry")
    
    # First check if we should use a tool for this intent
    tool_based_intents = [
        "generate_code", "explain_code", "debug_code", "implement_requirement",
        "analyze_code", "optimize_code", "refactor_code", "generate_tests"
    ]
    
    if intent in tool_based_intents and tool_registry:
        # Convert intent to tool name
        tool_name = intent
        
        # Check if the tool exists
        if not tool_registry.has_tool(tool_name):
            logger.warning(f"Tool not found for intent: {intent}")
            # Fall back to LLM-based execution
        else:
            # Prepare parameters from plan
            params = {}
            
            # General preparation for all tools - set defaults for required parameters
            language = plan.get("language", "python")  # Default to Python if not specified
            params["language"] = language
            
            # Special handling for different tools based on their expected parameters
            if tool_name == "generate_code":
                # Map plan parameters to tool parameters
                description = plan.get("description", plan.get("content", ""))
                requirements = plan.get("requirements", [])
                
                params = {
                    "language": language,
                    "description": description,
                    "requirements": requirements
                }
            
            # [... rest of the function ...]
            
            elif tool_name == "explain_code":
                # Map parameters for explain_code
                language = plan.get("language", "python")
                code = plan.get("code", "")
                
                params = {
                    "language": language,
                    "code": code
                }
            
            elif tool_name == "debug_code":
                # Map parameters for debug_code
                language = plan.get("language", "python")
                code = plan.get("code", "")
                error = plan.get("error", "")
                
                params = {
                    "language": language,
                    "code": code,
                    "error": error
                }
            
            elif tool_name == "implement_requirement":
                # Map parameters for implement_requirement
                language = plan.get("language", "python")
                requirement = plan.get("requirement", plan.get("content", ""))
                
                params = {
                    "language": language,
                    "requirement": requirement
                }
            
            elif tool_name == "analyze_code":
                # Map parameters for analyze_code
                language = plan.get("language", "python")
                code = plan.get("code", "")
                focus = plan.get("focus", ["readability", "performance"])
                
                params = {
                    "language": language,
                    "code": code,
                    "focus": focus
                }
            
            elif tool_name == "optimize_code":
                # Map parameters for optimize_code
                language = plan.get("language", "python")
                code = plan.get("code", "")
                focus = plan.get("focus", "performance")
                
                params = {
                    "language": language,
                    "code": code,
                    "focus": focus
                }
            
            elif tool_name == "generate_tests":
                # Map parameters for generate_tests
                language = plan.get("language", "python")
                code = plan.get("code", "")
                test_framework = plan.get("test_framework", "pytest")
                
                params = {
                    "language": language,
                    "code": code,
                    "framework": test_framework
                }
            
            else:
                # For other tools, add all parameters except excluded ones
                for key, value in plan.items():
                    if key not in ["content", "rag_context", "workflow_context", "intent", "code_elements"]:
                        params[key] = value
            
            # Execute the tool
            try:
                tool_result = await agent.use_tool(tool_name, params)
                
                # Add flag to update conversation history
                tool_result["update_history"] = True
                
                return tool_result
            except Exception as e:
                logger.error(f"Error executing tool for intent {intent}: {e}")
                # Fall back to LLM-based execution
    
    # For all other intents or if tool execution failed, use LLM
    context = {
        "content": plan.get("content", "")
    }
    
    # Add code if available
    if "code" in plan:
        context["code"] = plan["code"]
    
    # Add language if available
    if "language" in plan:
        context["language"] = plan["language"]
    
    # Add any RAG context
    if "rag_context" in plan and plan["rag_context"]:
        context["documents"] = plan["rag_context"]
    
    # Add any other relevant parameters from the plan
    for key, value in plan.items():
        if key not in ["content", "rag_context", "workflow_context", "intent", "code_elements"]:
            context[key] = value
    
    # Execute using LLM
    try:
        # Generate response using the LLM
        llm_response = await llm.execute_dev_task(
            task_type=intent,
            context=context,
            conversation_history=conversation_history
        )
        
        # Process the response based on intent
        if intent == "generate_code":
            # Extract code from LLM response
            code_match = re.search(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
            if code_match:
                code = code_match.group(1)
                
                # Create an entity in the context
                try:
                    language = plan.get("language", "python")
                    description = plan.get("description", plan.get("content", ""))
                    
                    entity = context_manager.create_entity(
                        entity_type="code_entity",
                        data={
                            "name": f"Generated_{language}_code",
                            "type": "function",
                            "language": language,
                            "content": code,
                            "description": description,
                            "created_by": agent.agent_id,
                            "created_at": time.time()
                        }
                    )
                    
                    # Add result metadata
                    result = {
                        "success": True,
                        "result": {
                            "entity_type": "code_entity",
                            "id": entity.id,
                            "language": language,
                            "code": code
                        },
                        "response": llm_response["text"],
                        "update_history": True
                    }
                    
                    return result
                except Exception as e:
                    logger.error(f"Error creating code entity: {e}")
        
        # For all other intents, simply return the LLM response
        return {
            "success": True,
            "result": llm_response["text"],
            "response": llm_response["text"],
            "update_history": True
        }
            
    except Exception as e:
        logger.error(f"Error executing plan with LLM for intent {intent}: {e}")
        return {
            "success": False,
            "error": f"Failed to execute {intent}: {str(e)}"
        }
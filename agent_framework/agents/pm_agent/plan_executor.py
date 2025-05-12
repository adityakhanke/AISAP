"""
Plan execution module for the PM Agent.
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
    
    Args:
        plan: The execution plan
        agent: The agent instance
        llm: LLM configuration to use
        tool_registry: Tool registry for accessing tools
        context_manager: Context manager for accessing shared context
        conversation_history: Conversation history for context
        
    Returns:
        Execution result
    """
    intent = plan.get("intent", "general_inquiry")
    
    # First check if we should use a tool for this intent
    tool_based_intents = [
        "create_requirement", "update_requirement", "list_requirements",
        "create_user_story", "update_user_story", "list_user_stories",
        "prioritize_requirements", "create_roadmap", "plan_sprint",
        "generate_prd", "generate_prd_section", "analyze_requirements"
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
            
            # Special handling for different tools based on their expected parameters
            if tool_name == "create_requirement":
                # Map 'content' to 'description' for create_requirement
                title = plan.get("title", "New Requirement")
                description = plan.get("content", "")
                priority = plan.get("priority", "medium")
                status = plan.get("status", "draft")
                tags = plan.get("tags", [])
                
                params = {
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "status": status,
                    "tags": tags
                }
            
            elif tool_name == "list_requirements":
                # Handle list_requirements parameters
                filter_text = plan.get("filter", "")
                status = plan.get("status")
                priority = plan.get("priority")
                tags = plan.get("tags")
                
                params = {
                    "filter": filter_text
                }
                
                if status:
                    params["status"] = status
                if priority:
                    params["priority"] = priority
                if tags:
                    params["tags"] = tags
            
            elif tool_name == "prioritize_requirements":
                # Handle prioritize_requirements parameters
                filter_text = plan.get("filter", "")
                method = plan.get("method", "value_effort")
                
                params = {
                    "filter": filter_text,
                    "method": method
                }
            
            elif tool_name == "analyze_requirements":
                # Handle analyze_requirements parameters
                filter_text = plan.get("filter", "")
                
                params = {
                    "filter": filter_text
                }
            
            elif tool_name == "create_roadmap":
                # Handle create_roadmap parameters
                timeframe = plan.get("timeframe", "6 months")
                context_content = plan.get("content", "")
                
                params = {
                    "timeframe": timeframe,
                    "content": context_content
                }
            
            elif tool_name == "create_user_story":
                # Handle create_user_story parameters
                title = plan.get("title", "New User Story")
                description = plan.get("content", "")
                priority = plan.get("priority", "medium")
                status = plan.get("status", "draft")
                
                params = {
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "status": status
                }
            
            elif tool_name == "generate_prd":
                # Handle generate_prd parameters
                title = plan.get("title", "Product Requirements Document")
                context_content = plan.get("content", "")
                
                params = {
                    "title": title,
                    "content": context_content
                }
            
            elif tool_name == "generate_prd_section":
                # Handle generate_prd_section parameters
                section_name = plan.get("section_name", "Requirements")
                context_content = plan.get("content", "")
                
                params = {
                    "section_name": section_name,
                    "content": context_content
                }
            
            elif tool_name == "plan_sprint":
                # Handle plan_sprint parameters
                sprint_name = plan.get("sprint_name", "Next Sprint")
                capacity = plan.get("capacity", "")
                context_content = plan.get("content", "")
                
                params = {
                    "sprint_name": sprint_name,
                    "capacity": capacity,
                    "content": context_content
                }
            
            else:
                # For other tools, add all parameters except excluded ones
                for key, value in plan.items():
                    if key not in ["content", "rag_context", "workflow_context", "intent", "entities", "themes"]:
                        params[key] = value
                
                # If a key is missing in params but tool expects it, try to derive from content
                if "description" not in params and "content" in plan:
                    params["description"] = plan["content"]
                
                if "title" not in params and "content" in plan:
                    # Try to extract a title from the content
                    content = plan["content"]
                    title = content.split("\n")[0] if "\n" in content else content[:50]
                    params["title"] = title
            
            # Execute the tool
            try:
                tool_result = agent.use_tool(tool_name, params)
                
                # Add flag to update conversation history
                tool_result["update_history"] = True
                
                return tool_result
            except Exception as e:
                logger.error(f"Error executing tool for intent {intent}: {e}")
                # Fall back to LLM-based execution
    
    # For all other intents or if tool execution failed, use LLM
    context = {
        "content": plan.get("content", ""),
        "intent": intent
    }
    
    # Add any RAG context
    if "rag_context" in plan and plan["rag_context"]:
        context["documents"] = plan["rag_context"]
    
    # Add any other relevant parameters from the plan
    for key, value in plan.items():
        if key not in ["content", "rag_context", "workflow_context", "intent", "entities", "themes"]:
            context[key] = value
    
    # Execute using LLM
    try:
        # Generate response using the LLM
        llm_response = await llm.execute_pm_task(
            task_type=intent,
            context=context,
            conversation_history=conversation_history
        )
        
        # Process the response based on intent
        if intent == "create_requirement":
            # Try to create an actual entity in the context
            try:
                # Extract a title and description from the LLM response
                title_match = re.search(r"^# (.+?)$", llm_response["text"], re.MULTILINE)
                title = title_match.group(1) if title_match else "New Requirement"

                # Create the entity
                entity = context_manager.create_entity(
                    entity_type="requirement",
                    data={
                        "title": title,
                        "description": llm_response["text"],
                        "priority": plan.get("priority", "medium"),
                        "status": "draft",
                        "created_by": agent.agent_id,
                        "created_at": time.time()
                    }
                )
                
                # Add result metadata
                result = {
                    "success": True,
                    "result": {
                        "entity_type": "requirement",
                        "id": entity.id,
                        "title": title,
                        "description": llm_response["text"]
                    },
                    "response": f"Created requirement: {title}",
                    "update_history": True
                }
                
                return result
            except Exception as e:
                logger.error(f"Error creating requirement entity: {e}")
        
        elif intent == "create_user_story":
            # Try to create an actual entity in the context
            try:
                # Extract a title from the LLM response
                title_match = re.search(r"^# (.+?)$", llm_response["text"], re.MULTILINE)
                title = title_match.group(1) if title_match else "New User Story"
                
                # Create the entity
                entity = context_manager.create_entity(
                    entity_type="user_story",
                    data={
                        "title": title,
                        "description": llm_response["text"],
                        "priority": plan.get("priority", "medium"),
                        "status": "draft",
                        "created_by": agent.agent_id,
                        "created_at": time.time()
                    }
                )
                
                # Add result metadata
                result = {
                    "success": True,
                    "result": {
                        "entity_type": "user_story",
                        "id": entity.id,
                        "title": title,
                        "description": llm_response["text"]
                    },
                    "response": f"Created user story: {title}",
                    "update_history": True
                }
                
                return result
            except Exception as e:
                logger.error(f"Error creating user story entity: {e}")
                
        elif intent == "create_project":
            # Try to create a project entity
            try:
                # Get or extract project name
                project_name = plan.get("name") or "New Project"
                
                # Create the entity
                entity = context_manager.create_entity(
                    entity_type="project",
                    data={
                        "name": project_name,
                        "description": llm_response["text"],
                        "created_by": agent.agent_id,
                        "created_at": time.time()
                    }
                )
                
                # Store as current project in agent's memory
                agent.working_memory.store("current_project", entity.id)
                
                # Add result metadata
                result = {
                    "success": True,
                    "result": {
                        "entity_type": "project",
                        "id": entity.id,
                        "name": project_name,
                        "description": llm_response["text"]
                    },
                    "response": f"Created project: {project_name}",
                    "update_history": True
                }
                
                return result
            except Exception as e:
                logger.error(f"Error creating project entity: {e}")
        
        # For other intents, return the raw response
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
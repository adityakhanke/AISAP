"""
Tools module for the Dev Agent.
Provides implementations for development-specific tools.
"""

import logging
import time
import re
import json
from typing import Dict, Any, List, Optional

from ...context.entity import EntityType, RelationType

logger = logging.getLogger(__name__)

def register_dev_tools(agent):
    """
    Register development-specific tools with the agent.
    
    Args:
        agent: Dev Agent instance
    """
    if not agent.tool_registry:
        logger.warning("No tool registry available, skipping tool registration")
        return
    
    # Code generation tools
    agent.tool_registry.register_function(
        name="generate_code",
        description="Generate code based on a description",
        func=lambda **kwargs: _generate_code(agent, **kwargs)
    )

    agent.tool_registry.register_function(
        name="explain_code",
        description="Explain what a piece of code does",
        func=lambda **kwargs: _explain_code(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="debug_code",
        description="Debug and fix code with issues",
        func=lambda **kwargs: _debug_code(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="implement_requirement",
        description="Implement code based on a requirement",
        func=lambda **kwargs: _implement_requirement(agent, **kwargs)
    )
    
    # Code analysis tools
    agent.tool_registry.register_function(
        name="analyze_code",
        description="Analyze code for quality, performance, and readability",
        func=lambda **kwargs: _analyze_code(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="optimize_code",
        description="Optimize code for performance or readability",
        func=lambda **kwargs: _optimize_code(agent, **kwargs)
    )
    
    agent.tool_registry.register_function(
        name="refactor_code",
        description="Refactor code to improve its structure",
        func=lambda **kwargs: _refactor_code(agent, **kwargs)
    )
    
    # Testing tools
    agent.tool_registry.register_function(
        name="generate_tests",
        description="Generate tests for a code implementation",
        func=lambda **kwargs: _generate_tests(agent, **kwargs)
    )
    
    logger.info(f"Registered development-specific tools for agent {agent.name}")

# Tool implementation functions

def _generate_code(agent, language: str = None, description: str = None, requirements: List[str] = None) -> Dict[str, Any]:
    """
    Generate code based on a description.
    """
    if not language:
        language = "python"  # Use a default language

    if not description:
        return {"success": False, "error": "Description is required"}

    if requirements is None:
        requirements = []

    return _generate_code_async(agent, language, description, requirements)
    

async def _generate_code_async(agent, language: str, description: str, requirements: List[str] = None) -> Dict[str, Any]:
    """
    Generate code based on a description (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language to use
        description: Description of what the code should do
        requirements: Optional list of requirements/constraints
        
    Returns:
        Generated code
    """
    if requirements is None:
        requirements = []
        
    # Prepare prompt for code generation
    context = {
        "language": language,
        "description": description,
        "requirements": requirements
    }
    
    # Generate code using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="generate_code",
            context=context
        )
        
        # Extract code block from response
        code_block = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            code_block = code_matches[0]
        else:
            # If no code block found, use entire text (less ideal)
            code_block = llm_response["text"]
        
        # Store code in context
        entity = agent.context_manager.create_entity(
            entity_type="code_entity",
            data={
                "name": _generate_entity_name(description, language),
                "type": "function" if "function" in description.lower() else "module",
                "language": language,
                "content": code_block,
                "description": description,
                "requirements": requirements,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        # Get current project if available
        project_id = agent.working_memory.retrieve("current_project")
        if project_id:
            # Create relationship with project
            agent.context_manager.create_relationship(
                from_entity_id=project_id,
                relation_type=RelationType.CONTAINS,
                to_entity_id=entity.id,
                metadata={"type": "code"}
            )
        
        # Extract explanation if available
        explanation = None
        explanation_section = re.search(r'(?:Explanation|Comments):(.*?)(?:$|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if explanation_section:
            explanation = explanation_section.group(1).strip()
        
        return {
            "success": True,
            "entity_type": "code_entity",
            "id": entity.id,
            "language": language,
            "code": code_block,
            "explanation": explanation,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        return {"success": False, "error": f"Failed to generate code: {str(e)}"}

def _explain_code(agent, language: str, code: str) -> Dict[str, Any]:
    """
    Explain what a piece of code does.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to explain
        
    Returns:
        Explanation of the code
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_explain_code_async(agent, language, code))
    return _explain_code_async(agent, language, code)

async def _explain_code_async(agent, language: str, code: str) -> Dict[str, Any]:
    """
    Explain what a piece of code does (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to explain
        
    Returns:
        Explanation of the code
    """
    # Prepare prompt for code explanation
    context = {
        "language": language,
        "code": code
    }
    
    # Generate explanation using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="explain_code",
            context=context
        )
        
        # Create explanation entity
        entity = agent.context_manager.create_entity(
            entity_type="code_explanation",
            data={
                "language": language,
                "code": code,
                "explanation": llm_response["text"],
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "code_explanation",
            "id": entity.id,
            "language": language,
            "code": code,
            "explanation": llm_response["text"],
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error explaining code: {e}")
        return {"success": False, "error": f"Failed to explain code: {str(e)}"}

def _debug_code(agent, language: str, code: str, error: str = None) -> Dict[str, Any]:
    """
    Debug and fix code with issues.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to debug
        error: Optional error message or description
        
    Returns:
        Debugged and fixed code
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_debug_code_async(agent, language, code, error))
    return _debug_code_async(agent, language, code, error)

async def _debug_code_async(agent, language: str, code: str, error: str = None) -> Dict[str, Any]:
    """
    Debug and fix code with issues (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to debug
        error: Optional error message or description
        
    Returns:
        Debugged and fixed code
    """
    # Prepare prompt for debugging
    context = {
        "language": language,
        "code": code,
        "error": error or "This code has issues that need to be fixed."
    }
    
    # Generate debug results using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="debug_code",
            context=context
        )
        
        # Extract fixed code block from response
        fixed_code = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            # Use the last code block as the fixed code
            fixed_code = code_matches[-1]
        
        # Extract issue explanation
        issue = None
        issue_section = re.search(r'(?:Issue|Problem|Bug):(.*?)(?:Fix|Solution|Explanation|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if issue_section:
            issue = issue_section.group(1).strip()
        
        # Extract explanation if available
        explanation = None
        explanation_section = re.search(r'(?:Explanation|Reason):(.*?)(?:$|Fix|Solution|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if explanation_section:
            explanation = explanation_section.group(1).strip()
        
        # Create debug result entity
        entity = agent.context_manager.create_entity(
            entity_type="debug_result",
            data={
                "language": language,
                "original_code": code,
                "fixed_code": fixed_code,
                "error": error,
                "issue": issue,
                "explanation": explanation,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "debug_result",
            "id": entity.id,
            "language": language,
            "original_code": code,
            "fixed_code": fixed_code,
            "issue": issue,
            "explanation": explanation,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error debugging code: {e}")
        return {"success": False, "error": f"Failed to debug code: {str(e)}"}

def _implement_requirement(agent, language: str, requirement: str) -> Dict[str, Any]:
    """
    Implement code based on a requirement.
    
    Args:
        agent: Dev Agent instance
        language: Programming language to use
        requirement: Requirement to implement
        
    Returns:
        Implementation of the requirement
    """
    if not language:
        return {"success": False, "error": "Language is required"}
    
    if not requirement:
        return {"success": False, "error": "Requirement is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_implement_requirement_async(agent, language, requirement))
    return _implement_requirement_async(agent, language, requirement)

async def _implement_requirement_async(agent, language: str, requirement: str) -> Dict[str, Any]:
    """
    Implement code based on a requirement (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language to use
        requirement: Requirement to implement
        
    Returns:
        Implementation of the requirement
    """
    # Check for existing requirement entity in context
    requirement_entity = None
    try:
        # Try to find by ID first
        if len(requirement) >= 32 and re.match(r'^[0-9a-f-]+$', requirement):
            # This might be an ID
            requirement_entity = agent.context_manager.get_entity(requirement)
        
        # If not found by ID, try to find by title
        if not requirement_entity:
            requirements = agent.context_manager.query_entities(
                entity_type="requirement",
                filters={"title": requirement}
            )
            if requirements:
                requirement_entity = requirements[0]
    except Exception:
        # Ignore errors in finding requirement entity
        pass
    
    # Prepare context for implementation
    context = {
        "language": language,
        "requirement": requirement_entity.data.get("description", requirement) if requirement_entity else requirement
    }
    
    # Generate implementation using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="implement_requirement",
            context=context
        )
        
        # Extract code block from response
        code_block = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            code_block = code_matches[0]
        else:
            # If no code block found, use entire text (less ideal)
            code_block = llm_response["text"]
        
        # Generate implementation name
        name = _generate_entity_name(
            requirement_entity.data.get("title", requirement) if requirement_entity else requirement,
            language
        )
        
        # Store implementation in context
        entity = agent.context_manager.create_entity(
            entity_type="code_entity",
            data={
                "name": name,
                "type": "implementation",
                "language": language,
                "content": code_block,
                "description": requirement_entity.data.get("description", requirement) if requirement_entity else requirement,
                "requirement_id": requirement_entity.id if requirement_entity else None,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        # Create relationship with requirement if available
        if requirement_entity:
            agent.context_manager.create_relationship(
                from_entity_id=requirement_entity.id,
                relation_type=RelationType.IMPLEMENTS,
                to_entity_id=entity.id,
                metadata={"created_by": agent.agent_id}
            )
        
        # Get current project if available
        project_id = agent.working_memory.retrieve("current_project")
        if project_id:
            # Create relationship with project
            agent.context_manager.create_relationship(
                from_entity_id=project_id,
                relation_type=RelationType.CONTAINS,
                to_entity_id=entity.id,
                metadata={"type": "implementation"}
            )
        
        return {
            "success": True,
            "entity_type": "code_entity",
            "id": entity.id,
            "name": name,
            "language": language,
            "code": code_block,
            "requirement_id": requirement_entity.id if requirement_entity else None,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error implementing requirement: {e}")
        return {"success": False, "error": f"Failed to implement requirement: {str(e)}"}

def _analyze_code(agent, language: str, code: str, focus: List[str] = None) -> Dict[str, Any]:
    """
    Analyze code for quality, performance, and readability.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to analyze
        focus: Optional focus areas (performance, security, readability, etc.)
        
    Returns:
        Analysis of the code
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_analyze_code_async(agent, language, code, focus))
    return _analyze_code_async(agent, language, code, focus)

async def _analyze_code_async(agent, language: str, code: str, focus: List[str] = None) -> Dict[str, Any]:
    """
    Analyze code for quality, performance, and readability (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to analyze
        focus: Optional focus areas (performance, security, readability, etc.)
        
    Returns:
        Analysis of the code
    """
    if focus is None:
        focus = ["readability", "performance"]
    
    # Prepare prompt for code analysis
    context = {
        "language": language,
        "code": code,
        "focus": focus
    }
    
    # Generate analysis using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="analyze_code",
            context=context
        )
        
        # Extract recommendations if available
        recommendations = []
        # Look for a recommendations section in the response
        rec_section = re.search(r'(?:Recommendations|Suggestions):(.*?)(?:$|##)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if rec_section:
            rec_text = rec_section.group(1).strip()
            # Extract bullet points or numbered items
            rec_items = re.findall(r'(?:^|\n)\s*(?:[\*\-\d]+\.?)\s*(.*?)(?:\n|$)', rec_text)
            if rec_items:
                recommendations = [item.strip() for item in rec_items if item.strip()]
            else:
                # Split by newlines if no bullet points found
                recommendations = [line.strip() for line in rec_text.split('\n') if line.strip()]
        
        # Create analysis entity
        entity = agent.context_manager.create_entity(
            entity_type="code_analysis",
            data={
                "language": language,
                "code": code,
                "focus": focus,
                "analysis": llm_response["text"],
                "recommendations": recommendations,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "code_analysis",
            "id": entity.id,
            "language": language,
            "code": code,
            "focus": focus,
            "analysis": llm_response["text"],
            "recommendations": recommendations,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return {"success": False, "error": f"Failed to analyze code: {str(e)}"}

def _optimize_code(agent, language: str, code: str, focus: str = "performance") -> Dict[str, Any]:
    """
    Optimize code for performance or readability.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to optimize
        focus: Focus area (performance, readability)
        
    Returns:
        Optimized code
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_optimize_code_async(agent, language, code, focus))
    return _optimize_code_async(agent, language, code, focus)

async def _optimize_code_async(agent, language: str, code: str, focus: str = "performance") -> Dict[str, Any]:
    """
    Optimize code for performance or readability (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to optimize
        focus: Focus area (performance, readability)
        
    Returns:
        Optimized code
    """
    # Prepare prompt for code optimization
    context = {
        "language": language,
        "code": code,
        "focus": focus
    }
    
    # Generate optimization using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="optimize_code",
            context=context
        )
        
        # Extract optimized code block from response
        optimized_code = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            # Use the last code block as the optimized code
            optimized_code = code_matches[-1]
        
        # Extract optimization explanation
        optimization = None
        opt_section = re.search(r'(?:Optimizations|Improvements|Changes):(.*?)(?:$|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if opt_section:
            optimization = opt_section.group(1).strip()
        
        # Create optimization entity
        entity = agent.context_manager.create_entity(
            entity_type="code_optimization",
            data={
                "language": language,
                "original_code": code,
                "optimized_code": optimized_code,
                "focus": focus,
                "optimization": optimization,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "code_optimization",
            "id": entity.id,
            "language": language,
            "original_code": code,
            "optimized_code": optimized_code,
            "focus": focus,
            "optimization": optimization,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error optimizing code: {e}")
        return {"success": False, "error": f"Failed to optimize code: {str(e)}"}

def _refactor_code(agent, language: str, code: str, goal: str = "readability") -> Dict[str, Any]:
    """
    Refactor code to improve its structure.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to refactor
        goal: Refactoring goal (readability, modularity, maintainability, etc.)
        
    Returns:
        Refactored code
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_refactor_code_async(agent, language, code, goal))
    return _refactor_code_async(agent, language, code, goal)

async def _refactor_code_async(agent, language: str, code: str, goal: str = "readability") -> Dict[str, Any]:
    """
    Refactor code to improve its structure (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to refactor
        goal: Refactoring goal (readability, modularity, maintainability, etc.)
        
    Returns:
        Refactored code
    """
    # Prepare prompt for code refactoring
    context = {
        "language": language,
        "code": code,
        "goal": goal
    }
    
    # Generate refactoring using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="optimize_code",  # Use optimize_code task type with refactoring focus
            context=context
        )
        
        # Extract refactored code block from response
        refactored_code = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            # Use the last code block as the refactored code
            refactored_code = code_matches[-1]
        
        # Extract refactoring explanation
        explanation = None
        ref_section = re.search(r'(?:Refactoring|Changes|Improvements):(.*?)(?:$|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if ref_section:
            explanation = ref_section.group(1).strip()
        
        # Create refactoring entity
        entity = agent.context_manager.create_entity(
            entity_type="code_refactoring",
            data={
                "language": language,
                "original_code": code,
                "refactored_code": refactored_code,
                "goal": goal,
                "explanation": explanation,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "code_refactoring",
            "id": entity.id,
            "language": language,
            "original_code": code,
            "refactored_code": refactored_code,
            "goal": goal,
            "explanation": explanation,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error refactoring code: {e}")
        return {"success": False, "error": f"Failed to refactor code: {str(e)}"}

def _generate_tests(agent, language: str, code: str, framework: str = None) -> Dict[str, Any]:
    """
    Generate tests for a code implementation.
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to test
        framework: Testing framework to use
        
    Returns:
        Generated tests
    """
    if not code:
        return {"success": False, "error": "Code is required"}
    
    # Execute asynchronously
    # import asyncio
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # return loop.run_until_complete(_generate_tests_async(agent, language, code, framework))
    return _generate_tests_async(agent, language, code, framework)

async def _generate_tests_async(agent, language: str, code: str, framework: str = None) -> Dict[str, Any]:
    """
    Generate tests for a code implementation (async implementation).
    
    Args:
        agent: Dev Agent instance
        language: Programming language of the code
        code: Code to test
        framework: Testing framework to use
        
    Returns:
        Generated tests
    """
    # Determine appropriate test framework if not specified
    if not framework:
        framework_mapping = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "java": "junit",
            "csharp": "xunit",
            "php": "phpunit",
            "ruby": "rspec",
            "go": "go test"
        }
        framework = framework_mapping.get(language.lower(), "pytest")
    
    # Prepare prompt for test generation
    context = {
        "language": language,
        "code": code,
        "framework": framework
    }
    
    # Generate tests using LLM
    try:
        llm_response = await agent.llm.execute_dev_task(
            task_type="generate_tests",
            context=context
        )
        
        # Extract test code block from response
        test_code = None
        code_matches = re.findall(r'```(?:\w+)?\n(.*?)```', llm_response["text"], re.DOTALL)
        
        if code_matches:
            # Use the last code block as the test code
            test_code = code_matches[-1]
        
        # Extract test explanation if available
        explanation = None
        exp_section = re.search(r'(?:Explanation|Description|About the tests):(.*?)(?:$|```)', llm_response["text"], re.DOTALL | re.IGNORECASE)
        if exp_section:
            explanation = exp_section.group(1).strip()
        
        # Create test entity
        entity = agent.context_manager.create_entity(
            entity_type="test",
            data={
                "language": language,
                "code": code,
                "test_code": test_code,
                "framework": framework,
                "explanation": explanation,
                "created_by": agent.agent_id,
                "created_at": time.time()
            }
        )
        
        return {
            "success": True,
            "entity_type": "test",
            "id": entity.id,
            "language": language,
            "code": code,
            "test_code": test_code,
            "framework": framework,
            "explanation": explanation,
            "response": llm_response["text"]
        }
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        return {"success": False, "error": f"Failed to generate tests: {str(e)}"}

def _generate_entity_name(description: str, language: str) -> str:
    """
    Generate a suitable name for a code entity based on its description.
    
    Args:
        description: Description of the code
        language: Programming language
        
    Returns:
        Suitable name for the entity
    """
    # Extract key terms from the description
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9]+', description)
    if not words:
        # Fallback if no words found
        return f"Code_{language}_{int(time.time())}"
    
    # Use first 3-5 significant words
    significant_words = [word for word in words if len(word) > 2 and word.lower() not in [
        'the', 'and', 'for', 'that', 'this', 'with', 'function', 'class', 'code'
    ]]
    
    if not significant_words:
        significant_words = words
    
    name_words = significant_words[:min(5, len(significant_words))]

    # Format based on language conventions
    if language.lower() in ['python', 'ruby']:
        # snake_case
        return '_'.join([word.lower() for word in name_words])
    elif language.lower() in ['javascript', 'typescript', 'java', 'csharp', 'php']:
        # camelCase
        return name_words[0].lower() + ''.join([word.capitalize() for word in name_words[1:]])
    else:
        # Generic
        return '_'.join([word.lower() for word in name_words])
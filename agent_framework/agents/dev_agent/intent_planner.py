"""
Intent determination and planning module for the Dev Agent.
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
    is_code_request = llm_analysis.get("is_code_related", False)
    primary_action = llm_analysis.get("primary_action", "")
    language = llm_analysis.get("programming_language", "")
    primary_intent = llm_analysis.get("primary_intent", "general_inquiry").lower()
    
    # Standardize intent names
    intent_mapping = {
        "generate code": "generate_code",
        "write code": "generate_code",
        "implement code": "generate_code",
        "create code": "generate_code",
        "code generation": "generate_code",
        "explain code": "explain_code",
        "analyze code": "analyze_code",
        "review code": "analyze_code",
        "debug code": "debug_code",
        "fix code": "debug_code",
        "troubleshoot code": "debug_code",
        "solve error": "debug_code",
        "implement requirement": "implement_requirement",
        "implement feature": "implement_requirement",
        "refactor code": "refactor_code",
        "optimize code": "optimize_code",
        "improve code": "optimize_code",
        "test code": "generate_tests",
        "create tests": "generate_tests",
        "general inquiry": "general_inquiry"
    }
    
    # Try to match to a known intent
    intent = "general_inquiry"
    
    if is_code_request and primary_action:
        combined_key = f"{primary_action} code"
        if combined_key in intent_mapping:
            intent = intent_mapping[combined_key]
        elif primary_action in intent_mapping:
            intent = intent_mapping[primary_action]
    
    if primary_intent in intent_mapping:
        intent = intent_mapping[primary_intent]
    
    # For other potential matches
    for key, value in intent_mapping.items():
        if key in content.lower():
            intent = value
            break
    
    # Initialize the plan with analysis data
    plan = {
        "content": content,
        "rag_context": analysis_result.get("rag_context"),
        "workflow_context": analysis_result.get("workflow_context"),
        "code_elements": llm_analysis.get("code_elements", {}),
        "language": language,
        "intent": intent
    }
    
    # Extract code blocks from content
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
    if code_blocks:
        plan["code"] = code_blocks[0]
        
        # Try to detect language from code block
        if not language:
            language_hint = re.search(r'```(\w+)', content)
            if language_hint:
                plan["language"] = language_hint.group(1).lower()
    
    # Enhance the plan with LLM-generated task-specific parameters
    plan_enhancement_prompt = f"""
    Based on the following user request to a Developer agent:
    
    USER REQUEST: {content}
    
    I need to create an execution plan for intent: {intent}
    
    What specific parameters or information should I include in my execution plan?
    Return a JSON object with recommended parameters for this development task.
    """
    
    enhancement_response = await llm.generate(
        prompt=plan_enhancement_prompt,
        system_prompt="You are a helpful assistant that recommends parameters for software development tasks. Respond in JSON format.",
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
    code_elements = analysis_result.get("code_elements", {})
    programming_language = analysis_result.get("programming_language")
    requirements = analysis_result.get("requirements", [])
    
    # Initialize the plan with analysis data
    plan = {
        "content": content,
        "rag_context": analysis_result.get("rag_context"),
        "workflow_context": analysis_result.get("workflow_context"),
        "code_elements": code_elements,
        "language": programming_language,
        "requirements": requirements
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
    code_block = command_info.get("code_block")
    
    # Add code block to plan if available
    if code_block:
        plan["code"] = code_block
    
    # Map commands to intents
    if command_type in ["generate", "create", "write"]:
        # Handle code generation commands
        plan.update({
            "intent": "generate_code",
            "language": command_args.get("language", plan.get("language", "python")),
            "description": command_info.get("content", "")
        })
        return "generate_code", plan
    
    elif command_type in ["debug", "fix", "solve", "troubleshoot"]:
        # Handle debugging commands
        error_description = None
        if "error" in command_info:
            error_description = command_info["error"]
        
        plan.update({
            "intent": "debug_code",
            "language": command_args.get("language", plan.get("language", "python")),
            "error": error_description
        })
        return "debug_code", plan
    
    elif command_type in ["explain", "describe", "analyze", "review"]:
        # Handle code explanation/analysis commands
        plan.update({
            "intent": "explain_code" if command_type == "explain" else "analyze_code",
            "language": command_args.get("language", plan.get("language", "python"))
        })
        return "explain_code" if command_type == "explain" else "analyze_code", plan
    
    elif command_type in ["refactor", "optimize", "improve"]:
        # Handle code optimization commands
        plan.update({
            "intent": "optimize_code",
            "language": command_args.get("language", plan.get("language", "python")),
            "focus": "performance" if "performance" in plan.get("requirements", []) else "readability"
        })
        return "optimize_code", plan
    
    elif command_type in ["test", "create test"]:
        # Handle test generation commands
        plan.update({
            "intent": "generate_tests",
            "language": command_args.get("language", plan.get("language", "python")),
            "test_framework": _detect_test_framework(plan)
        })
        return "generate_tests", plan
    
    elif command_type in ["implement"]:
        # Handle requirement implementation
        plan.update({
            "intent": "implement_requirement",
            "language": command_args.get("language", plan.get("language", "python")),
            "requirement": command_info.get("content", "")
        })
        return "implement_requirement", plan
    
    # If no specific command pattern was matched
    plan.update({
        "intent": "general_inquiry",
        "command_type": command_type,
        "command_target": command_target
    })
    return "general_inquiry", plan

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
    
    # Extract code blocks
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
    if code_blocks:
        plan["code"] = code_blocks[0]
        
        # Try to detect language from code block if not already known
        if not plan.get("language"):
            language_hint = re.search(r'```(\w+)', content)
            if language_hint:
                plan["language"] = language_hint.group(1).lower()
    
    # Check for code generation
    if re.search(r'(?:generate|create|write|implement)\s+(?:a\s+)?(?:code|function|class|method)', lower_content):
        plan.update({
            "intent": "generate_code",
            "language": plan.get("language", "python"),
            "description": content
        })
        return "generate_code", plan
    
    # Check for code explanation
    if re.search(r'(?:explain|describe|what\s+does\s+this\s+(?:code|function|class|method))', lower_content):
        plan.update({
            "intent": "explain_code",
            "language": plan.get("language", "python")
        })
        return "explain_code", plan
    
    # Check for code debugging
    if re.search(r'(?:debug|fix|solve|help\s+with|troubleshoot)\s+(?:this\s+)?(?:code|function|error|bug|issue)', lower_content):
        # Try to extract error message
        error_desc = None
        error_match = re.search(r'error[:\s]+(.*?)(?:\.|\n|$)', content, re.IGNORECASE)
        if error_match:
            error_desc = error_match.group(1).strip()
        
        plan.update({
            "intent": "debug_code",
            "language": plan.get("language", "python"),
            "error": error_desc
        })
        return "debug_code", plan
    
    # Check for code optimization/refactoring
    if re.search(r'(?:optimize|improve|refactor|clean\s+up)\s+(?:this\s+)?(?:code|function|implementation)', lower_content):
        plan.update({
            "intent": "optimize_code",
            "language": plan.get("language", "python"),
            "focus": "performance" if "performance" in plan.get("requirements", []) else "readability"
        })
        return "optimize_code", plan
    
    # Check for test generation
    if re.search(r'(?:create|generate|write)\s+(?:a\s+)?(?:test|tests|unit\s+test|integration\s+test)', lower_content):
        plan.update({
            "intent": "generate_tests",
            "language": plan.get("language", "python"),
            "test_framework": _detect_test_framework(plan)
        })
        return "generate_tests", plan
    
    # Check for code analysis
    if re.search(r'(?:analyze|review|evaluate|assess)\s+(?:this\s+)?(?:code|function|class|implementation)', lower_content):
        plan.update({
            "intent": "analyze_code",
            "language": plan.get("language", "python"),
            "focus": _detect_analysis_focus(content, plan)
        })
        return "analyze_code", plan
    
    # Check for requirement implementation
    if re.search(r'(?:implement|build|develop)\s+(?:a\s+)?(?:feature|requirement|functionality)', lower_content):
        plan.update({
            "intent": "implement_requirement",
            "language": plan.get("language", "python"),
            "requirement": content
        })
        return "implement_requirement", plan
    
    # Default to general inquiry
    plan.update({
        "intent": "general_inquiry",
        "content": content
    })
    return "general_inquiry", plan

def _detect_test_framework(plan: Dict[str, Any]) -> str:
    """
    Detect the appropriate test framework based on language and context.
    
    Args:
        plan: Execution plan
        
    Returns:
        Detected test framework
    """
    language = plan.get("language", "python").lower()
    content = plan.get("content", "")
    
    # Framework mapping by language
    framework_mapping = {
        "python": "pytest",
        "javascript": "jest",
        "typescript": "jest",
        "java": "junit",
        "csharp": "xunit",
        "php": "phpunit",
        "ruby": "rspec",
        "go": "go test",
        "rust": "cargo test"
    }
    
    # Check for explicit mention of frameworks
    frameworks = {
        "pytest": r'\bpytest\b',
        "unittest": r'\bunittest\b',
        "jest": r'\bjest\b',
        "mocha": r'\bmocha\b',
        "jasmine": r'\bjasmine\b',
        "junit": r'\bjunit\b',
        "xunit": r'\bxunit\b',
        "nunit": r'\bnunit\b',
        "phpunit": r'\bphpunit\b',
        "rspec": r'\brspec\b'
    }
    
    for framework, pattern in frameworks.items():
        if re.search(pattern, content, re.IGNORECASE):
            return framework
    
    # Default to language-appropriate framework
    return framework_mapping.get(language, "pytest")

def _detect_analysis_focus(content: str, plan: Dict[str, Any]) -> List[str]:
    """
    Detect the focus areas for code analysis.
    
    Args:
        content: User input text
        plan: Execution plan
        
    Returns:
        List of focus areas
    """
    focus_areas = []
    
    if re.search(r'\b(?:performance|efficient|optimize|speed|fast)\b', content, re.IGNORECASE):
        focus_areas.append("performance")
    
    if re.search(r'\b(?:security|secure|vulnerability|hack|attack|exploit)\b', content, re.IGNORECASE):
        focus_areas.append("security")
    
    if re.search(r'\b(?:readability|clean|maintainable|understandable|refactor)\b', content, re.IGNORECASE):
        focus_areas.append("readability")
    
    if re.search(r'\b(?:error|bug|issue|problem|failure)\b', content, re.IGNORECASE):
        focus_areas.append("bugs")
    
    if re.search(r'\b(?:pattern|architecture|design|structure)\b', content, re.IGNORECASE):
        focus_areas.append("architecture")
    
    # Default to readability if no focus areas detected
    if not focus_areas:
        focus_areas = ["readability", "performance"]
    
    return focus_areas
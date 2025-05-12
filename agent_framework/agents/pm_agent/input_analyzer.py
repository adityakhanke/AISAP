"""
Input analysis module for the PM Agent.
Provides both LLM-enhanced and traditional analysis methods.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

async def analyze_input_with_llm(content: str, llm: Any, rag_context: Optional[str] = None,
                            workflow_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze input using LLM for enhanced understanding.
    
    Args:
        content: User input text
        llm: LLM configuration to use
        rag_context: Optional RAG context
        workflow_context: Optional workflow context
        
    Returns:
        Analysis result
    """
    # Prepare the analysis prompt
    analysis_prompt = f"""
    Analyze the following input from a user to a Product Manager agent:
    
    USER INPUT: {content}
    
    Provide an analysis with the following information:
    1. Is this a command-style request? If so, what is the command and target?
    2. What entities (requirements, user stories, etc.) are mentioned?
    3. What is the primary intent of the user?
    4. What key themes or topics are present?
    
    Format your response as JSON.
    """
    
    # Add RAG context if available
    if rag_context:
        analysis_prompt += f"\n\nRELEVANT CONTEXT:\n{rag_context[:2000]}"  # Limit context size
    
    # Generate analysis using LLM
    llm_response = await llm.generate(
        prompt=analysis_prompt,
        system_prompt="You are an analytical assistant that helps identify the intent and entities in user requests. Respond in JSON format only.",
        params={"temperature": 0.3, "max_tokens": 32768}
    )
    
    # Parse the LLM's analysis
    analysis_text = llm_response["text"]
    
    # Extract JSON if it's wrapped in markdown code block
    if "```json" in analysis_text:
        import re
        json_match = re.search(r"```json\n(.*?)\n```", analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
    
    # Parse JSON
    llm_analysis = json.loads(analysis_text)
    
    # Combine with input data
    analysis_result = {
        "content": content,
        "rag_context": rag_context,
        "workflow_context": workflow_context,
        "llm_analysis": llm_analysis
    }
    
    return analysis_result

def analyze_input_traditional(content: str, rag_context: Optional[str] = None,
                          workflow_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze input using traditional pattern matching techniques.
    
    Args:
        content: User input text
        rag_context: Optional RAG context
        workflow_context: Optional workflow context
        
    Returns:
        Analysis result
    """
    # Extract command information
    command_info = parse_command(content)
    
    # Extract entities
    entities = extract_entities(content)
    
    # Identify themes
    themes = identify_themes(content)
    
    # Create analysis result
    return {
        "content": content,
        "rag_context": rag_context,
        "workflow_context": workflow_context,
        "command_info": command_info,
        "entities": entities,
        "themes": themes
    }

def parse_command(content: str) -> Dict[str, Any]:
    """
    Parse a command-style request from the input.
    
    Args:
        content: Input text
        
    Returns:
        Command information
    """
    # Default command info
    command_info = {
        "is_command": False,
        "command_type": None,
        "command_target": None,
        "command_args": {},
        "entity_ids": []
    }

    # Normalize and clean the content
    cleaned_content = content.strip().lower()

    # Try to extract commands from conversational phrases
    conversational_prefixes = [
        r'i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+',
        r'(?:please|can\s+you|could\s+you)\s+',
        r'(?:i\'d|i\s+would)\s+like\s+(?:you\s+)?to\s+',
        r'(?:let\'s|we\s+should|we\s+need\s+to)\s+'
    ]

    # Try to remove conversational prefixes to get to the command
    for prefix in conversational_prefixes:
        if re.match(prefix, cleaned_content, re.IGNORECASE):
            cleaned_content = re.sub(prefix, '', cleaned_content, flags=re.IGNORECASE)
            break

    # Command verbs we recognize
    command_verbs = [
        'create', 'update', 'delete', 'list', 'get', 'generate',
        'prioritize', 'analyze', 'plan', 'make', 'add', 'show', 'find'
    ]

    # Command targets and their standardized form
    target_mapping = {
        'requirement': 'requirement',
        'requirements': 'requirement',
        'req': 'requirement',
        'story': 'user_story',
        'stories': 'user_story',
        'user story': 'user_story',
        'user stories': 'user_story',
        'project': 'project',
        'roadmap': 'roadmap',
        'document': 'prd',
        'doc': 'prd',
        'documentation': 'prd',
        'prd': 'prd',
        'specification': 'prd',
        'spec': 'prd'
    }

    # Check for direct command format first
    command_pattern = r'^({})\s+(?:a|an|the)?\s*({}|for\s+(?:a|an)?\s*(?:new)?\s*(?:{}))(?:\s+(.+))?'.format(
        '|'.join(command_verbs),
        '|'.join(target_mapping.keys()),
        '|'.join(target_mapping.keys())
    )

    match = re.match(command_pattern, cleaned_content, re.IGNORECASE)

    if match:
        command_info["is_command"] = True
        command_info["command_type"] = match.group(1).lower()

        # Extract target from group 2, handling "for a new X" format
        target_text = match.group(2).lower()
        for target, standard_target in target_mapping.items():
            if target in target_text:
                command_info["command_target"] = standard_target
                break

        # If no target was matched, use a default based on command_type
        if not command_info["command_target"]:
            # Try to infer target from the command type and context
            if command_info["command_type"] in ["create", "add", "make"]:
                command_info["command_target"] = "requirement"  # Default to requirement
            elif command_info["command_type"] in ["list", "show", "find"]:
                command_info["command_target"] = "requirement"  # Default to requirement

        # Parse the rest of the content as arguments
        args_text = match.group(3) if match.group(3) else ''

    else:
        # If no direct command match, try to identify intents from the content
        for verb in command_verbs:
            if verb in cleaned_content:
                for target, standard_target in target_mapping.items():
                    if target in cleaned_content:
                        command_info["is_command"] = True
                        command_info["command_type"] = verb
                        command_info["command_target"] = standard_target
                        args_text = cleaned_content
                        break
                if command_info["is_command"]:
                    break

    # If we found a command, extract parameters
    if command_info["is_command"] and command_info["command_type"] and command_info["command_target"]:
        # Extract any IDs mentioned
        id_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
        id_matches = re.findall(id_pattern, content)
        if id_matches:
            command_info["entity_ids"] = id_matches

        # Extract key-value pairs
        # Format: key=value or key="value with spaces"
        kv_pattern = r'(\w+)=(?:"([^"]+)"|(\S+))'
        kv_matches = re.findall(kv_pattern, content)

        for key, quoted_value, simple_value in kv_matches:
            value = quoted_value if quoted_value else simple_value
            command_info["command_args"][key] = value

        # If no key-value pairs, use the whole content
        if not command_info["command_args"] and not command_info["entity_ids"]:
            # Use the cleaned content or args_text as appropriate
            description = args_text if locals().get('args_text') else content
            command_info["command_args"]["content"] = description

    return command_info

def extract_entities(content: str) -> Dict[str, List[str]]:
    """
    Extract entities mentioned in the content.
    
    Args:
        content: Input text
        
    Returns:
        Dictionary of entity types to lists of entity identifiers
    """
    entities = {
        "requirements": [],
        "user_stories": [],
        "features": [],
        "projects": []
    }
    
    # Extract requirement mentions
    req_pattern = r'requirement(?:s)?\s+(?:for|about|on)?\s+["\']?([^"\'\.,;]+)["\']?'
    req_matches = re.findall(req_pattern, content, re.IGNORECASE)
    if req_matches:
        entities["requirements"] = [match.strip() for match in req_matches]
    
    # Extract user story mentions
    story_pattern = r'(?:user\s+)?stor(?:y|ies)\s+(?:for|about|on)?\s+["\']?([^"\'\.,;]+)["\']?'
    story_matches = re.findall(story_pattern, content, re.IGNORECASE)
    if story_matches:
        entities["user_stories"] = [match.strip() for match in story_matches]
    
    # Extract feature mentions
    feature_pattern = r'feature(?:s)?\s+(?:for|about|on)?\s+["\']?([^"\'\.,;]+)["\']?'
    feature_matches = re.findall(feature_pattern, content, re.IGNORECASE)
    if feature_matches:
        entities["features"] = [match.strip() for match in feature_matches]
    
    # Extract project mentions
    project_pattern = r'project(?:s)?\s+(?:named|called)?\s+["\']?([^"\'\.,;]+)["\']?'
    project_matches = re.findall(project_pattern, content, re.IGNORECASE)
    if project_matches:
        entities["projects"] = [match.strip() for match in project_matches]
    
    return entities

def identify_themes(content: str) -> List[str]:
    """
    Identify key themes and topics in the content.
    
    Args:
        content: Input text
        
    Returns:
        List of identified themes
    """
    themes = []
    
    # Check for product management themes
    theme_patterns = {
        "roadmap": r'\b(?:roadmap|timeline|schedule|plan)\b',
        "prioritization": r'\b(?:priorit(?:y|ize|ization)|importance|value|effort|impact)\b',
        "user_needs": r'\b(?:user\s+needs|customer|persona|market|audience)\b',
        "documentation": r'\b(?:document|spec|specification|PRD|MRD|product\s+requirement)\b',
        "planning": r'\b(?:sprint|release|milestone|deadline|planning)\b',
        "metrics": r'\b(?:metric|KPI|measure|analytics|data|performance|success)\b'
    }
    
    for theme, pattern in theme_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            themes.append(theme)
    
    return themes
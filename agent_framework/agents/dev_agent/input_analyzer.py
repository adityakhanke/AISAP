"""
Input analysis module for the Dev Agent.
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
    Analyze the following input from a user to a Developer agent:
    
    USER INPUT: {content}
    
    Provide an analysis with the following information:
    1. Is this a code-related request? If so, what is the primary action (generate, explain, debug, etc.)?
    2. What programming language is mentioned or implied?
    3. What code elements (functions, classes, modules, etc.) are mentioned?
    4. What is the primary intent of the user?
    5. What technical requirements or constraints are mentioned?
    
    Format your response as JSON.
    """
    
    # Add RAG context if available
    if rag_context:
        analysis_prompt += f"\n\nRELEVANT CONTEXT:\n{rag_context[:2000]}"  # Limit context size
    
    # Generate analysis using LLM
    llm_response = await llm.generate(
        prompt=analysis_prompt,
        system_prompt="You are an analytical assistant that helps identify the intent and technical requirements in developer requests. Respond in JSON format only.",
        params={"temperature": 0.3, "max_tokens": 32768}
    )
    
    # Parse the LLM's analysis
    analysis_text = llm_response["text"]
    
    # Extract JSON if it's wrapped in markdown code block
    if "```json" in analysis_text:
        json_match = re.search(r"```json\n(.*?)\n```", analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(1)
    
    # Safely parse JSON with error handling
    try:
        llm_analysis = json.loads(analysis_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM analysis JSON: {e}. Using fallback analysis.")
        # Create a fallback analysis
        llm_analysis = {
            "is_code_related": "code" in content.lower(),
            "primary_action": "generate" if "generate" in content.lower() else "analyze",
            "programming_language": "python", # Default language
            "primary_intent": content[:100] + "..." if len(content) > 100 else content,
            "technical_requirements": []
        }
    
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
    
    # Extract code elements
    code_elements = extract_code_elements(content)
    
    # Detect programming language
    programming_language = detect_language(content)
    
    # Identify technical requirements
    requirements = identify_requirements(content)
    
    # Create analysis result
    return {
        "content": content,
        "rag_context": rag_context,
        "workflow_context": workflow_context,
        "command_info": command_info,
        "code_elements": code_elements,
        "programming_language": programming_language,
        "requirements": requirements
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
        "code_block": None
    }

    # Normalize and clean the content
    cleaned_content = content.strip().lower()

    # Try to extract commands from conversational phrases
    conversational_prefixes = [
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
        'generate', 'create', 'write', 'implement',
        'debug', 'fix', 'solve', 'troubleshoot',
        'explain', 'describe', 'analyze', 'review',
        'refactor', 'optimize', 'improve', 'test',
        'document', 'comment', 'find', 'identify'
    ]

    # Command targets and their standardized form
    target_mapping = {
        'code': 'code',
        'function': 'function',
        'class': 'class',
        'method': 'method',
        'module': 'module',
        'program': 'program',
        'script': 'script',
        'api': 'api',
        'algorithm': 'algorithm',
        'test': 'test',
        'bug': 'bug',
        'issue': 'issue',
        'error': 'error',
        'problem': 'problem'
    }

    # Check for direct command format first
    command_pattern = r'^({})\s+(?:a|an|the)?\s*({}|for\s+(?:a|an)?\s*(?:new)?\s*(?:{}))'.format(
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

    else:
        # If no direct command match, try to identify intents from the content
        for verb in command_verbs:
            if verb in cleaned_content:
                for target, standard_target in target_mapping.items():
                    if target in cleaned_content:
                        command_info["is_command"] = True
                        command_info["command_type"] = verb
                        command_info["command_target"] = standard_target
                        break
                if command_info["is_command"]:
                    break

    # Extract any code blocks in the content
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
    if code_blocks:
        command_info["code_block"] = code_blocks[0]

    # If we found a command, extract language information
    if command_info["is_command"]:
        # Look for language mentions
        languages = [
            'python', 'javascript', 'typescript', 'java', 'c++', 'cpp', 'c#', 'csharp',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'html', 'css'
        ]
        
        for lang in languages:
            if lang in cleaned_content.lower():
                command_info["command_args"]["language"] = lang
                break
            
            # Also check code blocks for language hints
            code_block_lang = re.search(r'```(\w+)', content)
            if code_block_lang and code_block_lang.group(1).lower() in languages:
                command_info["command_args"]["language"] = code_block_lang.group(1).lower()
                break

    return command_info

def extract_code_elements(content: str) -> Dict[str, List[str]]:
    """
    Extract code elements mentioned in the content.
    
    Args:
        content: Input text
        
    Returns:
        Dictionary of code element types to lists of element names
    """
    code_elements = {
        "functions": [],
        "classes": [],
        "variables": [],
        "modules": [],
        "apis": [],
        "methods": []
    }
    
    # Extract function mentions
    func_pattern = r'(?:function|method|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    func_matches = re.findall(func_pattern, content)
    if func_matches:
        code_elements["functions"] = [match for match in func_matches]
    
    # Extract class mentions
    class_pattern = r'(?:class)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    class_matches = re.findall(class_pattern, content)
    if class_matches:
        code_elements["classes"] = [match for match in class_matches]
    
    # Extract variable mentions
    var_pattern = r'(?:var|let|const|int|float|string|bool|variable)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    var_matches = re.findall(var_pattern, content)
    if var_matches:
        code_elements["variables"] = [match for match in var_matches]
    
    # Extract module/import mentions
    module_pattern = r'(?:import|from|require|include)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
    module_matches = re.findall(module_pattern, content)
    if module_matches:
        code_elements["modules"] = [match for match in module_matches]
    
    # Extract method mentions
    method_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    method_matches = re.findall(method_pattern, content)
    if method_matches:
        code_elements["methods"] = [f"{obj}.{method}" for obj, method in method_matches]
    
    # Extract API mentions
    api_pattern = r'(?:API|REST|GraphQL|endpoint)\s+([a-zA-Z_][a-zA-Z0-9_\-/]*)'
    api_matches = re.findall(api_pattern, content, re.IGNORECASE)
    if api_matches:
        code_elements["apis"] = [match for match in api_matches]
    
    return code_elements

def detect_language(content: str) -> Optional[str]:
    """
    Detect programming language mentioned or implied in the content.
    
    Args:
        content: Input text
        
    Returns:
        Detected programming language or None
    """
    # Check for explicit language mentions
    language_patterns = {
        'python': r'\b(?:python|py|pip|pandas|numpy|tensorflow|django|flask)\b',
        'javascript': r'\b(?:javascript|js|node|nodejs|npm|react|vue|angular)\b',
        'typescript': r'\b(?:typescript|ts|tsx|angular|nextjs)\b',
        'java': r'\b(?:java|spring|maven|gradle|servlet|jsp)\b',
        'c++': r'\b(?:c\+\+|cpp|clang)\b',
        'csharp': r'\b(?:c#|csharp|\.net|dotnet|asp\.net)\b',
        'go': r'\b(?:go|golang)\b',
        'rust': r'\b(?:rust|cargo|rustc)\b',
        'ruby': r'\b(?:ruby|rails|rake|gem)\b',
        'php': r'\b(?:php|laravel|symfony|composer)\b',
        'swift': r'\b(?:swift|swiftui|xcode|ios)\b',
        'kotlin': r'\b(?:kotlin|kt|android)\b',
        'scala': r'\b(?:scala|sbt)\b',
        'html': r'\b(?:html|dom|web|css|markup)\b',
        'sql': r'\b(?:sql|mysql|postgresql|sqlite|database|query)\b'
    }
    
    # Check the content for language indicators
    for language, pattern in language_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            return language
    
    # Check code blocks for language hints
    code_block_lang = re.search(r'```(\w+)', content)
    if code_block_lang:
        lang = code_block_lang.group(1).lower()
        if lang in language_patterns or lang in ['js', 'ts', 'py', 'cs', 'rb', 'cpp']:
            # Map abbreviations to full names
            lang_mapping = {
                'js': 'javascript',
                'ts': 'typescript',
                'py': 'python',
                'cs': 'csharp',
                'rb': 'ruby',
                'cpp': 'c++'
            }
            return lang_mapping.get(lang, lang)
    
    # Check for code syntax hints if no explicit language is found
    if '#!/usr/bin/python' in content or 'def ' in content or 'import ' in content:
        return 'python'
    elif 'function ' in content or 'const ' in content or 'let ' in content or 'var ' in content:
        return 'javascript'
    elif 'public class ' in content or 'private ' in content:
        return 'java'
    elif '<html>' in content or '<div>' in content:
        return 'html'
    
    return None

def identify_requirements(content: str) -> List[str]:
    """
    Identify technical requirements or constraints in the content.
    
    Args:
        content: Input text
        
    Returns:
        List of identified requirements
    """
    requirements = []
    
    # Check for performance requirements
    if re.search(r'\b(?:performance|fast|speed|efficient|optimize)\b', content, re.IGNORECASE):
        requirements.append("performance_optimization")
    
    # Check for security requirements
    if re.search(r'\b(?:secure|security|vulnerability|attack|protect|encrypt)\b', content, re.IGNORECASE):
        requirements.append("security")
    
    # Check for compatibility requirements
    if re.search(r'\b(?:compatible|compatibility|support|browser|platform|device)\b', content, re.IGNORECASE):
        requirements.append("compatibility")
    
    # Check for testing requirements
    if re.search(r'\b(?:test|testing|unit test|integration test|qa)\b', content, re.IGNORECASE):
        requirements.append("testing")
    
    # Check for documentation requirements
    if re.search(r'\b(?:document|documentation|comment|explain|clarity|readable)\b', content, re.IGNORECASE):
        requirements.append("documentation")
    
    # Check for specific version requirements
    version_match = re.search(r'\b(?:version|v)[\s:]?([0-9]+(?:\.[0-9]+)*)\b', content, re.IGNORECASE)
    if version_match:
        requirements.append(f"version_{version_match.group(1)}")
    
    # Check for framework requirements
    frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'express', 'nextjs']
    for framework in frameworks:
        if re.search(r'\b' + framework + r'\b', content, re.IGNORECASE):
            requirements.append(f"framework_{framework}")
    
    return requirements
"""
Dev Agent-specific Vertex AI LLM configuration for the Agent Framework.
Extends the Mistral Vertex configuration with task-specific parameters and prompts.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from ..providers.mistral_llm import MistralLLMConfig

logger = logging.getLogger(__name__)

class DevAgentLLMConfig(MistralLLMConfig):
    """
    Dev Agent-specific implementation of the Vertex AI LLM configuration.
    Provides specialized parameters and prompts for different development tasks.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Dev Agent Vertex AI LLM configuration.
        
        Args:
            model_name: Name of the model to use
            project_id: Google Cloud Project ID
            region: Google Cloud region
            **kwargs: Additional parameters for the base classes
        """
        # Remove settings to avoid duplicate parameters
        if 'settings' in kwargs:
            kwargs.pop('settings')
        
        # Use codestral-2501 as default model
        model_name = model_name or 'codestral-2501'
        
        # Pass parameters to base class
        super().__init__(
            model_name=model_name,
            project_id=project_id,  # This will be overridden with hardcoded value in MistralLLMConfig
            region=region,
            **kwargs
        )
        
        # Load built-in Dev-specific prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info(f"Initialized Dev Agent Vertex AI LLM configuration with model: {self.model_name}")
    
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates for different development tasks.
        All templates are defined in code, not in config.
        
        Returns:
            Dictionary of task-specific prompt templates
        """
        # All templates are hardcoded - none from config
        return {
            "generate_code": """
                As a Developer, generate code based on the following requirements:
                
                {context}
                
                The code should:
                - Be well-structured and follow best practices
                - Include comments explaining complex sections
                - Handle edge cases appropriately
                - Be optimized for readability and maintainability
                
                Programming language: {language}
            """.strip(),
            
            "explain_code": """
                As a Developer, explain the following code:
                
                ```{language}
                {code}
                ```
                
                Please provide:
                - A high-level overview of what the code does
                - Explanation of key functions and their purpose
                - Any design patterns or techniques used
                - Potential improvements or optimizations
            """.strip(),
            
            "debug_code": """
                As a Developer, debug the following code that has issues:
                
                ```{language}
                {code}
                ```
                
                Error/Issue: {error}
                
                Please:
                - Identify the root cause of the issue
                - Explain why this is causing problems
                - Provide a corrected version of the code
                - Suggest best practices to avoid similar issues
            """.strip(),
            
            "implement_requirement": """
                As a Developer, implement the following requirement:
                
                {context}
                
                Please create:
                - Well-structured code that fulfills the requirement
                - Appropriate error handling
                - Unit tests (if applicable)
                - Documentation comments
                
                Programming language: {language}
            """.strip(),
            
            "analyze_code": """
                As a Developer, analyze the following code for quality, performance, and security:
                
                ```{language}
                {code}
                ```
                
                Please provide:
                - Code quality assessment
                - Performance considerations
                - Security vulnerabilities (if any)
                - Suggestions for improvement
                - Refactoring recommendations
            """.strip(),
            
            "generate_tests": """
                As a Developer, create tests for the following code:
                
                ```{language}
                {code}
                ```
                
                Please generate:
                - Comprehensive unit tests
                - Test cases covering edge conditions
                - Mocks or stubs for dependencies (if needed)
                - Clear test descriptions and assertions
                
                Testing framework: {framework}
            """.strip(),
        }
    
    def get_task_prompt(self, task_type: str, context: Dict[str, Any]) -> str:
        """
        Get a formatted prompt for a specific development task.
        
        Args:
            task_type: Type of development task (e.g., 'generate_code')
            context: Context variables to insert into the prompt
            
        Returns:
            Formatted prompt for the task
        """
        if task_type not in self.prompt_templates:
            logger.warning(f"No prompt template found for task type: {task_type}")
            # Use a generic prompt if specific template not found
            return f"As a Developer, complete the following task: {context.get('content', '')}"
        
        # Get the template and format with context
        template = self.prompt_templates[task_type]
        
        # Convert context dict to a string if needed
        if isinstance(context, dict):
            if 'content' in context:
                context_str = context['content']
            else:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        else:
            context_str = str(context)
        
        # Handle the special case of code explanation/debugging
        if task_type in ["explain_code", "debug_code", "analyze_code"]:
            if "code" not in context and "content" in context:
                # Try to extract code blocks from content
                import re
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", context["content"], re.DOTALL)
                if code_blocks:
                    context["code"] = code_blocks[0]
                else:
                    context["code"] = context["content"]
        
        # Format the template
        return self.format_prompt(template, {"context": context_str, **context})
    
    def get_params_for_task(self, task_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters for a specific development task.
        
        Args:
            task_type: Type of development task
            
        Returns:
            Dictionary of task-specific parameters
        """
        # Try to get custom parameters from config
        settings = getattr(self, 'settings', {})
        task_params = settings.get('llm_task_params', {}).get('dev_agent', {}).get(task_type, {})
        
        # Start with default parameters
        params = self.get_default_params()
        
        # Default task-specific optimizations
        default_task_params = {
            "generate_code": {
                "temperature": 0.3,
                "max_tokens": 32768,
                "top_p": 0.95
            },
            "explain_code": {
                "temperature": 0.7,
                "max_tokens": 32768
            },
            "debug_code": {
                "temperature": 0.4,
                "max_tokens": 32768
            },
            "implement_requirement": {
                "temperature": 0.5,
                "max_tokens": 32768
            },
            "analyze_code": {
                "temperature": 0.4,
                "max_tokens": 32768
            },
            "generate_tests": {
                "temperature": 0.5,
                "max_tokens": 32768
            }
        }
        
        # Apply default task params if available
        if task_type in default_task_params:
            params.update(default_task_params[task_type])
        
        # Override with custom params from config if available
        if task_params:
            params.update(task_params)
        
        return params
    
    async def execute_dev_task(
        self,
        task_type: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        High-level method to execute a development task with the LLM.
        
        Args:
            task_type: Type of development task
            context: Context for the task
            system_prompt: Optional system instructions
            conversation_history: Optional conversation history
            
        Returns:
            Response from the LLM
        """
        # Get task-specific prompt and parameters
        prompt = self.get_task_prompt(task_type, context)
        params = self.get_params_for_task(task_type)
        
        # Use default system prompts - not from config
        if not system_prompt:
            # Standard system prompts for each task type
            task_system_prompts = {
                "generate_code": "You are a professional Software Developer focused on writing clean, efficient, and well-documented code that follows best practices.",
                "explain_code": "You are a code educator who excels at breaking down complex code into understandable explanations with clear examples.",
                "debug_code": "You are an expert debugger who identifies root causes of code issues and provides thorough explanations and fixes.",
                "implement_requirement": "You are a developer implementing requirements with production-quality code that's maintainable and robust.",
                "analyze_code": "You are a code reviewer who evaluates code quality, performance, and security with constructive feedback.",
                "generate_tests": "You are a test engineer creating comprehensive test suites that verify code functionality and edge cases."
            }

            # Get the appropriate system prompt for this task type
            system_prompt = task_system_prompts.get(
                task_type, 
                "You are a professional Software Developer assistant. Your goal is to help write, explain, and improve code with a focus on best practices, efficiency, and readability."
            )
        
        # Generate with or without conversation history
        if conversation_history:
            return await self.generate_with_context(
                prompt=prompt,
                context=conversation_history,
                system_prompt=system_prompt,
                params=params
            )
        else:
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                params=params
            )
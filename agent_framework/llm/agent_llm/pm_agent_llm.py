"""
PM Agent-specific LLM configuration for the Agent Framework.
Extends the Groq configuration with task-specific parameters and prompts.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from ..providers.groq_llm import GroqLLMConfig

logger = logging.getLogger(__name__)

class PMAgentLLMConfig(GroqLLMConfig):
    """
    PM Agent-specific implementation of the LLM configuration.
    Provides specialized parameters and prompts for different PM tasks.
    """
    
    def __init__(
        self,
        model_name: str = "llama-70b-4096",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the PM Agent LLM configuration.
        
        Args:
            model_name: Name of the model to use (defaults to Llama 70B)
            api_key: API key for Groq
            **kwargs: Additional parameters for the base classes
        """
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        
        # Load PM-specific prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info(f"Initialized PM Agent LLM configuration with model: {model_name}")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates for different PM tasks.
        
        Returns:
            Dictionary of task-specific prompt templates
        """
        return {
            "create_requirement": """
                As a Product Manager, create a detailed product requirement based on the following information:
                
                {context}
                
                The requirement should include:
                - A clear title
                - Detailed description
                - Acceptance criteria
                - Any constraints or dependencies
                
                Please format the requirement in a structured way that's easy to understand.
            """.strip(),
            
            "create_user_story": """
                As a Product Manager, create a user story based on the following information:
                
                {context}
                
                Format the user story using:
                - "As a [type of user]"
                - "I want to [perform some action]"
                - "So that [I can achieve some goal/value]"
                
                Please include acceptance criteria and story points estimate if possible.
            """.strip(),
            
            "prioritize_requirements": """
                As a Product Manager, analyze and prioritize the following requirements:
                
                {context}
                
                Please prioritize these requirements using the {method} approach and provide:
                - Prioritized list of requirements
                - Rationale for each priority decision
                - Recommendations for implementation sequence
            """.strip(),
            
            "generate_roadmap": """
                As a Product Manager, create a product roadmap based on the following information:
                
                {context}
                
                Please structure the roadmap with:
                - Clear timeline (phases or quarters)
                - Features and deliverables for each phase
                - Key milestones
                - Dependencies between items
                
                The roadmap should cover a {timeframe} timeframe.
            """.strip(),
            
            "generate_prd": """
                As a Product Manager, create a comprehensive Product Requirements Document (PRD) based on the following information:
                
                {context}
                
                The PRD should include:
                - Executive Summary
                - Product Overview
                - Goals and Objectives
                - User Personas
                - User Stories
                - Functional Requirements
                - Non-Functional Requirements
                - Dependencies and Constraints
                - Success Metrics
                
                Please make the document comprehensive but well-structured.
            """.strip(),
        }
    
    def get_task_prompt(self, task_type: str, context: Dict[str, Any]) -> str:
        """
        Get a formatted prompt for a specific PM task.
        
        Args:
            task_type: Type of PM task (e.g., 'create_requirement')
            context: Context variables to insert into the prompt
            
        Returns:
            Formatted prompt for the task
        """
        if task_type not in self.prompt_templates:
            logger.warning(f"No prompt template found for task type: {task_type}")
            # Use a generic prompt if specific template not found
            return f"As a Product Manager, complete the following task: {context.get('content', '')}"
        
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
        
        # Format the template
        return self.format_prompt(template, {"context": context_str, **context})
    
    def get_params_for_task(self, task_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters for a specific PM task.
        
        Args:
            task_type: Type of PM task
            
        Returns:
            Dictionary of task-specific parameters
        """
        # Start with default parameters
        params = self.get_default_params()
        
        # Override with task-specific optimizations
        if task_type == "create_requirement":
            params.update({
                "temperature": 0.6,  # More focused for requirements
                "max_tokens": 32768
            })
        elif task_type == "create_user_story":
            params.update({
                "temperature": 0.65,
                "max_tokens": 32768
            })
        elif task_type == "prioritize_requirements":
            params.update({
                "temperature": 0.4,  # More analytical for prioritization
                "max_tokens": 32768
            })
        elif task_type == "generate_roadmap":
            params.update({
                "temperature": 0.7,  # More creative for roadmap planning
                "max_tokens": 32768
            })
        elif task_type == "generate_prd":
            params.update({
                "temperature": 0.5,  # Balanced for documentation
                "max_tokens": 32768,  # Longer for comprehensive documents
                "top_p": 0.95
            })
        
        return params
    
    async def execute_pm_task(
        self,
        task_type: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        High-level method to execute a PM task with the LLM.
        
        Args:
            task_type: Type of PM task
            context: Context for the task
            system_prompt: Optional system instructions
            conversation_history: Optional conversation history
            
        Returns:
            Response from the LLM
        """
        # Get task-specific prompt and parameters
        prompt = self.get_task_prompt(task_type, context)
        params = self.get_params_for_task(task_type)
        
        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = "You are a professional Product Manager assistant. Your goal is to help create clear, comprehensive product specifications and documentation."
        
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
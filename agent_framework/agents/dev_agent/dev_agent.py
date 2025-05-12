"""
Developer (Dev) Agent implementation with LLM integration.
This is the main Dev Agent class file for code generation and analysis.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

from ...core.interfaces import ContextInterface, ToolInterface, RAGInterface
from ...core.agent import BaseAgent
from ...llm.agent_llm.dev_agent_llm import DevAgentLLMConfig
from ...llm.llm_factory import LLMFactory
from .tool import register_dev_tools

# Import Dev Agent-specific modules
from .input_analyzer import analyze_input_with_llm, analyze_input_traditional
from .intent_planner import determine_intent_with_llm, determine_intent_traditional
from .plan_executor import execute_plan_with_llm
from .response_generator import generate_response_with_llm, format_response_traditional

logger = logging.getLogger(__name__)

class DevAgent(BaseAgent):
    """
    Developer (Dev) Agent for code generation, analysis, and debugging tasks.
    Enhanced with LLM capabilities for more advanced interactions.
    
    Specializes in code generation, code explanation, debugging, and testing.
    """
    
    def __init__(self, agent_id: str, name: str, context_manager: ContextInterface, 
                tool_registry: Optional[ToolInterface] = None,
                rag: Optional[RAGInterface] = None, llm_config: Optional[DevAgentLLMConfig] = None,
                llm_api_key: Optional[str] = None,
                llm_model_name: Optional[str] = "codestral-latest"):
        """
        Initialize the Dev Agent.

        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            context_manager: Context manager for accessing shared context
            tool_registry: Registry of tools available to this agent
            rag: Optional contextual RAG instance
            llm_config: Optional pre-configured LLM configuration
            llm_api_key: Optional API key for LLM provider
            llm_model_name: Optional model name to use
        """
        super().__init__(agent_id, name, context_manager, tool_registry)
        self.rag = rag

        # Initialize LLM configuration
        if llm_config:
            self.llm = llm_config
        else:
            self.llm = LLMFactory.create(
                config_type="dev_agent",
                api_key=llm_api_key,
                model_name=llm_model_name
            )

        # Register Dev-specific tools
        self._register_tools()
        
        # Initialize working memory for context
        self.working_memory.store("current_project", None)
        self.working_memory.store("conversation_history", [])
        
        logger.info(f"Initialized Dev Agent with LLM: {name} (ID: {agent_id})")
    
    def _register_tools(self):
        """Register Dev-specific tools."""
        if not self.tool_registry:
            logger.warning("No tool registry available, skipping tool registration")
            return
            
        # Register tools using the dedicated module
        register_dev_tools(self)
    
    async def analyze_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the input to extract key information, enhanced with LLM.
        
        Args:
            input_data: Input data to analyze
            
        Returns:
            Analysis result
        """
        content = input_data.get("content", "")
        
        # Extract workflow context if available
        workflow_instance = input_data.get("workflow_instance")
        workflow_state = input_data.get("workflow_state")
        workflow_context = None
        
        if workflow_instance:
            # Get workflow context if this is part of a workflow
            workflow_context = input_data.get("context", {})
        
        # Get relevant context from RAG if available
        rag_context = None
        if self.rag and content:
            try:
                rag_context = self.rag.extract_document_for_llm(
                    content, 
                    agent_id=self.agent_id,
                    workflow_state=workflow_context
                )
                logger.debug(f"Retrieved RAG context: {len(rag_context) if rag_context else 0} chars")
            except Exception as e:
                logger.error(f"Error getting RAG context: {e}")
        
        # Try LLM-based analysis first
        try:
            analysis_result = await analyze_input_with_llm(
                content=content,
                llm=self.llm,
                rag_context=rag_context,
                workflow_context=workflow_context
            )
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error using LLM for analysis: {e}")
        
        # Fall back to traditional analysis
        return analyze_input_traditional(
            content=content,
            rag_context=rag_context,
            workflow_context=workflow_context
        )
    
    async def determine_intent(self, analysis_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the user's intent and create an execution plan, enhanced with LLM.
        
        Args:
            analysis_result: Result of input analysis
            
        Returns:
            Tuple of (intent, plan)
        """
        content = analysis_result.get("content", "")
        
        # Check if we have LLM analysis results
        if "llm_analysis" in analysis_result:
            try:
                # Use LLM-enhanced intent determination
                return await determine_intent_with_llm(
                    analysis_result=analysis_result,
                    llm=self.llm
                )
            except Exception as e:
                logger.error(f"Error determining intent with LLM: {e}")
        
        # Fall back to traditional intent determination
        return determine_intent_traditional(analysis_result)
    
    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan based on the determined intent, enhanced with LLM.
        
        Args:
            plan: Execution plan
            
        Returns:
            Execution result
        """
        try:
            # Execute plan using LLM and tools
            result = await execute_plan_with_llm(
                plan=plan,
                agent=self,
                llm=self.llm,
                tool_registry=self.tool_registry,
                context_manager=self.context_manager,
                conversation_history=self.working_memory.retrieve("conversation_history", [])
            )
            
            # Update conversation history if needed
            if "update_history" in result and result["update_history"]:
                self._update_conversation_history(
                    role="user",
                    content=plan.get("content", "")
                )
                
                self._update_conversation_history(
                    role="assistant",
                    content=result.get("response", "") or result.get("result", "")
                )
            
            return result
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return {
                "success": False,
                "error": f"Failed to execute plan: {str(e)}"
            }
    
    async def generate_response(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response based on the execution result, enhanced with LLM.
        
        Args:
            execution_result: Result of plan execution
            
        Returns:
            Response data
        """
        # Check for error
        if not execution_result.get("success", True):
            error = execution_result.get("error", "An unknown error occurred")
            return {
                "content": f"Sorry, I encountered an error: {error}",
                "error": error
            }
        
        # If there's a direct response already formatted, use it
        if "response" in execution_result:
            return {
                "content": execution_result["response"]
            }
        
        try:
            # Generate formatted response using LLM
            return await generate_response_with_llm(
                execution_result=execution_result,
                llm=self.llm
            )
        except Exception as e:
            logger.error(f"Error generating response with LLM: {e}")
            
            # Fall back to traditional formatting
            return format_response_traditional(execution_result)
    
    def _update_conversation_history(self, role: str, content: str) -> None:
        """
        Update the conversation history in working memory.
        
        Args:
            role: Role of the message ('user' or 'assistant')
            content: Content of the message
        """
        history = self.working_memory.retrieve("conversation_history", [])
        
        # Add the new message
        history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Limit history size
        max_history = 10
        if len(history) > max_history:
            history = history[-max_history:]
            
        # Store updated history
        self.working_memory.store("conversation_history", history)
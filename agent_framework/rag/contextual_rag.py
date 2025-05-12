"""
Context-aware RAG implementation that integrates with the agent framework.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from ..core.interfaces import ContextInterface, RAGInterface

logger = logging.getLogger(__name__)

class ContextualRAG(RAGInterface):
    """
    Context-aware RAG implementation that integrates the existing RAG with the agent framework.
    Enables context-sensitive retrievals based on the current agent and workflow state.
    """
    
    def __init__(self, context_manager: ContextInterface, base_rag: RAGInterface):
        """
        Initialize the contextual RAG.
        
        Args:
            context_manager: Context manager for accessing shared context
            base_rag: Base RAG implementation to use
        """
        self.context_manager = context_manager
        self.base_rag = base_rag
        
        # Cache for temporary optimization
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_last_cleared = time.time()
        
        logger.info("Initialized contextual RAG")
    
    def extract_document_for_llm(self, query_text: str, agent_id: Optional[str] = None,
                               workflow_state: Optional[Dict[str, Any]] = None,
                               k: int = 5) -> str:
        """
        Query the RAG system with context awareness and format for LLM consumption.
        
        Args:
            query_text: The search query
            agent_id: Optional ID of the requesting agent (for agent-specific context)
            workflow_state: Optional workflow state data
            k: Number of results to return
            
        Returns:
            Formatted context string for LLM consumption
        """
        # Get the enhanced context for retrieval
        enhanced_query, filters = self._enhance_query(query_text, agent_id, workflow_state)
        
        # Check cache for identical query
        cache_key = f"{enhanced_query}:{str(filters)}:{k}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                logger.info(f"Using cached result for query: {enhanced_query}")
                return cache_entry["result"]
        
        # Perform the search
        try:
            # Call the base RAG
            context_text = self.base_rag.extract_document_for_llm(
                enhanced_query, 
                agent_id=agent_id,
                workflow_state=workflow_state,
                k=k
            )
            
            # Update cache
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "result": context_text
            }
            
            # Periodically clean cache
            if time.time() - self.cache_last_cleared > 1800:  # 30 minutes
                self._clean_cache()
            
            return context_text
            
        except Exception as e:
            logger.error(f"Error in contextual RAG query: {e}")
            # Fall back to basic query without enhancement
            try:
                logger.info(f"Falling back to basic query: {query_text}")
                return self.base_rag.extract_document_for_llm(query_text, k=k)
            except Exception as e2:
                logger.error(f"Error in fallback query: {e2}")
                return f"Error retrieving context: {e2}\n\nOriginal query: {query_text}"
    
    def _enhance_query(self, query_text: str, agent_id: Optional[str] = None,
                    workflow_state: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance a query with contextual information.
        
        Args:
            query_text: Original query text
            agent_id: Optional ID of the requesting agent
            workflow_state: Optional workflow state data
            
        Returns:
            Tuple of (enhanced query, filters)
        """
        enhanced_query = query_text
        filters = {}
        
        # Add relevant context based on agent type
        if agent_id:
            # Get agent type from context if available
            agent_entities = self.context_manager.query_entities(
                entity_type="agent",
                filters={"id": agent_id},
                limit=1
            )
            
            if agent_entities:
                agent_type = agent_entities[0].data.get("type")
                
                # Enhance query based on agent type
                if agent_type == "pm_agent":
                    # PM Agent likely cares about requirements, user stories, etc.
                    product_entities = self.context_manager.query_entities(
                        entity_type="project",
                        limit=1
                    )
                    
                    if product_entities:
                        project_name = product_entities[0].data.get("name", "")
                        enhanced_query = f"{enhanced_query} {project_name} product requirements"
                
                elif agent_type == "dev_agent":
                    # Dev Agent likely cares about code, architecture, etc.
                    # Get current codebase context if available
                    code_entities = self.context_manager.query_entities(
                        entity_type="code_entity",
                        limit=5
                    )
                    
                    if code_entities:
                        code_terms = " ".join([
                            e.data.get("name", "") for e in code_entities 
                            if e.data.get("name")
                        ])
                        enhanced_query = f"{enhanced_query} {code_terms} code implementation"
        
        # Add workflow context if available
        if workflow_state:
            workflow_id = workflow_state.get("workflow_id")
            current_state = workflow_state.get("current_state")
            
            if workflow_id and current_state:
                # Get workflow information
                workflow_entities = self.context_manager.query_entities(
                    entity_type="workflow",
                    filters={"id": workflow_id},
                    limit=1
                )
                
                if workflow_entities:
                    workflow = workflow_entities[0].data
                    
                    # Find current state
                    states = workflow.get("states", [])
                    current_state_info = next((s for s in states if s.get("id") == current_state), None)
                    
                    if current_state_info:
                        state_name = current_state_info.get("name", "")
                        enhanced_query = f"{enhanced_query} {state_name} workflow step"
        
        logger.info(f"Enhanced query: '{query_text}' => '{enhanced_query}'")
        return enhanced_query, filters
    
    def _clean_cache(self) -> None:
        """Clean expired entries from the cache."""
        now = time.time()
        self.cache = {
            key: entry 
            for key, entry in self.cache.items() 
            if now - entry["timestamp"] < self.cache_ttl
        }
        self.cache_last_cleared = now
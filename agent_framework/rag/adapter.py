"""
Adapter for existing RAG implementations to integrate with the agent framework.
"""

import logging
import importlib
from typing import Dict, Any, Optional

from ..core.interfaces import RAGInterface

logger = logging.getLogger(__name__)

class RAGAdapter(RAGInterface):
    """
    Adapter for existing RAG implementations to integrate with the agent framework.
    Provides a standardized interface for different RAG implementations.
    """
    
    def __init__(self, rag_module: str = None, rag_class: str = None, rag_instance = None, **rag_params):
        """
        Initialize the RAG adapter.
        
        Args:
            rag_module: Optional module name containing the RAG implementation
            rag_class: Optional class name of the RAG implementation
            rag_instance: Optional pre-configured RAG instance
            **rag_params: Parameters to pass to the RAG constructor
        """
        self.rag_instance = None
        
        if rag_instance:
            # Use provided instance
            self.rag_instance = rag_instance
            logger.info(f"Using provided RAG instance of type {type(rag_instance).__name__}")
        elif rag_module and rag_class:
            # Import and instantiate the RAG class
            try:
                module = importlib.import_module(rag_module)
                rag_class_obj = getattr(module, rag_class)
                self.rag_instance = rag_class_obj(**rag_params)
                logger.info(f"Instantiated RAG class {rag_class} from module {rag_module}")
            except (ImportError, AttributeError) as e:
                logger.error(f"Error importing RAG class: {e}")
                raise ImportError(f"Could not import RAG class {rag_class} from {rag_module}: {e}")
            except Exception as e:
                logger.error(f"Error instantiating RAG class: {e}")
                raise ValueError(f"Could not instantiate RAG class {rag_class}: {e}")
        else:
            # Try to import our standard RAG and EnhancedRAG
            try:
                # Try enhanced RAG first
                try:
                    from rag_package.enhanced_rag import EnhancedRAG
                    self.rag_instance = EnhancedRAG(**rag_params)
                    logger.info("Using EnhancedRAG implementation")
                except ImportError:
                    # Fall back to standard RAG
                    from rag_package.rag import RAG
                    self.rag_instance = RAG(**rag_params)
                    logger.info("Using standard RAG implementation")
            except ImportError:
                logger.error("Could not import any RAG implementation")
                raise ImportError("No RAG implementation available. Please provide a valid RAG instance or class.")
        
        if self.rag_instance is None:
            raise ValueError("Failed to initialize RAG instance")
        
        # Determine available methods
        self.has_enhanced_query = hasattr(self.rag_instance, 'enhanced_query')
        self.has_extract_for_llm = hasattr(self.rag_instance, 'extract_document_for_llm')
        
        logger.info(f"RAG Adapter initialized with {type(self.rag_instance).__name__}")
        logger.info(f"Enhanced query: {'Available' if self.has_enhanced_query else 'Not available'}")
        logger.info(f"Extract for LLM: {'Available' if self.has_extract_for_llm else 'Not available'}")
    
    def query(self, query_text: str, k: int = 5, remove_duplicates: bool = True, 
            use_reranking: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: Query text
            k: Number of results to return
            remove_duplicates: If True, filters out results from the same file
            use_reranking: If True, uses cross-encoder to rerank results
            
        Returns:
            Query result
        """
        logger.info(f"Querying RAG with query: '{query_text}'")
        
        if self.has_enhanced_query:
            logger.info("Using enhanced_query method")
            return self.rag_instance.enhanced_query(query_text, k, remove_duplicates, use_reranking)
        else:
            logger.info("Using standard query method")
            return self.rag_instance.query(query_text, k, remove_duplicates, use_reranking)
    
    def extract_document_for_llm(self, query_text: str, agent_id: Optional[str] = None,
                               workflow_state: Optional[Dict[str, Any]] = None,
                               k: int = 5) -> str:
        """
        Extract document context formatted for LLM consumption.
        
        Args:
            query_text: Query text
            agent_id: Optional ID of the requesting agent (for context)
            workflow_state: Optional workflow state data
            k: Number of results to return
            
        Returns:
            Formatted context string
        """
        logger.info(f"Extracting document for LLM with query: '{query_text}'")
        
        if self.has_extract_for_llm:
            logger.info("Using extract_document_for_llm method")
            return self.rag_instance.extract_document_for_llm(query_text, k)
        else:
            logger.info("Using query method and formatting result")
            
            # Get query result
            result = self.query(query_text, k)
            
            # Format result for LLM
            context_text = f"Query: {query_text}\n\n"
            context_text += "RETRIEVED DOCUMENTS:\n\n"
            
            for i, doc in enumerate(result.get("results", [])):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                context_text += f"Document {i+1}:\n"
                context_text += f"Source: {metadata.get('source', 'Unknown')}\n"
                
                # Add other metadata as available
                if "title" in metadata:
                    context_text += f"Title: {metadata['title']}\n"
                
                context_text += "Content:\n"
                context_text += "----------------------------------------\n"
                context_text += f"{content}\n"
                context_text += "----------------------------------------\n\n"
            
            return context_text
    
    def get_rag_instance(self):
        """
        Get the underlying RAG instance.
        
        Returns:
            RAG instance
        """
        return self.rag_instance
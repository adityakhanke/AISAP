"""
Custom embedding functions for ChromaDB.
"""

from typing import List

from .embedding_manager import EmbeddingManager


class ChromaEmbeddingFunction:
    """Custom embedding function for ChromaDB that uses our EmbeddingManager."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize the embedding function with an embedding manager.
        
        Args:
            embedding_manager: The manager to use for generating embeddings
        """
        self.embedding_manager = embedding_manager
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts.
        
        Args:
            input: The texts to embed (renamed from 'texts' to match ChromaDB's interface)
        """
        return self.embedding_manager.get_embeddings(input)
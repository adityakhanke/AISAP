"""
RAG implementation using embedding models and vector storage.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from .storage.document import Document
from .storage.vector_store import ChromaVectorStore
from .embedding.embedding_manager import EmbeddingManager
from .text.file_parser import FileParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAG')

class RAG:
    """Retrieval-Augmented Generation system for files with improved error handling."""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                embedding_model: str = "nomic-ai/nomic-embed-code",
                cross_encoder_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                chunking_strategy: str = "semantic",
                chunk_size: int = 500,
                chunk_overlap: int = 50,
                force_model: bool = False,
                use_gpu: bool = True,
                max_length: int = 512,
                use_8bit: bool = False):  # Added use_8bit parameter
        """
        Initialize the RAG system with improved error handling.
        
        Args:
            persist_directory: Directory for ChromaDB storage
            embedding_model: Name of the embedding model to use
            cross_encoder_model: Name of the cross-encoder model for reranking (or None to disable)
            chunking_strategy: Strategy for chunking documents ('simple', 'semantic', or 'structure')
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks (for simple chunking)
            force_model: If True, use the specified model even if dimensions don't match
            use_gpu: If True and GPU is available, use GPU acceleration
            max_length: Maximum token length for large models
            use_8bit: If True, use 8-bit quantization for large models
        """
        logger.info(f"Initializing RAG with {embedding_model} model")
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            cross_encoder_name=cross_encoder_model,
            force_model=force_model,
            use_gpu=use_gpu,
            max_length=max_length,
            use_8bit=use_8bit  # Pass the 8-bit parameter
        )
        
        # Initialize vector store with dimension compatibility checking
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            embedding_manager=self.embedding_manager,
            force_model=force_model
        )
        
        # Initialize file parser with chunking settings
        self.file_parser = FileParser(
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def ingest_folder(self, folder_path: str, reset: bool = False, 
                     max_files: int = None, batch_size: int = 100):
        """
        Ingest all files from a folder.
        
        Args:
            folder_path: Path to the folder containing documents
            reset: If True, clear the existing collection before ingestion
            max_files: Maximum number of files to process (None for all)
            batch_size: Number of documents to add in each batch
        """
        if not os.path.isdir(folder_path):
            logger.error(f"Error: {folder_path} is not a valid directory")
            return
        
        # Clear existing documents if reset is True
        if reset:
            self.vector_store.clear_collection()
        
        # Find all files
        all_files = FileParser.find_all_files(folder_path, max_files)
        logger.info(f"Found {len(all_files)} files in {folder_path}")
        
        # Process each file
        start_time = time.time()
        total_chunks = 0
        for file_index, file_path in enumerate(all_files):
            logger.info(f"Processing file {file_index+1}/{len(all_files)}: {file_path}")
            
            # Use the file parser with configured chunking strategy
            documents = self.file_parser.parse_file(file_path)
            
            if documents:
                self.vector_store.add_documents(documents, batch_size)
                total_chunks += len(documents)
                logger.info(f"  Added {len(documents)} chunks from file")
            else:
                logger.warning(f"  No content extracted from file")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Ingestion complete in {elapsed_time:.2f} seconds.")
        logger.info(f"Processed {len(all_files)} files with {total_chunks} total chunks.")
        
        # Debug: Print collection info
        collection_info = self.vector_store.get_collection_info()
        logger.info(f"Collection now contains {collection_info['count']} documents from {collection_info['file_count']} files")
    
    def query(self, query_text: str, k: int = 3, 
            remove_duplicates: bool = True, 
            use_reranking: bool = True,
            max_retries: int = 2) -> dict:
        """
        Query the RAG system with improved error handling.
        
        Args:
            query_text: The search query
            k: Number of results to return
            remove_duplicates: If True, filters out results from the same file
            use_reranking: If True, uses cross-encoder to rerank results
            max_retries: Maximum number of retry attempts
        """
        collection_info = self.vector_store.get_collection_info()
        logger.info(f"Running query against collection with {collection_info['count']} documents")
        
        # Perform the search with all options and retries
        relevant_docs = self.vector_store.similarity_search(
            query_text, 
            k=k, 
            remove_duplicates=remove_duplicates,
            use_reranking=use_reranking,
            max_retries=max_retries
        )
        
        result = {
            "query": query_text,
            "results": []
        }
        
        for doc in relevant_docs:
            # Ensure file_id is never unknown in results
            if doc.metadata.get("file_id") is None or doc.metadata.get("file_id") == "unknown":
                # Try to find the file_id by querying the file collection with the source path
                source_path = doc.metadata.get("source")
                if source_path:
                    try:
                        # Query file collection by source path
                        file_results = self.vector_store.file_collection.get(
                            where={"source": source_path}
                        )
                        if file_results and file_results["ids"]:
                            # Update the file_id in the metadata
                            doc.metadata["file_id"] = file_results["ids"][0]
                    except Exception as e:
                        logger.error(f"Error finding file_id for source {source_path}: {e}")
            
            # Create a serializable version of metadata
            metadata_copy = doc.metadata.copy()
            
            # Handle entities list for JSON serialization
            if "entities" in metadata_copy and isinstance(metadata_copy["entities"], list):
                metadata_copy["entities"] = metadata_copy["entities"]
            
            result["results"].append({
                "content": doc.content,
                "metadata": metadata_copy,
                "distance": doc.metadata.get("distance", None),
                "reranker_score": doc.metadata.get("reranker_score", None)
            })
            
        # Include diagnostic info
        result["diagnostics"] = {
            "collection_size": collection_info['count'],
            "file_count": collection_info.get('file_count', 0),
            "results_returned": len(relevant_docs),
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dimension": self.embedding_manager.embedding_dimension,
            "cross_encoder": self.embedding_manager.cross_encoder_name if use_reranking else "disabled",
            "unique_sources": True if remove_duplicates else False,
            "gpu_enabled": hasattr(self.embedding_manager, 'use_gpu') and self.embedding_manager.use_gpu,
            "8bit_enabled": hasattr(self.embedding_manager, 'use_8bit') and self.embedding_manager.use_8bit
        }
            
        return result
    
    def get_file_chunks(self, file_id: str) -> List[Document]:
        """Get all chunks for a specific file by file_id."""
        return self.vector_store.get_file_chunks(file_id)
    
    def get_chunks_by_source(self, source_path: str) -> List[Document]:
        """Get all chunks for a specific file by its source path."""
        return self.vector_store.get_chunks_by_source(source_path)
    
    def get_all_files(self) -> List[Dict]:
        """Get metadata for all files in the store."""
        return self.vector_store.get_all_files()
    
    def reconstruct_file_content(self, file_id: str) -> str:
        """Reconstruct the full file content from its chunks."""
        chunks = self.get_file_chunks(file_id)
        if not chunks:
            return ""
        
        # Sort chunks by position in original file
        chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        # Combine chunk contents
        content = "".join(chunk.content for chunk in chunks)
        return content

    def extract_document_for_llm(self, query_text: str, k: int = 5, 
                            remove_duplicates: bool = True,
                            use_reranking: bool = True) -> str:
        """
        Query the RAG system and format the results in a way that's suitable for an LLM.
        Returns the full content of each document with clear separation and metadata.
        
        Args:
            query_text: The search query
            k: Number of results to return
            remove_duplicates: If True, filters out results from the same file
            use_reranking: If True, uses cross-encoder for reranking
        """
        # Get results
        result = self.query(query_text, k, remove_duplicates, use_reranking)
        
        # Format for LLM consumption
        llm_context = f"Query: {query_text}\n\n"
        llm_context += "RETRIEVED DOCUMENTS:\n\n"
        
        # Add each document with full content
        for i, doc in enumerate(result["results"]):
            source = doc["metadata"]["source"]
            title = doc["metadata"]["title"]
            file_id = doc["metadata"].get("file_id", "unknown")
            
            # Use reranker score if available, otherwise use distance
            if doc.get("reranker_score") is not None:
                score_type = "Relevance"
                score_value = doc["reranker_score"]
            else:
                score_type = "Similarity"
                score_value = 1 - doc["distance"] if doc["distance"] is not None else 0
                
            llm_context += f"Document {i+1} ({score_type} Score: {score_value:.4f}):\n"
            llm_context += f"Source: {source}\n"
            llm_context += f"Title: {title}\n"
            llm_context += f"File ID: {file_id}\n"
            
            # Add entities if available
            if doc["metadata"].get("entities"):
                entities = doc["metadata"]["entities"]
                if entities:
                    llm_context += f"Entities: {', '.join(entities[:5])}"
                    if len(entities) > 5:
                        llm_context += f" (+{len(entities)-5} more)"
                    llm_context += "\n"
            
            llm_context += "Content:\n"
            llm_context += "----------------------------------------\n"
            llm_context += f"{doc['content']}\n"
            llm_context += "----------------------------------------\n\n"
        
        # Add diagnostic information
        llm_context += f"Based on {result['diagnostics']['results_returned']} documents retrieved from "
        llm_context += f"a collection of {result['diagnostics']['collection_size']} documents across "
        llm_context += f"{result['diagnostics']['file_count']} files.\n"
        
        # Add more context about the search
        search_details = []
        if result['diagnostics'].get('embedding_model'):
            search_details.append(f"Embedding model: {result['diagnostics']['embedding_model']}")
        if remove_duplicates:
            search_details.append("Results are from unique source files")
        else:
            search_details.append("Results may include multiple chunks from the same file")
        if use_reranking and result['diagnostics'].get('cross_encoder') and result['diagnostics']['cross_encoder'] != 'disabled':
            search_details.append(f"Results reranked using {result['diagnostics']['cross_encoder']}")
        if result['diagnostics'].get('gpu_enabled'):
            search_details.append("GPU acceleration was used for embeddings")
        if result['diagnostics'].get('8bit_enabled'):
            search_details.append("8-bit quantization was used for model efficiency")
        
        llm_context += "\n" + "; ".join(search_details) + ".\n"
        
        return llm_context
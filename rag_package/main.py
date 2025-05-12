"""
Enhanced command-line interface for the RAG system with optimized retrieval.
"""

import os
import json
import argparse
import traceback
import logging
from typing import List, Dict, Any

from rag_package import RAG
from enhanced_rag import EnhancedRAG
from text.file_parser import FileParser
from embedding.model_utils import ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RAG-Main')

def main():
    parser = argparse.ArgumentParser(description="Enhanced RAG system with optimized retrieval")
    
    # Main operation modes
    parser.add_argument("--folder", type=str, help="Folder containing files to ingest (includes all subfolders)")
    parser.add_argument("--query", type=str, help="Query to run against the RAG system")
    parser.add_argument("--list-files", action="store_true", help="List all files in the vector store")
    parser.add_argument("--get-file", type=str, help="Get all chunks for a specific file (by file_id)")
    parser.add_argument("--get-file-by-path", type=str, help="Get all chunks for a specific file (by path)")
    parser.add_argument("--reconstruct-file", type=str, help="Reconstruct a file's content from its chunks (by file_id)")
    parser.add_argument("--fix-file-ids", action="store_true", help="Attempt to fix missing file IDs in the database")

    # Chunking options
    parser.add_argument("--chunking-strategy", type=str, default="structure", 
                       choices=["simple", "semantic", "structure"],
                       help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int, default=400, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap between chunks (for simple chunking)")
    
    # Embedding and search options
    parser.add_argument("--embedding-model", type=str, default="nomic-ai/nomic-embed-code", 
                       help="Model to use for embeddings")
    parser.add_argument("--cross-encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", 
                       help="CrossEncoder model to use for reranking, or 'none' to disable")
    parser.add_argument("--disable-reranking", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--disable-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length for embedding models")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization for large models")
    
    # Enhanced retrieval options
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced retrieval features")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, 
                       help="Weight for vector search in hybrid search (0-1), 0=pure keyword, 1=pure vector")
    parser.add_argument("--no-query-expansion", action="store_true", help="Disable query expansion")
    parser.add_argument("--no-hybrid-search", action="store_true", help="Disable hybrid search")
    parser.add_argument("--no-diversity", action="store_true", help="Disable diversity-based chunk selection")
    
    # Query options
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--allow-duplicates", action="store_true", 
                       help="Allow duplicate files in results (default: results are deduplicated by source file)")
    parser.add_argument("--output-file", type=str, help="Output file to save the query results")
    
    # System options
    parser.add_argument("--db-dir", type=str, default="./chroma_db", help="Directory for ChromaDB storage")
    parser.add_argument("--reset", action="store_true", help="Reset the collection before ingestion")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for adding documents to ChromaDB")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--fix-source-fields", action="store_true",
                   help="Fix documents with missing source fields")
    parser.add_argument("--optimize-memory", action="store_true", 
                   help="Optimize memory usage when using large models")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                   default='INFO', help="Set logging level")

    args = parser.parse_args()
    
    # Set logging level based on command line argument
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Setting log level to {args.log_level}")

    # Determine if cross-encoder should be disabled
    cross_encoder_model = None if args.disable_reranking or args.cross_encoder.lower() == 'none' else args.cross_encoder
    
    # Check if we're using a large model that might need optimization
    if "nomic" in args.embedding_model.lower() or "7B" in args.embedding_model:
        logger.info(f"Using large model: {args.embedding_model}")
        if args.optimize_memory:
            logger.info("Optimizing memory usage for large model")
            ModelOptimizer.optimize_memory()
    
    # Initialize RAG system with all configured options
    if args.enhanced:
        logger.info("Using Enhanced RAG with optimized retrieval features")
        rag = EnhancedRAG(
            persist_directory=args.db_dir,
            embedding_model=args.embedding_model,
            cross_encoder_model=cross_encoder_model,
            chunking_strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            hybrid_alpha=args.hybrid_alpha,
            enable_query_expansion=not args.no_query_expansion,
            enable_hybrid_search=not args.no_hybrid_search,
            enable_diversity=not args.no_diversity,
            use_gpu=not args.disable_gpu,
            max_length=args.max_length,
            use_8bit=args.use_8bit  # Added 8-bit quantization parameter
        )
        
        # Print configuration
        logger.info(f"Enhanced retrieval configuration:")
        logger.info(f"- Query expansion: {'disabled' if args.no_query_expansion else 'enabled'}")
        logger.info(f"- Hybrid search: {'disabled' if args.no_hybrid_search else 'enabled'} (alpha={args.hybrid_alpha})")
        logger.info(f"- Diversity selection: {'disabled' if args.no_diversity else 'enabled'}")
        logger.info(f"- Reranking: {'disabled' if args.disable_reranking else 'enabled'}")
        logger.info(f"- Chunking strategy: {args.chunking_strategy}")
        logger.info(f"- GPU acceleration: {'disabled' if args.disable_gpu else 'enabled'}")
        logger.info(f"- 8-bit quantization: {'enabled' if args.use_8bit else 'disabled'}")
    else:
        rag = RAG(
            persist_directory=args.db_dir,
            embedding_model=args.embedding_model,
            cross_encoder_model=cross_encoder_model,
            chunking_strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_gpu=not args.disable_gpu,
            max_length=args.max_length,
            use_8bit=args.use_8bit  # Added 8-bit quantization parameter
        )
        logger.info(f"Using standard RAG with {args.chunking_strategy} chunking")
        logger.info(f"- 8-bit quantization: {'enabled' if args.use_8bit else 'disabled'}")

    if args.folder:
        logger.info(f"Ingesting folder with all subfolders: {args.folder}")
        
        # Standard ignored directories and extensions
        ignored_dirs = ['.git', '.github', 'node_modules', '__pycache__', '.idea', '.vscode', 'venv', 'env', '.env']
        ignored_extensions = [
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', 
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
            '.mp3', '.mp4', '.wav', '.avi', '.mov',
            '.pdf', '.pyc', '.pyo', '.pyd', '.class', '.jar'
        ]
        
        logger.info(f"Automatically ignoring standard directories: {', '.join(ignored_dirs)}")
        logger.info("Automatically ignoring binary/media extensions")
        
        # Use the enhanced file parser for ingestion
        file_parser = FileParser(
            chunking_strategy=args.chunking_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Reset the collection if requested
        if args.reset:
            logger.info("Resetting the collection before ingestion")
            rag.vector_store.clear_collection()
        
        all_documents = file_parser.ingest_folder(
            args.folder,
            max_files=args.max_files,
            ignored_dirs=ignored_dirs,
            ignored_extensions=ignored_extensions,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        # Add the documents to the vector store
        if all_documents:
            logger.info(f"Adding {len(all_documents)} documents to the vector store")
            total_added = rag.vector_store.add_documents(all_documents, args.batch_size, args.verbose)
            logger.info(f"Ingestion complete. Added {total_added} chunks from folder {args.folder}")
        else:
            logger.warning("No documents were created during ingestion")
    
    # Fix missing file IDs if requested
    if args.fix_file_ids:
        logger.info("\nAttempting to fix missing file IDs...")
        # Get all files
        all_files = rag.get_all_files()
        file_paths_to_ids = {file['source']: file['file_id'] for file in all_files}
        
        # Get all chunks with unknown file IDs
        try:
            unknown_chunks = rag.vector_store.collection.get(
                where={"file_id": {"$exists": False}}
            )
            if not unknown_chunks or not unknown_chunks["ids"]:
                logger.info("No chunks with missing file IDs found.")
            else:
                logger.info(f"Found {len(unknown_chunks['ids'])} chunks with missing file IDs")
                fixed_count = 0
                
                for i, chunk_id in enumerate(unknown_chunks["ids"]):
                    source_path = unknown_chunks["metadatas"][i].get("source")
                    if source_path and source_path in file_paths_to_ids:
                        # Update the chunk with the correct file_id
                        file_id = file_paths_to_ids[source_path]
                        metadata = unknown_chunks["metadatas"][i].copy()
                        metadata["file_id"] = file_id
                        rag.vector_store.collection.update(
                            ids=[chunk_id],
                            metadatas=[metadata]
                        )
                        fixed_count += 1
                
                logger.info(f"Fixed {fixed_count} chunks with missing file IDs")
        except Exception as e:
            logger.error(f"Error fixing file IDs: {e}")
            traceback.print_exc()

    if args.list_files:
        files = rag.get_all_files()
        logger.info("\nFiles in the vector store:")
        for i, file in enumerate(files):
            logger.info(f"{i+1}. {file['source']} (ID: {file.get('file_id', 'unknown')}, Chunks: {file.get('num_chunks', 0)})")
    
    if args.get_file:
        chunks = rag.get_file_chunks(args.get_file)
        logger.info(f"\nChunks for file ID {args.get_file}:")
        if chunks:
            for i, chunk in enumerate(chunks):
                logger.info(f"\nChunk {i+1} (Index: {chunk.metadata.get('chunk_index', 'unknown')}):")
                logger.info("-" * 50)
                logger.info(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)
                logger.info("-" * 50)
        else:
            logger.warning("No chunks found for this file ID")
    
    if args.get_file_by_path:
        chunks = rag.get_chunks_by_source(args.get_file_by_path)
        logger.info(f"\nChunks for file path {args.get_file_by_path}:")
        if chunks:
            for i, chunk in enumerate(chunks):
                logger.info(f"\nChunk {i+1} (Index: {chunk.metadata.get('chunk_index', 'unknown')}):")
                logger.info("-" * 50)
                logger.info(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)
                logger.info("-" * 50)
        else:
            logger.warning("No chunks found for this file path")
    
    if args.reconstruct_file:
        content = rag.reconstruct_file_content(args.reconstruct_file)
        logger.info(f"\nReconstructed content for file ID {args.reconstruct_file}:")
        logger.info("-" * 50)
        if content:
            # Print first 500 characters and total length
            logger.info(content[:500] + "..." if len(content) > 500 else content)
            logger.info(f"\nTotal length: {len(content)} characters")
            
            # Option to save to file
            save_path = f"reconstructed_{args.reconstruct_file}.txt"
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Full content saved to {save_path}")
        else:
            logger.warning("No content could be reconstructed for this file ID")

    if args.query:
        # Determine search options
        remove_duplicates = not args.allow_duplicates
        use_reranking = not args.disable_reranking and args.cross_encoder.lower() != 'none'
        
        logger.info(f"Executing query: {args.query}")
        
        # Get results using appropriate method
        if args.enhanced:
            results = rag.enhanced_query(
                args.query, 
                k=args.top_k,
                remove_duplicates=remove_duplicates,
                use_reranking=use_reranking
            )
        else:
            results = rag.query(
                args.query, 
                k=args.top_k, 
                remove_duplicates=remove_duplicates,
                use_reranking=use_reranking
            )
                
        logger.info("\nQuery Results:")
        logger.info(f"Query: {results['query']}")
        logger.info(f"Diagnostic Info: {json.dumps(results['diagnostics'], indent=2)}")
        
        if results['results']:
            for i, result in enumerate(results['results']):
                # Display scores based on what's available
                score_info = []
                
                if args.enhanced:
                    if result['metadata'].get('reranker_score') is not None:
                        score_info.append(f"Reranker: {result['metadata']['reranker_score']:.4f}")
                    if result['metadata'].get('vector_score') is not None:
                        score_info.append(f"Vector: {result['metadata']['vector_score']:.4f}")
                    if result['metadata'].get('keyword_score') is not None:
                        score_info.append(f"Keyword: {result['metadata']['keyword_score']:.4f}")
                    if result['metadata'].get('combined_score') is not None:
                        score_info.append(f"Combined: {result['metadata']['combined_score']:.4f}")
                else:
                    if result.get('reranker_score') is not None:
                        score_info.append(f"Reranker: {result['reranker_score']:.4f}")
                    if result.get('distance') is not None:
                        score_info.append(f"Distance: {result['distance']:.4f}")
                
                scores_display = ", ".join(score_info)
                
                logger.info(f"\nResult {i+1} ({scores_display}):")
                
                # Get metadata (location differs between enhanced and regular RAG)
                if args.enhanced:
                    metadata = result['metadata']
                    content = result['content']
                else:
                    metadata = result['metadata']
                    content = result['content']
                
                logger.info(f"Source: {metadata['source']}")
                logger.info(f"Title: {metadata['title']}")
                logger.info(f"File ID: {metadata.get('file_id', 'unknown')}")
                
                # Print entities if available
                if metadata.get('entities'):
                    entities = metadata['entities']
                    if entities:
                        logger.info(f"Entities: {', '.join(entities[:5])}" + 
                             (f" (+{len(entities)-5} more)" if len(entities) > 5 else ""))

                logger.info("-" * 50)
                # Handle code specifically if available
                if metadata.get('is_code') or metadata.get('content_type') == 'code':
                    language = metadata.get('language', '')
                    logger.info(f"```{language}")
                    logger.info(content)
                    logger.info("```")
                else:
                    logger.info(content)
                logger.info("-" * 50)
                
            # Save results to file if requested
            if args.output_file:
                output_content = f"Query: {results['query']}\n\n"
                for i, result in enumerate(results['results']):
                    if args.enhanced:
                        metadata = result['metadata']
                        content = result['content']
                    else:
                        metadata = result['metadata']
                        content = result['content']
                        
                    output_content += f"Result {i+1}:\n"
                    output_content += f"Source: {metadata['source']}\n"
                    
                    # Handle code specifically if available
                    if metadata.get('is_code') or metadata.get('content_type') == 'code':
                        language = metadata.get('language', '')
                        output_content += f"```{language}\n"
                        output_content += f"{content}\n"
                        output_content += "```\n\n"
                    else:
                        output_content += f"{content}\n\n"
                
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                logger.info(f"\nResults saved to {args.output_file}")
                
        else:
            logger.warning("\nNo results found. Check the following:")
            logger.warning("1. Make sure your database directory contains embedded documents")
            logger.warning("2. Verify that the ingestion process completed successfully")
            logger.warning("3. Try with a different query that might match your content better")
            logger.warning(f"4. Use --reset flag if you want to start fresh")


if __name__ == "__main__":
    main()
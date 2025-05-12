"""
Enhanced FileParser with robust metadata extraction and validation.
"""

import os
import uuid
import json
import traceback
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

from ..storage.document import Document
from .text_processor import TextProcessor
from ..chunking.chunking_strategy import ChunkingStrategy

# Configure logging
logger = logging.getLogger('FileParser')

class FileParser:
    """Parser for extracting content from files with enhanced metadata and robust validation."""
    
    def __init__(self, chunking_strategy: str = "semantic", 
                chunk_size: int = 500, chunk_overlap: int = 50,
                optimize_processing: bool = False,
                validate_metadata: bool = True):
        """
        Initialize the parser with chunking settings.
        
        Args:
            chunking_strategy: Strategy to use for chunking ('simple', 'semantic', or 'structure')
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks (for simple chunking)
            optimize_processing: If True, use balanced optimizations that preserve quality
            validate_metadata: If True, validate and sanitize metadata
        """
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.optimize_processing = optimize_processing
        self.validate_metadata = validate_metadata
        
        # Track processed files for deduplication
        self._processed_files = set()
        
        # Cache for common operations - helps performance without sacrificing quality
        self._encoding_cache = {}
        
        logger.info(f"Initialized FileParser with strategy={chunking_strategy}, chunk_size={chunk_size}, validate_metadata={validate_metadata}")
        
    def parse_file(self, file_path: str) -> List[Document]:
        """Parse a file into chunks with enhanced metadata and robust validation."""
        # Check if already processed to avoid duplicates
        if file_path in self._processed_files:
            logger.info(f"Skipping already processed file: {file_path}")
            return []
            
        try:
            # Add to processed files set
            self._processed_files.add(file_path)
            logger.info(f"Processing file: {file_path}")
            
            # Extract file metadata first
            file_metadata = TextProcessor.extract_file_metadata(file_path)
            
            # Validate file metadata
            if self.validate_metadata:
                file_metadata = self._validate_metadata(file_metadata, "file")
            
            # Determine if this is a code file
            is_code_file = TextProcessor.is_code_file(file_path)
            file_metadata["is_code"] = is_code_file
            
            # Try different encodings if utf-8 fails - use cached successful encodings first
            encodings = self._get_cached_encoding(file_path) or ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    # Cache successful encoding for future files with same extension
                    self._cache_encoding(file_path, encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error opening file {file_path} with encoding {encoding}: {e}")
                    continue
            
            if content is None:
                logger.error(f"Error: Could not decode file {file_path} with any of the attempted encodings")
                return []
            
            # Skip files with no content
            if not content.strip():
                logger.warning(f"Skipping empty file: {file_path}")
                return []
            
            # Clean the text based on file type
            content = TextProcessor.clean_text(content, is_code=is_code_file)
            
            # Extract title from filename if possible
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            file_metadata["title"] = title
            
            # Add relative path information
            file_metadata["filename"] = filename
            file_metadata["dirname"] = os.path.dirname(file_path)
            
            # Generate a unique file_id for this file
            file_id = str(uuid.uuid4())
            file_metadata["file_id"] = file_id
            
            # Extract content type specific metadata
            content_metadata = {}
            language = None
            
            if is_code_file:
                # Get language information for code files
                language = TextProcessor.extract_language_from_path(file_path)
                file_metadata["language"] = language
                
                # Extract code-specific metadata with reasonable limits
                if self.optimize_processing:
                    # Use a reasonable content limit that maintains quality (20k chars)
                    max_content = 20000 if len(content) > 20000 else len(content)
                    content_metadata = TextProcessor.extract_code_metadata(content[:max_content], language)
                else:
                    content_metadata = TextProcessor.extract_code_metadata(content, language)
                    
                # Validate content metadata
                if self.validate_metadata:
                    content_metadata = self._validate_metadata(content_metadata, "code")
                    
            elif file_path.lower().endswith('.md'):
                # Extract markdown-specific metadata
                if self.optimize_processing:
                    # Use a reasonable content limit that maintains quality (20k chars)
                    max_content = 20000 if len(content) > 20000 else len(content)
                    content_metadata = TextProcessor.extract_markdown_metadata(content[:max_content])
                else:
                    content_metadata = TextProcessor.extract_markdown_metadata(content)
                
                # Validate content metadata
                if self.validate_metadata:
                    content_metadata = self._validate_metadata(content_metadata, "markdown")
            
            # Extract entities based on file type
            entities = []
            if self.optimize_processing:
                # Use a reasonable content limit that maintains quality (10k chars)
                entity_limit = 10000 if len(content) > 10000 else len(content)
                try:
                    extracted_entities = TextProcessor.extract_entities(content[:entity_limit], is_code=is_code_file)
                    if extracted_entities and len(extracted_entities) > 0:
                        entities = extracted_entities
                except Exception as e:
                    logger.warning(f"Error extracting entities from {file_path}: {e}")
            else:
                try:
                    extracted_entities = TextProcessor.extract_entities(content, is_code=is_code_file)
                    if extracted_entities and len(extracted_entities) > 0:
                        entities = extracted_entities
                except Exception as e:
                    logger.warning(f"Error extracting entities from {file_path}: {e}")
                
            if entities:
                file_metadata["entities"] = entities
            
            # Choose chunking strategy
            chunks = []
            try:
                if self.chunking_strategy == "simple":
                    text_chunks = ChunkingStrategy.simple_chunking(
                        content, self.chunk_size, self.chunk_overlap
                    )
                    for chunk in text_chunks:
                        chunks.append((chunk, {}))
                elif self.chunking_strategy == "semantic":
                    text_chunks = ChunkingStrategy.semantic_chunking(
                        content, self.chunk_size
                    )
                    for chunk in text_chunks:
                        chunks.append((chunk, {}))
                elif self.chunking_strategy == "structure":
                    chunks = ChunkingStrategy.structure_aware_chunking(
                        content, file_path, self.chunk_size
                    )
                else:
                    # Default to semantic chunking to maintain quality
                    text_chunks = ChunkingStrategy.semantic_chunking(
                        content, self.chunk_size
                    )
                    for chunk in text_chunks:
                        chunks.append((chunk, {}))
            except Exception as e:
                logger.error(f"Error during chunking of {file_path}: {e}")
                # Fallback to simple chunking if other methods fail
                try:
                    text_chunks = ChunkingStrategy.simple_chunking(
                        content, self.chunk_size, self.chunk_overlap
                    )
                    for chunk in text_chunks:
                        chunks.append((chunk, {}))
                except Exception as e2:
                    logger.error(f"Fallback chunking also failed for {file_path}: {e2}")
                    return []
            
            # Create Document objects from chunks
            documents = []
            for i, (chunk_text, chunk_metadata) in enumerate(chunks):
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                # Start with file-level metadata
                metadata = dict(file_metadata)
                
                # Add chunk-specific metadata
                metadata.update({
                    "chunk_index": i,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_text_length": len(chunk_text),
                    "total_chunks": len(chunks)
                })
                
                # Add code-specific metadata from file level (only for code files)
                if is_code_file and content_metadata:
                    # Keep full metadata for important fields
                    for key in ["functions", "classes", "methods", "imports", "dependencies"]:
                        if key in content_metadata and content_metadata[key]:
                            metadata[f"file_{key}"] = content_metadata[key]
                    
                    # Add complexity estimate
                    if "complexity_estimate" in content_metadata:
                        metadata["code_complexity"] = content_metadata["complexity_estimate"]
                
                # Add markdown-specific metadata from file level (only for markdown files)
                elif file_path.lower().endswith('.md') and content_metadata:
                    # Keep full metadata for important fields
                    for key in ["headers", "has_tables", "has_images", "estimated_reading_time", "word_count"]:
                        if key in content_metadata and content_metadata[key]:
                            metadata[f"md_{key}"] = content_metadata[key]
                
                # Add chunk metadata
                if chunk_metadata:
                    metadata.update(chunk_metadata)
                
                # Extract chunk-specific entities for sufficiently large chunks
                chunk_entities = []
                if len(chunk_text) > 200:
                    if self.optimize_processing:
                        # Use reasonable limits while maintaining quality
                        chunk_limit = 5000 if len(chunk_text) > 5000 else len(chunk_text)
                        try:
                            extracted_chunk_entities = TextProcessor.extract_entities(chunk_text[:chunk_limit], is_code=is_code_file)
                            if extracted_chunk_entities and len(extracted_chunk_entities) > 0:
                                chunk_entities = extracted_chunk_entities
                        except Exception as e:
                            logger.warning(f"Error extracting chunk entities: {e}")
                    else:
                        try:
                            extracted_chunk_entities = TextProcessor.extract_entities(chunk_text, is_code=is_code_file)
                            if extracted_chunk_entities and len(extracted_chunk_entities) > 0:
                                chunk_entities = extracted_chunk_entities
                        except Exception as e:
                            logger.warning(f"Error extracting chunk entities: {e}")
                
                if chunk_entities:
                    metadata["chunk_entities"] = chunk_entities
                
                # Add code-specific metadata for code chunks
                if is_code_file and language:
                    metadata["type"] = "code"
                    metadata["language"] = language
                    
                    # Extract code structures in this specific chunk
                    if len(chunk_text) > 100:
                        chunk_code_info = {}
                        try:
                            if self.optimize_processing:
                                # Use reasonable limits while maintaining quality
                                chunk_limit = 10000 if len(chunk_text) > 10000 else len(chunk_text)
                                chunk_code_info = TextProcessor.extract_code_metadata(chunk_text[:chunk_limit], language)
                            else:
                                chunk_code_info = TextProcessor.extract_code_metadata(chunk_text, language)
                                
                            # Keep only the most important metadata to avoid bloat
                            for key in ["functions", "classes", "imports"]:
                                if key in chunk_code_info and chunk_code_info[key]:
                                    metadata[f"chunk_{key}"] = chunk_code_info[key]
                        except Exception as e:
                            logger.warning(f"Error extracting code metadata from chunk: {e}")
                
                # Add content type
                if is_code_file:
                    metadata["content_type"] = "code"
                elif file_path.lower().endswith('.md'):
                    metadata["content_type"] = "markdown"
                else:
                    metadata["content_type"] = "text"

                metadata["source"] = file_path
                
                # Perform final metadata validation
                if self.validate_metadata:
                    metadata = self._validate_metadata(metadata, "chunk")
                
                documents.append(Document(
                    content=chunk_text,
                    metadata=metadata
                ))
            
            logger.info(f"Created {len(documents)} chunks from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            traceback.print_exc()
            return []
    
    def _validate_metadata(self, metadata: Dict[str, Any], metadata_type: str = "general") -> Dict[str, Any]:
        """
        Validate and sanitize metadata to ensure it works with ChromaDB.
        
        Args:
            metadata: The metadata dictionary to validate
            metadata_type: Type of metadata for context-specific validation
            
        Returns:
            Cleaned metadata dictionary
        """
        if not metadata:
            return {}
            
        valid_metadata = {}
        
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle lists
            if isinstance(value, list):
                if not value:  # Skip empty lists
                    continue
                    
                # Keep non-empty lists
                valid_metadata[key] = value
                continue
                
            # Handle dictionaries
            if isinstance(value, dict):
                if not value:  # Skip empty dictionaries
                    continue
                    
                # Recursively validate nested dictionaries
                valid_nested = self._validate_metadata(value, f"{metadata_type}_nested")
                if valid_nested:  # Only add if there's valid nested content
                    valid_metadata[key] = valid_nested
                continue
                
            # Handle primitives (str, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                valid_metadata[key] = value
                continue
                
            # Convert other types to strings
            try:
                valid_metadata[key] = str(value)
            except:
                logger.warning(f"Could not convert metadata field '{key}' to string")
        
        return valid_metadata
    
    def ingest_folder(self, folder_path: str, max_files: int = None, 
                     ignored_dirs: Optional[List[str]] = None,
                     ignored_extensions: Optional[List[str]] = None,
                     batch_size: int = 100, verbose: bool = False,
                     concurrent_workers: int = None,
                     optimize_processing: bool = None) -> List[Document]:
        """
        Ingest all files from a folder and its subfolders using parallel processing.
        
        Args:
            folder_path: Path to the folder to ingest
            max_files: Maximum number of files to process (None for all)
            ignored_dirs: List of directory names to ignore
            ignored_extensions: List of file extensions to ignore
            batch_size: Number of files to process in each batch
            verbose: Whether to print detailed progress information
            concurrent_workers: Number of concurrent workers (None = auto-detect)
            optimize_processing: Override the instance setting for optimization
            
        Returns:
            List of all documents created
        """
        if not os.path.isdir(folder_path):
            logger.error(f"Error: {folder_path} is not a valid directory")
            return []
        
        # Override optimization setting if specified
        if optimize_processing is not None:
            orig_optimize = self.optimize_processing
            self.optimize_processing = optimize_processing
            
        # Reset processed files tracking
        self._processed_files = set()
        
        # Find all files
        all_files = self.find_all_files(
            folder_path, 
            max_files=max_files,
            ignored_dirs=ignored_dirs,
            ignored_extensions=ignored_extensions,
            verbose=verbose
        )
        
        if not all_files:
            logger.warning(f"No files found in {folder_path} that match the criteria")
            return []
        
        # Determine concurrent workers if not specified
        if concurrent_workers is None:
            # Default to half of CPU count + 1 for balanced approach
            concurrent_workers = max(1, (multiprocessing.cpu_count() // 2) + 1)
            logger.info(f"Using {concurrent_workers} concurrent workers for processing")
        
        # Process files in parallel batches
        start_time = datetime.now()
        all_documents = []
        file_count = 0
        error_count = 0
        
        # Process in batches with progress tracking
        for batch_index in range(0, len(all_files), batch_size):
            batch = all_files[batch_index:batch_index + batch_size]
            batch_docs = []
            batch_start = datetime.now()
            
            logger.info(f"Processing batch {batch_index//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}")
            logger.info(f"Files {batch_index + 1} to {min(batch_index + len(batch), len(all_files))} of {len(all_files)}")
            
            # Use thread pool for I/O bound operations
            with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                # Map returns results in the same order as inputs
                batch_results = list(executor.map(self.parse_file, batch))
                
                for file_index, documents in enumerate(batch_results):
                    file_path = batch[file_index]
                    file_count += 1
                    
                    if documents:
                        batch_docs.extend(documents)
                        logger.info(f"Added {len(documents)} chunks from file {os.path.basename(file_path)}")
                    else:
                        error_count += 1
                        logger.warning(f"No content extracted from file {os.path.basename(file_path)}")
                
            # Add batch results to overall results
            if batch_docs:
                all_documents.extend(batch_docs)
                batch_duration = (datetime.now() - batch_start).total_seconds()
                logger.info(f"Batch complete: {len(batch)} files, {len(batch_docs)} chunks in {batch_duration:.2f}s")
                logger.info(f"Average: {batch_duration/len(batch):.3f}s per file, {len(batch_docs)/len(batch):.1f} chunks per file")

        # Calculate and report overall statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Ingestion complete in {elapsed:.2f} seconds.")
        logger.info(f"Processed {file_count} files with {len(all_documents)} total chunks.")
        if error_count > 0:
            logger.warning(f"Encountered errors with {error_count} files.")
        
        if file_count > 0:
            logger.info(f"Performance: {elapsed/file_count:.3f}s per file, {len(all_documents)/file_count:.1f} chunks per file")
            logger.info(f"Throughput: {file_count/elapsed:.1f} files per second, {len(all_documents)/elapsed:.1f} chunks per second")
        
        # Restore original optimization setting if it was overridden
        if optimize_processing is not None:
            self.optimize_processing = orig_optimize
            
        return all_documents
    
    def _get_cached_encoding(self, file_path: str) -> Optional[List[str]]:
        """Get successful encoding from cache based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return self._encoding_cache.get(ext)
    
    def _cache_encoding(self, file_path: str, encoding: str):
        """Cache successful encoding for future files with same extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            self._encoding_cache[ext] = [encoding, 'utf-8', 'latin-1']
    
    @staticmethod
    def find_all_files(directory: str, max_files: int = None, 
                     ignored_dirs: Optional[List[str]] = None,
                     ignored_extensions: Optional[List[str]] = None,
                     verbose: bool = False) -> List[str]:
        """
        Find all files in a directory and its subdirectories with improved filtering.
        
        Args:
            directory: Path to the directory to scan
            max_files: Maximum number of files to process (None = all)
            ignored_dirs: List of directory names to ignore (e.g., ['.git', 'node_modules'])
            ignored_extensions: List of file extensions to ignore
            verbose: Whether to print detailed progress information
            
        Returns:
            List of file paths
        """
        if ignored_dirs is None:
            ignored_dirs = ['.git', '.github', 'node_modules', '__pycache__', '.idea', '.vscode', 'venv', 'env', '.env']
        
        if ignored_extensions is None:
            ignored_extensions = [
                # Binaries
                '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', 
                # Archives
                '.zip', '.tar', '.gz', '.rar', '.7z',
                # Images
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
                # Audio/Video
                '.mp3', '.mp4', '.wav', '.avi', '.mov',
                # Other
                '.pdf', '.pyc', '.pyo', '.pyd', '.class', '.jar'
            ]
            
        all_files = []
        skipped_count = 0
        ignored_count = 0
        start_time = datetime.now()
        
        logger.info(f"Starting directory scan of {directory}")
        logger.info(f"Ignoring directories: {', '.join(ignored_dirs)}")
        logger.debug(f"Ignoring extensions: {', '.join(ignored_extensions)}")
        
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
            
            rel_path = os.path.relpath(root, directory)
            if rel_path == '.':
                rel_path = ''
                
            if verbose and files:
                logger.debug(f"Scanning directory: {rel_path or '.'} ({len(files)} files)")
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip files with ignored extensions
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ignored_extensions:
                    ignored_count += 1
                    continue
                
                # Skip very large files (>10MB)
                try:
                    size = os.path.getsize(file_path)
                    if size > 10 * 1024 * 1024:
                        if verbose:
                            logger.warning(f"Skipping large file: {file_path} ({size / (1024 * 1024):.2f} MB)")
                        skipped_count += 1
                        continue
                except Exception as e:
                    logger.warning(f"Error checking file size: {file_path} - {e}")
                    skipped_count += 1
                    continue
                
                all_files.append(file_path)
                
                # Progress reporting
                if len(all_files) % 100 == 0 and verbose:
                    logger.info(f"Found {len(all_files)} files so far...")
                
                # Stop if we've reached the maximum number of files
                if max_files and len(all_files) >= max_files:
                    if verbose:
                        logger.info(f"Reached maximum file limit ({max_files})")
                    break
            
            if max_files and len(all_files) >= max_files:
                break
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summarize results
        logger.info(f"Found {len(all_files)} files in {elapsed:.2f} seconds")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} large files")
        if ignored_count > 0:
            logger.info(f"Ignored {ignored_count} files with excluded extensions")
        
        # Print some examples of found files
        if verbose and all_files:
            logger.debug("Example files found:")
            for file in all_files[:5]:
                logger.debug(f"- {file}")
            if len(all_files) > 5:
                logger.debug(f"... and {len(all_files)-5} more")
                
        return all_files
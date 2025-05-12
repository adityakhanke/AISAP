"""
Vector store implementation using ChromaDB.
"""
import os
import sys
import json
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.errors import NotFoundError

from .document import Document
from ..embedding.embedding_manager import EmbeddingManager
from ..embedding.chroma_embeddings import ChromaEmbeddingFunction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VectorStore')

class ChromaVectorStore:
    """ChromaDB-based vector store with improved error handling."""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                collection_name: str = "document_store",
                embedding_manager: Optional[EmbeddingManager] = None,
                force_model: bool = False):
        """
        Initialize the vector store with dimension compatibility checking.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_manager: Manager for embeddings
            force_model: If True, use the specified model even if dimensions don't match
        """
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = None
        
        # Check if collection exists to determine embedding dimension
        collection_exists = False
        try:
            collection = self.client.get_collection(name=collection_name)
            collection_exists = True
            # Try to extract embedding dimension from collection metadata
            try:
                # Get collection metadata to determine embedding dimension
                coll_metadata = collection.metadata
                if coll_metadata:
                    self.embedding_dimension = coll_metadata.get("embedding_dimension")
                    logger.info(f"Found existing collection with embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                logger.error(f"Could not retrieve collection metadata: {e}")
        except NotFoundError:
            collection_exists = False
            logger.info("Collection does not exist yet, will be created")

        # Set up embedding manager with dimension check if collection exists
        if embedding_manager is None:
            # Default embedding manager - for simplicity we directly create one here
            # In production code you might want to handle this differently
            if collection_exists and self.embedding_dimension and not force_model:
                logger.info(f"Creating embedding manager compatible with collection dimension: {self.embedding_dimension}")
                # This will automatically select a compatible model
                self.embedding_manager = EmbeddingManager(
                    check_collection_dim=self.embedding_dimension
                )
            else:
                logger.info("Creating default embedding manager")
                self.embedding_manager = EmbeddingManager()
        else:
            # Use provided embedding manager, but check compatibility
            if collection_exists and self.embedding_dimension and not force_model:
                compatible, message = embedding_manager.check_model_compatibility(self.embedding_dimension)
                if not compatible:
                    logger.warning(f"Warning: {message}")
                    logger.info(f"Creating a compatible embedding manager instead")
                    self.embedding_manager = EmbeddingManager(
                        check_collection_dim=self.embedding_dimension
                    )
                else:
                    self.embedding_manager = embedding_manager
            else:
                self.embedding_manager = embedding_manager
                
        # After setting up embedding manager, get the dimension
        if self.embedding_dimension is None:
            self.embedding_dimension = self.embedding_manager.embedding_dimension
        
        # Create or get collection with the appropriate embedding function
        self._initialize_collections()
        
    def _initialize_collections(self):
        """Initialize the document and file collections."""
        embedding_function = ChromaEmbeddingFunction(self.embedding_manager)
        
        # Create or get main collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            count = self.collection.count()
            logger.info(f"Using existing collection '{self.collection_name}' with {count} documents")
            
            # Update embedding function to ensure compatibility
            self.collection._embedding_function = embedding_function
        except NotFoundError:
            logger.info(f"Creating new collection '{self.collection_name}'")
            # Create with the custom embedding function
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={
                    "description": "Collection for document chunks", 
                    "embedding_model": self.embedding_manager.model_name,
                    "embedding_dimension": self.embedding_dimension
                }
            )
        
        # Create or get file metadata collection
        try:
            self.file_collection = self.client.get_collection(name=f"{self.collection_name}_files")
            logger.info(f"Using existing file metadata collection with {self.file_collection.count()} files")
        except NotFoundError:
            logger.info(f"Creating new file metadata collection")
            self.file_collection = self.client.create_collection(
                name=f"{self.collection_name}_files",
                metadata={"description": "Collection for file metadata"}
            )

    def add_documents(self, documents: List[Document], batch_size: int = 50, verbose: bool = False):
        """
        Add documents to the vector store with balanced batch processing that maintains quality.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
            verbose: Whether to print detailed progress information
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        import time
        start_time = time.time()
        logger.info(f"Starting to add {len(documents)} documents to vector store")

        # Pre-processing step: Group documents by file_id for efficient processing
        document_groups = {}
        file_metadata = {}

        # Group documents by file_id
        for doc in documents:
            # Ensure we have required fields
            if "chunk_id" not in doc.metadata:
                # Generate a chunk ID if missing
                import uuid
                doc.metadata["chunk_id"] = str(uuid.uuid4())
                logger.debug(f"Generated chunk_id: {doc.metadata['chunk_id']}")

            # Ensure we have a file_id
            if "file_id" not in doc.metadata:
                import uuid
                doc.metadata["file_id"] = str(uuid.uuid4())
                logger.debug(f"Generated file_id: {doc.metadata['file_id']}")

            file_id = doc.metadata["file_id"]

            # Add to appropriate group
            if file_id not in document_groups:
                document_groups[file_id] = []
                # Extract file metadata once per file
                source = doc.metadata.get("source", "unknown_source")
                file_metadata[file_id] = {
                    "source": source,
                    "title": doc.metadata.get("title", os.path.basename(source)),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "chunks": []
                }

            document_groups[file_id].append(doc)

        logger.info(f"Pre-processing {len(documents)} documents across {len(document_groups)} files")

        # Update file metadata with all chunk IDs
        for file_id, file_docs in document_groups.items():
            for doc in file_docs:
                chunk_id = doc.metadata["chunk_id"]
                file_metadata[file_id]["chunks"].append(chunk_id)

        # Process in batches with metadata caching for performance
        # Cache processed metadata to avoid redundant work
        processed_metadata_cache = {}
        total_added = 0
        error_count = 0

        # Create a flat list of documents in their original order
        all_docs = []
        for docs in document_groups.values():
            all_docs.extend(docs)

        # Process in batches
        batch_count = 0
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i+batch_size]
            batch_count += 1
            logger.info(f"Processing batch {batch_count}/{(len(all_docs) + batch_size - 1)//batch_size} with {len(batch)} documents")

            # Prepare batch data
            ids = []
            texts = []
            metadatas = []

            for doc in batch:
                try:
                    doc_id = doc.metadata["chunk_id"]

                    # Check if we already processed this document's metadata
                    if doc_id in processed_metadata_cache:
                        metadata = processed_metadata_cache[doc_id]
                    else:
                        # Process metadata once per document
                        metadata = self._prepare_metadata_for_chroma(doc.metadata.copy())

                        # Cache only if not too large to avoid memory issues
                        if sys.getsizeof(metadata) < 100000:  # ~100KB limit
                            processed_metadata_cache[doc_id] = metadata

                    # VALIDATION: Check if metadata has any None values (ChromaDB will reject these)
                    if not metadata:
                        logger.warning(f"Skipping document with ID {doc_id} - all metadata was filtered out")
                        continue

                    ids.append(doc_id)
                    texts.append(doc.content)
                    metadatas.append(metadata)

                except Exception as e:
                    logger.error(f"Error processing document for ChromaDB: {e}")
                    error_count += 1
                    if verbose:
                        traceback.print_exc()
                    continue

            # Skip this batch if no valid documents
            if not ids:
                logger.warning(f"Batch {batch_count} has no valid documents, skipping")
                continue

            # Add documents to collection
            try:
                logger.info(f"Adding batch {batch_count} with {len(ids)} documents to ChromaDB")
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                total_added += len(ids)

                if verbose or batch_count % 5 == 0:  # Log every 5 batches or if verbose
                    logger.info(f"Added batch {batch_count}: {len(ids)} documents ({total_added}/{len(documents)} total)")

            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                error_count += 1
                if verbose:
                    traceback.print_exc()

                # Try to add documents one by one to identify problematic ones
                if verbose:
                    logger.info("Attempting to add documents individually to identify problem...")
                    for idx, (doc_id, text, metadata) in enumerate(zip(ids, texts, metadatas)):
                        try:
                            self.collection.add(
                                ids=[doc_id],
                                documents=[text],
                                metadatas=[metadata]
                            )
                            logger.info(f"Successfully added document {idx+1}/{len(ids)} (ID: {doc_id})")
                        except Exception as e2:
                            logger.error(f"Error with document {idx+1}/{len(ids)} (ID: {doc_id}): {e2}")
                            # Print the problematic metadata for debugging
                            logger.error(f"Problematic metadata: {json.dumps(metadata, default=str)}")

        # Process file metadata in reasonably sized batches
        file_ids_added = 0
        file_ids = list(file_metadata.keys())
        logger.info(f"Processing metadata for {len(file_ids)} files")

        for i in range(0, len(file_ids), batch_size):
            batch_file_ids = file_ids[i:i+batch_size]
            file_batch_count = i // batch_size + 1
            logger.info(f"Processing file metadata batch {file_batch_count}/{(len(file_ids) + batch_size - 1)//batch_size}")

            for file_id in batch_file_ids:
                try:
                    metadata = file_metadata[file_id]

                    # Skip files with no valid chunks
                    if "chunks" not in metadata or not metadata["chunks"]:
                        logger.warning(f"File {file_id} has no valid chunks, skipping")
                        continue

                    # Process metadata for ChromaDB
                    metadata_copy = self._prepare_metadata_for_chroma(metadata.copy())

                    # Validate the metadata
                    if not metadata_copy:
                        logger.warning(f"Skipping file {file_id} - all metadata was filtered out")
                        continue

                    # Check if file_id already exists
                    existing = self.file_collection.get(ids=[file_id])
                    if existing and existing["ids"]:
                        # Update existing entry
                        logger.debug(f"Updating existing file metadata for {file_id}")
                        self.file_collection.update(
                            ids=[file_id],
                            metadatas=[metadata_copy]
                        )
                    else:
                        # Add new entry
                        logger.debug(f"Adding new file metadata for {file_id}")
                        self.file_collection.add(
                            ids=[file_id],
                            documents=[metadata["source"]],  # Use file path as document content
                            metadatas=[metadata_copy]
                        )

                    file_ids_added += 1

                except Exception as e:
                    logger.error(f"Error adding file metadata to ChromaDB: {e}")
                    error_count += 1
                    if verbose:
                        traceback.print_exc()

        elapsed_time = time.time() - start_time
        logger.info(f"Added {total_added} documents and {file_ids_added} file metadata entries in {elapsed_time:.2f} seconds")

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during document addition")

        logger.info(f"Document count in collection: {self.collection.count()}")
        logger.info(f"File count in collection: {self.file_collection.count()}")

        return total_added

    def _prepare_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively prepare metadata for ChromaDB by converting lists and other
        non-primitive types to JSON strings, and handling None values.

        Args:
            metadata: The metadata dictionary to process

        Returns:
            A new metadata dictionary with all lists converted to JSON strings and None values handled
        """
        processed_metadata = {}

        for key, value in metadata.items():
            # Skip None values - ChromaDB doesn't accept them
            if value is None:
                logger.warning(f"Skipping None value for metadata field '{key}'")
                continue

            # Handle lists - convert them to JSON strings
            if isinstance(value, list):
                # Skip empty lists
                if not value:
                    logger.debug(f"Skipping empty list for metadata field '{key}'")
                    continue

                # Preserve the original field name pattern for entities
                if key == "entities":
                    processed_metadata["entities_json"] = json.dumps(value)
                    # Don't keep the original list
                    continue
                else:
                    # For other lists, add a _json suffix to the key
                    processed_metadata[f"{key}_json"] = json.dumps(value)
                    # Remove the original list from metadata
                    continue

            # Handle nested dictionaries
            elif isinstance(value, dict):
                # Skip empty dictionaries
                if not value:
                    logger.debug(f"Skipping empty dictionary for metadata field '{key}'")
                    continue

                nested_result = self._prepare_metadata_for_chroma(value)
                # Only add if the nested processing returned something
                if nested_result:
                    processed_metadata[key] = nested_result

            # Handle other non-serializable types (if any)
            elif not isinstance(value, (str, int, float, bool)):
                try:
                    # Convert to string for non-standard types
                    processed_metadata[key] = str(value)
                except:
                    # If we can't convert it to a string, skip it
                    logger.warning(f"Could not convert metadata field '{key}' to a serializable format")
                    continue
            else:
                # Primitive types can be kept as is
                processed_metadata[key] = value

        return processed_metadata

    def _restore_json_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore JSON-encoded fields in metadata back to Python objects.

        Args:
            metadata: The metadata dictionary with JSON-encoded fields

        Returns:
            A metadata dictionary with JSON fields restored to Python objects
        """
        restored_metadata = metadata.copy()
        
        # Look for fields ending with _json and restore them
        json_fields = [key for key in metadata.keys() if key.endswith('_json')]
        for json_field in json_fields:
            try:
                # Get the original field name (without _json suffix)
                original_field = json_field[:-5]  # Remove the _json suffix
                
                # Special case for entities
                if json_field == "entities_json":
                    restored_metadata["entities"] = json.loads(metadata[json_field])
                else:
                    # For other fields, restore with original name
                    restored_metadata[original_field] = json.loads(metadata[json_field])
                
                # Remove the JSON field
                del restored_metadata[json_field]
            except Exception as e:
                logger.error(f"Error restoring JSON field {json_field}: {e}")
        
        return restored_metadata
    
    def similarity_search(self, query: str, k: int = 3, 
                         remove_duplicates: bool = True,
                         use_reranking: bool = True,
                         max_retries: int = 2) -> List[Document]:
        """
        Find the most similar documents to the query with improved error handling.
        
        Args:
            query: The search query
            k: Number of results to return
            remove_duplicates: If True, filters out results from the same file
            use_reranking: If True, uses cross-encoder to rerank results
            max_retries: Maximum number of retry attempts for query
        """
        for attempt in range(max_retries + 1):
            try:
                # Check if the collection is empty
                count = self.collection.count()
                if count == 0:
                    logger.warning("The collection is empty. No results can be returned.")
                    return []
                    
                logger.info(f"Running similarity search on collection with {count} documents...")
                
                # Run the query with a higher k if we'll be filtering or reranking
                # This ensures we still get k diverse results after filtering
                fetch_k = min(k * 3, count) if (remove_duplicates or use_reranking) else min(k, count)
                
                # Run the query using the embedding manager
                query_embedding = None
                if self.embedding_manager:
                    query_embedding = self.embedding_manager.get_query_embedding(query)
                    
                # Run the query
                if query_embedding:
                    logger.info(f"Using pre-computed query embedding for search")
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=fetch_k
                    )
                else:
                    logger.info(f"Using text query for search")
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=fetch_k
                    )
                
                # Debug information
                logger.info(f"Query returned {len(results['documents'][0]) if results['documents'] else 0} documents")
                if results['distances'] and results['distances'][0]:
                    logger.debug(f"Top distance scores: {results['distances'][0][:3]}")
                
                documents = []
                if results and results['documents'] and results['documents'][0]:
                    for i, content in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                        
                        # Add distance to metadata
                        metadata["distance"] = distance
                        
                        # Restore all JSON encoded fields
                        restored_metadata = self._restore_json_fields(metadata)
                        
                        documents.append(Document(content=content, metadata=restored_metadata))
                else:
                    logger.warning("No documents returned from query")
                
                # Perform reranking if enabled and we have a cross-encoder
                if use_reranking and self.embedding_manager and self.embedding_manager.cross_encoder and len(documents) > 1:
                    logger.info("Reranking results with cross-encoder...")
                    documents = self.embedding_manager.rerank(query, documents, fetch_k)
                    logger.info("Reranking complete")
                
                # Filter out duplicates from the same source file
                if remove_duplicates and documents:
                    seen_sources = set()
                    unique_documents = []
                    
                    for doc in documents:
                        source = doc.metadata.get("source", "unknown")
                        if source not in seen_sources:
                            seen_sources.add(source)
                            unique_documents.append(doc)
                            
                            # Stop if we have enough unique documents
                            if len(unique_documents) >= k:
                                break
                    
                    logger.info(f"Filtered to {len(unique_documents)} unique source documents")
                    return unique_documents[:k]
                
                return documents[:k]
            
            except Exception as e:
                logger.error(f"Error querying ChromaDB (attempt {attempt+1}/{max_retries+1}): {e}")
                traceback.print_exc()
                
                if "dimension" in str(e).lower() and attempt < max_retries:
                    logger.warning("Dimension mismatch detected. Attempting to fix...")
                    # Try to reload embedding manager with appropriate dimensions
                    try:
                        error_msg = str(e)
                        # Try to extract dimensions from error message
                        # Example: "Collection expecting embedding with dimension of 384, got 1024"
                        import re
                        match = re.search(r"dimension of (\d+)", error_msg)
                        if match:
                            expected_dim = int(match.group(1))
                            logger.info(f"Collection expects dimension: {expected_dim}")
                            # Reinitialize embedding manager with correct dimension
                            self.embedding_manager = EmbeddingManager(check_collection_dim=expected_dim)
                            logger.info(f"Reinitialized embedding manager with model: {self.embedding_manager.model_name}")
                            # Also update the embedding function
                            embedding_function = ChromaEmbeddingFunction(self.embedding_manager)
                            self.collection._embedding_function = embedding_function
                    except Exception as fix_error:
                        logger.error(f"Error fixing dimensions: {fix_error}")
                elif attempt < max_retries:
                    logger.info(f"Retrying query...")
                    
        # If we get here, all attempts failed
        return []
    
    def get_file_chunks(self, file_id: str) -> List[Document]:
        """Get all chunks for a specific file."""
        try:
            # Get the file metadata first
            file_metadata = self.file_collection.get(ids=[file_id])
            
            if not file_metadata or not file_metadata["ids"]:
                logger.warning(f"No file found with ID: {file_id}")
                return []
            
            # Get chunks from metadata - now stored as a JSON string
            chunk_ids_json = file_metadata["metadatas"][0].get("chunks_json", "[]")
            chunk_ids = json.loads(chunk_ids_json)
            
            logger.info(f"Retrieving {len(chunk_ids)} chunks for file ID: {file_id}")
            
            # Get all chunks
            chunks = self.collection.get(ids=chunk_ids)
            
            documents = []
            for i, content in enumerate(chunks["documents"]):
                metadata = chunks["metadatas"][i]
                
                # Restore all JSON encoded fields
                restored_metadata = self._restore_json_fields(metadata)
                        
                documents.append(Document(content=content, metadata=restored_metadata))
            
            # Sort by chunk_index if available, otherwise by start_char
            documents.sort(key=lambda x: x.metadata.get("chunk_index", x.metadata.get("start_char", 0)))
            
            logger.info(f"Retrieved and sorted {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting file chunks: {e}")
            traceback.print_exc()
            return []
    
    def get_chunks_by_source(self, source_path: str) -> List[Document]:
        """Get all chunks from a specific source file path."""
        try:
            logger.info(f"Retrieving chunks for source path: {source_path}")
            
            # Query using metadata filter
            results = self.collection.get(
                where={"source": source_path}
            )
            
            documents = []
            if results and results["documents"]:
                for i, content in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]
                    
                    # Restore all JSON encoded fields
                    restored_metadata = self._restore_json_fields(metadata)
                        
                    documents.append(Document(content=content, metadata=restored_metadata))
                
                # Sort by start_char to preserve original document order
                documents.sort(key=lambda x: x.metadata.get("chunk_index", 0))
                
                logger.info(f"Found {len(documents)} chunks for source: {source_path}")
            else:
                logger.warning(f"No chunks found for source: {source_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting chunks by source: {e}")
            return []
    
    def get_all_files(self) -> List[Dict]:
        """Get metadata for all files in the store."""
        try:
            logger.info("Retrieving metadata for all files")
            # Get all file metadata
            results = self.file_collection.get()
            
            files = []
            if results and results["ids"]:
                for i, file_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    metadata["file_id"] = file_id
                    
                    # Parse chunks from JSON
                    if "chunks_json" in metadata:
                        try:
                            chunks = json.loads(metadata["chunks_json"])
                            metadata["num_chunks"] = len(chunks)
                        except:
                            metadata["num_chunks"] = 0
                    else:
                        metadata["num_chunks"] = 0
                        
                    files.append(metadata)
                
                logger.info(f"Found {len(files)} files in the store")
            else:
                logger.warning("No files found in the store")
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting all files: {e}")
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            logger.info("Clearing all documents from collections")
            self.collection.delete(where={})
            self.file_collection.delete(where={})
            logger.info("Collections cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
            
    def get_collection_info(self):
        """Get information about the collection with more details."""
        try:
            count = self.collection.count()
            file_count = self.file_collection.count()
            
            model_name = "unknown"
            model_dimension = None
            if self.embedding_manager:
                model_name = self.embedding_manager.model_name
                model_dimension = self.embedding_manager.embedding_dimension
            
            collection_metadata = {}
            try:
                collection_metadata = self.collection.metadata or {}
            except:
                pass
                
            logger.info(f"Collection stats - Documents: {count}, Files: {file_count}, Model: {model_name}")
            
            return {
                "count": count,
                "file_count": file_count,
                "directory": self.persist_directory,
                "embedding_model": model_name,
                "embedding_dimension": model_dimension,
                "collection_metadata": collection_metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
"""
Enhanced RAG implementation with optimized retrieval for LLMs.
"""

import os
import time
import numpy as np
import logging
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass

from .storage.document import Document
from .text.text_processor import TextProcessor
from .rag import RAG  # Import the base RAG class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnhancedRAG')

# Try to import BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 package not available. Hybrid search will use a simplified keyword method.")


@dataclass
class SearchResult:
    """Class to hold search result info for easier manipulation."""
    document: Document
    vector_score: float = 0.0
    keyword_score: float = 0.0
    reranker_score: float = 0.0
    combined_score: float = 0.0


class EnhancedRAG(RAG):
    """Enhanced RAG system with optimized retrieval strategies for LLMs."""
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                embedding_model: str = "nomic-ai/nomic-embed-code",
                cross_encoder_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                chunking_strategy: str = "semantic",
                chunk_size: int = 500,
                chunk_overlap: int = 50,
                force_model: bool = False,
                hybrid_alpha: float = 0.7,  # Weight for vector search in hybrid (0-1)
                enable_query_expansion: bool = True,
                enable_hybrid_search: bool = True,
                enable_diversity: bool = True,
                use_gpu: bool = True,
                max_length: int = 512,
                use_8bit: bool = False):  # Added 8-bit parameter
        """
        Initialize the Enhanced RAG system with retrieval optimizations.
        
        Args:
            persist_directory: Directory for ChromaDB storage
            embedding_model: Name of the embedding model to use
            cross_encoder_model: Name of the cross-encoder model for reranking (or None to disable)
            chunking_strategy: Strategy for chunking documents ('simple', 'semantic', or 'structure')
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks (for simple chunking)
            force_model: If True, use the specified model even if dimensions don't match
            hybrid_alpha: Weight for vector search in hybrid (0-1), where 0=pure keyword, 1=pure vector
            enable_query_expansion: Whether to use query expansion
            enable_hybrid_search: Whether to use hybrid search
            enable_diversity: Whether to use diversity-based chunk selection
            use_gpu: If True and GPU is available, use GPU acceleration
            max_length: Maximum token length for large models
            use_8bit: If True, use 8-bit quantization for large models
        """
        # Initialize the parent class
        logger.info(f"Initializing Enhanced RAG with {embedding_model} model")
        super().__init__(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_model=force_model,
            use_gpu=use_gpu,
            max_length=max_length,
            use_8bit=use_8bit  # Pass 8-bit parameter to parent
        )
        
        # Set additional parameters
        self.hybrid_alpha = hybrid_alpha
        self.enable_query_expansion = enable_query_expansion
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_diversity = enable_diversity
        
        logger.info(f"Enhanced RAG configuration:")
        logger.info(f"- Hybrid alpha: {hybrid_alpha}")
        logger.info(f"- Query expansion: {'enabled' if enable_query_expansion else 'disabled'}")
        logger.info(f"- Hybrid search: {'enabled' if enable_hybrid_search else 'disabled'}")
        logger.info(f"- Diversity: {'enabled' if enable_diversity else 'disabled'}")
        logger.info(f"- 8-bit quantization: {'enabled' if use_8bit else 'disabled'}")
        
        # Load stop words for query expansion
        self.stop_words = set()
        try:
            import nltk
            try:
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('english'))
            except:
                try:
                    nltk.download('stopwords')
                    from nltk.corpus import stopwords
                    self.stop_words = set(stopwords.words('english'))
                except:
                    logger.warning("Unable to load NLTK stopwords. Using a basic set.")
                    self.stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "from"}
        except ImportError:
            logger.warning("NLTK not available. Using a basic set of stopwords.")
            self.stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "from"}
        
        # Initialize cache for frequent operations
        self.cache = {
            "synonyms": {},       # Cache for synonyms
            "keyword_index": None, # Cache for BM25 index
            "document_lengths": {}  # Cache for document lengths (used in diversity calculation)
        }
    
    def get_synonyms(self, term: str, max_synonyms: int = 3) -> List[str]:
        """Get synonyms for a term with caching."""
        # Check cache first
        if term in self.cache["synonyms"]:
            logger.debug(f"Using cached synonyms for '{term}'")
            return self.cache["synonyms"][term][:max_synonyms]
        
        synonyms = []
        # Try using NLTK's WordNet
        try:
            from nltk.corpus import wordnet
            for syn in wordnet.synsets(term)[:2]:  # Limit to 2 synsets
                for lemma in syn.lemmas()[:2]:     # Limit to 2 lemmas per synset
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != term.lower():
                        synonyms.append(synonym)
        except:
            # Fallback to TextProcessor if available
            if hasattr(TextProcessor, "expand_query"):
                expanded = TextProcessor.expand_query(term)
                synonyms = [word for word in expanded.split() if word.lower() != term.lower()]
        
        # Limit and cache results
        result = list(set(synonyms))[:max_synonyms]
        self.cache["synonyms"][term] = result
        logger.debug(f"Found synonyms for '{term}': {result}")
        return result
    
    def structured_llm_context(self, query: str, results: List[Dict]) -> str:
        """Create a structured context that's easier for the LLM to process."""
        context = f"Query: {query}\n\n"
        
        # Group chunks by source document
        doc_groups = defaultdict(list)
        for result in results:
            source = result["metadata"].get("source", "unknown")
            doc_groups[source].append(result)
        
        logger.info(f"Creating structured context from {len(results)} chunks across {len(doc_groups)} documents")
        
        # Present chunks grouped by document
        for i, (source, chunks) in enumerate(doc_groups.items()):
            # Sort chunks by position in document
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            
            # Add document header with relevance
            title = chunks[0]["metadata"].get("title", os.path.basename(source))
            top_score = max(chunk["metadata"].get("combined_score", 0) for chunk in chunks)
            context += f"Document {i+1}: {title} (Relevance: {top_score:.2f})\n"
            context += f"Source: {source}\n\n"
            
            # Add each chunk with its relevant section header
            for j, chunk in enumerate(chunks):
                # Add section header if available
                header = chunk["metadata"].get("header", "")
                if header:
                    context += f"Section: {header}\n"
                
                # Add chunk content
                context += f"{chunk['content']}\n\n"
                
                # Add separator between chunks from same document
                if j < len(chunks) - 1:
                    context += "---\n\n"
            
            # Add separator between documents
            if i < len(doc_groups) - 1:
                context += "=".ljust(50, "=") + "\n\n"
        
        return context
    
    def extract_document_for_llm(self, query_text: str, k: int = 5, 
                               remove_duplicates: bool = True,
                               use_reranking: bool = True) -> str:
        """Enhanced version of document extraction for LLM consumption."""
        # Get enhanced search results
        logger.info(f"Extracting documents for LLM with query: {query_text}")
        result = self.enhanced_query(query_text, k, remove_duplicates, use_reranking)
        
        # Create structured context for LLM
        llm_context = self.structured_llm_context(query_text, result["results"])
        
        # Add diagnostic information
        diag = result["diagnostics"]
        llm_context += f"\nBased on {diag['results_returned']} documents retrieved from "
        llm_context += f"a collection of {diag['collection_size']} documents across "
        llm_context += f"{diag['file_count']} files.\n"
        
        # Add details about the search approach
        search_details = []
        if diag.get('embedding_model'):
            search_details.append(f"Primary embedding model: {diag['embedding_model']}")
        if diag.get('query_expansion'):
            search_details.append("Query expansion was applied")
        if diag.get('hybrid_search'):
            vector_weight = diag.get('hybrid_alpha', 0.7) * 100
            keyword_weight = (1 - diag.get('hybrid_alpha', 0.7)) * 100
            search_details.append(f"Hybrid search with {vector_weight:.0f}% vector and {keyword_weight:.0f}% keyword weighting")
        if diag.get('diversity_enabled'):
            search_details.append("Diversity-based selection was applied")
        if diag.get('cross_encoder') and diag['cross_encoder'] != 'disabled':
            search_details.append(f"Results reranked using {diag['cross_encoder']}")
        if diag.get('unique_sources'):
            search_details.append("Results are from unique source files")
        if diag.get('gpu_enabled'):
            search_details.append("GPU acceleration was used for embeddings")
        if diag.get('8bit_enabled'):
            search_details.append("8-bit quantization was used for model efficiency")
        
        search_timing = f"Total retrieval time: {diag.get('total_time', 0):.2f} seconds"
        
        llm_context += "\n" + "; ".join(search_details) + ".\n"
        llm_context += search_timing + "\n"
        
        return llm_context
    
    def expand_query(self, query: str) -> str:
        """Simple query expansion with keywords and synonyms."""
        if not self.enable_query_expansion:
            return query
            
        logger.info(f"Expanding query: '{query}'")
        
        # Extract key terms (exclude stop words and short words)
        key_terms = [word for word in query.lower().split() 
                   if word not in self.stop_words and len(word) > 3]
        
        # Get synonyms for key terms
        expanded_terms = set()
        for term in key_terms[:3]:  # Limit to top 3 terms for latency
            for syn in self.get_synonyms(term, max_synonyms=2):
                expanded_terms.add(syn)
        
        # Create expanded query
        expanded_query = query
        if expanded_terms:
            # Append synonyms to original query
            expanded_query += " " + " ".join(expanded_terms)
            logger.info(f"Expanded query: '{expanded_query}'")
        
        return expanded_query
    
    def keyword_search(self, query: str, documents: List[Document], k: int = 10) -> List[SearchResult]:
        """
        Perform keyword-based search on documents.
        Uses BM25 if available or falls back to TF-IDF.
        """
        if not documents:
            logger.warning("No documents provided for keyword search")
            return []

        logger.info(f"Performing keyword search on {len(documents)} documents")
        results = []
        doc_texts = [doc.content for doc in documents]
        query_terms = query.lower().split()

        if BM25_AVAILABLE:
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in doc_texts]

            # Create or update BM25 index
            bm25 = BM25Okapi(tokenized_docs)

            # Get scores
            scores = bm25.get_scores(query_terms)
            logger.debug(f"BM25 scores range: min={min(scores) if len(scores) > 0 else 'N/A'}, max={max(scores) if len(scores) > 0 else 'N/A'}")

            # Create results with normalized scores
            if len(scores) > 0:
                # Use numpy's max for array handling if scores is a numpy array
                if isinstance(scores, np.ndarray):
                    max_score = np.max(scores) if scores.size > 0 else 1.0
                else:
                    max_score = max(scores) if scores else 1.0

                for i, score in enumerate(scores):
                    normalized_score = score / max_score if max_score > 0 else 0
                    results.append(SearchResult(
                        document=documents[i],
                        keyword_score=normalized_score
                    ))
            else:
                # Fallback if no scores are returned
                for i, _ in enumerate(documents):
                    results.append(SearchResult(
                        document=documents[i],
                        keyword_score=0.0
                    ))
        else:
            logger.info("Using fallback TF-IDF-like scoring (BM25 not available)")
            # Simple TF-IDF-like scoring
            scores = []
            for doc in doc_texts:
                doc_lowercase = doc.lower()
                score = 0
                for term in query_terms:
                    if term in self.stop_words:
                        continue
                    # Count term occurrences and weight by inverse document frequency
                    term_count = doc_lowercase.count(term)
                    term_in_docs = sum(1 for d in doc_texts if term in d.lower())
                    idf = np.log(len(doc_texts) / (1 + term_in_docs))
                    score += term_count * idf
                scores.append(score)

            # Normalize scores safely
            if len(scores) > 0:
                # Use numpy's max for array handling if scores is a numpy array
                if isinstance(scores, np.ndarray):
                    max_score = np.max(scores) if scores.size > 0 else 1.0
                else:
                    max_score = max(scores) if scores else 1.0

                for i, score in enumerate(scores):
                    normalized_score = score / max_score if max_score > 0 else 0
                    results.append(SearchResult(
                        document=documents[i],
                        keyword_score=normalized_score
                    ))
            else:
                # Fallback if no scores are returned
                for i, _ in enumerate(documents):
                    results.append(SearchResult(
                        document=documents[i],
                        keyword_score=0.0
                    ))

        # Sort by keyword score
        results.sort(key=lambda x: x.keyword_score, reverse=True)
        logger.info(f"Keyword search returned {len(results)} ranked documents")
        return results[:k]

    def hybrid_search(self, query: str, k: int = 5, max_retries: int = 2) -> List[SearchResult]:
        """
        Combine vector search with keyword search for better results.
        Alpha controls the balance: 0 = pure keyword, 1 = pure vector search
        """
        # Only use hybrid if enabled, otherwise fall back to vector search
        if not self.enable_hybrid_search:
            logger.info("Using vector search only (hybrid search disabled)")
            vector_results = self.vector_store.similarity_search(query, k=k, max_retries=max_retries)
            return [SearchResult(document=doc, vector_score=1-doc.metadata.get("distance", 0))
                   for doc in vector_results]

        # Use a larger k for the initial retrieval to ensure diversity
        retrieval_k = min(k * 3, 30)  # Limit to 30 to avoid excessive computation
        logger.info(f"Performing hybrid search with alpha={self.hybrid_alpha}, retrieving {retrieval_k} initial results")

        # Get vector search results first
        vector_results = self.vector_store.similarity_search(
            query, k=retrieval_k, remove_duplicates=False, use_reranking=False, max_retries=max_retries
        )
        
        if not vector_results:
            logger.warning("Vector search returned no results")
            return []
        
        # Initialize results dict with vector search results
        results_dict = {}
        for doc in vector_results:
            doc_id = doc.metadata.get("chunk_id")
            distance = doc.metadata.get("distance", 0.5)
            vector_score = 1 - distance  # Convert distance to similarity
            
            results_dict[doc_id] = SearchResult(
                document=doc,
                vector_score=vector_score,
                combined_score=self.hybrid_alpha * vector_score
            )
        
        # Perform keyword search on the results from vector search
        logger.info(f"Performing keyword search on {len(vector_results)} vector search results")
        keyword_results = self.keyword_search(query, vector_results, k=retrieval_k)
        
        # Combine scores
        for result in keyword_results:
            doc_id = result.document.metadata.get("chunk_id")
            if doc_id in results_dict:
                # Update existing entry
                results_dict[doc_id].keyword_score = result.keyword_score
                results_dict[doc_id].combined_score = (
                    self.hybrid_alpha * results_dict[doc_id].vector_score + 
                    (1 - self.hybrid_alpha) * result.keyword_score
                )
            else:
                # This shouldn't happen as keyword search is done on vector results,
                # but just in case
                result.combined_score = (1 - self.hybrid_alpha) * result.keyword_score
                results_dict[doc_id] = result
        
        # Sort by combined score
        results = list(results_dict.values())
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.info(f"Hybrid search returned {len(results)} ranked documents")
        return results[:k*2]  # Return top k*2 for diversity selection
    
    def calculate_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate content similarity between two documents."""
        # Use cached embeddings if available
        emb1 = doc1.metadata.get("embedding")
        emb2 = doc2.metadata.get("embedding")
        
        if emb1 is not None and emb2 is not None:
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
            return similarity
        
        # Fallback to simple content similarity
        text1 = doc1.content.lower()
        text2 = doc2.content.lower()
        
        # Get document lengths (with caching)
        doc1_id = doc1.metadata.get("chunk_id", "unknown")
        doc2_id = doc2.metadata.get("chunk_id", "unknown")
        
        if doc1_id not in self.cache["document_lengths"]:
            self.cache["document_lengths"][doc1_id] = set(text1.split())
        if doc2_id not in self.cache["document_lengths"]:
            self.cache["document_lengths"][doc2_id] = set(text2.split())
        
        words1 = self.cache["document_lengths"][doc1_id]
        words2 = self.cache["document_lengths"][doc2_id]
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0
        return intersection / union
    
    def diversity_score(self, candidate: SearchResult, selected: List[SearchResult]) -> float:
        """Calculate how diverse a candidate is compared to already selected documents."""
        if not selected:
            return 1.0
        
        # Calculate average similarity to selected documents
        similarities = [self.calculate_similarity(candidate.document, s.document) for s in selected]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Convert to diversity score (1 - similarity)
        diversity = 1 - avg_similarity
        
        # Weight by the combined score to balance relevance and diversity
        return diversity * candidate.combined_score
    
    def select_diverse_chunks(self, candidates: List[SearchResult], k: int = 5) -> List[SearchResult]:
        """Select diverse chunks to avoid redundancy."""
        if not self.enable_diversity or len(candidates) <= k:
            logger.info(f"Returning top {min(k, len(candidates))} chunks (diversity disabled or not enough candidates)")
            return candidates[:k]
        
        logger.info(f"Selecting {k} diverse chunks from {len(candidates)} candidates")
        
        # Start with the highest-ranked chunk
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # Iteratively add the most diverse chunk
        while len(selected) < k and remaining:
            next_chunk = max(remaining, key=lambda x: self.diversity_score(x, selected))
            selected.append(next_chunk)
            remaining.remove(next_chunk)
        
        logger.info(f"Selected {len(selected)} diverse chunks")
        return selected
    
    def enhanced_query(self, query_text: str, k: int = 5, remove_duplicates: bool = True, use_reranking: bool = True) -> Dict:
        """Enhanced query with optimized retrieval techniques.
        
        Args:
            query_text: The search query
            k: Number of results to return
            remove_duplicates: If True, filters out results from the same file
            use_reranking: If True, uses cross-encoder to rerank results
        """
        start_time = time.time()
        retrieval_times = {}
        logger.info(f"Running enhanced query: '{query_text}'")
        
        # Expand query if enabled
        if self.enable_query_expansion:
            expanded_query = self.expand_query(query_text)
            logger.info(f"Original query: '{query_text}'")
            logger.info(f"Expanded query: '{expanded_query}'")
            query_for_search = expanded_query
        else:
            query_for_search = query_text
        
        # Time the retrieval step
        retrieval_start = time.time()
        
        # Do hybrid search
        results = self.hybrid_search(query_for_search, k=k*2)
        
        retrieval_times["retrieval"] = time.time() - retrieval_start
        logger.info(f"Retrieved {len(results)} initial results in {retrieval_times['retrieval']:.3f}s")
        
        # Apply reranking if requested
        if use_reranking:
            rerank_start = time.time()
            results = self.optimized_reranking(query_text, results, k=k*2 if self.enable_diversity else k)
            retrieval_times["reranking"] = time.time() - rerank_start
            logger.info(f"Reranked results in {retrieval_times['reranking']:.3f}s")
        
        # Select diverse chunks if enabled
        if self.enable_diversity:
            diversity_start = time.time()
            results = self.select_diverse_chunks(results, k=k)
            retrieval_times["diversity"] = time.time() - diversity_start
            logger.info(f"Selected diverse chunks in {retrieval_times['diversity']:.3f}s")
        
        # Filter for unique sources if requested
        if remove_duplicates:
            dedup_start = time.time()
            seen_sources = set()
            filtered_results = []
            
            for result in results:
                source = result.document.metadata.get("source", "unknown")
                if source not in seen_sources:
                    seen_sources.add(source)
                    filtered_results.append(result)
                    
                    # Stop if we have enough unique documents
                    if len(filtered_results) >= k:
                        break
            
            retrieval_times["deduplication"] = time.time() - dedup_start
            results = filtered_results
            logger.info(f"Deduplicated to {len(results)} results in {retrieval_times['deduplication']:.3f}s")
        else:
            # Limit to k results
            results = results[:k]
        
        # Prepare final result dict
        result = {
            "query": query_text,
            "results": []
        }
        
        # Convert from SearchResult to dict for serialization
        for search_result in results:
            doc = search_result.document
            
            # Create a serializable version of metadata
            metadata_copy = doc.metadata.copy()
            
            # Handle entities list for JSON serialization
            if "entities" in metadata_copy and isinstance(metadata_copy["entities"], list):
                metadata_copy["entities"] = metadata_copy["entities"]
            
            # Add scores to metadata
            metadata_copy["vector_score"] = search_result.vector_score
            metadata_copy["keyword_score"] = search_result.keyword_score
            metadata_copy["reranker_score"] = search_result.reranker_score
            metadata_copy["combined_score"] = search_result.combined_score
            
            result["results"].append({
                "content": doc.content,
                "metadata": metadata_copy
            })
        
        # Get collection info
        collection_info = self.vector_store.get_collection_info()
        
        # Include diagnostic info
        total_time = time.time() - start_time
        result["diagnostics"] = {
            "collection_size": collection_info['count'],
            "file_count": collection_info.get('file_count', 0),
            "results_returned": len(results),
            "embedding_model": self.embedding_manager.model_name,
            "query_expansion": self.enable_query_expansion,
            "hybrid_search": self.enable_hybrid_search,
            "hybrid_alpha": self.hybrid_alpha,
            "diversity_enabled": self.enable_diversity,
            "cross_encoder": self.embedding_manager.cross_encoder_name if use_reranking else "disabled",
            "unique_sources": True if remove_duplicates else False,
            "gpu_enabled": hasattr(self.embedding_manager, 'use_gpu') and self.embedding_manager.use_gpu,
            "8bit_enabled": hasattr(self.embedding_manager, 'use_8bit') and self.embedding_manager.use_8bit,
            "total_time": total_time,
            "retrieval_times": retrieval_times
        }
        
        logger.info(f"Enhanced query completed in {total_time:.2f}s, returning {len(result['results'])} results")
        return result
    
    def optimized_reranking(self, query: str, results: List[SearchResult], k: int = 5) -> List[SearchResult]:
        """Optimized reranking that only processes a filtered subset."""
        if not self.embedding_manager.cross_encoder or len(results) <= k:
            return results[:k]
        
        # Limit to top candidates to reduce computation
        max_to_rerank = min(k * 2, len(results))
        candidates = results[:max_to_rerank]
        
        logger.info(f"Reranking {len(candidates)} candidates")
        
        # Apply full reranking on this smaller set
        pairs = [(query, result.document.content) for result in candidates]
        scores = self.embedding_manager.cross_encoder.predict(pairs)
        
        for idx, score in enumerate(scores):
            candidates[idx].reranker_score = float(score)
            # Update the combined score with the reranker score (higher weight)
            candidates[idx].combined_score = float(score)
        
        # Sort by reranker score
        candidates.sort(key=lambda x: x.reranker_score, reverse=True)
        return candidates[:k]
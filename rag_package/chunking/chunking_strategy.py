"""
Chunking strategies for breaking documents into semantic units.
"""

import os
import re
from typing import List, Tuple, Dict, Any

# Try to import NLTK for better sentence splitting
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class ChunkingStrategy:
    """Class for different document chunking strategies."""
    
    @staticmethod
    def simple_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Simple chunking by character count with overlap.
        """
        chunks = []
        content_length = len(text)
        for i in range(0, content_length, chunk_size - chunk_overlap):
            chunk_end = min(i + chunk_size, content_length)
            chunk = text[i:chunk_end]
            # Skip empty chunks
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    @staticmethod
    def semantic_chunking(text: str, max_chunk_size: int = 500, min_chunk_size: int = 100) -> List[str]:
        """
        Break text into semantically coherent chunks, respecting paragraph boundaries.
        """
        if not text:
            return []
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is too long, split it by sentences
            if len(paragraph) > max_chunk_size:
                if NLTK_AVAILABLE:
                    sentences = sent_tokenize(paragraph)
                    sentence_chunks = ChunkingStrategy._process_sentences(sentences, max_chunk_size)
                    chunks.extend(sentence_chunks)
                else:
                    # Fall back to simple chunking if NLTK is not available
                    for i in range(0, len(paragraph), max_chunk_size - 50):
                        chunk = paragraph[i:i + max_chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)
            else:
                # If adding this paragraph exceeds max size, save current chunk and start new one
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    joiner = "\n\n" if current_chunk else ""
                    current_chunk += joiner + paragraph
        
        # Add the final chunk if it's not empty
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
        
        return chunks
    
    @staticmethod
    def _process_sentences(sentences: List[str], max_chunk_size: int) -> List[str]:
        """Helper method to process sentences into chunks."""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If this sentence alone exceeds max size, split it further
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split long sentence
                words = sentence.split()
                current_sentence = ""
                for word in words:
                    if len(current_sentence) + len(word) + 1 > max_chunk_size:
                        chunks.append(current_sentence.strip())
                        current_sentence = word
                    else:
                        joiner = " " if current_sentence else ""
                        current_sentence += joiner + word
                
                if current_sentence:
                    current_chunk = current_sentence
            else:
                # If adding this sentence exceeds max size, save current chunk and start new one
                if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    joiner = " " if current_chunk else ""
                    current_chunk += joiner + sentence
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    @staticmethod
    def structure_aware_chunking(text: str, file_path: str, max_chunk_size: int = 500) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunking that respects document structure like headers.
        Returns a list of (chunk, metadata) tuples.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        chunks_with_metadata = []
        
        if file_ext == '.md':
            # Process Markdown with header awareness
            chunks_with_metadata = ChunkingStrategy._chunk_markdown(text, max_chunk_size)
        elif file_ext == '.html' or file_ext == '.htm':
            # Process HTML with tag structure awareness
            chunks_with_metadata = ChunkingStrategy._chunk_html(text, max_chunk_size)
        else:
            # Default to semantic chunking for other file types
            chunks = ChunkingStrategy.semantic_chunking(text, max_chunk_size)
            for chunk in chunks:
                chunks_with_metadata.append((chunk, {}))
        
        return chunks_with_metadata
    
    @staticmethod
    def _chunk_markdown(text: str, max_chunk_size: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Process Markdown text into chunks with header information."""
        # Find all headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headers = list(header_pattern.finditer(text))
        
        chunks_with_metadata = []
        
        # If no headers found, use semantic chunking
        if not headers:
            chunks = ChunkingStrategy.semantic_chunking(text, max_chunk_size)
            for chunk in chunks:
                chunks_with_metadata.append((chunk, {}))
            return chunks_with_metadata
        
        # Process each section (from one header to the next)
        for i in range(len(headers)):
            header_match = headers[i]
            header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            section_start = header_match.start()
            section_end = headers[i+1].start() if i < len(headers) - 1 else len(text)
            section_text = text[section_start:section_end]
            
            # Create metadata for this section
            section_metadata = {
                "header": header_text,
                "header_level": header_level
            }
            
            # If section is small enough, keep it as one chunk
            if len(section_text) <= max_chunk_size:
                chunks_with_metadata.append((section_text, section_metadata))
            else:
                # Otherwise, split into smaller chunks while preserving the header info
                sub_chunks = ChunkingStrategy.semantic_chunking(section_text, max_chunk_size)
                for chunk in sub_chunks:
                    chunks_with_metadata.append((chunk, section_metadata))
        
        return chunks_with_metadata
    
    @staticmethod
    def _chunk_html(text: str, max_chunk_size: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Process HTML text into chunks with structure awareness."""
        # This is a simplified version - a real implementation would use an HTML parser
        # First, try to extract title
        title_match = re.search(r'<title>(.+?)</title>', text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else None
        
        # Find heading tags
        heading_pattern = re.compile(r'<h[1-6][^>]*>(.+?)</h[1-6]>', re.IGNORECASE | re.DOTALL)
        headings = list(heading_pattern.finditer(text))
        
        chunks_with_metadata = []
        
        # If no structure found, strip tags and use semantic chunking
        if not headings:
            # Simple tag stripping (not perfect but works for basic HTML)
            plain_text = re.sub(r'<[^>]+>', ' ', text)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            
            chunks = ChunkingStrategy.semantic_chunking(plain_text, max_chunk_size)
            for chunk in chunks:
                metadata = {"title": title} if title else {}
                chunks_with_metadata.append((chunk, metadata))
            return chunks_with_metadata
        
        # Process content between headings
        for i in range(len(headings)):
            heading_match = headings[i]
            heading_text = re.sub(r'<[^>]+>', '', heading_match.group(1)).strip()
            
            section_start = heading_match.start()
            section_end = headings[i+1].start() if i < len(headings) - 1 else len(text)
            section_html = text[section_start:section_end]
            
            # Strip HTML tags (simple approach)
            section_text = re.sub(r'<[^>]+>', ' ', section_html)
            section_text = re.sub(r'\s+', ' ', section_text).strip()
            
            # Create metadata
            section_metadata = {
                "heading": heading_text,
                "title": title
            }
            
            # Split if necessary
            if len(section_text) <= max_chunk_size:
                chunks_with_metadata.append((section_text, section_metadata))
            else:
                sub_chunks = ChunkingStrategy.semantic_chunking(section_text, max_chunk_size)
                for chunk in sub_chunks:
                    chunks_with_metadata.append((chunk, section_metadata))
        
        return chunks_with_metadata
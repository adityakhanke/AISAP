# Enhanced RAG (Retrieval-Augmented Generation) System

An optimized implementation of RAG with advanced chunking strategies, specialized embeddings for code, hybrid search, query expansion, and diversity-based retrieval.

## Features

- **Multiple Embedding Models**: Supports various embedding models with automatic compatibility detection
  - Default: `nomic-ai/nomic-embed-code` (optimized for both text and code)
  - Compatible with any Hugging Face or SentenceTransformer model
- **Advanced Chunking Strategies**:
  - Semantic: Creates chunks based on natural language boundaries
  - Structure-aware: Respects document structures like Markdown headers or code blocks
  - Simple: Traditional character-based chunking with overlap
- **Enhanced Retrieval**:
  - Hybrid search combining vector similarity with keyword (BM25) search
  - Query expansion with synonyms and domain-specific terms
  - Diversity-based result selection to reduce redundancy
  - Cross-encoder reranking for precision
- **Performance Optimizations**:
  - 8-bit quantization for large models
  - Automatic GPU/CPU device mapping
  - Batched processing with reasonable defaults
  - Caching for repeated operations
- **Rich Metadata Extraction**:
  - Code-specific: functions, classes, imports, complexity metrics
  - Content-aware: headings, entities, document structure
  - Hierarchical storage for efficient retrieval

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Required packages (install via `pip`):

```bash
pip install --upgrade transformers>=4.30.0
pip install --upgrade torch>=2.0.0
pip install --upgrade accelerate>=0.19.0
pip install sentence-transformers
pip install chromadb
pip install rank_bm25
pip install nltk
pip install spacy
```

Optionally, download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Using the setup script

The included setup script will install all required dependencies:

```bash
./script.sh
```

## Quick Start

### 1. Ingest documents

```bash
python main.py --enhanced --embedding-model nomic-ai/nomic-embed-code \
  --max-length 1024 --batch-size 16 --folder "./docs" --reset \
  --db-dir "./chroma_db"
```

### 2. Query the system

```bash
python main.py --enhanced --embedding-model nomic-ai/nomic-embed-code \
  --query "How does the hybrid search algorithm work?" --top-k 5 \
  --db-dir "./chroma_db"
```

## Core Components

### `RAG` Class (Base Implementation)

The foundation of the system that handles:
- Document ingestion and storage
- Basic similarity search and querying
- Cross-encoder reranking
- File management and reconstruction

### `EnhancedRAG` Class (Advanced Features)

Extends the base RAG implementation with:
- Hybrid search that combines vector and keyword matching
- Query expansion with contextual synonyms
- Diversity-based chunk selection
- Structured context for LLM consumption
- Diagnostic information and performance metrics

### Storage

- `ChromaVectorStore`: Persistent vector database with error handling
- `Document`: Data class for text chunks and associated metadata

### Embedding

- `EmbeddingManager`: Handles embedding generation with model compatibility checking
- Supports GPU acceleration and 8-bit quantization for large models

### Text Processing

- `TextProcessor`: Utilities for text cleaning, entity extraction, and metadata processing
- `FileParser`: Robust file handling with metadata extraction and validation
- `ChunkingStrategy`: Document segmentation strategies optimized for different content types

## Command Line Options

### Main Operations

- `--folder PATH`: Folder containing files to ingest (includes all subfolders)
- `--query TEXT`: Query to run against the RAG system
- `--list-files`: List all files in the vector store
- `--get-file ID`: Get all chunks for a specific file (by file_id)
- `--get-file-by-path PATH`: Get all chunks for a specific file (by path)
- `--reconstruct-file ID`: Reconstruct a file's content from its chunks

### Enhanced Retrieval Options

- `--enhanced`: Use enhanced retrieval features
- `--hybrid-alpha FLOAT`: Weight for vector search in hybrid search (0-1), 0=pure keyword, 1=pure vector
- `--no-query-expansion`: Disable query expansion
- `--no-hybrid-search`: Disable hybrid search
- `--no-diversity`: Disable diversity-based chunk selection

### Embedding and Model Options

- `--embedding-model NAME`: Model to use for embeddings (default: nomic-ai/nomic-embed-code)
- `--cross-encoder NAME`: CrossEncoder model to use for reranking, or 'none' to disable
- `--disable-reranking`: Disable cross-encoder reranking
- `--disable-gpu`: Disable GPU acceleration
- `--max-length INT`: Maximum token length for embedding models
- `--use-8bit`: Use 8-bit quantization for large models

### Chunking Options

- `--chunking-strategy {simple,semantic,structure}`: Chunking strategy to use
- `--chunk-size INT`: Size of text chunks
- `--chunk-overlap INT`: Overlap between chunks (for simple chunking)

### Query Options

- `--top-k INT`: Number of results to return
- `--allow-duplicates`: Allow duplicate files in results
- `--output-file PATH`: Output file to save the query results

### System Options

- `--db-dir PATH`: Directory for ChromaDB storage
- `--reset`: Reset the collection before ingestion
- `--max-files INT`: Maximum number of files to process
- `--batch-size INT`: Batch size for adding documents to ChromaDB
- `--verbose`: Enable verbose output
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging level

## Examples

### Ingest Different Document Types

```bash
# Process code files
python main.py --enhanced --embedding-model nomic-ai/nomic-embed-code \
  --chunking-strategy structure --folder "./src/code" --db-dir "./chroma_db"

# Process documentation
python main.py --enhanced --folder "./documentation" \
  --chunking-strategy semantic --db-dir "./chroma_db"
```

### Different Query Methods

```bash
# Simple query
python main.py --query "How does diversity selection work?" --top-k 3 --db-dir "./chroma_db"

# Enhanced query with detailed output
python main.py --enhanced --query "How does diversity selection work?" \
  --top-k 5 --output-file "results.txt" --db-dir "./chroma_db" --verbose
```

### Optimizing for Different Hardware

```bash
# For high-end GPU
python main.py --enhanced --embedding-model nomic-ai/nomic-embed-code \
  --max-length 1024 --batch-size 32 --db-dir "./chroma_db"

# For limited GPU memory
python main.py --enhanced --embedding-model all-MiniLM-L6-v2 \
  --batch-size 16 --disable-reranking --db-dir "./chroma_db"

# For CPU-only environments
python main.py --disable-gpu --embedding-model all-MiniLM-L6-v2 \
  --batch-size 8 --db-dir "./chroma_db"
```

## Advanced Usage

### Using with LLMs

The `extract_document_for_llm` method provides context in a structured format optimized for LLMs:

```python
from rag_package import EnhancedRAG

rag = EnhancedRAG(persist_directory="./chroma_db")
context = rag.extract_document_for_llm("How does hybrid search work?")

# Pass the context to your LLM
response = my_llm_model.generate(context)
```

### Customizing Retrieval Parameters

```python
from rag_package import EnhancedRAG

# Create with custom settings
rag = EnhancedRAG(
    persist_directory="./chroma_db",
    embedding_model="intfloat/e5-large-v2",
    chunking_strategy="semantic",
    hybrid_alpha=0.3,  # Lower weight for vector search
    enable_diversity=True,
    use_8bit=True
)

# Run a query with specific parameters
results = rag.enhanced_query(
    "code example for diversity algorithm",
    k=7,
    remove_duplicates=False,
    use_reranking=True
)
```

## Performance Considerations

- **Memory Usage**: Large embedding models require substantial RAM or VRAM
  - Use `--use-8bit` for reduced memory requirements
  - For limited hardware, use smaller models like `all-MiniLM-L6-v2`
- **Disk Space**: Vector embeddings require approximately 4-16KB per document chunk
- **Processing Speed**: 
  - GPU significantly accelerates embedding generation
  - Batch size affects throughput and memory usage
  - Cross-encoder reranking improves quality but adds latency

## Troubleshooting

- **Dimension Mismatch Errors**: Ensure consistent embedding models when adding to existing collections
- **Out of Memory**: Reduce batch size, use smaller models, or enable 8-bit quantization
- **Slow Performance**: 
  - Enable GPU acceleration if available
  - Increase batch size on capable hardware
  - Consider disabling some features (reranking, diversity) for speed
- **Poor Results**: 
  - Try different chunking strategies for your content type
  - Adjust hybrid search weights with `--hybrid-alpha`
  - Experiment with different embedding models for your domain

## License

[MIT License](LICENSE)
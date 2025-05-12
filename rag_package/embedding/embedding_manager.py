"""
EmbeddingManager using Nomic Embed Code (or any specified HF/SentenceTransformer model) for unified text+code embeddings.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
from ..storage.document import Document
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EmbeddingManager')

class EmbeddingManager:
    """Manager for embedding and reranking using HF and SentenceTransformer models, defaulting to nomic-ai/nomic-embed-code."""
    MODEL_DIMENSIONS = {
        "nomic-ai/nomic-embed-code": 4096,
        "nomic-ai/nomic-embed-text-v1": 4096,
        "intfloat/e5-large-v2": 1024,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384
    }

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-code",
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
        max_length: int = 512,
        force_model: bool = False,
        check_collection_dim: Optional[int] = None,
        use_8bit: bool = True  # Added parameter for 8-bit quantization
    ):
        self.model_name = model_name
        self.cross_encoder_name = cross_encoder_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.max_length = max_length
        self.force_model = force_model
        self.check_collection_dim = check_collection_dim
        self.use_8bit = use_8bit and self.use_gpu  # Only use 8-bit if GPU is available

        self.embedding_model: Optional[SentenceTransformer] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.hf_model: Optional[AutoModel] = None
        self.embedding_dimension: int = self.MODEL_DIMENSIONS.get(model_name, None) or 0
        self.cross_encoder: Optional[CrossEncoder] = None

        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        logger.info(f"Using device: {self.device}, 8-bit quantization: {self.use_8bit}")
        self._initialize_embedding_model()
        if cross_encoder_name:
            self._load_cross_encoder()

    def _initialize_embedding_model(self):
        # Check collection dimension compatibility
        if self.check_collection_dim and not self.force_model:
            for model_name, dim in self.MODEL_DIMENSIONS.items():
                if dim == self.check_collection_dim:
                    logger.info(f"Selected model {model_name} to match collection dimension {self.check_collection_dim}")
                    self.model_name = model_name
                    self.embedding_dimension = dim
                    break

        is_nomic_model = 'nomic' in self.model_name.lower()
        use_8bit_for_model = self.use_8bit and is_nomic_model

        # Try SentenceTransformer
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            if use_8bit_for_model:
                logger.info("SentenceTransformer with 8-bit quantization is not supported directly. Skipping to AutoModel fallback.")
                raise NotImplementedError("Skip to fallback")
            else:
                self.embedding_model = SentenceTransformer(self.model_name)
                if self.use_gpu:
                    self.embedding_model = self.embedding_model.to(self.device)
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer: dim={self.embedding_dimension}")
                return
        except Exception as e:
            logger.error(f"SentenceTransformer load failed: {e}")

        # Fallback to Hugging Face AutoModel
        logger.info(f"Loading HF model via AutoModel: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        try:
            quant_config = None
            if use_8bit_for_model:
                logger.info("Using 8-bit quantization with AutoModel")
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )

            self.hf_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto" if self.use_gpu else {"": "cpu"},
                quantization_config=quant_config,
                low_cpu_mem_usage=True
            )
            self.embedding_dimension = self.hf_model.config.hidden_size
            logger.info(f"Loaded HF model: dim={self.embedding_dimension}, 8-bit={use_8bit_for_model}")
        except Exception as e:
            logger.error(f"Error loading model with 8-bit quantization: {e}")
            logger.info("Falling back to full precision AutoModel")
            self.hf_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto" if self.use_gpu else {"": "cpu"},
                low_cpu_mem_usage=True
            )
            self.embedding_dimension = self.hf_model.config.hidden_size
            logger.info(f"Loaded HF model in full precision: dim={self.embedding_dimension}")
        finally:
            logger.info(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


    def _load_cross_encoder(self):
        try:
            logger.info(f"Loading cross-encoder: {self.cross_encoder_name}")
            self.cross_encoder = CrossEncoder(self.cross_encoder_name)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.cross_encoder = None

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (prose or code snippets)."""
        if not texts:
            return []

        # Use SentenceTransformer if available
        if self.embedding_model:
            logger.debug(f"Generating embeddings for {len(texts)} texts using SentenceTransformer")
            embs = self.embedding_model.encode(texts, show_progress_bar=False)
            return embs.tolist() if isinstance(embs, np.ndarray) else embs

        # Otherwise use HF model + mean pooling
        logger.debug(f"Generating embeddings for {len(texts)} texts using HuggingFace model")
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.hf_model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1)
        emb = emb.cpu().numpy()
        return emb.tolist()

    def get_query_embedding(self, query: str) -> List[float]:
        """Embed a single query string."""
        if not query:
            return [0.0] * self.embedding_dimension
        return self.get_embeddings([query])[0]

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using a cross-encoder."""
        if not self.cross_encoder or not docs:
            return docs[:top_k]
        logger.debug(f"Reranking {len(docs)} documents with cross-encoder")
        pairs = [(query, doc.content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            docs[i].metadata["reranker_score"] = float(score)
        return sorted(docs, key=lambda d: d.metadata["reranker_score"], reverse=True)[:top_k]

    def check_model_compatibility(self, collection_dim: Optional[int] = None) -> Tuple[bool, str]:
        """Check if embedding dimension matches a collection."""
        if collection_dim is None:
            return True, "No dimension to check"
        if self.embedding_dimension != collection_dim:
            return False, f"Dimension mismatch: expected {collection_dim}, got {self.embedding_dimension}"
        return True, "Compatible dimensions"
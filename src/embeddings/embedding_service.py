"""
Embedding service for generating vector embeddings from text.

Handles model loading, caching, and embedding generation.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence transformers.
    
    Attributes:
        model_name: Name of the sentence transformer model
        dimension: Dimensionality of the embeddings
        model: Loaded sentence transformer model
    """
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code: bool = True
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model identifier
            trust_remote_code: Whether to trust remote code in model
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._trust_remote_code = trust_remote_code
        self._dimension: Optional[int] = None
        
        logger.info(f"Embedding service initialized with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self._trust_remote_code
            )
            logger.info("Embedding model loaded successfully")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Normalize text (remove extra whitespace)
        normalized_text = " ".join(text.strip().split())
        
        # Generate embedding
        embedding = self.model.encode(
            normalized_text,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Normalize all texts
        normalized_texts = [" ".join(text.strip().split()) for text in texts]
        
        # Generate embeddings in batch
        embeddings = self.model.encode(
            normalized_texts,
            convert_to_tensor=False,
            show_progress_bar=len(texts) > 10  # Show progress for large batches
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Embedding model unloaded from memory")

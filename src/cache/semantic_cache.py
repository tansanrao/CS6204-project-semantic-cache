"""
Semantic Cache for LLM responses.

Uses embeddings for semantic similarity matching with PostgreSQL and Qdrant.
"""

import uuid
import hashlib
from typing import Optional, Dict, Any
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.config import Config
from src.embeddings.embedding_service import EmbeddingService
from src.db.postgres import PostgresManager
from src.db.qdrant import QdrantManager

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache for LLM responses using embeddings for similarity matching.
    
    Architecture:
    - EmbeddingService: Generates embeddings from text
    - QdrantManager: Stores vectors and performs similarity search
    - PostgresManager: Stores actual prompt-response pairs
    - Single UUID links entries across both databases
    
    Attributes:
        config: Configuration object
        embedding_service: Service for generating embeddings
        postgres: PostgreSQL manager
        qdrant: Qdrant manager
        similarity_threshold: Minimum similarity for cache hit (0-1)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        similarity_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the semantic cache.
        
        Args:
            config: Configuration object (uses defaults if None)
            similarity_threshold: Override default similarity threshold
            embedding_model: Override default embedding model
        """
        # Use provided config or create default
        if config is None:
            from config.config import default_config
            config = default_config
        
        self.config = config
        
        # Override similarity threshold if provided
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = config.cache.similarity_threshold
        
        # Initialize embedding service
        model_name = embedding_model or config.cache.embedding_model
        self.embedding_service = EmbeddingService(
            model_name=model_name,
            trust_remote_code=config.cache.trust_remote_code
        )
        
        # Initialize database managers
        self.postgres = PostgresManager(config.postgres.to_dict())
        self.qdrant = QdrantManager(
            host=config.qdrant.host,
            port=config.qdrant.port,
            collection_name=config.qdrant.collection_name
        )
        
        # Initialize storage
        self._init_storage()
        
        logger.info(
            f"Semantic cache initialized with similarity threshold: {self.similarity_threshold}"
        )
    
    def _init_storage(self) -> None:
        """Initialize database storage."""
        self.postgres.init_table()
        self.qdrant.init_collection(self.embedding_service.dimension)
    
    def _generate_cache_id(self, prompt: str, model: str) -> uuid.UUID:
        """
        Generate a deterministic UUID for a prompt-model combination.
        
        This ensures the same prompt+model always gets the same UUID.
        
        Args:
            prompt: The input prompt
            model: Model identifier
            
        Returns:
            UUID for the cache entry
        """
        # Create deterministic hash
        content = f"{prompt}|{model}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        
        # Convert to UUID (deterministic)
        uuid_str = f"{hash_digest[:8]}-{hash_digest[8:12]}-{hash_digest[12:16]}-{hash_digest[16:20]}-{hash_digest[20:32]}"
        return uuid.UUID(uuid_str)
    
    def get(
        self,
        prompt: str,
        model: str,
        return_metadata: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a prompt if similar prompt exists.
        
        Process:
        1. Generate embedding for prompt
        2. Search Qdrant for similar vectors
        3. If match found above threshold, retrieve from PostgreSQL
        4. Update access statistics
        
        Args:
            prompt: The input prompt
            model: The model name
            return_metadata: Include cache metadata in response
            
        Returns:
            Dictionary with 'response', 'similarity_score', and optionally 'metadata'
            Returns None if no cache hit
        """
        try:
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(prompt)
            
            # Search for similar prompts in Qdrant
            results = self.qdrant.search_similar(
                query_vector=embedding,
                limit=1,
                score_threshold=self.similarity_threshold
            )
            
            if not results:
                logger.debug("No similar prompts found in cache")
                return None
            
            # Get the best match
            best_match = results[0]
            entry_id = uuid.UUID(best_match.id)
            similarity_score = best_match.score
            
            logger.debug(f"Found similar prompt with score: {similarity_score:.4f}")
            
            # Retrieve full response from PostgreSQL
            entry = self.postgres.get_entry(entry_id)
            
            if not entry:
                logger.warning(f"Entry {entry_id} found in Qdrant but not in PostgreSQL")
                return None
            
            # Build response
            response_data = {
                "response": entry["response"],
                "cached": True,
                "similarity_score": similarity_score
            }
            
            if return_metadata:
                response_data["metadata"] = {
                    "cached_prompt": entry["prompt"],
                    "model": entry["model"],
                    "created_at": entry["created_at"].isoformat(),
                    "access_count": entry["access_count"],
                    "cache_metadata": entry.get("metadata")
                }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}", exc_info=True)
            return None
    
    def set(
        self,
        prompt: str,
        response: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> uuid.UUID:
        """
        Store a prompt-response pair in the cache.
        
        Process:
        1. Generate deterministic UUID for entry
        2. Generate embedding for prompt
        3. Store in PostgreSQL with UUID as primary key
        4. Store vector in Qdrant with same UUID
        
        Args:
            prompt: The input prompt
            response: The LLM response
            model: The model name
            metadata: Optional metadata to store
            
        Returns:
            UUID of the cache entry
        """
        try:
            # Generate cache ID
            entry_id = self._generate_cache_id(prompt, model)
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(prompt)
            
            # Store in PostgreSQL
            self.postgres.insert_entry(
                entry_id=entry_id,
                prompt=prompt,
                response=response,
                model=model,
                metadata=metadata
            )
            
            # Store in Qdrant
            self.qdrant.upsert_vector(
                entry_id=entry_id,
                vector=embedding,
                payload={
                    "model": model,
                    "prompt_preview": prompt[:200]  # Store preview for debugging
                }
            )
            
            logger.debug(f"Cached entry with ID: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}", exc_info=True)
            raise
    
    def clear(self) -> None:
        """Clear all cache entries from both databases."""
        try:
            self.postgres.clear_all()
            self.qdrant.clear_collection()
            
            # Reinitialize
            self._init_storage()
            
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
            raise
    
    def delete(self, entry_id: uuid.UUID) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            entry_id: UUID of the entry to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            postgres_deleted = self.postgres.delete_entry(entry_id)
            qdrant_deleted = self.qdrant.delete_vector(entry_id)
            
            return postgres_deleted and qdrant_deleted
        except Exception as e:
            logger.error(f"Error deleting entry {entry_id}: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with comprehensive cache statistics
        """
        try:
            postgres_stats = self.postgres.get_stats()
            qdrant_stats = self.qdrant.get_collection_info()
            
            return {
                "postgres": postgres_stats,
                "qdrant": qdrant_stats,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.dimension
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {}

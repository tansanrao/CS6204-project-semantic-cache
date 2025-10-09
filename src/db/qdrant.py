"""
Qdrant vector database manager for semantic search.

Handles all Qdrant operations for vector storage and similarity search.
"""

import uuid
from typing import List, Dict, Any, Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manager for Qdrant vector database operations.
    
    Handles collection management, vector storage, and similarity search.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "llm_cache"
    ):
        """
        Initialize Qdrant manager.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        logger.info(f"Qdrant manager initialized for collection: {collection_name}")
    
    def init_collection(self, dimension: int) -> None:
        """
        Create collection if it doesn't exist or verify dimensions match.
        
        Args:
            dimension: Dimensionality of vectors to store
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            # Verify dimensions match
            collection_info = self.client.get_collection(self.collection_name)
            vectors_config = collection_info.config.params.vectors
            
            # Handle both dict and object access patterns
            if isinstance(vectors_config, dict):
                existing_dim = vectors_config.get('size') or list(vectors_config.values())[0].size
            else:
                existing_dim = vectors_config.size
            
            if existing_dim != dimension:
                logger.warning(
                    f"Collection '{self.collection_name}' has dimension {existing_dim}, "
                    f"but expected {dimension}. Recreating collection."
                )
                self.recreate_collection(dimension)
            else:
                logger.info(
                    f"Using existing collection '{self.collection_name}' with dimension {dimension}"
                )
        else:
            # Create new collection
            self.create_collection(dimension)
    
    def create_collection(self, dimension: int) -> None:
        """
        Create a new collection.
        
        Args:
            dimension: Dimensionality of vectors
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        logger.info(f"Created collection '{self.collection_name}' with dimension {dimension}")
    
    def recreate_collection(self, dimension: int) -> None:
        """
        Delete and recreate collection (clean slate).
        
        Args:
            dimension: Dimensionality of vectors
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        
        self.create_collection(dimension)
    
    def upsert_vector(
        self,
        entry_id: uuid.UUID,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> None:
        """
        Insert or update a vector in the collection.
        
        Args:
            entry_id: Unique UUID for the vector point
            vector: The embedding vector
            payload: Metadata to store with the vector
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(entry_id),
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.debug(f"Upserted vector with ID: {entry_id}")
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None
    ) -> List[ScoredPoint]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of scored points with similarity scores
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        logger.debug(f"Found {len(results)} similar vectors")
        return results
    
    def delete_vector(self, entry_id: uuid.UUID) -> bool:
        """
        Delete a vector from the collection.
        
        Args:
            entry_id: UUID of the vector to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[str(entry_id)]
            )
            logger.debug(f"Deleted vector with ID: {entry_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector {entry_id}: {e}")
            return False
    
    def clear_collection(self) -> None:
        """Delete all vectors from the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Cleared collection '{self.collection_name}'")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "points_count": 0,
                "vectors_count": 0,
                "indexed_vectors_count": 0,
            }

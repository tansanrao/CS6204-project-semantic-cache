"""
Semantic Cache for LLM responses using OpenAI embeddings, Qdrant, and PostgreSQL.
Similar to GPTCache implementation with semantic similarity matching.
"""

import hashlib
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class SemanticCache:
    """
    Semantic cache for LLM responses using embeddings for similarity matching.
    
    Uses:
    - nomic-embed-text-v1.5 model for generating embeddings (local)
    - Qdrant for vector similarity search
    - PostgreSQL for storing actual prompt-response pairs
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        postgres_config: Optional[Dict[str, str]] = None,
        collection_name: str = "llm_cache",
        similarity_threshold: float = 0.95,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code: bool = True
    ):
        """
        Initialize the semantic cache.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            postgres_config: PostgreSQL connection config (defaults to localhost)
            collection_name: Name of Qdrant collection
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            embedding_model: Sentence transformer model to use
            trust_remote_code: Whether to trust remote code for the model
        """
        # Initialize sentence transformer for embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            trust_remote_code=trust_remote_code
        )
        self.embedding_model_name = embedding_model
        
        # Get embedding dimension from the model
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Initialize PostgreSQL connection
        if postgres_config is None:
            postgres_config = {
                "host": "localhost",
                "port": "5432",
                "database": "devdb",
                "user": "postgres",
                "password": "postgres"
            }
        self.postgres_config = postgres_config
        
        # Initialize storage
        self._init_qdrant_collection()
        self._init_postgres_table()
    
    def _init_qdrant_collection(self):
        """Initialize Qdrant collection if it doesn't exist or recreate if dimensions don't match."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Check if collection exists
        if self.collection_name in collection_names:
            # Get collection info to check dimensions
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Handle both dict and object access patterns for vectors config
            vectors_config = collection_info.config.params.vectors
            if isinstance(vectors_config, dict):
                existing_dim = vectors_config.get('size') or list(vectors_config.values())[0].size
            else:
                existing_dim = vectors_config.size
            
            if existing_dim != self.embedding_dim:
                print(f"Collection '{self.collection_name}' exists with {existing_dim} dimensions, but model uses {self.embedding_dim} dimensions.")
                print(f"Deleting and recreating collection...")
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                print(f"Recreated Qdrant collection: {self.collection_name} with dimension {self.embedding_dim}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name} with dimension {self.embedding_dim}")
        else:
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {self.collection_name} with dimension {self.embedding_dim}")
    
    def _init_postgres_table(self):
        """Initialize PostgreSQL table if it doesn't exist."""
        conn = psycopg2.connect(**self.postgres_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS llm_cache (
                        id SERIAL PRIMARY KEY,
                        cache_key VARCHAR(64) UNIQUE NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        model VARCHAR(255) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                
                # Create index on cache_key for faster lookups
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_key 
                    ON llm_cache(cache_key)
                """)
                
                conn.commit()
                print("PostgreSQL table initialized")
        finally:
            conn.close()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformer model."""
        # Encode returns numpy array, convert to list
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def _generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique cache key for prompt + model combination."""
        content = f"{prompt}|{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cache_key_to_uuid(self, cache_key: str) -> str:
        """Convert a cache key (SHA256 hash) to a UUID string."""
        # Use the first 32 characters of the hash to create a UUID
        # This is deterministic and ensures the same cache_key always maps to the same UUID
        uuid_str = f"{cache_key[:8]}-{cache_key[8:12]}-{cache_key[12:16]}-{cache_key[16:20]}-{cache_key[20:32]}"
        return uuid_str
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for better cache matching."""
        # Remove extra whitespace and normalize
        return " ".join(prompt.strip().split())
    
    def get(
        self,
        prompt: str,
        model: str,
        return_metadata: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a prompt if semantically similar prompt exists.
        
        Args:
            prompt: The input prompt
            model: The model name
            return_metadata: Whether to return cache metadata
            
        Returns:
            Dictionary with 'response' and optionally 'metadata' if found, None otherwise
        """
        normalized_prompt = self._normalize_prompt(prompt)
        
        # Generate embedding for the prompt
        embedding = self._get_embedding(normalized_prompt)
        
        # Search for similar prompts in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=1,
            score_threshold=self.similarity_threshold
        )
        
        if not search_results:
            return None
        
        # Get the cache key from the best match
        cache_key = search_results[0].payload["cache_key"]
        similarity_score = search_results[0].score
        
        # Retrieve full response from PostgreSQL
        conn = psycopg2.connect(**self.postgres_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE llm_cache 
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE cache_key = %s
                    RETURNING *
                """, (cache_key,))
                
                result = cur.fetchone()
                conn.commit()
                
                if result:
                    response_data = {
                        "response": result["response"],
                        "cached": True,
                        "similarity_score": similarity_score
                    }
                    
                    if return_metadata:
                        response_data["metadata"] = {
                            "cached_prompt": result["prompt"],
                            "model": result["model"],
                            "created_at": result["created_at"].isoformat(),
                            "access_count": result["access_count"],
                            "cache_metadata": result["metadata"]
                        }
                    
                    return response_data
                
                return None
        finally:
            conn.close()
    
    def set(
        self,
        prompt: str,
        response: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store a prompt-response pair in the cache.
        
        Args:
            prompt: The input prompt
            response: The LLM response
            model: The model name
            metadata: Optional metadata to store with the cache entry
        """
        normalized_prompt = self._normalize_prompt(prompt)
        cache_key = self._generate_cache_key(normalized_prompt, model)
        
        # Generate embedding
        embedding = self._get_embedding(normalized_prompt)
        
        # Store in PostgreSQL
        conn = psycopg2.connect(**self.postgres_config)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO llm_cache (cache_key, prompt, response, model, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) 
                    DO UPDATE SET 
                        response = EXCLUDED.response,
                        accessed_at = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata
                    RETURNING id
                """, (
                    cache_key,
                    normalized_prompt,
                    response,
                    model,
                    json.dumps(metadata) if metadata else None
                ))
                
                result = cur.fetchone()
                record_id = result[0]
                conn.commit()
        finally:
            conn.close()
        
        # Store in Qdrant (convert cache_key to UUID format)
        point_id = self._cache_key_to_uuid(cache_key)
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,  # Use UUID format for Qdrant point ID
                    vector=embedding,
                    payload={
                        "cache_key": cache_key,
                        "model": model,
                        "prompt_preview": normalized_prompt[:200]  # Store preview for debugging
                    }
                )
            ]
        )
    
    def clear(self):
        """Clear all cache entries."""
        # Clear Qdrant collection
        self.qdrant_client.delete_collection(collection_name=self.collection_name)
        self._init_qdrant_collection()
        
        # Clear PostgreSQL table
        conn = psycopg2.connect(**self.postgres_config)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM llm_cache")
                conn.commit()
                print("Cache cleared")
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = psycopg2.connect(**self.postgres_config)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses_per_entry,
                        MAX(accessed_at) as last_access
                    FROM llm_cache
                """)
                stats = cur.fetchone()
                
                # Get Qdrant collection info
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                
                return {
                    "total_entries": stats["total_entries"],
                    "total_accesses": stats["total_accesses"],
                    "avg_accesses_per_entry": float(stats["avg_accesses_per_entry"]) if stats["avg_accesses_per_entry"] else 0,
                    "last_access": stats["last_access"].isoformat() if stats["last_access"] else None,
                    "qdrant_points": collection_info.points_count
                }
        finally:
            conn.close()

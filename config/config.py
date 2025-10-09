"""
Centralized configuration for the semantic cache project.

This module provides a single source of truth for all configuration values,
with support for environment variables and validation.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class VLLMConfig:
    """Configuration for vLLM service."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy-key"
    model_name: str = "openai/gpt-oss-20b"
    test_prompt: str = "What is 2+2?"
    test_max_tokens: int = 8000


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL database."""
    host: str = "localhost"
    port: int = 5432
    database: str = "devdb"
    user: str = "postgres"
    password: str = "postgres"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for psycopg2 connection."""
        return {
            "host": self.host,
            "port": str(self.port),
            "database": self.database,
            "user": self.user,
            "password": self.password
        }


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "llm_cache"


@dataclass
class CacheConfig:
    """Configuration for semantic cache behavior."""
    similarity_threshold: float = 0.95
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    trust_remote_code: bool = True


@dataclass
class Config:
    """Master configuration for the entire project."""
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.
        
        Environment variables follow the pattern: {SECTION}_{KEY}
        Example: VLLM_BASE_URL, POSTGRES_HOST, etc.
        """
        vllm = VLLMConfig(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "dummy-key"),
            model_name=os.getenv("VLLM_MODEL_NAME", "openai/gpt-oss-20b"),
        )
        
        postgres = PostgresConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "devdb"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
        
        qdrant = QdrantConfig(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "llm_cache"),
        )
        
        cache = CacheConfig(
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")),
            embedding_model=os.getenv("CACHE_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        )
        
        return cls(vllm=vllm, postgres=postgres, qdrant=qdrant, cache=cache)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.cache.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {self.cache.similarity_threshold}")
        
        if self.postgres.port <= 0 or self.postgres.port > 65535:
            raise ValueError(f"Invalid PostgreSQL port: {self.postgres.port}")
        
        if self.qdrant.port <= 0 or self.qdrant.port > 65535:
            raise ValueError(f"Invalid Qdrant port: {self.qdrant.port}")


# Default configuration instance
default_config = Config()

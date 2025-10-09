"""
Basic tests for the semantic cache system.

Run with: pytest tests/test_cache.py
"""

import sys
import os
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from src.cache.semantic_cache import SemanticCache


def test_config_creation():
    """Test that configuration can be created."""
    config = Config()
    assert config.vllm.base_url is not None
    assert config.postgres.host is not None
    assert config.qdrant.host is not None
    assert config.cache.similarity_threshold > 0.0


def test_config_validation():
    """Test configuration validation."""
    config = Config()
    config.validate()  # Should not raise
    
    # Test invalid threshold
    config.cache.similarity_threshold = 1.5
    try:
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_cache_id_generation():
    """Test that cache IDs are deterministic."""
    config = Config()
    cache = SemanticCache(config=config)
    
    # Same prompt+model should give same ID
    id1 = cache._generate_cache_id("test prompt", "test-model")
    id2 = cache._generate_cache_id("test prompt", "test-model")
    
    assert id1 == id2
    assert isinstance(id1, uuid.UUID)
    
    # Different prompt should give different ID
    id3 = cache._generate_cache_id("different prompt", "test-model")
    assert id1 != id3


def test_embedding_service():
    """Test embedding service initialization."""
    from src.embeddings.embedding_service import EmbeddingService
    
    service = EmbeddingService()
    
    # Test embedding generation
    embedding = service.generate_embedding("test text")
    
    assert isinstance(embedding, list)
    assert len(embedding) == service.dimension
    assert all(isinstance(x, float) for x in embedding)


def test_postgres_manager():
    """Test PostgreSQL manager initialization."""
    from src.db.postgres import PostgresManager
    
    config = Config()
    manager = PostgresManager(config.postgres.to_dict())
    
    # Should be able to create manager without error
    assert manager.table_name == "llm_cache"


def test_qdrant_manager():
    """Test Qdrant manager initialization."""
    from src.db.qdrant import QdrantManager
    
    config = Config()
    manager = QdrantManager(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name
    )
    
    # Should be able to create manager without error
    assert manager.collection_name == config.qdrant.collection_name


if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    
    print("✓ test_config_creation")
    test_config_creation()
    
    print("✓ test_config_validation")
    test_config_validation()
    
    print("✓ test_cache_id_generation")
    test_cache_id_generation()
    
    print("✓ test_embedding_service")
    test_embedding_service()
    
    print("✓ test_postgres_manager")
    test_postgres_manager()
    
    print("✓ test_qdrant_manager")
    test_qdrant_manager()
    
    print("\n✅ All tests passed!")

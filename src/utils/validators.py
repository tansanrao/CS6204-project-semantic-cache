"""
Validators and health check utilities.

Provides functions to test connectivity and health of all services.
"""

import logging
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
import psycopg2
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def test_vllm_connection(
    base_url: str,
    api_key: str,
    model: str,
    test_prompt: str = "What is 2+2?",
    max_tokens: int = 50,
    timeout: int = 30
) -> bool:
    """
    Test vLLM server connectivity and model availability.
    
    Args:
        base_url: vLLM server base URL
        api_key: API key (can be dummy for vLLM)
        model: Model name to test
        test_prompt: Test prompt to send
        max_tokens: Maximum tokens for test
        timeout: Request timeout in seconds
        
    Returns:
        True if vLLM is healthy and responds correctly
    """
    try:
        print(f"  → Testing vLLM at {base_url}")
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        
        # Send test prompt
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Verify response
        if response and response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content
            print(f"  ✓ vLLM is healthy")
            print(f"  ✓ Model '{model}' loaded and responding")
            print(f"  → Test response: {answer[:100]}...")
            return True
        else:
            print(f"  ✗ vLLM returned invalid response")
            return False
            
    except Exception as e:
        print(f"  ✗ vLLM health check failed: {e}")
        logger.error(f"vLLM health check error: {e}", exc_info=True)
        return False


def test_postgres_connection(config: Dict[str, Any]) -> bool:
    """
    Test PostgreSQL connectivity.
    
    Args:
        config: PostgreSQL connection configuration
        
    Returns:
        True if connection successful
    """
    try:
        print(f"  → Testing PostgreSQL at {config.get('host')}:{config.get('port')}")
        conn = psycopg2.connect(**config)
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"  ✓ PostgreSQL connected: {version.split(',')[0]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ PostgreSQL connection failed: {e}")
        logger.error(f"PostgreSQL connection error: {e}", exc_info=True)
        return False


def test_qdrant_connection(host: str, port: int) -> bool:
    """
    Test Qdrant connectivity.
    
    Args:
        host: Qdrant host
        port: Qdrant port
        
    Returns:
        True if connection successful
    """
    try:
        print(f"  → Testing Qdrant at {host}:{port}")
        client = QdrantClient(host=host, port=port)
        
        # Get collections to verify connection
        collections = client.get_collections()
        print(f"  ✓ Qdrant connected: {len(collections.collections)} collections found")
        return True
        
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        logger.error(f"Qdrant connection error: {e}", exc_info=True)
        return False


def test_database_connections(
    postgres_config: Dict[str, Any],
    qdrant_host: str,
    qdrant_port: int
) -> bool:
    """
    Test all database connections.
    
    Args:
        postgres_config: PostgreSQL configuration
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        
    Returns:
        True if all connections successful
    """
    print("\n🔍 Testing database connections...")
    
    postgres_ok = test_postgres_connection(postgres_config)
    qdrant_ok = test_qdrant_connection(qdrant_host, qdrant_port)
    
    if postgres_ok and qdrant_ok:
        print("  ✓ All database connections successful\n")
        return True
    else:
        print("  ✗ Some database connections failed\n")
        return False


def recreate_databases(
    postgres_config: Dict[str, Any],
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    embedding_dimension: int = 768
) -> None:
    """
    Recreate all database tables and collections (clean slate).
    
    Args:
        postgres_config: PostgreSQL configuration
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        collection_name: Qdrant collection name
        embedding_dimension: Dimension of embedding vectors
    """
    print("\n🔄 Recreating databases...")
    
    # Recreate PostgreSQL table
    try:
        from src.db.postgres import PostgresManager
        
        pg_manager = PostgresManager(postgres_config)
        pg_manager.recreate_table()
        print(f"  ✓ PostgreSQL table recreated")
    except Exception as e:
        print(f"  ✗ Failed to recreate PostgreSQL table: {e}")
        raise
    
    # Recreate Qdrant collection
    try:
        from src.db.qdrant import QdrantManager
        
        qdrant_manager = QdrantManager(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        qdrant_manager.recreate_collection(embedding_dimension)
        print(f"  ✓ Qdrant collection recreated (dimension: {embedding_dimension})")
    except Exception as e:
        print(f"  ✗ Failed to recreate Qdrant collection: {e}")
        raise
    
    print("  ✓ All databases recreated successfully\n")


def validate_system(
    vllm_base_url: str,
    vllm_api_key: str,
    vllm_model: str,
    postgres_config: Dict[str, Any],
    qdrant_host: str,
    qdrant_port: int,
    test_prompt: str = "What is 2+2?"
) -> bool:
    """
    Validate entire system health.
    
    Args:
        vllm_base_url: vLLM base URL
        vllm_api_key: vLLM API key
        vllm_model: Model name
        postgres_config: PostgreSQL configuration
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        test_prompt: Test prompt for vLLM
        
    Returns:
        True if all systems healthy
    """
    print("\n" + "="*60)
    print("🏥 SYSTEM HEALTH CHECK")
    print("="*60)
    
    # Test databases
    db_ok = test_database_connections(postgres_config, qdrant_host, qdrant_port)
    
    # Test vLLM
    vllm_ok = test_vllm_connection(
        vllm_base_url,
        vllm_api_key,
        vllm_model,
        test_prompt
    )
    
    print("="*60)
    if db_ok and vllm_ok:
        print("✅ ALL SYSTEMS HEALTHY")
        print("="*60 + "\n")
        return True
    else:
        print("❌ SYSTEM HEALTH CHECK FAILED")
        print("="*60 + "\n")
        return False

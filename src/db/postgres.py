"""
PostgreSQL database manager for cache storage.

Handles all PostgreSQL operations for storing and retrieving cache entries.
"""

import uuid
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class PostgresManager:
    """
    Manager for PostgreSQL operations in the semantic cache.
    
    Handles cache entry storage, retrieval, and statistics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL manager.
        
        Args:
            config: PostgreSQL connection configuration
        """
        self.config = config
        self.table_name = "llm_cache"
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(**self.config)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_table(self) -> None:
        """Create the cache table if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        model VARCHAR(255) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                
                # Create index for faster lookups
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_id 
                    ON {self.table_name}(id)
                """)
                
                conn.commit()
                logger.info(f"PostgreSQL table '{self.table_name}' initialized")
    
    def recreate_table(self) -> None:
        """Drop and recreate the cache table (clean slate)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE")
                conn.commit()
                logger.info(f"Dropped table '{self.table_name}'")
        
        self.init_table()
    
    def insert_entry(
        self,
        entry_id: uuid.UUID,
        prompt: str,
        response: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert a new cache entry.
        
        Args:
            entry_id: Unique UUID for the cache entry
            prompt: The input prompt
            response: The LLM response
            model: Model identifier
            metadata: Optional metadata dictionary
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name} (id, prompt, response, model, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) 
                    DO UPDATE SET 
                        response = EXCLUDED.response,
                        accessed_at = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata
                """, (
                    str(entry_id),
                    prompt,
                    response,
                    model,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
                logger.debug(f"Inserted cache entry with ID: {entry_id}")
    
    def get_entry(self, entry_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cache entry by ID and update access statistics.
        
        Args:
            entry_id: UUID of the cache entry
            
        Returns:
            Dictionary with entry data or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    UPDATE {self.table_name}
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE id = %s
                    RETURNING *
                """, (str(entry_id),))
                
                result = cur.fetchone()
                conn.commit()
                
                if result:
                    return dict(result)
                return None
    
    def delete_entry(self, entry_id: uuid.UUID) -> bool:
        """
        Delete a cache entry.
        
        Args:
            entry_id: UUID of the cache entry
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE id = %s
                """, (str(entry_id),))
                
                deleted = cur.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.debug(f"Deleted cache entry with ID: {entry_id}")
                
                return deleted
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries deleted
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name}")
                count = cur.rowcount
                conn.commit()
                logger.info(f"Cleared {count} entries from cache")
                return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from PostgreSQL.
        
        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses_per_entry,
                        MAX(accessed_at) as last_access,
                        MIN(created_at) as first_entry
                    FROM {self.table_name}
                """)
                stats = cur.fetchone()
                
                return {
                    "total_entries": stats["total_entries"] or 0,
                    "total_accesses": stats["total_accesses"] or 0,
                    "avg_accesses_per_entry": float(stats["avg_accesses_per_entry"] or 0),
                    "last_access": stats["last_access"].isoformat() if stats["last_access"] else None,
                    "first_entry": stats["first_entry"].isoformat() if stats["first_entry"] else None,
                }
    
    def list_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List cache entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of cache entries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, prompt, model, created_at, accessed_at, access_count
                    FROM {self.table_name}
                    ORDER BY accessed_at DESC
                    LIMIT %s
                """, (limit,))
                
                return [dict(row) for row in cur.fetchall()]

"""Tests for the Qdrant vector store helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.cache.qdrant import QdrantVectorStore


@pytest.mark.anyio
async def test_ensure_collection_creates_when_missing() -> None:
    """A new collection should be created when absent."""
    client = MagicMock()
    client.collection_exists.return_value = False
    store = QdrantVectorStore(client, "cache_entries", 768, 0.86)

    await store.ensure_collection()

    client.create_collection.assert_called_once()


@pytest.mark.anyio
async def test_ensure_collection_validates_existing_collection() -> None:
    """Existing collections with mismatched dimensions should raise errors."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.dict.return_value = {
        "config": {"params": {"vectors": {"size": 512, "distance": "Cosine"}}}
    }
    store = QdrantVectorStore(client, "cache_entries", 768, 0.86)

    with pytest.raises(ValueError):
        await store.ensure_collection()

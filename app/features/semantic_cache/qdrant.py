"""Helpers for managing the Qdrant vector collection."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class QdrantVectorStore:
    """Wrapper around the Qdrant client with utility helpers."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dimension: int,
        similarity_threshold: float,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dimension = dimension
        self._similarity_threshold = similarity_threshold

    async def ensure_collection(self) -> None:
        """Create the collection when it does not yet exist."""
        await asyncio.to_thread(self._ensure_collection_sync)

    async def upsert_point(
        self,
        point_id: UUID,
        vector: Sequence[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update the vector payload for an entry."""
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self._collection_name,
            points=[
                rest.PointStruct(
                    id=str(point_id),
                    vector=list(vector),
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        vector: Sequence[float],
        model: str,
        params_fingerprint: str,
        not_before: datetime,
        limit: int,
    ) -> list[rest.ScoredPoint]:
        """Run a similarity search constrained by model and TTL."""
        qdrant_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="model",
                    match=rest.MatchValue(value=model),
                ),
                rest.FieldCondition(
                    key="params_fingerprint",
                    match=rest.MatchValue(value=params_fingerprint),
                ),
                rest.FieldCondition(
                    key="expires_at_ts",
                    range=rest.Range(gte=not_before.timestamp()),
                ),
            ]
        )
        return await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection_name,
            query_vector=list(vector),
            limit=limit,
            with_payload=True,
            score_threshold=self._similarity_threshold,
            query_filter=qdrant_filter,
        )

    def _ensure_collection_sync(self) -> None:
        """Synchronous helper that performs collection bootstrap."""
        if self._client.collection_exists(self._collection_name):
            self._validate_collection()
            return
        vectors_config = rest.VectorParams(
            size=self._dimension,
            distance=rest.Distance.COSINE,
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=vectors_config,
        )

    def _validate_collection(self) -> None:
        """Ensure an existing collection matches the expected shape."""
        info = self._client.get_collection(self._collection_name)
        params = info.dict()["config"]["params"]
        size = params["vectors"]["size"]
        distance = params["vectors"]["distance"]
        if size != self._dimension:
            raise ValueError(
                f"Qdrant collection '{self._collection_name}' has dimension {size}, "
                f"expected {self._dimension}."
            )
        if distance.lower() != rest.Distance.COSINE.value.lower():
            raise ValueError(
                f"Qdrant collection '{self._collection_name}' uses distance "
                f"{distance}, expected cosine."
            )

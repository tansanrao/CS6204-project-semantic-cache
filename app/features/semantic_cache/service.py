"""High-level semantic cache orchestration."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from qdrant_client.http.models import ScoredPoint

from .embedding import EmbeddingService
from .policy.manager import PolicyRuntimeConfig, TTLPolicyManager, build_policy_manager
from .qdrant import QdrantVectorStore
from .repository import CacheRepository
from .types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheEntryCreate,
    CacheHit,
    CacheLookupResult,
    CacheQuery,
    FeedbackEventCreate,
    PolicyRewardCreate,
    RefreshOutcome,
    TTLDecisionCreate,
)

_REWARD_ALPHA = 0.2
_REWARD_BETA = 0.3
_REWARD_GAMMA = 0.2
_REWARD_DELTA = 0.6
_REWARD_ETA = 0.4
_REWARD_KAPPA = 0.2


logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)


@dataclass(frozen=True, slots=True)
class SemanticCacheSettings:
    """Configuration knobs for the semantic cache service."""

    similarity_threshold: float = 0.86
    search_limit: int = 5
    ttl_seconds: tuple[int, ...] = (60, 300, 900, 1800, 3600, 7200)
    default_ttl_bucket: int = 2
    policy_enabled: bool = True
    policy_type: str = "linucb"
    policy_feature_dimension: int = 256
    policy_allow_feature_growth: bool = True
    policy_snapshot_path: str | None = None
    policy_autosave_interval: int = 50
    linucb_alpha: float = 0.6
    linucb_regularization: float = 1.0
    linucb_min_propensity: float = 1e-3
    ts_regularization: float = 1.0
    ts_sampling_variance: float = 1.0
    ts_min_propensity: float = 1e-3

    def ttl_for_bucket(self, bucket: int | None) -> tuple[int, int]:
        """Return (bucket_index, ttl_seconds) with validation."""
        if bucket is None:
            bucket = self.default_ttl_bucket
        if not 0 <= bucket < len(self.ttl_seconds):
            raise ValueError(f"Invalid TTL bucket {bucket}.")
        return bucket, self.ttl_seconds[bucket]


class SemanticCacheService:
    """Provide lookup and storage operations backed by Postgres and Qdrant."""

    def __init__(
        self,
        repository: CacheRepository,
        vector_store: QdrantVectorStore,
        embedder: EmbeddingService,
        *,
        settings: SemanticCacheSettings | None = None,
        embedding_model_name: str,
        embedding_mode: str,
        embedding_dimension: int,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._embedder = embedder
        self._settings = settings or SemanticCacheSettings()
        self._embedding_model_name = embedding_model_name
        self._embedding_mode = embedding_mode
        self._embedding_dimension = embedding_dimension
        self._policy_manager: TTLPolicyManager | None = None

        if self._settings.policy_enabled:
            snapshot_path = (
                Path(self._settings.policy_snapshot_path)
                if self._settings.policy_snapshot_path
                else None
            )
            runtime_config = PolicyRuntimeConfig(
                policy_type=self._settings.policy_type,
                num_actions=len(self._settings.ttl_seconds),
                feature_dimension=self._settings.policy_feature_dimension,
                allow_feature_growth=self._settings.policy_allow_feature_growth,
                linucb_alpha=self._settings.linucb_alpha,
                linucb_regularization=self._settings.linucb_regularization,
                linucb_min_propensity=self._settings.linucb_min_propensity,
                ts_regularization=self._settings.ts_regularization,
                ts_sampling_variance=self._settings.ts_sampling_variance,
                ts_min_propensity=self._settings.ts_min_propensity,
                snapshot_path=snapshot_path,
                autosave_interval=self._settings.policy_autosave_interval,
            )
            self._policy_manager = build_policy_manager(runtime_config)

        policy_label = (
            self._settings.policy_type if self._settings.policy_enabled else "disabled"
        )
        collection_name = getattr(self._vector_store, "_collection_name", "unknown")
        logger.info(
            "semantic cache initialized policy=%s collection=%s embedding_model=%s "
            "embedding_mode=%s dimension=%d",
            policy_label,
            collection_name,
            self._embedding_model_name,
            self._embedding_mode,
            self._embedding_dimension,
        )

    async def bootstrap(self) -> None:
        """Ensure storage backends are ready for use."""
        collection_name = getattr(self._vector_store, "_collection_name", "unknown")
        logger.info("semantic cache bootstrap start collection=%s", collection_name)
        await self._repository.create_schema()
        await self._vector_store.ensure_collection()
        logger.info("semantic cache bootstrap complete collection=%s", collection_name)

    def normalize_request(
        self, path: str, payload: dict[str, Any]
    ) -> CacheQuery | None:
        """Normalize a JSON payload into a cache query."""
        lowered = path.lower().strip("/")
        if lowered.endswith("chat/completions"):
            return _normalize_chat_completion(path, payload)
        if lowered.endswith("completions"):
            return _normalize_legacy_completion(path, payload)
        return None

    async def lookup(self, query: CacheQuery) -> CacheLookupResult:
        """Search Qdrant and return the best matching cache entry."""
        logger.info(
            "cache lookup start request_fingerprint=%s model=%s",
            query.request_fingerprint,
            query.model,
        )
        vector = await self._embed_prompt(query)
        points = await self._vector_store.search(
            vector=vector,
            model=query.model,
            params_fingerprint=query.params_fingerprint,
            not_before=datetime.now(UTC),
            limit=self._settings.search_limit,
        )
        decision = await self._materialize_hit(query, points)
        similarity = None
        if decision.hit is not None:
            similarity = decision.hit.similarity
        elif decision.stale_similarity is not None:
            similarity = decision.stale_similarity
        logger.info(
            "cache lookup decision=%s request_fingerprint=%s prompt_hash=%s "
            "similarity=%s candidates=%d neighbor_stale_rate=%s reasons=%s",
            decision.status.value,
            query.request_fingerprint,
            query.prompt_hash,
            f"{similarity:.4f}" if similarity is not None else "n/a",
            len(points),
            (
                f"{decision.neighbor_stale_rate:.4f}"
                if decision.neighbor_stale_rate is not None
                else "n/a"
            ),
            ",".join(decision.reasons) if decision.reasons else "-",
        )
        return decision

    async def select_ttl_bucket(
        self,
        features: Mapping[str, float],
    ) -> tuple[int, float]:
        """Return the TTL bucket index and propensity for the supplied features."""
        summary = self._summarize_features(features)
        if self._policy_manager is None:
            bucket = self._settings.default_ttl_bucket
            logger.info(
                "ttl selection policy=disabled bucket=%d propensity=1.0 %s",
                bucket,
                summary,
            )
            return bucket, 1.0
        action, propensity = await self._policy_manager.select_action(features)
        logger.info(
            "ttl selection policy=%s bucket=%d propensity=%.4f %s",
            self._settings.policy_type,
            action,
            propensity,
            summary,
        )
        return action, propensity

    async def store(
        self,
        query: CacheQuery,
        response_payload: dict[str, Any],
        *,
        ttl_bucket: int | None = None,
    ) -> CacheEntry:
        """Persist a cache miss response into Postgres and Qdrant."""
        vector = await self._embed_prompt(query)
        bucket_index, ttl_seconds = self._settings.ttl_for_bucket(ttl_bucket)

        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        entry_id = uuid4()

        payload = CacheEntryCreate(
            id=entry_id,
            request_fingerprint=query.request_fingerprint,
            prompt_hash=query.prompt_hash,
            model=query.model,
            parameters=query.parameters,
            prompt_text=query.prompt_text,
            response_payload=response_payload,
            ttl_bucket=bucket_index,
            ttl_seconds=ttl_seconds,
            created_at=created_at,
            expires_at=expires_at,
            embedding_model=self._embedding_model_name,
            embedding_mode=self._embedding_mode,
            embedding_dimension=self._embedding_dimension,
            similarity_threshold=self._settings.similarity_threshold,
        )

        await self._repository.insert_entry(payload)
        await self._vector_store.upsert_point(
            entry_id,
            vector,
            {
                "cache_entry_id": str(entry_id),
                "model": query.model,
                "params_fingerprint": query.params_fingerprint,
                "request_fingerprint": query.request_fingerprint,
                "prompt_hash": query.prompt_hash,
                "created_at_ts": created_at.timestamp(),
                "expires_at_ts": expires_at.timestamp(),
                "ttl_bucket": bucket_index,
                "ttl_seconds": ttl_seconds,
            },
        )
        stored = await self._repository.get_entry(entry_id)
        if stored is None:
            raise RuntimeError("Inserted cache entry could not be reloaded.")
        logger.info(
            "cache store entry_id=%s request_fingerprint=%s prompt_hash=%s "
            "bucket=%d ttl_seconds=%d expires_at=%s",
            entry_id,
            query.request_fingerprint,
            query.prompt_hash,
            bucket_index,
            ttl_seconds,
            expires_at.isoformat(),
        )
        return stored

    async def log_ttl_decision(
        self,
        *,
        query: CacheQuery,
        entry: CacheEntry,
        policy_name: str,
        policy_version: str,
        ttl_bucket: int,
        features: dict[str, float],
        propensity: float | None,
    ) -> UUID:
        """Record the policy decision that produced a cached entry."""
        payload = TTLDecisionCreate(
            id=uuid4(),
            cache_entry_id=entry.id,
            request_fingerprint=query.request_fingerprint,
            prompt_hash=query.prompt_hash,
            policy_name=policy_name,
            policy_version=policy_version,
            ttl_bucket=ttl_bucket,
            features=features,
            propensity=propensity,
            created_at=datetime.now(UTC),
        )
        await self._repository.insert_ttl_decision(payload)
        logger.info(
            "ttl decision recorded decision_id=%s cache_entry_id=%s bucket=%d "
            "propensity=%s policy=%s/%s %s",
            payload.id,
            entry.id,
            ttl_bucket,
            f"{propensity:.4f}" if propensity is not None else "n/a",
            policy_name,
            policy_version,
            self._summarize_features(features),
        )
        return payload.id

    async def log_feedback_event(
        self,
        *,
        cache_entry_id: UUID,
        event_type: str,
        score: float,
        details: dict[str, Any] | None = None,
        ttl_decision_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Persist cache feedback events for policy learning."""
        payload = FeedbackEventCreate(
            id=uuid4(),
            cache_entry_id=cache_entry_id,
            ttl_decision_id=ttl_decision_id,
            event_type=event_type,
            score=score,
            details=details,
            created_at=created_at or datetime.now(UTC),
        )
        await self._repository.insert_feedback_event(payload)
        logger.info(
            "cache feedback recorded cache_entry_id=%s event_type=%s score=%.3f "
            "decision_id=%s details_keys=%s",
            payload.cache_entry_id,
            payload.event_type,
            payload.score,
            payload.ttl_decision_id or "n/a",
            sorted(payload.details.keys()) if payload.details else [],
        )

    async def record_hit(self, entry_id: UUID) -> None:
        """Increment hit metrics for the supplied entry."""
        await self._repository.mark_hit(entry_id, datetime.now(UTC))
        logger.info("cache hit recorded entry_id=%s", entry_id)

    async def log_refresh_outcome(self, outcome: RefreshOutcome) -> None:
        """Persist refresh feedback and reward attribution."""
        feedback_details: dict[str, Any] = dict(outcome.feedback_details or {})
        logger.info(
            "refresh outcome cache_entry_id=%s decision_id=%s stale=%s elapsed=%d "
            "ttl=%d bonus_applicable=%s feedback_event=%s score=%.3f",
            outcome.cache_entry_id,
            outcome.ttl_decision_id or "n/a",
            outcome.stale,
            outcome.elapsed_seconds,
            outcome.ttl_seconds,
            outcome.bonus_applicable,
            outcome.feedback_event.value,
            outcome.feedback_score,
        )

        extension_detail = await self._maybe_extend_ttl(outcome)
        if extension_detail is not None:
            adjustments = feedback_details.setdefault("ttl_adjustments", [])
            adjustments.append(extension_detail)
            logger.info(
                "refresh ttl extension cache_entry_id=%s previous_bucket=%s "
                "new_bucket=%s new_ttl_seconds=%s",
                outcome.cache_entry_id,
                extension_detail.get("previous_bucket"),
                extension_detail.get("new_bucket"),
                extension_detail.get("new_ttl_seconds"),
            )

        guardrail_detail = await self._apply_guardrails(outcome)
        if guardrail_detail is not None:
            guardrails = feedback_details.setdefault("guardrails", [])
            guardrails.append(guardrail_detail)
            logger.info(
                "refresh ttl guardrail cache_entry_id=%s detail=%s",
                outcome.cache_entry_id,
                guardrail_detail,
            )

        await self.log_feedback_event(
            cache_entry_id=outcome.cache_entry_id,
            event_type=outcome.feedback_event.value,
            score=outcome.feedback_score,
            details=feedback_details or None,
            ttl_decision_id=outcome.ttl_decision_id,
            created_at=outcome.obtained_at,
        )

        if outcome.ttl_decision_id is None:
            return

        reward_value = self._compute_reward(outcome)
        payload = PolicyRewardCreate(
            id=uuid4(),
            ttl_decision_id=outcome.ttl_decision_id,
            reward=reward_value,
            attribution_rule="stale_check_v1",
            created_at=outcome.obtained_at,
        )
        await self._repository.insert_policy_reward(payload)
        logger.info(
            "policy reward recorded decision_id=%s reward=%.4f",
            payload.ttl_decision_id,
            reward_value,
        )

        if self._policy_manager is not None:
            decision = await self._repository.get_ttl_decision(outcome.ttl_decision_id)
            if decision is not None:
                await self._policy_manager.update(
                    action=decision.ttl_bucket,
                    features=decision.features,
                    reward=reward_value,
                )

    async def _maybe_extend_ttl(self, outcome: RefreshOutcome) -> dict[str, Any] | None:
        """Extend TTL when refresh confirms content remains current."""
        if outcome.stale or not outcome.bonus_applicable:
            return None
        entry = await self._repository.get_entry(outcome.cache_entry_id)
        if entry is None:
            return None
        current_bucket = entry.ttl_bucket
        last_bucket_index = len(self._settings.ttl_seconds) - 1
        if current_bucket >= last_bucket_index:
            return None
        new_bucket = current_bucket + 1
        new_ttl_seconds = self._settings.ttl_seconds[new_bucket]
        new_expires_at = datetime.now(UTC) + timedelta(seconds=new_ttl_seconds)
        await self._repository.update_entry_ttl(
            entry_id=entry.id,
            ttl_bucket=new_bucket,
            ttl_seconds=new_ttl_seconds,
            expires_at=new_expires_at,
        )
        return {
            "action": "extend_bucket",
            "previous_bucket": current_bucket,
            "new_bucket": new_bucket,
            "new_ttl_seconds": new_ttl_seconds,
        }

    async def _apply_guardrails(self, outcome: RefreshOutcome) -> dict[str, Any] | None:
        """Reduce TTL when refresh detects early staleness."""
        if not outcome.stale:
            return None
        ttl_seconds = max(outcome.ttl_seconds, 1)
        elapsed_ratio = min(max(outcome.elapsed_seconds, 0) / ttl_seconds, 1.0)
        if elapsed_ratio >= 0.25:
            return None
        entry = await self._repository.get_entry(outcome.cache_entry_id)
        if entry is None:
            return {
                "action": "early_stale",
                "elapsed_ratio": elapsed_ratio,
            }
        current_bucket = entry.ttl_bucket
        if current_bucket <= 0:
            return {
                "action": "early_stale",
                "elapsed_ratio": elapsed_ratio,
                "previous_bucket": current_bucket,
                "new_bucket": current_bucket,
            }
        new_bucket = current_bucket - 1
        new_ttl_seconds = self._settings.ttl_seconds[new_bucket]
        new_expires_at = datetime.now(UTC) + timedelta(seconds=new_ttl_seconds)
        await self._repository.update_entry_ttl(
            entry_id=entry.id,
            ttl_bucket=new_bucket,
            ttl_seconds=new_ttl_seconds,
            expires_at=new_expires_at,
        )
        return {
            "action": "shrink_bucket",
            "reason": "early_stale",
            "elapsed_ratio": elapsed_ratio,
            "previous_bucket": current_bucket,
            "new_bucket": new_bucket,
            "new_ttl_seconds": new_ttl_seconds,
        }

    @staticmethod
    def _compute_reward(outcome: RefreshOutcome) -> float:
        """Compute reward following DESIGN §6 heuristics."""
        ttl_seconds = max(outcome.ttl_seconds, 1)
        elapsed_ratio = min(max(outcome.elapsed_seconds, 0) / ttl_seconds, 1.0)

        hit_component = _REWARD_ALPHA * 1.0
        cost_component = _REWARD_BETA * outcome.cost_savings
        latency_component = _REWARD_GAMMA * outcome.latency_savings
        stale_penalty = _REWARD_DELTA * (1.0 if outcome.stale else 0.0)

        base_reward = hit_component + cost_component + latency_component - stale_penalty

        if outcome.stale:
            early_penalty = 1.0 - elapsed_ratio
            return base_reward - (_REWARD_ETA * early_penalty)

        if outcome.bonus_applicable:
            return base_reward + (_REWARD_KAPPA * elapsed_ratio)
        return base_reward

    @staticmethod
    def _summarize_features(
        features: Mapping[str, float],
        *,
        limit: int = 5,
    ) -> str:
        """Return a compact description of the feature payload for logging."""
        if not features:
            return "features=count=0"
        sorted_items = sorted(
            features.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        head = ", ".join(f"{name}={value:.3f}" for name, value in sorted_items[:limit])
        if not head:
            return f"features=count={len(features)}"
        return f"features=count={len(features)} top=[{head}]"

    async def _embed_prompt(self, query: CacheQuery) -> list[float]:
        """Compute an embedding vector for the cache query."""
        vectors = await self._embedder.embed([query.embedding_input])
        if not vectors:
            raise ValueError("Embedding service returned no vectors.")
        if len(vectors[0]) != self._embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{self._embedding_dimension}, received {len(vectors[0])}"
            )
        return vectors[0]

    async def _estimate_neighbor_stale_rate(
        self,
        points: Sequence[ScoredPoint],
    ) -> float | None:
        """Approximate stale-rate of nearest neighbors identified in Qdrant."""
        prompt_hashes: list[str] = []
        for point in points:
            payload = point.payload or {}
            prompt_hash = payload.get("prompt_hash")
            if isinstance(prompt_hash, str):
                prompt_hashes.append(prompt_hash)
        if not prompt_hashes:
            return None
        stale_rates = await self._repository.fetch_prompt_stale_rates(prompt_hashes)
        if not stale_rates:
            return None
        return sum(stale_rates.values()) / len(stale_rates)

    async def _materialize_hit(
        self, query: CacheQuery, points: list[ScoredPoint]
    ) -> CacheLookupResult:
        """Convert Qdrant scored points into cache entries."""
        neighbor_stale_rate = await self._estimate_neighbor_stale_rate(points)
        stale_candidate: tuple[CacheEntry, float] | None = None
        reasons: list[str] = []
        for point in points:
            payload = point.payload or {}
            entry_id_raw = payload.get("cache_entry_id")
            if not entry_id_raw:
                reasons.append("missing_entry_id")
                continue
            try:
                entry_id = UUID(entry_id_raw)
            except ValueError:
                reasons.append("invalid_entry_id")
                continue

            entry = await self._repository.get_entry(entry_id)
            if entry is None:
                reasons.append("entry_not_found")
                continue
            if entry.expires_at <= datetime.now(UTC):
                reasons.append("expired_entry")
                stale_candidate = stale_candidate or (entry, point.score)
                continue
            await self.record_hit(entry_id)
            refreshed = await self._repository.get_entry(entry_id)
            if refreshed is None:
                reasons.append("entry_not_found_after_hit")
                continue
            return CacheLookupResult(
                status=CacheDecisionStatus.HIT,
                query=query,
                neighbor_stale_rate=neighbor_stale_rate,
                hit=CacheHit(entry=refreshed, similarity=point.score),
                reasons=tuple(reasons),
            )
        if stale_candidate is not None:
            entry, similarity = stale_candidate
            return CacheLookupResult(
                status=CacheDecisionStatus.STALE_HIT,
                query=query,
                neighbor_stale_rate=neighbor_stale_rate,
                stale_entry=entry,
                stale_similarity=similarity,
                reasons=tuple(reasons or ("expired_entry",)),
            )
        return CacheLookupResult(
            status=CacheDecisionStatus.MISS,
            query=query,
            neighbor_stale_rate=neighbor_stale_rate,
            reasons=tuple(reasons or ("no_matches",)),
        )


def _normalize_chat_completion(path: str, payload: dict[str, Any]) -> CacheQuery | None:
    """Flatten a Chat Completions request into a cache query."""
    if payload.get("stream"):
        return None
    model = payload.get("model")
    messages = payload.get("messages")
    if not model or not isinstance(messages, list):
        return None

    prompt_segments = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_segments.append(f"{role}:{_stringify_content(content)}")

    prompt_text = "\n".join(prompt_segments)
    parameters = {k: v for k, v in payload.items() if k not in {"messages"}}
    return _build_query(path, model, prompt_text, parameters)


def _normalize_legacy_completion(
    path: str, payload: dict[str, Any]
) -> CacheQuery | None:
    """Flatten legacy completion payloads that use a prompt field."""
    if payload.get("stream"):
        return None
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not model or prompt is None:
        return None
    if isinstance(prompt, list):
        prompt_text = "\n".join(str(item) for item in prompt)
    else:
        prompt_text = str(prompt)
    parameters = {k: v for k, v in payload.items() if k not in {"prompt"}}
    return _build_query(path, model, prompt_text, parameters)


def _build_query(
    path: str,
    model: str,
    prompt_text: str,
    parameters: dict[str, Any],
) -> CacheQuery:
    """Construct the CacheQuery dataclass with consistent hashing."""
    normalized_prompt = prompt_text.strip()
    prompt_hash = sha256(normalized_prompt.encode("utf-8")).hexdigest()
    params_fingerprint = _stable_hash(parameters)
    request_fingerprint = _stable_hash(
        {
            "path": path.lower(),
            "model": model,
            "params_fingerprint": params_fingerprint,
            "prompt_hash": prompt_hash,
        }
    )
    return CacheQuery(
        request_path=path,
        model=model,
        prompt_text=normalized_prompt,
        embedding_input=normalized_prompt,
        request_fingerprint=request_fingerprint,
        params_fingerprint=params_fingerprint,
        prompt_hash=prompt_hash,
        parameters=parameters,
    )


def _stable_hash(payload: Any) -> str:
    """Serialize the payload deterministically and return its SHA-256 hash."""
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _stringify_content(content: Any) -> str:
    """Convert OpenAI content structures into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(
                    item.get("text")
                    or item.get("content")
                    or json.dumps(item, sort_keys=True)
                )
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    return str(content)

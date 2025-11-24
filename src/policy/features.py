"""Prompt featurization utilities for TTL bandit policies."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Protocol


RECENCY_KEYWORDS = (
    "today",
    "tonight",
    "now",
    "latest",
    "breaking",
    "this week",
    "this month",
    "recent",
    "current",
    "update",
    "news",
    "earnings",
    "weather",
    "price",
    "score",
    "release",
)

STABILITY_TERMS = (
    "how to",
    "tutorial",
    "reference",
    "history",
    "guide",
    "steps",
    "explain",
    "manual",
)

RELATIVE_TIME_PATTERN = re.compile(
    r"\b(?:yesterday|tomorrow|last\s+night|last\s+week|next\s+week|today|tonight|"
    r"this\s+(?:morning|afternoon|evening)|earlier\s+today|recently)\b",
    flags=re.IGNORECASE,
)

ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TEXTUAL_DATE_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)(?:\s+\d{1,2})(?:,\s*\d{4})?\b",
    flags=re.IGNORECASE,
)

URL_PATTERN = re.compile(r"https?://\S+")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
CURRENCY_PATTERN = re.compile(r"[$€£¥₹]")
TICKER_PATTERN = re.compile(r"\b[A-Z]{3,5}\b")


class EntityRecognizer(Protocol):
    """Strategy protocol for extracting entity spans."""

    def __call__(self, text: str) -> Iterable["EntitySpan"]:
        """Return entity spans present in the text."""


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """Named entity span returned by recognizers."""

    label: str
    text: str


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    """Optional metadata used during feature extraction."""

    prompt_hash: str
    route: str | None = None
    tenant_id: str | None = None
    rate_limit_pressure: float | None = None
    upstream_latency_p95_ms: float | None = None
    nearest_neighbor_stale_rate: float | None = None


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Container for numeric features."""

    values: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        """Return a shallow copy for downstream consumers."""
        return dict(self.values)


class FeatureExtractor:
    """Featurize prompts for contextual TTL policy decisions."""

    def __init__(
        self,
        *,
        entity_recognizer: EntityRecognizer | None = None,
        recency_keywords: tuple[str, ...] = RECENCY_KEYWORDS,
        stability_terms: tuple[str, ...] = STABILITY_TERMS,
    ) -> None:
        self._entity_recognizer = entity_recognizer
        self._recency_keywords = tuple(
            {kw.lower().strip(): kw for kw in recency_keywords}.keys()
        )
        self._stability_terms = tuple(
            {term.lower().strip(): term for term in stability_terms}.keys()
        )

    def extract(
        self, prompt: str, *, context: ExtractionContext | None = None
    ) -> FeatureVector:
        """Return engineered features for the provided prompt."""
        normalized = prompt.strip()
        lowered = normalized.lower()

        feature_map: dict[str, float] = {}

        feature_map["text.length_chars"] = float(len(normalized))

        tokens = [token for token in re.split(r"\s+", normalized) if token]
        token_count = len(tokens)
        feature_map["text.length_tokens"] = float(token_count)

        uppercase_tokens = sum(
            1 for token in tokens if token.isupper() and len(token) > 1
        )
        feature_map["structure.uppercase_ratio"] = (
            uppercase_tokens / token_count if token_count else 0.0
        )

        number_matches = NUMBER_PATTERN.findall(normalized)
        feature_map["structure.number_count"] = float(len(number_matches))
        feature_map["structure.number_ratio"] = (
            len(number_matches) / token_count if token_count else 0.0
        )

        feature_map["structure.mean_token_length"] = (
            len(normalized) / token_count if token_count else 0.0
        )

        feature_map["structure.contains_code_block"] = (
            1.0 if "```" in normalized else 0.0
        )
        feature_map["structure.url_count"] = float(len(URL_PATTERN.findall(normalized)))
        feature_map["structure.contains_url"] = (
            1.0 if feature_map["structure.url_count"] else 0.0
        )
        feature_map["structure.contains_currency_symbol"] = (
            1.0 if CURRENCY_PATTERN.search(normalized) else 0.0
        )
        feature_map["structure.contains_ticker_like"] = (
            1.0 if TICKER_PATTERN.search(normalized) else 0.0
        )

        recency_hits = sum(1 for kw in self._recency_keywords if kw in lowered)
        feature_map["recency.keyword_hits"] = float(recency_hits)
        feature_map["recency.relative_time_mentions"] = float(
            len(RELATIVE_TIME_PATTERN.findall(normalized))
        )
        feature_map["recency.explicit_dates"] = float(
            len(ISO_DATE_PATTERN.findall(normalized))
            + len(TEXTUAL_DATE_PATTERN.findall(normalized))
        )

        stability_hits = sum(1 for term in self._stability_terms if term in lowered)
        feature_map["stability.keyword_hits"] = float(stability_hits)
        feature_map["stability.code_like_tokens"] = (
            1.0 if any(token.endswith(":") for token in tokens) else 0.0
        )

        if self._entity_recognizer is not None:
            entity_counts: dict[str, float] = {}
            total_entities = 0
            for span in self._entity_recognizer(normalized):
                label = span.label.upper()
                entity_counts[label] = entity_counts.get(label, 0.0) + 1.0
                total_entities += 1
            for label, count in entity_counts.items():
                feature_map[f"entity.count.{label}"] = count
            feature_map["entity.total"] = float(total_entities)
        else:
            feature_map["entity.total"] = 0.0

        # Omitting the entire context, have no solid understanding of why these features are required?
        if context is not None:
            if context.route:
                feature_map[f"route.{context.route}"] = 1.0
            if context.tenant_id:
                feature_map[f"tenant.{context.tenant_id}"] = 1.0
            if context.rate_limit_pressure is not None:
                feature_map["ops.rate_limit_pressure"] = float(
                    context.rate_limit_pressure
                )
            if context.upstream_latency_p95_ms is not None:
                feature_map["ops.upstream_latency_p95_ms"] = float(
                    context.upstream_latency_p95_ms
                )
                # embeddings of neighbors should have similar staleness
            if context.nearest_neighbor_stale_rate is not None:
                feature_map["nn.avg_stale_rate"] = float(
                    max(0.0, min(context.nearest_neighbor_stale_rate, 1.0))
                )
            else:
                feature_map["nn.avg_stale_rate"] = -1.0
        else:
            feature_map["nn.avg_stale_rate"] = -1.0

        # Z-score clipping helper: ensure values remain bounded for exploration.
        for key in ("structure.number_count", "structure.number_ratio"):
            feature_map[key] = float(_clip(feature_map[key], -3.0, 3.0))

        return FeatureVector(values=feature_map)


def _clip(value: float, lower: float, upper: float) -> float:
    """Clamp a value into the provided range."""
    if math.isnan(value):
        return 0.0
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value

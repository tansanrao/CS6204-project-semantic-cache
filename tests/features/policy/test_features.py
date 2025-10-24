from __future__ import annotations

from app.features.semantic_cache.policy import build_default_entity_recognizer
from app.features.semantic_cache.policy.features import (
    EntitySpan,
    ExtractionContext,
    FeatureExtractor,
)


class DummyEntityRecognizer:
    """Return deterministic entities for testing."""

    def __call__(self, text: str):  # noqa: D401 - Protocol compatibility
        yield EntitySpan(label="ORG", text="NVIDIA")
        yield EntitySpan(label="PRODUCT", text="RTX 5090")


def test_feature_extractor_returns_expected_signals() -> None:
    prompt = (
        "Breaking news: Today, Oct 22, 2025, NVIDIA reports Q3 earnings.\n"
        "Stock ticker NVDA hit $450. Visit https://example.com/report for details.\n"
        "How to interpret results? Provide steps.\n"
    )
    extractor = FeatureExtractor(entity_recognizer=DummyEntityRecognizer())
    context = ExtractionContext(
        prompt_hash="hash123",
        route="chat",
        tenant_id="tenant_a",
        rate_limit_pressure=0.75,
        upstream_latency_p95_ms=480.0,
        nearest_neighbor_stale_rate=0.25,
    )

    vector = extractor.extract(prompt, context=context)
    features = vector.as_dict()

    assert features["recency.keyword_hits"] >= 3
    assert features["recency.explicit_dates"] == 1
    assert features["structure.url_count"] == 1.0
    assert features["structure.contains_currency_symbol"] == 1.0
    assert features["structure.contains_ticker_like"] == 1.0
    assert features["entity.total"] == 2.0
    assert features["entity.count.ORG"] == 1.0
    assert features["entity.count.PRODUCT"] == 1.0
    assert features["route.chat"] == 1.0
    assert features["tenant.tenant_a"] == 1.0
    assert features["ops.rate_limit_pressure"] == 0.75
    assert features["ops.upstream_latency_p95_ms"] == 480.0
    assert 0.0 <= features["nn.avg_stale_rate"] <= 1.0

    # Sanity check on ratios to ensure clipping keeps them bounded.
    assert 0.0 <= features["structure.uppercase_ratio"] <= 1.0
    assert -3.0 <= features["structure.number_ratio"] <= 3.0


def test_default_entity_recognizer_returns_spans() -> None:
    recognizer = build_default_entity_recognizer()
    spans = list(recognizer("OpenAI released GPT-5 in San Francisco on Oct 10, 2025."))
    assert isinstance(spans, list)
    for span in spans:
        assert isinstance(span, EntitySpan)

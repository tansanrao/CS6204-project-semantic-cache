# Reinforcement Learning TTL Policy — Implementation Snapshot

## Purpose
- Capture the currently delivered Flask semantic cache + TTL policy features so
  contributors share an accurate baseline before scheduling new work.

## Current Scope (24 Oct 2025)
- Semantic cache backed by Postgres and Qdrant with TTL-aware storage on cache
  misses.
- Contextual bandit policy (LinUCB baseline, Thompson optional) that selects TTL
  buckets and returns logged propensities.
- Refresh outcome handling that records feedback, computes rewards, and adjusts
  TTL buckets when entries prove fresh or stale.

## Implemented Areas

### Persistence & Data Pipeline
- Alembic migration `78f0d4afbc8a_add_ttl_policy_tables.py` adds ttl decision,
  feedback, and policy reward tables. `app/cache/models.py` mirrors the schema
  for other SQL engines.
- `app/cache/repository.py` writes cache entries, TTL decisions, feedback, and
  rewards. It aggregates bucket metrics, fetches prompt stale-rate estimates,
  and assembles refresh candidates.
- `app/tasks/jobs.py` builds `RefreshBatch` payloads that combine candidate
  entries with recent bucket aggregates for workers.

### Feature Engineering
- `app/policy/features.py` emits prompt features for length, structure,
  recency/stability cues, entity counts, operational context, and nearest-neigh
  stale-rate signals.
- `app/policy/ner.py` provides a default entity recognizer that prefers spaCy
  pipelines and falls back to regex heuristics when models are unavailable.
- `FeatureExtractor` normalizes feature values and clips numeric ratios so the
  bandit policy receives bounded floats.

### Bandit Policy Core
- `app/policy/bandit.py` implements `LinUCBPolicy` with snapshot persistence,
  optional prior weights, and propensity estimation, plus a Thompson variant.
- `app/policy/manager.py` wraps policies in `TTLPolicyManager` for thread-safe
  selection, reward updates, and autosave snapshots via `PolicyRuntimeConfig`.
- `SemanticCacheService.select_ttl_bucket` delegates to the manager (or defaults)
  and logs feature summaries for observability.
- `SemanticCacheService._compute_reward` maps `RefreshOutcome` signals into
  rewards aligned with DESIGN §6 coefficients.

### Proxy & Online Integration
- `app/proxy/service.py` normalizes OpenAI-compatible requests, consults the
  semantic cache, executes TTL selection on misses, persists cache entries, and
  logs policy decisions plus feedback with request-scoped metadata.
- Responses include `ttl_proxy` metadata with cache status, TTL bucket,
  propensity, and identifiers, alongside `X-Cache` + `X-Cache-Status` headers.
- `flask_app.py` wires configuration, request logging hooks, the proxy
  blueprint, semantic cache initialization (database, Qdrant, embeddings), and
  a background async runner for bridging blocking contexts.

### Refresh & Reward Loop
- `app/tasks/worker.py` defines `RefreshWorker`, which pulls batches via the
  repository, invokes a `RefreshPipeline`, and hands `RefreshOutcome` objects to
  result handlers.
- `SemanticCacheService.log_refresh_outcome` stores feedback events, computes
  rewards, updates TTL decisions, triggers bandit updates, and adjusts TTL
  buckets through `_maybe_extend_ttl` and `_apply_guardrails`.
- Guardrails shrink TTL buckets when stale detections happen before 25 % of the
  assigned TTL elapses. Confirmed-fresh outcomes can promote entries when
  `bonus_applicable` is set.

### Observability & Logging
- `app/logging.py` configures structured key=value logging, request-id
  propagation, and before/after hooks that emit request lifecycle events.
- `ProxyService._log_cache_decision` and semantic cache loggers emit status,
  reasons, latency, and feature summaries suitable for downstream ingestion.
- `app/runtime/async_runner.py` ensures async operations run with logging
  context restored when bridging from synchronous Flask handlers.

### Tests & Verification
- `tests/features/test_proxy.py` exercises cache hits/misses, TTL metadata,
  auth guards, and SSE streaming pass-through using stub backends.
- `tests/features/policy/test_features.py` validates feature extraction signals
  and the default entity recognizer.
- `tests/features/policy/test_bandit.py` covers LinUCB/Thompson policies,
  snapshot persistence, and manager autosave flows.
- `tests/ingest/ttl_refresh/test_jobs.py` plus `test_worker.py` verify refresh
  batch construction, worker coordination, and reward logging via the cache
  service.

## Notes
- Keep this document aligned with the living codebase; add future work only
  after it lands in the Flask app.
- When architecture shifts beyond this scope, update both `DESIGN.md` and this
  snapshot as part of the change.

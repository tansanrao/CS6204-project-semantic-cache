# Reinforcement Learning TTL Policy — Implementation Plan

## Purpose
- Track day-to-day progress while building the contextual bandit TTL policy described in `DESIGN.md` v0.2.
- Capture status for schema changes, feature engineering, policy algorithms, and observability so contributors can pick up the next actionable task.

## Current Focus (22 Oct 2025)
- [x] Draft schema migrations and repository interfaces for policy decision logging (`ttl_decision`, `feedback_event`, `policy_reward`). (22 Oct 2025 — Alembic stub `alembic/versions/78f0d4afbc8a_add_ttl_policy_tables.py`, applied via `uv run alembic upgrade head` against `semantic_cache`)
- [x] Prototype feature featurization module (`app/features/semantic_cache/policy/features.py`) with deterministic unit tests. (22 Oct 2025 — covered by `tests/features/policy/test_features.py`)
- [x] Implement LinUCB policy class with warm-start loading and checkpoint persistence. (22 Oct 2025 — validated via `tests/features/policy/test_bandit.py`)

## Workstreams & Milestones

### 1. Persistence & Data Pipeline
- [x] Define SQLAlchemy tables (cache decisions, rewards, feedback) + alembic/SQL migration stubs.
- [x] Extend repository layer to insert/look up decisions, rewards, feedback events.
- [x] Add batch job skeletons for pulling stale candidates and computing aggregate metrics. (22 Oct 2025 — `app/ingest/ttl_refresh/jobs.py`, repository helpers + `tests/ingest/ttl_refresh/test_jobs.py`)

### 2. Feature Engineering
- [x] Implement text preprocessing utilities (recency cues, structure metrics, normalization).
- [x] Integrate NER / entity type extraction (initially spaCy placeholder, pluggable backend). (22 Oct 2025 — default spaCy/regex recognizer wired via `FeatureExtractor`; see `app/features/semantic_cache/policy/ner.py` & proxy wiring.)
- [x] Wire Qdrant nearest-neighbor stats into feature vectors. (22 Oct 2025 — neighbor stale-rate estimate surfaced through `CacheLookupResult.neighbor_stale_rate` and logged with TTL decisions.)
- [ ] Cover featurization with tests under `tests/features/policy/`.
- [ ] Add global feature scaling/normalization and clip logic for additional numeric dimensions as described in DESIGN §2.2.
- [ ] Thread route/tenant/rate-limit/latency context from proxy into feature extraction so operational signals land in the bandit input vector.

### 3. Bandit Policy Core
- [x] LinUCB policy with per-bucket models, exploration parameter tuning, serialization hooks.
- [x] Thompson Sampling variant scaffolding for future experiments. (22 Oct 2025 — `ThompsonSamplingPolicy` + manager wiring in `app/features/semantic_cache/policy/` with unit coverage under `tests/features/policy/test_bandit.py`.)
- [x] Reward computation module reflecting §6 of `DESIGN.md` (base components, penalties, bonuses). (22 Oct 2025 — `_compute_reward` now feeds live policy updates via `SemanticCacheService.log_refresh_outcome`.)
- [x] Propensity logging for IPS / doubly-robust evaluation. (22 Oct 2025 — proxy records policy propensities on decisions, persisted through `TTLDecisionCreate`.)
- [ ] Implement warm-start training flow (multiclass logistic classifier) and expose prior weight seeding for LinUCB / Thompson policies.

### 4. Online Integration
- [x] Embed policy selection on cache misses inside proxy service. (22 Oct 2025 — `ProxyService.forward` queries `SemanticCacheService.select_ttl_bucket` and stores decisions with features.)
- [x] Emit structured decision logs (JSON) with bucket, features hash, propensity. (22 Oct 2025 — Proxy now records TTL decisions via `SemanticCacheService.log_ttl_decision`)
- [ ] Ensure cached entries record bucket + feature snapshot for future updates.
- [ ] Enqueue bucket-dependent freshness checks on cache hits (ρ(Bi) sampling) so refresh worker receives work items.
- [ ] Start background refresh worker(s) in FastAPI lifespan to process queued stale-hit evaluations.

### 5. Freshness & Reward Loop
- [x] Implement refresh worker to compare cached vs refreshed content, compute deltas, and determine staleness. (22 Oct 2025 — `app/ingest/ttl_refresh/worker.py`, covered by `tests/ingest/ttl_refresh/test_worker.py`)
- [x] Apply TTL extension rules when refreshed content agrees with cache. (22 Oct 2025 — `SemanticCacheService._maybe_extend_ttl` promotes buckets and updates TTL metadata.)
- [x] Persist refresh feedback and rewards derived from refresh outcomes. (22 Oct 2025 — `SemanticCacheService.log_refresh_outcome`, validations in `tests/features/test_semantic_cache_refresh.py`)
- [x] Guardrail logic for oversize TTL detection (stale-rate thresholds, survival curves). (22 Oct 2025 — early-stale guardrails shrink buckets via `SemanticCacheService._apply_guardrails` and annotate feedback.)
- [ ] Implement refresh pipeline stages (classify → retrieve → synthesize → compare) and ensure stale confirmations overwrite cached payloads with updated responses.

### 6. Observability & Ops
- [x] Simplified logging to Python defaults while keeping cache + policy telemetry at
      info/warning/error levels only. (24 Oct 2025 — `logging.basicConfig` in
      `app/main.py`, module loggers in proxy/semantic cache/policy emit `proxy decision`
      summaries without request-scoped helpers.)
- [ ] Expand guardrail handling with bucket-level stale/stability metrics (hazard curves, drift signals, canary probes) and use aggregates to adjust exploration priors.
- [ ] Prometheus/Grafana metrics: hit rate, stale rate by bucket, survival curves, early-stale ratio.
- [ ] CLI / admin endpoint to reload policy weights without deploy.
- [ ] Documentation updates (`DESIGN.md`, operations runbooks).

### 7. Validation & Experiments
- [ ] Unit/integration test matrix (policy math, featurization, reward shaping, end-to-end flows).
- [ ] Shadow deployment experiment plan (traffic sampling, evaluation windows).
- [ ] Success criteria: ≥5% latency savings, stale-hit rate ≤3% in monitored slices.

## Notes
- Update task checkboxes as milestones complete; add dates where helpful.
- If scope shifts, reflect changes both here and in `DESIGN.md`.
- Latest verification (22 Oct 2025): `uv run pytest tests/features/policy`.
- Proxy service integration tests (22 Oct 2025): `uv run pytest tests/features/test_proxy.py::test_proxy_logs_decision_and_feedback_events`.
- Semantic cache bootstrap runs `alembic upgrade head` against `PROXY_DATABASE_DSN` when using PostgreSQL (22 Oct 2025).

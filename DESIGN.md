# Contextual Bandit TTL Policy — Design v0.2

*Scope:* Nail down Option A (contextual bandit), clarify stale-hit handling and auto-refresh, define reward computation via cache-vs-refresh comparison, and propose signals/algorithms to detect oversized TTLs.

---

## 0) Terms

* **Hit:** Valid cache entry (not expired) served immediately.
* **Stale-hit:** Cache entry served but flagged stale by online checks (see §3), triggering refresh pipeline.
* **Miss:** No valid cache entry → call upstream LLM.
* **Bucket `Bi`:** One of 6 TTLs (B0..B5).
* **Context x:** Prompt features (§2.1).

---

## 1) Contextual Bandit Overview

We use a **contextual bandit** over 6 discrete actions (TTL buckets). Online learning selects a bucket per request based on prompt features.

**Policy choices** (choose one to implement first):

1. **LinUCB** (fast, interpretable): per-action linear model with uncertainty bonus.
2. **Thompson Sampling (Bayesian linear regression)**: per-action posterior over weights; sample and pick argmax.

*Implementation note (22 Oct 2025):* Both LinUCB and Thompson Sampling are wired through a shared `TTLPolicyManager`. Runtime selection is controlled via `semantic_cache_policy_type`, with snapshots persisted whenever policy weights are updated.

### 1.1 Runtime policy orchestration

`TTLPolicyManager` is the in-memory controller that exposes bandit decisions to the proxy. It coordinates feature indexing, computes propensities for IPS / DR evaluation, and persists checkpoints (JSON snapshots) on a configurable cadence. The proxy calls `SemanticCacheService.select_ttl_bucket(...)` during cache misses, which delegates to the manager for bucket selection and propensity logging.

**Warm start:** Pretrain a multiclass logistic classifier from heuristics and offline logs; use its weights as prior for Bayesian TS or as initial θ in LinUCB.

---

## 2) Features & Preprocessing

### 2.1 Features (x)

* **Recency cues:** tokens: today/now/latest/this week; relative time phrases; presence of explicit dates; newsy terms (earnings, score, weather, election, breaking, price, exchange rate, version, release).
* **Entity types:** PERSON/ORG/PRODUCT/EVENT/GPE; count and types.
* **Stability cues:** math/how-to/reference/historical; code blocks/stack traces; URL presence.
* **Nearest-neighbor freshness:** average stale rate of top-k (k=20) nearest prompts in Qdrant.
* **Prompt structure:** length, #numbers, #uppercase tokens, presence of currency/tickers/versions.
* **Route/tenant:** optional bias features.
* **Operational context:** current rate limit pressure, trailing p95 upstream latency (for cost-aware exploration).

### 2.2 Normalization

* Standardize numeric features (z-score, clipped to ±3). One-hot encode sparse categorical signals; keep feature space ≤ 1k dims.

---

## 3) Online Decision & Refresh Flow

**Request path**

1. Receive `(prompt, model, params)`.
2. **Lookup**: Qdrant semantic search (k=5), filter `expires_at > now()` and matching model/params; choose best candidate by similarity.
3. If candidate similarity ≥ τ_hit (e.g., 0.86) → **Hit** → return cached response immediately (**low latency path**). Attach metadata: `{cache: hit, ttl_bucket, age_s, similarity}`.
4. Else → **Miss** → compute features x → policy selects bucket `Bi` → call upstream LLM → store entry with `expires_at=now()+Bi` and `ttl_bucket=Bi` → return.

**Stale-hit detection (async + selective sync)**

* For served hits, enqueue a **freshness check** job according to bucket-dependent sampling rate `ρ(Bi)` (e.g., B0=0.5, B1=0.25, B2=0.15, B3=0.1, B4=0.05, B5=0.02). Higher for high-risk domains.
* Check performs **Refresh Pipeline** (§4). If delta indicates staleness, mark **stale-hit**; trigger auto-update to user if still on session stream (optional) or on next touch.

---

## 4) Refresh Pipeline (for suspected stale-hit or periodic scouts)

Goal: cheaply validate whether cached content has gone stale and repair it.

**4.1 Cheap validator model**

* Use a smaller/cheaper LLM for *information retrieval + synthesis*.
* Steps:

  1. **Query Classification:** Decide if external recency is needed (news/finance/weather/sports/etc.).
  2. **Retrieve:** Call web search tool; fetch 1–3 top documents.
  3. **Synthesize:** Ask small LLM to answer *the original prompt* from retrieved pages.

**4.2 Comparison & Decision**

* Compute a textual **diff score** between cached response `R_cache` and refreshed `R_new`:

  * `semantic_delta = 1 - sim( embed(R_cache), embed(R_new) )` (use same embedding model).
  * `fact_delta` via LLM judge with rubric: contradictory numbers/dates/entities → score in [0,1].
* Define **stale** if `(semantic_delta > σ_sem OR fact_delta > σ_fact)` and refreshed answer cites sources newer than `created_at`.
* If **not stale** (i.e., answers materially equivalent), the TTL was **too conservative** → see §5.2 (TTL tighten/extend).

**4.3 Write-back**

* If stale: update `response_text` to `R_new`, set `created_at=now()`, recompute `expires_at=now()+Bi` (keep original bucket for reward attribution), and log `feedback_event(user_accepted|auto_refresh)`.

---

## 5) TTL Adaptation Rules

### 5.1 When we correctly classify as stale (cached ≠ refreshed)

* No immediate bucket change for that entry (it was correct). But **policy learns** via negative reward for too-large buckets when stale occurs early (see §6.2).

### 5.2 When refreshed ≈ cached (false stale or overly conservative TTL)

* If `R_new` ≈ `R_cache` within thresholds for `k_confirm` successive checks while entry is near or past `t_fraction` of its TTL (e.g., >70%), then:

  * **Entry-level extension:** increase `expires_at` by `λ_extend * remaining_or_full_TTL` (e.g., 1.5× up to B_{i+1} cap) and log `feedback_event` with positive score.
  * **Policy hint:** emit positive reward for larger buckets in similar contexts (see reward shaping §6.3).

*Implementation note (22 Oct 2025):* When `RefreshOutcome.bonus_applicable` is set, `SemanticCacheService._maybe_extend_ttl` automatically promotes the entry to the next bucket (capped at the highest bucket) and records the adjustment inside refresh feedback metadata.

### 5.3 “TTLs too large” detection (population level)

Signals and actions:

* **Elevated stale-hit rate per bucket:** if `stale_rate(Bi)` > target (e.g., 5%) over rolling window N decisions, **shrink** exploration into `Bi` by decreasing its prior mean (TS) or increasing α (LinUCB) penalty; optionally put a hard cap: `max(Bi)`.
* **Time-to-contradiction hazard:** estimate survival curve S(t) (Kaplan–Meier) from time until first detected contradiction; if hazard h(t) spikes before bucket’s expiry, that bucket is oversized for that context slice.
* **Early-stale ratio:** proportion of stale detections before 25% of TTL elapsed. If > threshold, downshift bucket suggestions for similar contexts.
* **Drift detector:** monitor embedding centroid drift of web snapshots referenced by prompts; rising drift correlates with shorter optimal TTL.
* **Canary prompts:** maintain a small set of recurring prompts per domain; bisection search TTL by doubling/halving to find safe upper bounds.

Implementation notes (22 Oct 2025):

* `app/ingest/ttl_refresh/worker.py` coordinates refresh batches using the repository helpers and downstream pipelines.
* `SemanticCacheService.log_refresh_outcome` records refresh feedback (`feedback_event`) and reward attribution (`policy_reward`) for policy updates.
* Early-stale guardrails shrink TTL buckets automatically via `SemanticCacheService._apply_guardrails`, which triggers whenever a stale detection occurs before 25% of the assigned TTL has elapsed. Guardrail actions are annotated in refresh feedback for observability.

---

## 6) Reward Model

We compute reward for each decision (bucket choice) using logged outcomes and refresh comparisons.

### 6.1 Base components

Let

* `hit` ∈ {0,1}, `stale` ∈ {0,1} from §4 decision.
* `Δcost` = `cost_miss − cost_hit` (positive if cache saved money/time).
* `Δlat` = `latency_miss − latency_hit`.

Base reward:

```
r_base =  α * hit + β * norm(Δcost) + γ * norm(Δlat) − δ * stale
```

Defaults: α=0.2, β=0.3, γ=0.2, δ=0.6.

### 6.2 TTL-size penalty/bonus by timing

* If **stale==1**, add penalty scaled by **how early** staleness appeared:

```
  e = 1 - min( elapsed / ttl_seconds, 1.0 )     # early=close to 1
  r = r_base − η * e            # η≈0.4; stronger hit if staled early
```

* If **not stale** and refreshed≈cached after ≥ t_fraction (e.g., 70%) of TTL, add small **bonus** encouraging longer TTL for similar contexts:

```
  r = r_base + κ * min( elapsed / ttl_seconds, 1.0 )   # κ≈0.2
```

### 6.3 Counterfactual shaping

* Use the classifier’s softmax over buckets as a **propensity**; log `π(a|x)` for IPS/DR evaluation. Optionally apply
  Doubly-Robust estimator combining a simple value model.

---

## 7) Batch Jobs & SQL Sketches

**7.1 Pull stale candidates from DB**

```sql
-- Entries recently served as hits and eligible for check
SELECT ce.id, ce.prompt_text, ce.response_text, ce.created_at, ce.ttl_bucket, ce.ttl_seconds
FROM cache_entry ce
JOIN LATERAL (
  SELECT 1
) AS _ ON TRUE
WHERE ce.created_at > now() - interval '7 days'  
  AND ce.expires_at > now()  -- still within TTL
ORDER BY ce.created_at DESC
LIMIT 500;
```

**7.2 Record refresh outcome**

```sql
INSERT INTO feedback_event (cache_entry_id, event_type, score, details)
VALUES ($1, CASE WHEN $2 THEN 'user_accepted' ELSE 'eval_disagreed' END, $3,
        jsonb_build_object('semantic_delta',$4,'fact_delta',$5,'ref_urls',$6));
```

**7.3 Compute per-decision reward**

```sql
INSERT INTO policy_reward (ttl_decision_id, reward, attribution_rule)
VALUES ($1, $2, 'stale_check_v1');
```

---

## 8) Pseudocode (Python)

```python
# Policy interface
class TTLPolicy:
    def choose(self, x) -> int: ...  # returns bucket index 0..5
    def update(self, x, a, r): ...

# Online path
def serve(prompt, model, params):
    cand = qdrant_lookup(prompt, model, params)
    if cand and cand.sim >= TAU and cand.expires_at > now():
        enqueue_freshness_check(cand, sample_rate=cand_bucket_rate(cand.ttl_bucket))
        return cand.response_text, {"cache": "hit", "bucket": cand.ttl_bucket}

    x = featurize(prompt)
    a = policy.choose(x)
    seconds = BUCKET_SECONDS[a]
    upstream = call_llm(prompt, model, params)
    cache_put(prompt, upstream, a, seconds, x)
    log_decision(prompt, a, x)
    return upstream, {"cache": "miss", "bucket": a}

# Freshness check worker
def check_freshness(entry):
    need_recency = classify_recency(entry.prompt_text)
    R_new = refresh(entry.prompt_text, need_recency)  # small LLM + retrieval
    deltas = compare(entry.response_text, R_new)
    stale = is_stale(deltas, entry.created_at, R_new)

    r = compute_reward(entry, stale, deltas)
    policy.update(entry.features, entry.ttl_bucket, r)

    if stale:
        cache_update(entry.id, R_new)
        log_feedback(entry.id, 'auto_refresh', score=-1.0, details=deltas)
    else:
        maybe_extend_ttl(entry)
        log_feedback(entry.id, 'confirmed_fresh', score=+0.5, details=deltas)
```

---

## 9) Thresholds & Defaults (tune later)

* `τ_hit` (semantic sim): 0.86 (cosine) for nomic embeddings.
* Refresh thresholds: `σ_sem=0.25`, `σ_fact=0.35` (initial); require at least one retrieved source newer than `created_at` for staleness.
* `ρ(Bi)` sampling: {0.5, 0.25, 0.15, 0.10, 0.05, 0.02}.
* Extension rule: if 2 consecutive confirmations near TTL end → extend by 1.5× up to next bucket cap.
* Guardrails: max stale-hit rate per domain = 5%; circuit break to heuristic TTLs if exceeded over 500 recent decisions.

---

## 10) How to know TTLs are too large (summary)

* Bucket-level stale-hit > target.
* Early-stale ratio high.
* Survival/hazard shows high failure probability well before expiry.
* Canary prompts trip frequently at current TTL.
* Upstream drift metrics (web snapshots, news intensity) spike for domain.

**Action:** downshift priors and reduce exploration into large buckets for contexts matching the failing slice; optionally compress the bucket set for that domain (e.g., drop B5 temporarily).

---

## 11) Next Steps

* Implement refresh pipeline skeleton (classify→retrieve→synthesize→compare) with pluggable backends.
* Wire reward computation with timing-weighted penalty/bonus.
* Add bucket-level dashboards: stale-rate, survival curves, early-stale ratio.
* Run a shadow deployment on sampled traffic; tune thresholds and sampling rates.

---

## 12) OpenAI-Compatible Proxy Service

We expose a lightweight Flask service that fronts the vLLM gateway while preserving OpenAI semantics.

### Runtime overview

* `flask_app.py` bootstraps configuration, logging hooks, and the proxy blueprint in `app/proxy/`.
* `/v1/**` routes remain OpenAI-compatible and are handled synchronously with a shared `httpx.Client` for connection pooling and streaming.
* Streaming (`text/event-stream`) responses are relayed chunk-by-chunk without buffering to preserve token streaming behaviour.
* Each response attaches a `ttl_proxy` metadata block capturing cache status, TTL decision details, and policy propensity for client diagnostics.

### Authentication and configuration

* Runtime settings live in `app/config.py` as a dataclass loader. Environment variables retain the `PROXY_` prefix; notable options:
  * `PROXY_VLLM_BASE_URL`: base URL of the vLLM gateway (default `http://localhost:8000/v1`).
  * `PROXY_VLLM_API_KEY`: optional bearer token injected toward the backend.
  * `PROXY_INBOUND_API_KEYS`: optional comma-separated list of client tokens. When present, inbound requests must send `Authorization: Bearer <token>`; the proxy forwards the backend key instead.
  * `PROXY_REQUEST_TIMEOUT_SECONDS`: timeout applied to the shared `httpx` client.

### Error handling

* Backend responses (status/body/headers) are forwarded verbatim except hop-by-hop headers.
* Auth failures are rejected locally with 401/403 before contacting vLLM.

### Testing

* Feature coverage resides in `tests/features/test_proxy.py`, which drives the proxy with `httpx.MockTransport` to assert pass-through JSON calls, SSE streaming, auth guardrails, and cache hits.

### Deployment notes

* Local workflow: `uv sync --frozen --python 3.11` followed by `uv run python flask_app.py` (or `uv run flask --app flask_app run --debug`).
* Docker compose stacks remain unchanged; the proxy binds to the same HTTP port (default 8000) and forwards to the existing vLLM gateway.

### Observability & logging

* Concise key=value logging is configured via `app/logging.py`, which installs request hooks and a formatter that emits `timestamp level logger event key=value...` lines. Namespace-specific handlers (`app.proxy`, `app.semantic_cache`, `app.semantic_cache.repository`, `app.policy`) are attached explicitly so INFO logs survive any root logger reconfiguration that happens during reloads.
* Flask `before_request`/`after_request` handlers assign or propagate `X-Request-ID`, log lifecycle events (`proxy.request_start`, `proxy.request_complete`), and append the request id to responses.
* Proxy, semantic cache, repository, and policy modules emit scoped events (for example `proxy.cache_decision`, `semantic_cache.lookup_decision`, `semantic_cache.policy_reward`) by attaching `record.kv` payloads so downstream systems can parse hit/miss telemetry, TTL decisions, and rewards. The semantic cache runner re-enables these loggers before each async operation to guard against Flask disabling them while handling streaming responses.
* `REQUEST_ID_VAR` in `app/logging.py` exposes the current request id for downstream propagation (database rows, Qdrant payloads, vLLM headers) and log correlation.

---

## 13) Semantic Cache Implementation Snapshot — 22 Oct 2025

*Embedding runtime*

* The Flask app loads `nomic-ai/nomic-embed-text-v1.5` (SentenceTransformer, `prompt_name="clustering"`) during bootstrap when `PROXY_SEMANTIC_CACHE_ENABLED=true`. Embeddings are Matryoshka-trimmed to 768 dims and normalized before similarity calculations.
* `EmbeddingService` defers model initialization to first use and runs encode calls via `asyncio.to_thread` to avoid blocking the event loop.

*Storage backends*

* Postgres schema lives in `app/cache/models.py`. Table `cache_entry` holds request hashes, response payloads, TTL metadata, and embedding provenance (model/mode/dimension). `cache_entry.request_fingerprint` is unique; indexes exist for `model` and `expires_at` to speed lookups and expiry sweeps.
* Qdrant bootstrap uses `Distance.COSINE`, vector size 768, and stores payload metadata (`cache_entry_id`, `model`, `params_fingerprint`, `expires_at_ts`, etc.). Collection name defaults to `cache_entries` and is validated for dimensional drift during startup.

*Application wiring*

* `ProxyService.forward` now normalizes OpenAI chat/completions requests, probes Qdrant (k=5) for hits above `τ_hit=0.86`, and short-circuits the response path with cached payloads (adds `X-Cache` headers). On misses, successful JSON responses are persisted back through Postgres + Qdrant.
* Startup (`app.main.create_app`) initializes the semantic cache stack (async SQLAlchemy engine, `QdrantClient`, embedder) and runs Alembic migrations (`alembic upgrade head`) against `PROXY_DATABASE_DSN` when `semantic_cache_bootstrap=True` before the server begins accepting traffic.

*Configurability*

* New env toggles (`PROXY_SEMANTIC_CACHE_*`) control enablement, bootstrap, similarity threshold, search fan-out, default TTL bucket, and bucket durations.
* Default TTL buckets map to `[60, 300, 900, 1800, 3600, 7200]` seconds (bucket index 2 currently used for misses until the bandit policy lands).

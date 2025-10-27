# 6204 Project: Semantic Cache
## Quick Summary
We adapt cache TTLs for LLM responses using a contextual bandit (Hybrid LinUCB) over 6 discrete TTL buckets. The policy picks a TTL based on prompt/context features, then learns from freshness checks and outcomes. We target **cost & latency reduction** while keeping **stale-hit < 5%** as constraints. A two-stage stale detection pipeline (retrieve→synthesize→compare) provides reliable feedback. 

Why this approach?
- **High variance freshness:** Some prompts are stable for hours (how‑to, reference) while others stale quickly (news, prices). One-size TTL wastes money or risks staleness.
- **Fast online learning:** Bandits balance exploration/exploitation with low compute overhead versus full RL. LinUCB is interpretable and easy to debug.
- **Operational fit:** TTL decisions are per‑request, single‑step rewards; no long credit assignment required.

## System Overview
**Inputs:** (prompt, model, params) + operational signals.  
**Policy:** Hybrid LinUCB over TTL buckets `[1, 10, 60, 180 ,720, 1440]` mins.  
**Cache:** Qdrant cosine distance based similarity → hit/miss. Cache hits compared with TTL for staleness check.
**Feedback:** Sampled freshness checks by bucket; rewards based on staleness penalties. 

## Key Design Choices:
### Policy: Hybrid LinUCB (v1)
- **What:** Shared/global weights + per-bucket parameters; UCB for exploration.
- **Why:**
    - **Interpretable** coefficients support feature attribution & audits.
    - **Data efficiency** by sharing across buckets; stable under nonstationary traffic.
    - **Operationally simple** (no sampling temperature tuning).
### Two-Stage Stale Detection
- **What:** (1) cheap validator (LLM) retrieves 1–3 sources and synthesizes a fresh answer; (2) compare cached vs new via **semantic delta** + **QA factual delta**. (cosine distance for semantic delta, LLM judge for factual delta)
- **Why:** Low-latency, high-precision stale signals from hybrid semantic+factual to train the bandit; avoids false positives from superficial text changes.

### Storage & Embeddings
- **What:** Nomic v1.5 embeddings (Matryoshka-trimmed to 512 dims); Qdrant (cosine) + Postgres metadata.
- **Why:** Fast approximate search, good semantic recall at modest cost; payload indexing supports efficient expiry scans and policy analytics.


## Online Flow
1. **Lookup:** Qdrant for a valid (not expired) near-duplicate with `τ_hit = 0.86`. (This is tunable) If hit → return immediately; enqueue freshness check with bucket‑dependent rate `ρ(Bi)`.
2. **Miss path:** Compute features → bandit selects TTL bucket `Bi` → call upstream LLM → write-through cache with `expires_at = now + Bi`.
3. **Freshness checks (for hits):** Sampled by `ρ(Bi)`; run validator, compute deltas:
    - `semantic_delta = 1 − cos(embed(R_cache), embed(R_new))`
    - `fact_delta` via QA-based factual metric Mark **stale** if `(semantic_delta > σ_sem OR fact_delta > σ_fact)` and sources are newer than `created_at`.
4. **Stale-hit Refresh Path**: compare semantic + factual similarity, ensure stale hit actually stale, provide feedback to classifier with appropriate reward.
**Defaults:** `σ_sem = 0.25`, `σ_fact = 0.35`, `ρ(Bi) = {0.5, 0.25, 0.15, 0.10, 0.05, 0.02}`.

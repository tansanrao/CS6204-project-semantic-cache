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

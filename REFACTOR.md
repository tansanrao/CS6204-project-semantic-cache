# Flask Proxy Refactor Plan

This plan replaces the existing FastAPI-based service with a streamlined Flask application that fronts the vLLM gateway, preserves the semantic cache and contextual bandit features, and improves operational observability through concise, human-readable logs.

---

## 1. Objectives
- **Swap FastAPI for Flask** to simplify the serving stack while keeping OpenAI-compatible proxy semantics (non-streaming + streaming completions).
- **Retain semantic cache + RL TTL policy** but refactor the code into clearer, modular components that are easy to reason about.
- **Improve developer UX**: lightweight dependency graph, minimal boilerplate, fewer moving parts, pragmatic testing.
- **Logging-first observability**: tail-able, readable log lines covering request lifecycle, cache behaviour, vector DB latency, and policy decisions. JSON is not required; emphasis is on short key=value lists.
- **Avoid bloat**: keep dependencies lean, tests only where behaviour is tricky (bandit math, cache expiry logic), no corporate ceremony.

---

## 2. Guiding Principles
1. **Single runnable module:** one Flask app (`flask_app.py`) bootstrapping configuration, logging, routes, and background tasks.
2. **Plain modules over frameworks:** keep business logic in `app/cache`, `app/policy`, `app/proxy` with simple functions/classes.
3. **Dependency Light:** rely on Flask + httpx + SQLAlchemy/Qdrant client. Remove Starlette/FastAPI, Pydantic models, and ASGI middleware.
4. **Concise logging:** standard `logging` emitting lines like `proxy request_id=... method=... status=... cache=hit latency_ms=...`.
5. **Preserve design intent:** maintain contextual bandit policy, TTL buckets, refresh pathways, but simplify surfaces/interfaces.

---

## 3. Target Architecture

```
flask_app.py
└── create_app(settings)                   # config + logging + blueprint registration
    ├── setup_logging()                    # human-readable formatter
    ├── register_routes(app)               # /v1/* OpenAI proxy endpoints
    ├── init_cache_stack(settings)         # Postgres + Qdrant + embedding service
    ├── init_policy_manager(settings)      # LinUCB/Thompson manager
    └── background tasks (refresh worker)  # via threading.Timer or asyncio if needed
```

### Core packages
- `app/config.py`: environment settings (minimal, dataclasses or dotenv).
- `app/logging.py`: logging setup, request-id helper, log utility functions.
- `app/proxy/`
  - `router.py`: Flask Blueprint for `/v1/*`.
  - `service.py`: request handling, semantic cache integration, streaming helper.
  - `openai_adapter.py`: normalize OpenAI payloads.
- `app/cache/`
  - `service.py`: semantic cache orchestration (lookup, store, TTL updates).
  - `repository.py`: Postgres operations.
  - `vector.py`: Qdrant interactions.
  - `models.py`: dataclasses for cache entities.
- `app/policy/`
  - `manager.py`: contextual bandit runtime.
  - `linucb.py`, `thompson.py`: algorithms (reuse existing logic, simplify).
  - `features.py`: featurization utilities.
- `app/tasks/refresh.py`: optional background refresh worker (reused from existing code but simplified).

### Flask specifics
- Use `blueprints` for proxy routes (`Blueprint("proxy", __name__)`).
- Leverage Flask `before_request` / `after_request` to assign request IDs and emit logs (instead of ASGI middleware).
- Streaming via Flask `Response` generator hooking into httpx streaming.

---

## 4. Logging Strategy
- Configure root logger with plain formatter: `%(asctime)s %(levelname)s %(name)s %(message)s`, and attach dedicated handlers to the `app.proxy`, `app.cache`, `app.semantic_cache`, and `app.policy` loggers so INFO-level cache telemetry survives any subsequent root-level reconfiguration.
- Helpers to emit key=value sequences:
  - `log_request_start(request_id, method, path, client_ip)`
  - `log_request_complete(request_id, status, cache_status, latency_ms, bytes_out)`
  - `log_cache_event(request_id, event, **fields)` e.g. `cache_hit`, `cache_miss`, `cache_store`.
  - `log_policy_event(request_id, action, propensity, feature_summary)`
- When cache operations hop onto the background asyncio runner, explicitly re-enable the namespace loggers before executing each coroutine to guard against Flask temporarily disabling them during request handling.
- Request ID propagation:
  - `g.request_id` assigned in `before_request` (reuse inbound `X-Request-ID` or `uuid4()`).
  - Inject into downstream calls (headers to vLLM, repo logs).

Example log:
```
proxy request_id=123 method=POST path=/v1/chat/completions cache=hit status=200 latency_ms=58.2 bytes_out=487 ua=openai/py
cache lookup request_id=123 decision=hit similarity=0.934 ttl_bucket=2 qdrant_ms=12.7
policy select request_id=456 action=3 propensity=0.41 top_features=recency=0.8,numbers=0.4
```

---

## 5. Migration Steps

### 5.1 Remove FastAPI stack
- Delete `/app/main.py`, FastAPI router/middleware modules, any Starlette dependencies.
- Clean `pyproject.toml` to drop FastAPI-related packages.
- Update entrypoint command to `uv run python flask_app.py` (or `uv run flask --app flask_app run`).

### 5.2 Introduce Flask app
1. Add `flask_app.py` with application factory & CLI runner.
2. Implement request ID hooks and logging based on plan above.
3. Register blueprint covering `/v1` routes; replicate existing path matching semantics.

### 5.3 Proxy layer rewrite
- Translate `ProxyService.forward` into Flask view function(s):
  - Accepts request, builds httpx request, handles streaming response via generator.
  - Integrate semantic cache lookup & store (same logic, but reorganized).
  - Emit log lines at key checkpoints.
- Ensure OpenAI compatibility (headers, SSE streaming).

### 5.4 Semantic cache & policy refactor
- Move/rename modules into new packages; replace settings dependency injection with direct config objects.
- Simplify dataclasses & return types (ditch Pydantic models).
- Keep synchronous API; use `asyncio` only where necessary (embedding if still async).
- Extract Qdrant + Postgres interactions into straightforward functions; remove `async` SQLAlchemy if possible (switch to sync engine for Flask simplicity) — evaluate effort; otherwise wrap async with `asyncio.run`.

### 5.5 Background refresh
- Re-assess worker complexity. For initial refactor, keep manual trigger or simple `Thread` loop.
- Ensure logs for refresh outcomes follow the concise format.

### 5.6 Configuration cleanup
- Replace Pydantic settings with simple `dataclass` + env parsing (e.g., `python-dotenv` or manual).
- Update `.env` / `example.env` references.
- Modify docs (`DESIGN.md`, `IMPLEMENTATION_PLAN.md`, `AGENTS.md` if needed) to reflect Flask architecture and new commands.

---

## 6. Testing Approach
- Keep existing critical tests focusing on:
  - Policy selection math.
  - Cache lookup/store logic.
  - Proxy streaming integration (mocked).
- Remove or downscope tests tied to FastAPI specifics.
- Add one high-level test covering Flask route (optional smoke).

---

## 7. Deliverables
1. **Flask application (`flask_app.py`)** with logging, request ID propagation, and blueprint registration.
2. **Refactored modules** under `app/cache`, `app/policy`, `app/proxy`, and simplified configuration/logging utilities.
3. **Updated documentation** (README snippet, DESIGN.md adjustments, IMPLEMENTATION_PLAN.md tasks).
4. **Lean dependency manifest** without FastAPI/Starlette.
5. **Retained schema & migrations** but confirm compatibility with new code paths.

---

## 8. Open Questions / TODOs
- Decide whether to keep async SQLAlchemy/Qdrant clients or switch to synchronous alternatives.
- Revisit refresh worker scheduling strategy under Flask (thread vs external job).
- Validate embedding service integration (async vs sync) with Flask request lifecycle.
- Confirm streaming compatibility with vLLM using Flask + httpx.

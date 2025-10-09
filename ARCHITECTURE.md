# Architecture Documentation

## System Overview

The Semantic Cache system provides intelligent caching for Large Language Model (LLM) responses using vector embeddings to match semantically similar prompts.

## Core Components

### 1. Configuration Layer (`config/`)

**Purpose**: Centralized configuration management

**Key Classes**:
- `Config`: Master configuration container
- `VLLMConfig`: vLLM service configuration
- `PostgresConfig`: PostgreSQL database configuration
- `QdrantConfig`: Qdrant vector DB configuration
- `CacheConfig`: Cache behavior configuration

**Design Decisions**:
- Environment variable support for deployment flexibility
- Validation methods to catch configuration errors early
- Dataclass-based for type safety and IDE support

### 2. Embedding Service (`src/embeddings/`)

**Purpose**: Generate vector embeddings from text

**Key Class**: `EmbeddingService`

**Features**:
- Lazy loading of transformer model (only loads when first used)
- Automatic dimension detection
- Text normalization (whitespace cleanup)
- Batch processing support

**Model**: `nomic-ai/nomic-embed-text-v1.5` (768 dimensions)

**Why This Model**:
- High quality embeddings
- Open source, no API keys needed
- Good balance of speed and accuracy
- Well-supported by sentence-transformers

### 3. Database Layer (`src/db/`)

**Purpose**: Isolated database operations

#### PostgreSQL Manager (`postgres.py`)

**Responsibilities**:
- Store prompt-response pairs
- Track access statistics
- Provide CRUD operations
- Handle transactions safely

**Schema**:
```sql
CREATE TABLE llm_cache (
    id UUID PRIMARY KEY,           -- Unified ID (links to Qdrant)
    prompt TEXT NOT NULL,          -- Normalized prompt
    response TEXT NOT NULL,        -- LLM response
    model VARCHAR(255) NOT NULL,   -- Model identifier
    metadata JSONB,                -- Additional metadata
    created_at TIMESTAMP,          -- Creation time
    accessed_at TIMESTAMP,         -- Last access time
    access_count INTEGER           -- Number of accesses
)
```

**Key Features**:
- Context manager for safe connection handling
- Atomic updates for access statistics
- JSONB for flexible metadata storage

#### Qdrant Manager (`qdrant.py`)

**Responsibilities**:
- Store embedding vectors
- Perform similarity search
- Manage collection lifecycle
- Handle vector operations

**Collection Structure**:
- **Vector Size**: 768 dimensions
- **Distance Metric**: Cosine similarity
- **Point ID**: UUID string (same as PostgreSQL ID)
- **Payload**: model, prompt preview

**Key Features**:
- Automatic dimension detection and validation
- Cosine similarity for semantic matching
- Configurable similarity threshold

### 4. Cache Layer (`src/cache/`)

**Purpose**: Core caching logic

#### SemanticCache (`semantic_cache.py`)

**Responsibilities**:
- Orchestrate cache operations
- Coordinate embedding, Qdrant, and PostgreSQL
- Implement cache get/set logic
- Manage statistics

**Key Algorithm**:

```python
def get(prompt, model):
    # 1. Generate embedding
    embedding = embedding_service.generate(prompt)
    
    # 2. Search Qdrant for similar vectors
    results = qdrant.search(embedding, threshold)
    
    if no_results:
        return None  # Cache miss
    
    # 3. Get entry ID from best match
    entry_id = results[0].id
    
    # 4. Retrieve from PostgreSQL (updates stats)
    entry = postgres.get_entry(entry_id)
    
    return entry
```

```python
def set(prompt, response, model):
    # 1. Generate deterministic UUID
    entry_id = generate_uuid(prompt, model)
    
    # 2. Generate embedding
    embedding = embedding_service.generate(prompt)
    
    # 3. Store in PostgreSQL (with UUID)
    postgres.insert(entry_id, prompt, response, model)
    
    # 4. Store vector in Qdrant (same UUID)
    qdrant.upsert(entry_id, embedding, metadata)
```

**UUID Generation**:
- Deterministic: SHA256(prompt + model) → UUID
- Same prompt+model always generates same UUID
- Links PostgreSQL and Qdrant entries
- Makes cache lookups idempotent

#### CachedLLMClient (`cache_client.py`)

**Responsibilities**:
- Wrap OpenAI-compatible LLM clients
- Intercept chat completion calls
- Check cache before LLM calls
- Store responses in cache
- Track statistics

**Key Features**:
- Transparent caching (drop-in replacement)
- Optional cache bypass
- Verbose mode for debugging
- Compatible response format

### 5. Validation & Health Checks (`src/utils/`)

**Purpose**: System health verification

**Key Functions**:

- `test_vllm_connection()`: Verify vLLM is up and model loaded
- `test_postgres_connection()`: Check PostgreSQL connectivity
- `test_qdrant_connection()`: Check Qdrant connectivity
- `recreate_databases()`: Clean slate initialization
- `validate_system()`: Complete system health check

**Used By**: Cell 1 in notebook for initialization

## Data Flow

### Cache Miss Flow

```
User Prompt
    ↓
[CachedLLMClient]
    ↓
[Convert messages to prompt string]
    ↓
[Generate embedding via EmbeddingService]
    ↓
[Search Qdrant via QdrantManager]
    ↓
No similar vectors found (< threshold)
    ↓
[Call LLM via OpenAI client]
    ↓
[Store response in PostgreSQL via PostgresManager]
    ↓
[Store embedding in Qdrant via QdrantManager]
    ↓
Return response to user
```

### Cache Hit Flow

```
User Prompt
    ↓
[CachedLLMClient]
    ↓
[Convert messages to prompt string]
    ↓
[Generate embedding via EmbeddingService]
    ↓
[Search Qdrant via QdrantManager]
    ↓
Similar vector found (>= threshold)
    ↓
[Extract UUID from result]
    ↓
[Retrieve from PostgreSQL via PostgresManager]
    ↓
[Update access statistics]
    ↓
Return cached response to user
```

## Key Design Patterns

### 1. Separation of Concerns
- Each component has a single, well-defined responsibility
- Database operations isolated from cache logic
- Embedding generation separated from storage

### 2. Dependency Injection
- Components receive dependencies via constructor
- Easy to swap implementations (e.g., different embedding models)
- Facilitates testing with mocks

### 3. Context Managers
- Safe resource handling (database connections)
- Automatic cleanup even on errors
- Pythonic and idiomatic

### 4. Lazy Loading
- Embedding model loaded only when first used
- Reduces startup time
- Saves memory if embeddings not needed

### 5. Deterministic ID Generation
- Same input always produces same UUID
- Makes operations idempotent
- Simplifies debugging and data consistency

## Configuration Management

### Priority Order
1. Explicit parameters to constructors
2. Environment variables
3. Default values in config.py

### Example
```python
# Method 1: Use defaults
config = Config()

# Method 2: Override specific values
config = Config()
config.cache.similarity_threshold = 0.90

# Method 3: Environment variables
# Set CACHE_SIMILARITY_THRESHOLD=0.90
config = Config.from_env()

# Method 4: Direct parameter
cache = SemanticCache(
    config=config,
    similarity_threshold=0.90  # Overrides config
)
```

## Error Handling Strategy

### Database Errors
- Connection errors logged and raised
- Transaction rollback on failures
- Context managers ensure cleanup

### Cache Errors
- Cache failures don't block LLM calls
- Graceful degradation (bypass cache)
- Detailed logging for debugging

### Embedding Errors
- Model loading failures raised immediately
- Encoding errors logged with context
- No silent failures

## Performance Considerations

### Embedding Generation
- **First call**: ~100-500ms (model loading)
- **Subsequent calls**: ~10-50ms per prompt
- **Optimization**: Lazy loading, batch processing available

### Vector Search
- **Typical**: 5-20ms for small collections (<10k vectors)
- **Scales**: Sub-linear with HNSW index
- **Threshold**: Higher threshold = faster (fewer candidates)

### PostgreSQL Lookups
- **Typical**: 1-5ms with UUID primary key
- **Index**: Automatic on primary key
- **Optimization**: Connection pooling available

### Overall Cache Hit
- **Total latency**: ~15-75ms (embedding + search + lookup)
- **Compared to LLM**: 10-100x faster
- **Trade-off**: Storage cost vs. speed gain

## Scalability Considerations

### Current Scale
- **Design target**: 10k-100k cached prompts
- **Vector DB**: Qdrant handles millions of vectors
- **SQL DB**: PostgreSQL scales to billions of rows

### Bottlenecks
1. **Embedding generation**: CPU-bound, consider GPU
2. **Vector search**: Memory-bound, Qdrant is efficient
3. **Storage**: Disk space grows with cache size

### Future Improvements
1. **Cache eviction**: LRU or TTL-based cleanup
2. **Distributed embedding**: Multiple workers
3. **Connection pooling**: Reuse DB connections
4. **Async operations**: Non-blocking cache checks

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock external dependencies
- Focus on logic correctness

### Integration Tests
- Test component interactions
- Use test databases
- Verify data consistency

### End-to-End Tests
- Full workflow from prompt to response
- Real databases (test environment)
- Performance benchmarks

## Monitoring & Observability

### Current Metrics
- Cache hit/miss rate
- Average similarity scores
- Access frequency per entry
- Total cache size

### Logging Levels
- **DEBUG**: Detailed operation logs
- **INFO**: Major events (cache init, hits/misses)
- **WARNING**: Degraded performance, fallbacks
- **ERROR**: Failures, exceptions

### Future Enhancements
- Prometheus metrics export
- Grafana dashboards
- Distributed tracing
- Performance profiling

## Security Considerations

### Data Privacy
- Prompts and responses stored in plain text
- No built-in encryption (use at DB level)
- Consider PII scrubbing for production

### API Keys
- vLLM uses dummy key (no authentication)
- PostgreSQL password in config (use secrets manager)
- No external API calls for embeddings

### Network Security
- All connections over localhost by default
- Use TLS for production deployments
- Firewall rules for database access

## Future Roadmap

### Short Term
- [ ] Cache eviction policies (LRU, TTL)
- [ ] Connection pooling
- [ ] Async API support
- [ ] More comprehensive tests

### Medium Term
- [ ] Multiple embedding model support
- [ ] Cache warming utilities
- [ ] Performance profiling tools
- [ ] Docker deployment guide

### Long Term
- [ ] Distributed cache support
- [ ] Cache replication
- [ ] Advanced analytics dashboard
- [ ] ML-based cache optimization

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-09

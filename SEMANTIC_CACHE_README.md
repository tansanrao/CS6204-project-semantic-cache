# Semantic Cache for LLM

This implementation provides a semantic caching layer for LLM responses, similar to GPTCache, using:
- **OpenAI's text-embedding-3-small** model for generating embeddings
- **Qdrant** for vector similarity search
- **PostgreSQL** for storing prompt-response pairs

## Architecture

```
User Query
    ↓
[Generate Embedding] ← OpenAI text-embedding-3-small
    ↓
[Search Qdrant] ← Find similar prompts (cosine similarity)
    ↓
  Cache Hit? (similarity > threshold)
    ↓ Yes                    ↓ No
[Get from PostgreSQL]    [Query LLM]
    ↓                        ↓
Return cached response   [Store in PostgreSQL + Qdrant]
                            ↓
                        Return LLM response
```

## Setup

### 1. Start the databases

The databases are already configured in `db-stack/docker-compose.yaml`:

```bash
cd db-stack
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Qdrant on ports 6333 (HTTP) and 6334 (gRPC)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `psycopg2-binary`: PostgreSQL adapter
- `qdrant-client`: Qdrant vector database client

### 3. Set up OpenAI API Key

You need an OpenAI API key to use the embedding model:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

## Usage

### Basic Usage

```python
from openai import OpenAI
from semantic_cache import SemanticCache
from cached_llm_client import CachedLLMClient

# Initialize your LLM client (vLLM in this case)
llm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

# Initialize the semantic cache
cache = SemanticCache(
    openai_api_key="your-openai-api-key",
    similarity_threshold=0.95  # 95% similarity required for cache hit
)

# Wrap your client with caching
cached_client = CachedLLMClient(
    llm_client=llm_client,
    cache=cache,
    verbose=True  # Show cache hits/misses
)

# Use it like a normal OpenAI client
response = cached_client.chat_completions_create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"Cached: {response.cached}")
```

### Configuration Options

#### SemanticCache Parameters

- **openai_api_key**: Your OpenAI API key for embeddings
- **openai_base_url**: OpenAI API base URL (default: `https://api.openai.com/v1`)
- **qdrant_host**: Qdrant server host (default: `localhost`)
- **qdrant_port**: Qdrant server port (default: `6333`)
- **postgres_config**: PostgreSQL connection config (default: localhost with provided credentials)
- **collection_name**: Qdrant collection name (default: `llm_cache`)
- **similarity_threshold**: Minimum cosine similarity for cache hit (default: `0.95`, range: 0-1)
- **embedding_model**: OpenAI embedding model (default: `text-embedding-3-small`)

#### Adjusting Similarity Threshold

The `similarity_threshold` determines how similar a query must be to trigger a cache hit:

- **0.95-1.0** (strict): Only very similar queries hit cache - fewer cache hits, more accurate
- **0.85-0.95** (moderate): Reasonable similarity required - good balance
- **0.70-0.85** (loose): More cache hits, but may return responses for less similar queries

```python
# Strict caching - only nearly identical queries
cache = SemanticCache(openai_api_key="...", similarity_threshold=0.98)

# Moderate caching - recommended for most use cases
cache = SemanticCache(openai_api_key="...", similarity_threshold=0.90)

# Loose caching - more aggressive caching
cache = SemanticCache(openai_api_key="...", similarity_threshold=0.80)
```

### Cache Statistics

```python
stats = cached_client.get_stats()

print(f"Total Requests: {stats['client_stats']['total_requests']}")
print(f"Cache Hits: {stats['client_stats']['cache_hits']}")
print(f"Hit Rate: {stats['client_stats']['hit_rate']:.1%}")
print(f"Total Cache Entries: {stats['cache_stats']['total_entries']}")
```

### Cache Management

```python
# Clear all cache entries
cached_client.clear_cache()

# Reset statistics
cached_client.reset_stats()

# Disable caching temporarily
cached_client.enable_cache = False
```

## How It Works

### 1. Prompt Normalization
When a prompt is received, it's normalized by:
- Trimming whitespace
- Collapsing multiple spaces
- This helps match similar prompts even with formatting differences

### 2. Embedding Generation
The normalized prompt is sent to OpenAI's `text-embedding-3-small` model to generate a 1536-dimensional vector embedding that captures the semantic meaning.

### 3. Similarity Search
The embedding is used to search Qdrant for similar prompts using cosine similarity. If a prompt with similarity > threshold is found, it's a cache hit.

### 4. Storage
- **Qdrant**: Stores embeddings for fast similarity search
- **PostgreSQL**: Stores the actual prompt-response pairs, metadata, and access statistics

### 5. Cache Hit/Miss
- **Hit**: Return the cached response immediately
- **Miss**: Query the LLM, store the response, and return it

## Example Scenarios

### Exact Match
```python
# First call - cache miss
response1 = cached_client.chat_completions_create(
    model="...",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
# Result: Cache MISS → queries LLM

# Second call - exact match - cache hit
response2 = cached_client.chat_completions_create(
    model="...",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
# Result: Cache HIT (similarity: 1.000)
```

### Semantic Similarity
```python
# Original query (already cached from above)
# "What is the capital of France?"

# Semantically similar query
response = cached_client.chat_completions_create(
    model="...",
    messages=[{"role": "user", "content": "What's the capital city of France?"}]
)
# Result: Cache HIT (similarity: 0.97)
```

### Different Query
```python
response = cached_client.chat_completions_create(
    model="...",
    messages=[{"role": "user", "content": "What is the capital of Germany?"}]
)
# Result: Cache MISS (similarity too low)
```

## Performance Benefits

1. **Reduced Latency**: Cached responses are returned in ~10-50ms vs. 1-5 seconds for LLM calls
2. **Cost Savings**: No LLM API calls for cached responses
3. **Resource Efficiency**: Reduces load on your LLM server
4. **Semantic Matching**: Unlike exact string matching, catches paraphrased or reformulated queries

## Database Schema

### PostgreSQL Table: `llm_cache`

```sql
CREATE TABLE llm_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(64) UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model VARCHAR(255) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);
```

### Qdrant Collection: `llm_cache`
- **Vector Size**: 1536 (text-embedding-3-small dimensions)
- **Distance Metric**: Cosine similarity
- **Payload**: cache_key, model, prompt_preview

## Troubleshooting

### Connection Issues

If you get connection errors:

```bash
# Check if databases are running
docker ps

# Check Qdrant
curl http://localhost:6333/collections

# Check PostgreSQL
docker exec -it postgres18 psql -U postgres -d devdb -c "SELECT COUNT(*) FROM llm_cache;"
```

### Import Errors

If you get import errors for `psycopg2` or `qdrant_client`:

```bash
pip install psycopg2-binary qdrant-client
```

### OpenAI API Errors

Make sure you're using a valid OpenAI API key and have credits available.

## Advanced Usage

### Custom Metadata

```python
# Store custom metadata with cache entries
cache.set(
    prompt="What is the capital of France?",
    response="Paris",
    model="gpt-4",
    metadata={
        "user_id": "12345",
        "session_id": "abc-def",
        "timestamp": "2024-01-01T00:00:00Z"
    }
)
```

### Retrieve with Metadata

```python
result = cache.get(prompt="...", model="...", return_metadata=True)
if result:
    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")
```

### Direct Cache Access

```python
# Access the cache directly (bypassing the client wrapper)
from semantic_cache import SemanticCache

cache = SemanticCache(openai_api_key="...")

# Manual cache operations
result = cache.get(prompt="...", model="...")
cache.set(prompt="...", response="...", model="...")
stats = cache.get_stats()
cache.clear()
```

## Comparison with GPTCache

This implementation is inspired by GPTCache but simplified:

| Feature | This Implementation | GPTCache |
|---------|-------------------|-----------|
| Embedding Model | OpenAI text-embedding-3-small | Configurable (multiple models) |
| Vector Store | Qdrant | Multiple options (Milvus, Qdrant, etc.) |
| Scalar Store | PostgreSQL | Multiple options (SQLite, Redis, etc.) |
| Similarity Algorithm | Cosine similarity | Multiple algorithms |
| Setup Complexity | Low (2 services) | Medium (configurable) |
| Customization | Moderate | High |

## License

MIT License - feel free to use in your projects!

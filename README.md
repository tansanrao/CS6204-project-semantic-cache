# Semantic Cache for LLM Responses

A production-ready semantic caching system for Large Language Model (LLM) responses using vector embeddings, designed to reduce API costs and improve response times through intelligent similarity matching.

## 🌟 Features

- **Semantic Similarity Matching**: Finds cached responses for semantically similar prompts, not just exact matches
- **Unified UUID System**: Single identifier links cache entries across PostgreSQL and Qdrant
- **Local Embeddings**: Uses `nomic-embed-text-v1.5` model (768 dimensions) - no API keys required
- **Dual Storage**: PostgreSQL for prompt-response pairs, Qdrant for vector similarity search
- **Configurable Thresholds**: Adjust similarity requirements for cache hits
- **Comprehensive Statistics**: Track cache performance with detailed metrics
- **Clean Architecture**: Modular, well-documented, and easy to extend

## 🏗️ Architecture

```
User Query
    ↓
[Generate Embedding] ← Local Model (nomic-embed-text-v1.5)
    ↓
[Search Qdrant] ← Vector Similarity (Cosine)
    ↓
Cache Hit? (similarity >= threshold)
    ↓ Yes                           ↓ No
[Get from PostgreSQL]           [Query LLM]
    ↓                                ↓
[Update Access Stats]           [Store in Cache]
    ↓                                ↓
Return Cached Response          Return LLM Response
```

## 📁 Project Structure

```
6204-project/
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── semantic_cache.py  # Core cache logic
│   │   └── cache_client.py    # LLM client wrapper
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py        # PostgreSQL manager
│   │   └── qdrant.py          # Qdrant manager
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   └── utils/
│       ├── __init__.py
│       └── validators.py      # Health checks
├── notebooks/
│   └── exploration.ipynb      # Interactive testing
├── db-stack/
│   └── docker-compose.yaml    # PostgreSQL & Qdrant
├── llm-stack/
│   └── docker-compose.yaml    # vLLM server
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Docker & Docker Compose
- NVIDIA GPU (for vLLM)

### 2. Start Database Services

```bash
cd db-stack
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Qdrant on port 6333

### 3. Start LLM Service (Optional)

```bash
cd llm-stack
docker-compose up -d
```

This starts:
- vLLM server on port 8000 with `openai/gpt-oss-20b` model

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Notebook

Open `notebooks/exploration.ipynb` and run **Cell 1** for complete initialization:
- ✅ Recreates databases (clean slate)
- ✅ Tests vLLM connectivity
- ✅ Initializes semantic cache
- ✅ Ready to use!

## 💡 Usage

### In Jupyter Notebook

```python
# Cell 1 handles all setup - just run it!

# Then use the cached client:
response = cached_client.chat_completions_create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
print(f"Cached: {response.cached}")
print(f"Similarity: {response.similarity_score}")
```

### In Python Scripts

```python
from openai import OpenAI
from config.config import Config
from src.cache.semantic_cache import SemanticCache
from src.cache.cache_client import CachedLLMClient

# Load configuration
config = Config()

# Initialize LLM client
llm_client = OpenAI(
    base_url=config.vllm.base_url,
    api_key=config.vllm.api_key
)

# Initialize cache
cache = SemanticCache(config=config)

# Create cached client
cached_client = CachedLLMClient(
    llm_client=llm_client,
    cache=cache,
    verbose=True
)

# Use it!
response = cached_client.chat_completions_create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## ⚙️ Configuration

Edit `config/config.py` or set environment variables:

```python
# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=openai/gpt-oss-20b

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=devdb

QDRANT_HOST=localhost
QDRANT_PORT=6333

# Cache Configuration
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

## 📊 Performance

- **Cache Hit**: ~10-50ms (embedding + vector search + DB lookup)
- **Cache Miss**: Full LLM latency + ~50ms overhead
- **Storage**: ~1KB per cached entry (text + metadata)
- **Vector Dimension**: 768 (nomic-embed-text-v1.5)

## 🧪 Testing

The notebook includes comprehensive tests:

1. **First Call Test**: Cache miss, LLM query
2. **Exact Match Test**: Cache hit with 1.0 similarity
3. **Semantic Match Test**: Cache hit with similar wording
4. **Different Question Test**: Cache miss, new LLM query
5. **Statistics View**: Comprehensive cache metrics
6. **Cache Inspection**: View all cached entries

## 🔧 Advanced Features

### Custom Similarity Threshold

```python
config.cache.similarity_threshold = 0.90  # More lenient
# OR
cache = SemanticCache(config=config, similarity_threshold=0.99)  # More strict
```

### Direct Cache Access

```python
# Get cache stats
stats = cache.get_stats()

# Clear cache
cache.clear()

# Delete specific entry
cache.delete(entry_uuid)

# List entries
entries = cache.postgres.list_entries(limit=50)
```

### Disable Caching

```python
cached_client = CachedLLMClient(
    llm_client=llm_client,
    cache=cache,
    enable_cache=False  # Bypass cache
)
```

## 🛠️ Development

### Adding New Embedding Models

Edit `config/config.py`:

```python
config.cache.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

The system automatically detects embedding dimensions.

### Custom Database Configurations

All database operations are isolated in `src/db/`:
- `postgres.py`: PostgreSQL operations
- `qdrant.py`: Qdrant operations

Easily extend or modify without affecting cache logic.

## 📝 Key Design Decisions

1. **Unified UUID System**: Deterministic UUID generation from prompt+model hash ensures same entry ID across both databases
2. **Separation of Concerns**: Embedding, storage, and cache logic are completely decoupled
3. **Local Embeddings**: No API dependencies for embeddings (faster, cheaper, private)
4. **Configuration-First**: All settings centralized in one place
5. **Clean Slate Init**: Cell 1 recreates databases for reproducible testing

## 🤝 Contributing

Feel free to:
- Add new embedding models
- Improve cache eviction strategies
- Add cache warming utilities
- Enhance statistics and visualization
- Write additional tests

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **nomic-ai** for the excellent embedding model
- **Qdrant** for the vector database
- **vLLM** for fast LLM inference
- **Sentence Transformers** library

---

**Made with ❤️ for efficient LLM caching**

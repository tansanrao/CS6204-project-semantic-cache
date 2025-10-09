# Semantic Cache Benchmark

This directory contains a comprehensive benchmark script for testing the semantic cache performance with hundreds of prompts.

## Overview

The `benchmark_cache.py` script:
- **Populates** the cache with hundreds of diverse prompts across multiple categories
- **Tests** cache performance with exact matches, similar prompts, and new queries
- **Measures** insertion times, query latencies, cache hit rates, and similarity scores
- **Generates** detailed statistics and saves results to JSON

## Usage

### Basic Usage

```bash
# Run with default settings (500 prompts, mock responses)
uv run python tests/benchmark_cache.py

# Specify number of prompts
uv run python tests/benchmark_cache.py --num-prompts 1000

# Clear cache before benchmarking
uv run python tests/benchmark_cache.py --clear-cache

# Adjust similarity threshold
uv run python tests/benchmark_cache.py --similarity-threshold 0.85

# Specify output file
uv run python tests/benchmark_cache.py --output my_results.json

# Use actual LLM (requires vLLM server running)
uv run python tests/benchmark_cache.py --use-llm --num-prompts 100

# Use actual LLM with custom model
uv run python tests/benchmark_cache.py --use-llm --llm-model "meta-llama/Llama-3-8B" --max-tokens 150
```

### Combined Options

```bash
# Full benchmark with 800 prompts, cleared cache, and custom threshold (mock responses)
uv run python tests/benchmark_cache.py \
  --num-prompts 800 \
  --clear-cache \
  --similarity-threshold 0.92 \
  --output benchmark_800.json

# Full benchmark WITH actual LLM calls
uv run python tests/benchmark_cache.py \
  --use-llm \
  --num-prompts 200 \
  --clear-cache \
  --similarity-threshold 0.92 \
  --max-tokens 100 \
  --output benchmark_llm_200.json
```

## Prerequisites

### 1. Database Services

Ensure the database services are running:

```bash
# Start PostgreSQL and Qdrant
cd db-stack
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 2. LLM Server (Optional - only needed for `--use-llm`)

If you want to benchmark with actual LLM calls:

```bash
# Start vLLM server
cd llm-stack
docker-compose up -d

# Verify LLM is running (should return model info)
curl http://localhost:8000/v1/models

# Check logs
docker-compose logs -f vllm
```

**Note**: The LLM server may take several minutes to start up as it loads the model into memory.

## Benchmark Details

### Two Benchmark Modes

#### 1. Mock Mode (Default)
- Fast and doesn't require LLM server
- Uses generated mock responses
- Good for testing cache infrastructure
- Measures embedding and database performance

#### 2. LLM Mode (`--use-llm`)
- Requires running vLLM server
- Makes actual API calls to LLM
- Measures end-to-end performance including:
  - LLM inference time
  - Network latency
  - Cache vs LLM speed comparison
  - Token savings

### Prompt Categories

The benchmark generates diverse prompts across 5 categories:
- **Coding**: Programming concepts, algorithms, languages
- **Math**: Mathematical concepts, calculations, proofs
- **General**: Technology and computing topics
- **Science**: Scientific concepts and processes
- **Business**: Business strategies and concepts

### Query Types

The benchmark tests three types of queries:
1. **Exact matches** (from original prompts)
2. **Similar prompts** (semantic variations of originals)
3. **New prompts** (completely new queries)

### Metrics Collected

#### Cache Statistics
- Total prompts inserted
- Cache hits vs misses
- Hit rate percentage

#### Performance Metrics
- Average insertion time
- Average query time (overall, hits, misses)
- Query latency percentiles (P50, P95, P99)
- Min/max query times

#### Similarity Metrics
- Average similarity score for cache hits
- Min/max similarity scores
- Score distribution

## Output

### Console Output

The script provides real-time progress updates and a formatted summary:

**Mock Mode Example:**
```
🚀 Starting Semantic Cache Benchmark
Configuration:
  • Number of prompts: 500
  • Similarity threshold: 0.90
  • Clear cache: False
  • Use actual LLM: False

================================================================================
 SEMANTIC CACHE BENCHMARK RESULTS
================================================================================

📊 CACHE POPULATION:
  • Total prompts inserted: 500
  • Average insertion time: 125.45 ms
  • Mode: Using mock responses

🔍 QUERY PERFORMANCE:
  • Total queries: 255
  • Cache hits: 182
  • Cache misses: 73
  • Hit rate: 71.37%

⏱️  QUERY LATENCY:
  • Average query time: 45.23 ms
  • Average HIT query time: 38.12 ms
  • Average MISS query time: 62.45 ms
  • Min query time: 15.23 ms
  • Max query time: 145.67 ms
  • P50 (median): 42.34 ms
  • P95: 98.76 ms
  • P99: 132.45 ms

🎯 SIMILARITY SCORES (for cache hits):
  • Average: 0.9650
  • Min: 0.9012
  • Max: 1.0000

📝 RESPONSE METRICS:
  • Average response length: 96 chars

⏲️  TOTAL DURATION: 68.45 seconds
================================================================================
```

**LLM Mode Example (`--use-llm`):**
```
================================================================================
 SEMANTIC CACHE BENCHMARK RESULTS
================================================================================

📊 CACHE POPULATION:
  • Total prompts inserted: 200
  • Average insertion time: 1250.45 ms
  • Mode: Using actual LLM calls

🔍 QUERY PERFORMANCE:
  • Total queries: 105
  • Cache hits: 78
  • Cache misses: 27
  • Hit rate: 74.29%
  • Actual LLM calls made: 27
  • Estimated tokens saved: ~7,800

⏱️  QUERY LATENCY:
  • Average query time: 245.23 ms
  • Average HIT query time: 45.12 ms
  • Average MISS query time: 1150.45 ms
  • Speedup (cache hit vs miss): 25.50x faster
  • Min query time: 35.23 ms
  • Max query time: 1845.67 ms
  • P50 (median): 52.34 ms
  • P95: 1298.76 ms
  • P99: 1532.45 ms

🎯 SIMILARITY SCORES (for cache hits):
  • Average: 0.9720
  • Min: 0.9102
  • Max: 1.0000

📝 RESPONSE METRICS:
  • Average response length: 312 chars

⏲️  TOTAL DURATION: 285.45 seconds
================================================================================
```

### JSON Output

Detailed results are saved to a JSON file (default: `benchmark_results.json`):

```json
{
  "summary": {
    "total_prompts_inserted": 500,
    "total_queries": 255,
    "cache_hits": 182,
    "cache_misses": 73,
    "hit_rate": 71.37,
    ...
  },
  "insert_results": [...],
  "query_results": [...],
  "timestamp": "2025-10-09T10:30:45.123456"
}
```

## Interpreting Results

### Good Performance Indicators
- **Hit rate**: 60-80% (depends on similarity threshold)
- **Hit query time**: <50ms for typical workloads
- **Average similarity score**: >0.95 for semantic matches
- **P95 query time**: <100ms

### Performance Optimization

If performance is suboptimal:

1. **Low hit rate**: 
   - Lower similarity threshold (e.g., 0.85 instead of 0.95)
   - Check prompt diversity

2. **High query latency**:
   - Check database connection latency
   - Verify Qdrant indexing is complete
   - Monitor system resources

3. **Slow insertions**:
   - Check embedding service performance
   - Verify database connection pool settings

## Troubleshooting

### Database Connection Issues

```bash
# Check if databases are running
docker ps

# Restart databases if needed
cd db-stack
docker-compose restart

# Check logs
docker-compose logs postgres
docker-compose logs qdrant
```

### Memory Issues

For large benchmarks (>1000 prompts), monitor memory usage:

```bash
# Monitor system resources
htop

# Check Docker container resources
docker stats
```

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Test Prompts

Modify the `generate_test_prompts()` function in `benchmark_cache.py` to add your own prompt templates and topics.

### Integration with CI/CD

```bash
# Run benchmark and check performance thresholds
python tests/benchmark_cache.py --num-prompts 500 --output ci_benchmark.json

# Parse results and verify metrics
python -c "
import json
with open('ci_benchmark.json') as f:
    data = json.load(f)
    hit_rate = data['summary']['hit_rate']
    assert hit_rate > 50, f'Hit rate too low: {hit_rate}%'
    print(f'✓ Benchmark passed (hit rate: {hit_rate:.1f}%)')
"
```

## Performance Baselines

Based on typical hardware (4-core CPU, 16GB RAM, SSD):

| Metric | Expected Range |
|--------|---------------|
| Insertion time | 100-200ms |
| Query time (hit) | 30-60ms |
| Query time (miss) | 50-100ms |
| Hit rate (0.95 threshold) | 60-75% |
| P95 query latency | <150ms |

Your results may vary based on:
- Hardware specifications
- Network latency
- Database configuration
- Embedding model size
- Number of cached entries

## Next Steps

- Experiment with different similarity thresholds
- Test with domain-specific prompts
- Benchmark with production-like query patterns
- Monitor cache performance over time
- Optimize based on your specific use case

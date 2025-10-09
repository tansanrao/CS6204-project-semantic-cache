"""
Semantic Cache Benchmark Script

This script populates the semantic cache with hundreds of prompts and benchmarks
the cache performance including:
- Cache hit rates
- Query response times
- Embedding generation times
- Database operation times
- LLM API calls
- Similarity score distributions

Usage:
    python tests/benchmark_cache.py [--num-prompts 500] [--clear-cache] [--use-llm]
"""

import sys
import os
import time
import argparse
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from src.cache.semantic_cache import SemanticCache
from src.cache.cache_client import CachedLLMClient
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark operation."""
    operation: str
    duration_ms: float
    cache_hit: bool
    similarity_score: float
    prompt_length: int
    response_length: int
    timestamp: str


@dataclass
class BenchmarkSummary:
    """Summary statistics from the benchmark run."""
    total_prompts_inserted: int
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    
    avg_insert_time_ms: float
    avg_query_time_ms: float
    avg_hit_query_time_ms: float
    avg_miss_query_time_ms: float
    
    min_query_time_ms: float
    max_query_time_ms: float
    p50_query_time_ms: float
    p95_query_time_ms: float
    p99_query_time_ms: float
    
    avg_similarity_score: float
    min_similarity_score: float
    max_similarity_score: float
    
    total_llm_calls: int
    avg_response_length: int
    
    total_duration_seconds: float


# Generate diverse test prompts
def generate_test_prompts(num_prompts: int) -> List[Tuple[str, str]]:
    """
    Generate diverse test prompts for benchmarking.
    
    Args:
        num_prompts: Number of prompts to generate
        
    Returns:
        List of (prompt, category) tuples
    """
    prompts = []
    
    # Categories with templates
    categories = {
        "coding": [
            "Write a Python function to {}",
            "How do I implement {} in {}?",
            "Explain the concept of {} in programming",
            "Debug this code: {}",
            "What is the best way to {} in {}?",
            "Compare {} and {} for {}",
            "Write unit tests for a function that {}",
        ],
        "math": [
            "What is the solution to {}?",
            "Explain the mathematical concept of {}",
            "Calculate {} using {}",
            "Prove that {} equals {}",
            "What is the formula for {}?",
            "How do you derive {}?",
        ],
        "general": [
            "What is {}?",
            "Explain {} in simple terms",
            "How does {} work?",
            "What are the benefits of {}?",
            "Compare {} and {}",
            "Describe the history of {}",
        ],
        "science": [
            "How does {} function in {}?",
            "What causes {}?",
            "Explain the theory of {}",
            "What is the relationship between {} and {}?",
            "Describe the process of {}",
        ],
        "business": [
            "What is the best strategy for {}?",
            "How can I improve {} in my business?",
            "Explain the concept of {} in business",
            "What are the key factors for {}?",
            "How do companies approach {}?",
        ]
    }
    
    # Fill-in values for templates
    coding_topics = [
        "binary search", "sorting algorithms", "hash tables", "recursion",
        "dynamic programming", "graph traversal", "linked lists", "tree structures",
        "REST APIs", "database optimization", "caching strategies", "authentication",
        "error handling", "async programming", "design patterns", "code refactoring"
    ]
    
    languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
    
    math_topics = [
        "calculus", "linear algebra", "statistics", "probability",
        "differential equations", "trigonometry", "number theory", "combinatorics",
        "game theory", "optimization", "complex numbers", "matrices"
    ]
    
    general_topics = [
        "artificial intelligence", "machine learning", "neural networks", "blockchain",
        "quantum computing", "cloud computing", "cybersecurity", "data science",
        "natural language processing", "computer vision", "IoT", "edge computing"
    ]
    
    science_topics = [
        "photosynthesis", "DNA replication", "cellular respiration", "evolution",
        "genetics", "ecosystems", "climate change", "renewable energy",
        "neuroscience", "quantum mechanics", "thermodynamics", "electromagnetism"
    ]
    
    business_topics = [
        "customer retention", "market analysis", "product development", "branding",
        "supply chain management", "digital marketing", "financial planning",
        "team management", "innovation", "competitive advantage", "scalability"
    ]
    
    # Generate prompts by filling templates
    topics_map = {
        "coding": (coding_topics, languages),
        "math": (math_topics,),
        "general": (general_topics,),
        "science": (science_topics,),
        "business": (business_topics,)
    }
    
    import random
    random.seed(42)  # For reproducibility
    
    for i in range(num_prompts):
        # Select category
        category = list(categories.keys())[i % len(categories)]
        templates = categories[category]
        template = random.choice(templates)
        
        # Fill template
        topics = topics_map[category]
        placeholder_count = template.count("{}")
        
        if category == "coding" and placeholder_count == 2:
            topic = random.choice(topics[0])
            lang = random.choice(topics[1])
            prompt = template.format(topic, lang)
        else:
            fill_values = [random.choice(topics[0]) for _ in range(placeholder_count)]
            prompt = template.format(*fill_values)
        
        prompts.append((prompt, category))
    
    return prompts


def generate_similar_prompts(base_prompts: List[Tuple[str, str]], similarity_factor: float = 0.3) -> List[str]:
    """
    Generate similar versions of existing prompts to test cache hits.
    
    Args:
        base_prompts: Original prompts
        similarity_factor: Fraction of prompts to create variations for
        
    Returns:
        List of similar prompts
    """
    import random
    random.seed(43)
    
    similar_prompts = []
    
    # Select subset to create variations
    num_similar = int(len(base_prompts) * similarity_factor)
    selected = random.sample(base_prompts, num_similar)
    
    variations = [
        lambda p: p.replace("What", "Can you tell me what"),
        lambda p: p.replace("How", "I need to know how"),
        lambda p: p.replace("Explain", "Please explain"),
        lambda p: f"{p} Please provide details.",
        lambda p: f"Question: {p}",
        lambda p: p.replace("?", " in detail?"),
        lambda p: f"{p} I'm curious about this.",
    ]
    
    for prompt, category in selected:
        variation = random.choice(variations)
        similar_prompts.append(variation(prompt))
    
    return similar_prompts


def populate_cache(
    cache: SemanticCache,
    prompts: List[Tuple[str, str]],
    model: str = "test-model",
    llm_client: Optional[CachedLLMClient] = None,
    use_llm: bool = False
) -> List[BenchmarkResult]:
    """
    Populate cache with prompts and measure insertion performance.
    
    Args:
        cache: SemanticCache instance
        prompts: List of (prompt, category) tuples
        model: Model name to use
        llm_client: CachedLLMClient instance (required if use_llm=True)
        use_llm: If True, call actual LLM; if False, use mock responses
        
    Returns:
        List of benchmark results for insertions
    """
    logger.info(f"Populating cache with {len(prompts)} prompts (use_llm={use_llm})...")
    results = []
    
    for i, (prompt, category) in enumerate(prompts):
        try:
            start_time = time.time()
            
            if use_llm and llm_client:
                # Call actual LLM through cached client
                response_obj = llm_client.chat_completions_create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8000,
                    temperature=0.7
                )
                response = response_obj.choices[0].message.content
            else:
                # Generate mock response
                response = f"This is a cached response for the {category} query: '{prompt[:50]}...'"
                
                # Insert into cache (when not using LLM client, we insert directly)
                cache.set(
                    prompt=prompt,
                    response=response,
                    model=model,
                    metadata={"category": category}
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            results.append(BenchmarkResult(
                operation="insert",
                duration_ms=duration_ms,
                cache_hit=False,
                similarity_score=0.0,
                prompt_length=len(prompt),
                response_length=len(response),
                timestamp=datetime.now().isoformat()
            ))
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Inserted {i + 1}/{len(prompts)} prompts")
                
        except Exception as e:
            logger.error(f"Error inserting prompt {i}: {e}")
    
    logger.info(f"✓ Successfully populated cache with {len(results)} prompts")
    return results


def benchmark_queries(
    cache: SemanticCache,
    original_prompts: List[Tuple[str, str]],
    similar_prompts: List[str],
    model: str = "test-model",
    llm_client: Optional[CachedLLMClient] = None,
    use_llm: bool = False
) -> List[BenchmarkResult]:
    """
    Benchmark cache query performance with both exact and similar prompts.
    
    Args:
        cache: SemanticCache instance
        original_prompts: Original prompts used to populate cache
        similar_prompts: Similar prompts to test cache hits
        model: Model name
        llm_client: CachedLLMClient instance (required if use_llm=True)
        use_llm: If True, call actual LLM; if False, use cache.get()
        
    Returns:
        List of benchmark results for queries
    """
    import random
    
    logger.info("Benchmarking cache queries...")
    results = []
    
    # Mix original and similar prompts
    query_prompts = []
    
    # Sample some original prompts (should be exact matches)
    num_original = min(100, len(original_prompts))
    sampled_original = random.sample(original_prompts, num_original)
    query_prompts.extend([(p, "original") for p, _ in sampled_original])
    
    # Add similar prompts (should trigger semantic matching)
    query_prompts.extend([(p, "similar") for p in similar_prompts])
    
    # Add completely new prompts (should miss)
    new_prompts = [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "When was the internet invented?",
        "What is the speed of light?",
        "Who wrote Hamlet?",
    ]
    query_prompts.extend([(p, "new") for p in new_prompts])
    
    # Shuffle for realistic query pattern
    random.shuffle(query_prompts)
    
    logger.info(f"Testing {len(query_prompts)} queries...")
    
    for i, (prompt, query_type) in enumerate(query_prompts):
        try:
            start_time = time.time()
            
            if use_llm and llm_client:
                # Call through cached LLM client
                response_obj = llm_client.chat_completions_create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,  # Increased for reasoning models
                    temperature=0.7
                )
                cache_hit = response_obj.cached
                similarity_score = response_obj.similarity_score
                response = response_obj.choices[0].message.content
            else:
                # Direct cache query
                result = cache.get(prompt=prompt, model=model)
                cache_hit = result is not None
                if cache_hit and result:
                    similarity_score = result.get("similarity_score", 0.0)
                    response = result.get("response", "")
                else:
                    similarity_score = 0.0
                    response = ""
            
            duration_ms = (time.time() - start_time) * 1000
            
            results.append(BenchmarkResult(
                operation=f"query_{query_type}",
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                similarity_score=similarity_score,
                prompt_length=len(prompt),
                response_length=len(response),
                timestamp=datetime.now().isoformat()
            ))
            
            if (i + 1) % 50 == 0:
                hits = sum(1 for r in results if r.cache_hit)
                hit_rate = hits / len(results) * 100
                logger.info(f"  Processed {i + 1}/{len(query_prompts)} queries (Hit rate: {hit_rate:.1f}%)")
                
        except Exception as e:
            logger.error(f"Error querying prompt {i}: {e}")
    
    logger.info(f"✓ Completed {len(results)} queries")
    return results


def analyze_results(insert_results: List[BenchmarkResult], query_results: List[BenchmarkResult]) -> BenchmarkSummary:
    """
    Analyze benchmark results and generate summary statistics.
    
    Args:
        insert_results: Results from insertion benchmark
        query_results: Results from query benchmark
        
    Returns:
        BenchmarkSummary with comprehensive statistics
    """
    logger.info("Analyzing benchmark results...")
    
    # Query analysis
    query_times = [r.duration_ms for r in query_results]
    hit_queries = [r for r in query_results if r.cache_hit]
    miss_queries = [r for r in query_results if not r.cache_hit]
    
    hit_times = [r.duration_ms for r in hit_queries]
    miss_times = [r.duration_ms for r in miss_queries]
    
    similarity_scores = [r.similarity_score for r in hit_queries]
    
    # Response length analysis
    response_lengths = [r.response_length for r in insert_results + query_results if r.response_length > 0]
    
    # Calculate statistics
    summary = BenchmarkSummary(
        total_prompts_inserted=len(insert_results),
        total_queries=len(query_results),
        cache_hits=len(hit_queries),
        cache_misses=len(miss_queries),
        hit_rate=float(len(hit_queries) / len(query_results) * 100) if query_results else 0.0,
        
        avg_insert_time_ms=float(np.mean([r.duration_ms for r in insert_results])) if insert_results else 0.0,
        avg_query_time_ms=float(np.mean(query_times)) if query_times else 0.0,
        avg_hit_query_time_ms=float(np.mean(hit_times)) if hit_times else 0.0,
        avg_miss_query_time_ms=float(np.mean(miss_times)) if miss_times else 0.0,
        
        min_query_time_ms=float(np.min(query_times)) if query_times else 0.0,
        max_query_time_ms=float(np.max(query_times)) if query_times else 0.0,
        p50_query_time_ms=float(np.percentile(query_times, 50)) if query_times else 0.0,
        p95_query_time_ms=float(np.percentile(query_times, 95)) if query_times else 0.0,
        p99_query_time_ms=float(np.percentile(query_times, 99)) if query_times else 0.0,
        
        avg_similarity_score=float(np.mean(similarity_scores)) if similarity_scores else 0.0,
        min_similarity_score=float(np.min(similarity_scores)) if similarity_scores else 0.0,
        max_similarity_score=float(np.max(similarity_scores)) if similarity_scores else 0.0,
        
        total_llm_calls=len(miss_queries),
        avg_response_length=int(np.mean(response_lengths)) if response_lengths else 0,
        
        total_duration_seconds=sum(r.duration_ms for r in insert_results + query_results) / 1000
    )
    
    return summary


def print_summary(summary: BenchmarkSummary, use_llm: bool = False) -> None:
    """Print formatted benchmark summary."""
    print("\n" + "=" * 80)
    print(" SEMANTIC CACHE BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\n📊 CACHE POPULATION:")
    print(f"  • Total prompts inserted: {summary.total_prompts_inserted}")
    print(f"  • Average insertion time: {summary.avg_insert_time_ms:.2f} ms")
    if use_llm:
        print(f"  • Mode: Using actual LLM calls")
    else:
        print(f"  • Mode: Using mock responses")
    
    print(f"\n🔍 QUERY PERFORMANCE:")
    print(f"  • Total queries: {summary.total_queries}")
    print(f"  • Cache hits: {summary.cache_hits}")
    print(f"  • Cache misses: {summary.cache_misses}")
    print(f"  • Hit rate: {summary.hit_rate:.2f}%")
    if use_llm:
        print(f"  • Actual LLM calls made: {summary.total_llm_calls}")
        tokens_saved = (summary.total_queries - summary.total_llm_calls) * summary.avg_response_length
        print(f"  • Estimated tokens saved: ~{tokens_saved:,.0f}")
    
    print(f"\n⏱️  QUERY LATENCY:")
    print(f"  • Average query time: {summary.avg_query_time_ms:.2f} ms")
    print(f"  • Average HIT query time: {summary.avg_hit_query_time_ms:.2f} ms")
    print(f"  • Average MISS query time: {summary.avg_miss_query_time_ms:.2f} ms")
    if use_llm:
        speedup = summary.avg_miss_query_time_ms / summary.avg_hit_query_time_ms if summary.avg_hit_query_time_ms > 0 else 0
        print(f"  • Speedup (cache hit vs miss): {speedup:.2f}x faster")
    print(f"  • Min query time: {summary.min_query_time_ms:.2f} ms")
    print(f"  • Max query time: {summary.max_query_time_ms:.2f} ms")
    print(f"  • P50 (median): {summary.p50_query_time_ms:.2f} ms")
    print(f"  • P95: {summary.p95_query_time_ms:.2f} ms")
    print(f"  • P99: {summary.p99_query_time_ms:.2f} ms")
    
    print(f"\n🎯 SIMILARITY SCORES (for cache hits):")
    print(f"  • Average: {summary.avg_similarity_score:.4f}")
    print(f"  • Min: {summary.min_similarity_score:.4f}")
    print(f"  • Max: {summary.max_similarity_score:.4f}")
    
    if summary.avg_response_length > 0:
        print(f"\n📝 RESPONSE METRICS:")
        print(f"  • Average response length: {summary.avg_response_length} chars")
    
    print(f"\n⏲️  TOTAL DURATION: {summary.total_duration_seconds:.2f} seconds")
    print("=" * 80)


def save_results(summary: BenchmarkSummary, insert_results: List[BenchmarkResult], 
                query_results: List[BenchmarkResult], output_file: str = "benchmark_results.json") -> None:
    """Save detailed benchmark results to JSON file."""
    results_dict = {
        "summary": asdict(summary),
        "insert_results": [asdict(r) for r in insert_results],
        "query_results": [asdict(r) for r in query_results],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"✓ Detailed results saved to {output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark semantic cache performance")
    parser.add_argument("--num-prompts", type=int, default=500, 
                       help="Number of prompts to generate and insert (default: 500)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache before running benchmark")
    parser.add_argument("--similarity-threshold", type=float, default=0.90,
                       help="Similarity threshold for cache hits (default: 0.90)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for detailed results (default: benchmark_results.json)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use actual LLM calls instead of mock responses (requires vLLM server)")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="LLM model to use (overrides config default)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Max tokens for LLM responses (default: 100)")
    
    args = parser.parse_args()
    
    print("\n🚀 Starting Semantic Cache Benchmark")
    print(f"Configuration:")
    print(f"  • Number of prompts: {args.num_prompts}")
    print(f"  • Similarity threshold: {args.similarity_threshold}")
    print(f"  • Clear cache: {args.clear_cache}")
    print(f"  • Use actual LLM: {args.use_llm}")
    if args.use_llm:
        print(f"  • Max tokens: {args.max_tokens}")
    
    # Initialize configuration
    config = Config()
    config.cache.similarity_threshold = args.similarity_threshold
    
    # Initialize cache
    logger.info("Initializing semantic cache...")
    cache = SemanticCache(config=config)
    
    # Initialize LLM client if needed
    llm_client = None
    model_name = args.llm_model or config.vllm.model_name
    
    if args.use_llm:
        logger.info("Initializing LLM client...")
        try:
            openai_client = OpenAI(
                base_url=config.vllm.base_url,
                api_key=config.vllm.api_key
            )
            
            # Test connection
            logger.info(f"Testing connection to {config.vllm.base_url}...")
            test_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info(f"✓ LLM connection successful (model: {model_name})")
            
            # Create cached client
            llm_client = CachedLLMClient(
                llm_client=openai_client,
                cache=cache,
                enable_cache=True,
                verbose=False
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to LLM server: {e}")
            print(f"\n❌ Error: Could not connect to LLM server at {config.vllm.base_url}")
            print("Please ensure the vLLM server is running:")
            print("  cd llm-stack && docker-compose up -d")
            return
    
    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing existing cache...")
        cache.clear()
    
    # Check initial cache stats
    initial_stats = cache.get_stats()
    logger.info(f"Initial cache size: {initial_stats.get('postgres', {}).get('total_entries', 0)} entries")
    
    # Generate test prompts
    start_time = time.time()
    
    logger.info("Generating test prompts...")
    prompts = generate_test_prompts(args.num_prompts)
    similar_prompts = generate_similar_prompts(prompts, similarity_factor=0.3)
    
    # Run benchmarks
    insert_results = populate_cache(cache, prompts, model=model_name, 
                                   llm_client=llm_client, use_llm=args.use_llm)
    query_results = benchmark_queries(cache, prompts, similar_prompts, model=model_name,
                                     llm_client=llm_client, use_llm=args.use_llm)
    
    # Analyze results
    summary = analyze_results(insert_results, query_results)
    
    # Display results
    print_summary(summary, use_llm=args.use_llm)
    
    # Save detailed results
    save_results(summary, insert_results, query_results, args.output)
    
    # Final cache stats
    final_stats = cache.get_stats()
    print(f"\n📈 Final Cache Statistics:")
    print(f"  • Total entries: {final_stats.get('postgres', {}).get('total_entries', 0)}")
    print(f"  • Total accesses: {final_stats.get('postgres', {}).get('total_accesses', 0)}")
    print(f"  • Qdrant vectors: {final_stats.get('qdrant', {}).get('vectors_count', 0)}")
    
    if args.use_llm and llm_client:
        print(f"\n📊 LLM Client Statistics:")
        stats = llm_client.get_stats()
        client_stats = stats.get('client_stats', {})
        print(f"  • Total requests: {client_stats.get('total_requests', 0)}")
        print(f"  • Cache hits: {client_stats.get('cache_hits', 0)}")
        print(f"  • Cache misses: {client_stats.get('cache_misses', 0)}")
        hit_rate_pct = client_stats.get('hit_rate', 0.0) * 100
        print(f"  • Hit rate: {hit_rate_pct:.2f}%")
    
    print(f"\n✅ Benchmark completed in {time.time() - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()

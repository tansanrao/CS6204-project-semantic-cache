"""
Cached LLM Client - wraps OpenAI client with semantic caching.
"""

from typing import Optional, Dict, Any, List
from openai import OpenAI
from semantic_cache import SemanticCache


class CachedLLMClient:
    """
    A wrapper around OpenAI client that adds semantic caching.
    
    Automatically checks cache before making LLM calls and stores responses.
    """
    
    def __init__(
        self,
        llm_client: OpenAI,
        cache: SemanticCache,
        enable_cache: bool = True,
        verbose: bool = False
    ):
        """
        Initialize cached LLM client.
        
        Args:
            llm_client: OpenAI client instance for LLM
            cache: SemanticCache instance
            enable_cache: Whether to enable caching (can disable for testing)
            verbose: Print cache hit/miss information
        """
        self.llm_client = llm_client
        self.cache = cache
        self.enable_cache = enable_cache
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """
        Create chat completion with caching.
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments passed to OpenAI API
            
        Returns:
            OpenAI ChatCompletion object with added 'cached' attribute
        """
        self.stats["total_requests"] += 1
        
        # Extract the prompt (combine all messages for caching)
        prompt = self._messages_to_prompt(messages)
        
        # Check cache
        if self.enable_cache:
            cached_result = self.cache.get(prompt, model)
            
            if cached_result:
                self.stats["cache_hits"] += 1
                if self.verbose:
                    print(f"✓ Cache HIT (similarity: {cached_result['similarity_score']:.4f})")
                
                # Create a mock response object that looks like OpenAI response
                return self._create_mock_response(
                    cached_result["response"],
                    model,
                    cached=True,
                    similarity_score=cached_result["similarity_score"]
                )
        
        # Cache miss - make actual LLM call
        self.stats["cache_misses"] += 1
        if self.verbose:
            print("✗ Cache MISS - calling LLM")
        
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        # Store in cache
        if self.enable_cache:
            response_text = response.choices[0].message.content
            metadata = {
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "model": model
            }
            self.cache.set(prompt, response_text, model, metadata)
        
        # Add cached flag to response
        response.cached = False
        return response
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to a single prompt string for caching."""
        # Concatenate all messages with role prefixes
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)
    
    def _create_mock_response(
        self,
        content: str,
        model: str,
        cached: bool = True,
        similarity_score: float = 1.0
    ):
        """Create a mock OpenAI response object from cached content."""
        from types import SimpleNamespace
        
        # Create nested structure similar to OpenAI response
        message = SimpleNamespace(
            content=content,
            role="assistant"
        )
        
        choice = SimpleNamespace(
            finish_reason="stop",
            index=0,
            message=message
        )
        
        response = SimpleNamespace(
            id="cached",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            cached=cached,
            similarity_score=similarity_score
        )
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self.cache.get_stats()
        
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_requests"]
        
        return {
            "client_stats": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": hit_rate
            },
            "cache_stats": cache_stats
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

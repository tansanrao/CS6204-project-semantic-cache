"""
Cached LLM Client wrapper.

Wraps OpenAI-compatible LLM clients with semantic caching.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from src.cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str
    content: str


@dataclass
class ChatChoice:
    """Represents a chat completion choice."""
    finish_reason: str
    index: int
    message: ChatMessage


@dataclass
class ChatCompletion:
    """
    Represents a chat completion response.
    
    Compatible with OpenAI's ChatCompletion structure but with added caching info.
    """
    id: str
    choices: List[ChatChoice]
    created: int
    model: str
    object: str
    cached: bool = False
    similarity_score: float = 0.0


class CachedLLMClient:
    """
    Wrapper around OpenAI-compatible LLM clients that adds semantic caching.
    
    Automatically checks cache before making LLM calls and stores responses.
    
    Attributes:
        llm_client: The underlying OpenAI client
        cache: SemanticCache instance
        enable_cache: Whether caching is enabled
        verbose: Whether to print cache hit/miss info
        stats: Dictionary tracking cache statistics
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
            llm_client: OpenAI client instance
            cache: SemanticCache instance
            enable_cache: Enable/disable caching
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
        
        logger.info(f"Cached LLM client initialized (cache {'enabled' if enable_cache else 'disabled'})")
    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ChatCompletion:
        """
        Create chat completion with caching.
        
        Process:
        1. Convert messages to cache key
        2. Check cache for similar prompts
        3. If hit: return cached response
        4. If miss: call LLM and cache response
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments passed to OpenAI API
            
        Returns:
            ChatCompletion object with 'cached' and 'similarity_score' attributes
        """
        self.stats["total_requests"] += 1
        
        # Convert messages to single prompt string for caching
        prompt = self._messages_to_prompt(messages)
        
        # Check cache if enabled
        if self.enable_cache:
            cached_result = self.cache.get(prompt, model)
            
            if cached_result:
                self.stats["cache_hits"] += 1
                
                if self.verbose:
                    print(f"✓ Cache HIT (similarity: {cached_result['similarity_score']:.4f})")
                
                return self._create_response(
                    content=cached_result["response"],
                    model=model,
                    cached=True,
                    similarity_score=cached_result["similarity_score"]
                )
        
        # Cache miss - make actual LLM call
        self.stats["cache_misses"] += 1
        
        if self.verbose:
            print("✗ Cache MISS - calling LLM")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Store in cache if enabled
            if self.enable_cache:
                response_text = response.choices[0].message.content
                metadata = {
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "model": model
                }
                self.cache.set(prompt, response_text, model, metadata)
            
            # Convert to our ChatCompletion format
            return self._convert_openai_response(response, cached=False)
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages list to a single prompt string for caching.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Combined prompt string
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)
    
    def _create_response(
        self,
        content: str,
        model: str,
        cached: bool = True,
        similarity_score: float = 1.0
    ) -> ChatCompletion:
        """
        Create a ChatCompletion response.
        
        Args:
            content: Response text
            model: Model identifier
            cached: Whether this is a cached response
            similarity_score: Similarity score for cached responses
            
        Returns:
            ChatCompletion object
        """
        message = ChatMessage(content=content, role="assistant")
        choice = ChatChoice(finish_reason="stop", index=0, message=message)
        
        return ChatCompletion(
            id="cached" if cached else "generated",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            cached=cached,
            similarity_score=similarity_score
        )
    
    def _convert_openai_response(
        self,
        response: Any,
        cached: bool = False
    ) -> ChatCompletion:
        """
        Convert OpenAI response to our ChatCompletion format.
        
        Args:
            response: OpenAI ChatCompletion response
            cached: Whether this is from cache
            
        Returns:
            ChatCompletion object
        """
        # Extract first choice
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = ChatMessage(
                content=choice.message.content,
                role=choice.message.role
            )
            chat_choice = ChatChoice(
                finish_reason=choice.finish_reason,
                index=choice.index,
                message=message
            )
            
            return ChatCompletion(
                id=response.id,
                choices=[chat_choice],
                created=response.created,
                model=response.model,
                object=response.object,
                cached=cached,
                similarity_score=0.0
            )
        else:
            raise ValueError("Invalid OpenAI response: no choices")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with client and cache statistics
        """
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_requests"]
        
        return {
            "client_stats": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": hit_rate,
                "cache_enabled": self.enable_cache
            },
            "cache_stats": self.cache.get_stats() if self.enable_cache else {}
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.enable_cache:
            self.cache.clear()
            logger.info("Cache cleared")
        else:
            logger.warning("Cache not enabled, nothing to clear")
    
    def reset_stats(self) -> None:
        """Reset client statistics."""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Client statistics reset")

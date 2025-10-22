"""Embedding service wrapping the Hugging Face nomic model."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Load and execute the semantic embedding model."""

    def __init__(
        self,
        model_name: str,
        mode: str,
        dimension: int,
        device: str = "cpu",
        *,
        model_loader: Callable[[str], SentenceTransformer] | None = None,
    ) -> None:
        self._model_name = model_name
        self._mode = mode
        self._dimension = dimension
        self._device = device
        self._model_loader = model_loader or _load_sentence_transformer
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized embeddings for the provided texts."""
        if not texts:
            return []

        model = await self._ensure_model()
        prepared = [_normalize_text(self._mode, text) for text in texts]

        embeddings = await asyncio.to_thread(
            _encode_texts,
            model,
            prepared,
            self._device,
            self._mode,
            self._dimension,
        )
        return embeddings

    async def _ensure_model(self) -> SentenceTransformer:
        """Lazily load the embedding model and cache it on the instance."""
        if self._model is not None:
            return self._model

        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    self._model_loader, self._model_name
                )
        return self._model


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model with remote code enabled."""
    return SentenceTransformer(model_name, trust_remote_code=True)


def _normalize_text(mode: str, text: str) -> str:
    """Apply light normalization for model input."""
    collapsed = " ".join(text.split())
    prefix = f"{mode}:"
    if not collapsed:
        return prefix
    if collapsed.lower().startswith(prefix.lower()):
        return collapsed
    # Nomic Matryoshka models expect an explicit task instruction prefix.
    return f"{prefix} {collapsed}"


def _encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    device: str,
    mode: str,
    dimension: int,
) -> list[list[float]]:
    """Run the actual encode call in a background thread."""
    prompt_kwargs: dict[str, Any] = {}
    prompts = getattr(model, "prompts", None)
    if prompts is None or mode in prompts:
        prompt_kwargs["prompt_name"] = mode
    try:
        embeddings: Any = model.encode(
            list(texts),
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False,
            **prompt_kwargs,
        )
    except TypeError:
        # Older sentence-transformers versions may not expose prompt_name.
        embeddings = model.encode(
            list(texts),
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except ValueError as exc:
        # Some models do not ship prompt templates; fall back to vanilla encode.
        if "Prompt name" in str(exc):
            embeddings = model.encode(
                list(texts),
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            raise

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)

    if dimension > array.shape[1]:
        raise ValueError(
            f"Requested dimension {dimension} exceeds model output {array.shape[1]}."
        )

    sliced = array[:, :dimension]
    return sliced.tolist()


def load_embedding_service(
    model_name: str,
    mode: str,
    dimension: int,
    device: str = "cpu",
) -> EmbeddingService:
    """Helper to construct the embedding service with sane defaults."""
    return EmbeddingService(
        model_name=model_name,
        mode=mode,
        dimension=dimension,
        device=device,
    )

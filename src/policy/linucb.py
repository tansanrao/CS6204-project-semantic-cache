"""Contextual bandit implementations for TTL selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .features import FeatureVector


@dataclass(frozen=True, slots=True)
class LinUCBConfig:
    """Configuration for a LinUCB policy instance."""

    alpha: float
    num_actions: int
    dimension: int
    regularization: float = 1.0
    min_propensity: float = 1e-3
    allow_feature_growth: bool = True


@dataclass(frozen=True, slots=True)
class LinUCBSnapshot:
    """Serializable representation of policy state."""

    config: LinUCBConfig
    feature_index: dict[str, int]
    matrices: list[list[list[float]]]
    responses: list[list[float]]


class FeatureIndexer:
    """Assign stable indices to sparse feature names."""

    def __init__(self, dimension: int, *, allow_growth: bool = True) -> None:
        self._dimension = dimension
        self._allow_growth = allow_growth
        self._index: dict[str, int] = {}

    @property
    def dimension(self) -> int:  # noqa: D401 - simple alias
        return self._dimension

    @property
    def index(self) -> Mapping[str, int]:
        """Expose the feature-to-index mapping."""
        return self._index

    def ensure(self, feature_name: str) -> int:
        """Guarantee the feature has an index, growing the mapping if allowed."""
        if feature_name in self._index:
            return self._index[feature_name]
        if not self._allow_growth:
            raise ValueError(
                f"Feature growth disabled; missing feature '{feature_name}'"
            )
        if len(self._index) >= self._dimension:
            raise ValueError("Feature space exhausted; increase config.dimension")
        next_index = len(self._index)
        self._index[feature_name] = next_index
        return next_index

    def vectorize(
        self, features: Mapping[str, float], *, grow: bool = True
    ) -> np.ndarray:
        """Convert sparse feature mapping into a dense numpy vector."""
        vector = np.zeros(self._dimension, dtype=np.float64)
        for name, value in features.items():
            if name not in self._index:
                if not grow:
                    continue
                try:
                    index = self.ensure(name)
                except ValueError:
                    # Feature space exhausted; skip silently to avoid crashes in prod.
                    continue
            else:
                index = self._index[name]
            vector[index] = float(value)
        return vector

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, int], *, dimension: int, allow_growth: bool
    ) -> "FeatureIndexer":
        """Instantiate an indexer from a stored mapping."""
        indexer = cls(dimension=dimension, allow_growth=allow_growth)
        indexer._index.update(mapping)
        return indexer


class LinUCBPolicy:
    """Per-action linear UCB policy with optional warm start and persistence."""

    def __init__(
        self,
        config: LinUCBConfig,
        *,
        feature_indexer: FeatureIndexer | None = None,
        prior_weights: Sequence[Mapping[str, float]] | None = None,
    ) -> None:
        if config.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if config.dimension <= 0:
            raise ValueError("dimension must be positive")

        self._config = config
        self._indexer = feature_indexer or FeatureIndexer(
            dimension=config.dimension,
            allow_growth=config.allow_feature_growth,
        )
        self._A = [
            np.eye(config.dimension, dtype=np.float64) * config.regularization
            for _ in range(config.num_actions)
        ]
        self._b = [
            np.zeros(config.dimension, dtype=np.float64)
            for _ in range(config.num_actions)
        ]

        if prior_weights is not None:
            if len(prior_weights) != config.num_actions:
                raise ValueError("prior_weights length must match num_actions")
            self._apply_prior(prior_weights)

    @property
    def config(self) -> LinUCBConfig:
        """Return immutable configuration."""
        return self._config

    @property
    def feature_index(self) -> Mapping[str, int]:
        """Expose the locked feature mapping."""
        return self._indexer.index

    def choose(
        self, features: FeatureVector | Mapping[str, float]
    ) -> tuple[int, float]:
        """Select the TTL bucket index using LinUCB scoring."""
        vector = self._vectorize(features)
        scores = np.zeros(self._config.num_actions, dtype=np.float64)
        for action in range(self._config.num_actions):
            theta = np.linalg.solve(self._A[action], self._b[action])
            exploration = self._config.alpha * math.sqrt(
                float(vector.T @ np.linalg.solve(self._A[action], vector))
            )
            scores[action] = float(theta.T @ vector) + exploration
        chosen = int(np.argmax(scores))
        return chosen, float(scores[chosen])

    def update(
        self,
        action: int,
        reward: float,
        features: FeatureVector | Mapping[str, float],
    ) -> None:
        """Update posterior matrices with observed reward."""
        if not 0 <= action < self._config.num_actions:
            raise ValueError("action index out of range")
        vector = self._vectorize(features, grow=False)
        x = vector.reshape(-1, 1)
        self._A[action] += x @ x.T
        self._b[action] += reward * vector

    def propensity(
        self, action: int, features: FeatureVector | Mapping[str, float]
    ) -> float:
        """Return the softmax propensity approximation for logging."""
        scores = self._score_all(features)
        logits = scores - np.max(scores)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return float(max(self._config.min_propensity, probs[action]))

    def snapshot(self) -> LinUCBSnapshot:
        """Generate a serializable snapshot of the policy state."""
        matrices_payload = [matrix.tolist() for matrix in self._A]
        response_payload = [vector.tolist() for vector in self._b]
        return LinUCBSnapshot(
            config=self._config,
            feature_index=dict(self._indexer.index),
            matrices=matrices_payload,
            responses=response_payload,
        )

    def save(self, path: Path) -> None:
        """Persist the policy state to a JSON file."""
        snapshot = self.snapshot()
        payload = {
            "config": asdict(snapshot.config),
            "feature_index": snapshot.feature_index,
            "matrices": snapshot.matrices,
            "responses": snapshot.responses,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinUCBPolicy":
        """Restore a policy from a JSON snapshot."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = LinUCBConfig(**payload["config"])
        indexer = FeatureIndexer.from_mapping(
            payload["feature_index"],
            dimension=config.dimension,
            allow_growth=config.allow_feature_growth,
        )
        policy = cls(config=config, feature_indexer=indexer)
        policy._A = [
            np.array(matrix, dtype=np.float64) for matrix in payload["matrices"]
        ]
        policy._b = [
            np.array(vector, dtype=np.float64) for vector in payload["responses"]
        ]
        return policy

    def _apply_prior(self, prior_weights: Sequence[Mapping[str, float]]) -> None:
        """Seed per-action b vectors using provided weight dictionaries."""
        for action, weights in enumerate(prior_weights):
            for feature_name, weight in weights.items():
                index = self._indexer.ensure(feature_name)
                self._b[action][index] = weight * self._config.regularization

    def _vectorize(
        self,
        features: FeatureVector | Mapping[str, float],
        *,
        grow: bool | None = None,
    ) -> np.ndarray:
        """Return dense representation with optional growth control."""
        mapping = features.values if isinstance(features, FeatureVector) else features
        if grow is None:
            grow = self._config.allow_feature_growth
        return self._indexer.vectorize(mapping, grow=grow)

    def _score_all(self, features: FeatureVector | Mapping[str, float]) -> np.ndarray:
        """Compute LinUCB scores for all actions."""
        vector = self._vectorize(features)
        scores = np.zeros(self._config.num_actions, dtype=np.float64)
        for action in range(self._config.num_actions):
            theta = np.linalg.solve(self._A[action], self._b[action])
            exploration = self._config.alpha * math.sqrt(
                float(vector.T @ np.linalg.solve(self._A[action], vector))
            )
            scores[action] = float(theta.T @ vector) + exploration
        return scores

"""Runtime management helpers for contextual bandit policies."""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from policy.linucb import (
    LinUCBConfig,
    LinUCBPolicy
)

LOG = logging.getLogger("app.semantic_cache.policy")

@dataclass(frozen=True, slots=True)
class PolicyRuntimeConfig:
    """Configuration describing the desired bandit policy backend."""

    policy_type: str
    num_actions: int
    feature_dimension: int
    allow_feature_growth: bool
    linucb_alpha: float
    linucb_regularization: float
    linucb_min_propensity: float
    ts_regularization: float
    ts_sampling_variance: float
    ts_min_propensity: float
    snapshot_path: Path | None = None
    autosave_interval: int = 50


class TTLPolicyManager:
    """Thread-safe wrapper around contextual bandit policies."""

    def __init__(
        self,
        *,
        policy: LinUCBPolicy ,
        snapshot_path: Path | None,
        autosave_interval: int,
    ) -> None:
        self._policy = policy
        self._snapshot_path = snapshot_path
        self._autosave_interval = max(1, autosave_interval)
        self._lock = asyncio.Lock()
        self._updates_since_flush = 0

    async def select_action(self, features: Mapping[str, float]) -> tuple[int, float]:
        """Choose an action index and return its propensity."""
        async with self._lock:
            action, _score = self._policy.choose(features)
            propensity = self._policy.propensity(action, features)
            LOG.info(
                "semantic_cache.policy.select",
                extra={
                    "extra": {
                        "action": action,
                        "propensity": round(propensity, 4),
                        "feature_summary": self._summarize_features(features),
                    }
                },
            )
            return action, propensity

    async def update(
        self, action: int, features: Mapping[str, float], reward: float
    ) -> None:
        """Apply a reward update and persist snapshots when configured."""
        async with self._lock:
            self._policy.update(action, reward, features)
            self._updates_since_flush += 1
            LOG.info(
                "semantic_cache.policy.update",
                extra={
                    "extra": {
                        "action": action,
                        "reward": round(reward, 4),
                        "updates_since_flush": self._updates_since_flush,
                        "feature_summary": self._summarize_features(features),
                    }
                },
            )
            if (
                self._snapshot_path
                and self._updates_since_flush >= self._autosave_interval
            ):
                self._persist_locked()

    async def flush(self) -> None:
        """Force persistence of the current policy state."""
        async with self._lock:
            if self._snapshot_path is None:
                return
            self._persist_locked()
            LOG.info(
                "semantic_cache.policy.flush",
                extra={
                    "extra": {
                        "snapshot_path": str(self._snapshot_path),
                    }
                },
            )

    def _persist_locked(self) -> None:
        if self._snapshot_path is None:
            return
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._snapshot_path.with_suffix(".tmp")
        self._policy.save(tmp_path)
        tmp_path.replace(self._snapshot_path)
        self._updates_since_flush = 0
        LOG.info(
            "semantic_cache.policy.snapshot_persisted",
            extra={
                "extra": {
                    "snapshot_path": str(self._snapshot_path),
                }
            },
        )

    @staticmethod
    def _summarize_features(features: Mapping[str, float], limit: int = 5) -> str:
        if not features:
            return "features=count=0"
        sorted_items = sorted(
            features.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        head = ", ".join(f"{name}={value:.3f}" for name, value in sorted_items[:limit])
        if not head:
            return f"features=count={len(features)}"
        return f"features=count={len(features)} top=[{head}]"


def build_policy_manager(config: PolicyRuntimeConfig) -> TTLPolicyManager:
    """Instantiate a policy manager based on the supplied configuration."""
    snapshot_path = config.snapshot_path

    if config.policy_type == "linucb":
        policy = _load_or_initialize_linucb(config, snapshot_path)
    else:
        raise ValueError(f"Unsupported policy_type '{config.policy_type}'")

    return TTLPolicyManager(
        policy=policy,
        snapshot_path=snapshot_path,
        autosave_interval=config.autosave_interval,
    )


def load_or_initialize_linucb(
    config: PolicyRuntimeConfig,
    snapshot_path: Path | None,
) -> LinUCBPolicy:
    linucb_config = LinUCBConfig(
        alpha=config.linucb_alpha,
        num_actions=config.num_actions,
        dimension=config.feature_dimension,
        regularization=config.linucb_regularization,
        min_propensity=config.linucb_min_propensity,
        allow_feature_growth=config.allow_feature_growth,
    )
    if snapshot_path and snapshot_path.exists():
        print('Loading LinUCB policy from snapshot:', snapshot_path)
        return LinUCBPolicy.load(snapshot_path)
    return LinUCBPolicy(config=linucb_config)

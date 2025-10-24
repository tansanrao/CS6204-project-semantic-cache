from __future__ import annotations

from pathlib import Path

import pytest

from app.features.semantic_cache.policy.bandit import (
    LinUCBConfig,
    LinUCBPolicy,
    ThompsonSamplingConfig,
    ThompsonSamplingPolicy,
)
from app.features.semantic_cache.policy.features import FeatureVector
from app.features.semantic_cache.policy.manager import (
    PolicyRuntimeConfig,
    build_policy_manager,
)


def test_linucb_prefers_action_with_positive_reward(tmp_path: Path) -> None:
    config = LinUCBConfig(alpha=0.1, num_actions=2, dimension=8)
    policy = LinUCBPolicy(config=config)
    feature = FeatureVector(values={"bias": 1.0, "freshness": 0.2})

    # Register feature indices.
    policy.choose(feature)

    for _ in range(10):
        policy.update(action=1, reward=1.0, features=feature)
        policy.update(action=0, reward=-0.5, features=feature)

    action, score = policy.choose(feature)
    assert action == 1
    assert score > 0.0

    propensity = policy.propensity(action=1, features=feature)
    assert 0.0 < propensity <= 1.0

    snapshot_path = tmp_path / "linucb.json"
    policy.save(snapshot_path)
    restored = LinUCBPolicy.load(snapshot_path)
    action_restored, _ = restored.choose(feature)
    assert action_restored == 1


def test_linucb_prior_weights_bias_initial_choice() -> None:
    config = LinUCBConfig(alpha=0.0, num_actions=2, dimension=4, regularization=1.0)
    prior = (
        {"bias": 0.8},
        {"bias": -0.2},
    )
    policy = LinUCBPolicy(config=config, prior_weights=prior)
    feature = FeatureVector(values={"bias": 1.0})

    action, _ = policy.choose(feature)
    assert action == 0


def test_thompson_sampling_prefers_action_with_positive_reward(tmp_path: Path) -> None:
    config = ThompsonSamplingConfig(
        num_actions=2,
        dimension=6,
        sampling_variance=1e-6,
    )
    policy = ThompsonSamplingPolicy(config=config)
    feature = FeatureVector(values={"bias": 1.0, "fresh": 0.5})

    policy.choose(feature)

    for _ in range(10):
        policy.update(action=1, reward=1.0, features=feature)
        policy.update(action=0, reward=-0.5, features=feature)

    action, score = policy.choose(feature)
    assert action == 1
    assert score != 0.0

    propensity = policy.propensity(action=1, features=feature)
    assert propensity > 0.5

    snapshot_path = tmp_path / "thompson.json"
    policy.save(snapshot_path)
    restored = ThompsonSamplingPolicy.load(snapshot_path)
    restored_action, _ = restored.choose(feature)
    assert restored_action == 1


@pytest.mark.asyncio
async def test_policy_manager_updates_and_persists(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "policy.json"
    runtime_config = PolicyRuntimeConfig(
        policy_type="linucb",
        num_actions=2,
        feature_dimension=8,
        allow_feature_growth=True,
        linucb_alpha=0.2,
        linucb_regularization=1.0,
        linucb_min_propensity=1e-3,
        ts_regularization=1.0,
        ts_sampling_variance=1.0,
        ts_min_propensity=1e-3,
        snapshot_path=snapshot_path,
        autosave_interval=1,
    )
    manager = build_policy_manager(runtime_config)
    features = {"bias": 1.0}

    action, propensity = await manager.select_action(features)
    assert 0 <= action < runtime_config.num_actions
    assert 0.0 < propensity <= 1.0

    await manager.update(action, features, reward=0.75)
    await manager.flush()

    assert snapshot_path.exists()

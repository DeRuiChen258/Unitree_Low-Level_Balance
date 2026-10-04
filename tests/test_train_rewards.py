"""奖励单测：分量有限、权重生效、终止条件。"""

from __future__ import annotations

import numpy as np

from cb_train.envs.terminations import check_termination
from cb_train.rewards import RewardConfig, compute_reward


def _reward(weights: dict[str, float]):
    config = RewardConfig(weights=weights, params={"velocity_tracking": {"sigma": 0.35}})
    return compute_reward(
        config=config,
        base_lin_vel=np.array([1.0, 0.0, 0.0]),
        base_ang_vel=np.zeros(3),
        rpy=(0.0, 0.0, 0.0),
        base_height=0.75,
        command=np.array([1.0, 0.0, 0.0]),
        action=np.zeros(29),
        last_action=np.zeros(29),
        torque=np.zeros(29),
        contacts=(True, True),
        expected_contacts=(True, True),
        foot_speed=0.0,
        progress=0.01,
        skill_delta_error=0.0,
    )


def test_reward_components_and_weight() -> None:
    """分量有限；提高存活权重提高总奖励。"""
    low, components = _reward({"alive": 0.5})
    high, _ = _reward({"alive": 2.0})
    assert np.isfinite(low) and np.isfinite(high)
    assert high > low
    assert set(components) >= {"velocity_tracking", "upright", "alive", "progress"}


def test_termination_conditions() -> None:
    """低高度/NaN/大倾角触发终止。"""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    assert check_termination(base_height=0.2, base_quat=quat, joint_pos=np.zeros(29), joint_vel=np.zeros(29), config={})[0]
    bad = np.zeros(29)
    bad[0] = np.nan
    assert check_termination(base_height=0.8, base_quat=quat, joint_pos=bad, joint_vel=np.zeros(29), config={})[1] == "nan"
    tilted = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0, 0.0])
    assert check_termination(base_height=0.8, base_quat=tilted, joint_pos=np.zeros(29), joint_vel=np.zeros(29), config={})[0]

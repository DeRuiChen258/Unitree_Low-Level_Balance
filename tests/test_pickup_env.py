"""PickupBalanceEnv 单测：观测/动作/奖励/终止。"""

from __future__ import annotations

import numpy as np

from cb_pickup.env import PickupBalanceEnv, PickupEnvConfig
from cb_pickup.scenarios import SCENARIOS

SCENE = "/home/violet/Workspace/Code/Embedded_code/unitree_workspace/mujoco_menagerie/unitree_g1/scene.xml"


def test_env_reset_step_reward() -> None:
    """reset/step 形状正确、奖励有限、primitive ID 可用。"""
    env = PickupBalanceEnv(PickupEnvConfig(scene_path=SCENE, primitive_step_s=0.1), scenario=SCENARIOS["front"], seed=0)
    obs, info = env.reset()
    assert obs.ndim == 1 and obs.size == env.observation_dim
    assert np.isfinite(obs).all()
    for action in (1, 2, "REACH"):
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "stability_margin" in info
    env.close()

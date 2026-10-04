"""把环境状态适配到 `cb_train.rewards.compute_reward`。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cb_common.types import RobotState

from ..rewards import RewardConfig, compute_reward


def compute_reward_adapter(
    *,
    config: Mapping[str, Any] | RewardConfig,
    state: RobotState,
    command: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    torque: np.ndarray | None,
    expected_contacts: tuple[bool, bool],
    target: np.ndarray,
    teacher_target: np.ndarray,
    dt: float = 0.02,
) -> tuple[float, dict[str, float]]:
    """计算奖励：脚滑用接触帧踝速度近似，progress 用前向速度 × dt。"""
    cfg = config if isinstance(config, RewardConfig) else RewardConfig.from_config(config)
    contact = state.contact
    foot_speed = 0.0
    if any(contact):
        ankle_speed = float(np.linalg.norm(state.joint_vel[[4, 5, 10, 11]])) * 0.12
        foot_speed = ankle_speed
    progress = float(state.base_lin_vel[0]) * dt
    skill_error = float(np.linalg.norm(np.asarray(target) - np.asarray(teacher_target)))
    return compute_reward(
        config=cfg,
        base_lin_vel=state.base_lin_vel,
        base_ang_vel=state.base_ang_vel,
        rpy=state.rpy(),
        base_height=float(state.base_pos[2]),
        command=command,
        action=action,
        last_action=last_action,
        torque=torque,
        contacts=contact,
        expected_contacts=expected_contacts,
        foot_speed=foot_speed,
        progress=progress,
        skill_delta_error=skill_error,
    )

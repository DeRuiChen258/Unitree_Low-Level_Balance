"""奖励项（每项独立可开关，第 6.3 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    """奖励权重与参数（来自 configs/reward.yaml）。"""

    weights: dict[str, float]
    params: dict[str, dict[str, float]]

    @classmethod
    def from_config(cls, reward_cfg: Mapping[str, Any]) -> RewardConfig:
        """解析 reward.yaml。"""
        section = dict(reward_cfg.get("reward", {}))
        weights: dict[str, float] = {}
        params: dict[str, dict[str, float]] = {}
        for name, item in section.items():
            item = dict(item)
            weights[name] = float(item.get("weight", 0.0))
            params[name] = {
                key: float(value)
                for key, value in item.items()
                if key not in ("enabled", "weight") and isinstance(value, (int, float))
            }
        return cls(weights=weights, params=params)


def compute_reward(
    *,
    config: RewardConfig,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    rpy: tuple[float, float, float],
    base_height: float,
    command: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    torque: np.ndarray | None,
    contacts: tuple[bool, bool],
    expected_contacts: tuple[bool, bool],
    foot_speed: float,
    progress: float,
    skill_delta_error: float,
) -> tuple[float, dict[str, float]]:
    """计算总奖励与分量（无量纲，权重来自配置）。"""
    vx, vy, yaw_rate = float(command[0]), float(command[1]), float(command[2])
    vel_err = float(np.hypot(base_lin_vel[0] - vx, base_lin_vel[1] - vy))
    yaw_err = float(abs(base_ang_vel[2] - yaw_rate))
    roll, pitch = float(rpy[0]), float(rpy[1])
    tilt = float(roll * roll + pitch * pitch)
    sigma_v = config.params.get("velocity_tracking", {}).get("sigma", 0.35)
    sigma_yaw = config.params.get("yaw_tracking", {}).get("sigma", 0.30)
    sigma_tilt = config.params.get("upright", {}).get("sigma_rad", 0.20)
    target_h = config.params.get("height", {}).get("target_m", 0.74)
    sigma_h = config.params.get("height", {}).get("sigma_m", 0.12)
    sigma_skill = config.params.get("skill_tracking", {}).get("sigma_rad", 0.35)
    components = {
        "velocity_tracking": float(np.exp(-(vel_err**2) / max(sigma_v**2, 1e-6))),
        "yaw_tracking": float(np.exp(-(yaw_err**2) / max(sigma_yaw**2, 1e-6))),
        "upright": float(np.exp(-tilt / max(sigma_tilt**2, 1e-6))),
        "height": float(np.exp(-((base_height - target_h) ** 2) / max(sigma_h**2, 1e-6))),
        "action_rate": float(-np.mean((np.asarray(action) - np.asarray(last_action)) ** 2)),
        "torque_sq": float(-np.mean(np.square(torque))) if torque is not None else 0.0,
        "feet_slip": float(-foot_speed),
        "contact_schedule": float(np.mean([a == b for a, b in zip(contacts, expected_contacts, strict=False)])),
        "alive": 1.0,
        "progress": float(progress),
        "skill_tracking": float(np.exp(-(skill_delta_error**2) / max(sigma_skill**2, 1e-6))),
    }
    total = sum(config.weights.get(name, 0.0) * value for name, value in components.items())
    return float(total), components

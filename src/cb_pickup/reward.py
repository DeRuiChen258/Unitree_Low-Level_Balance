"""PickupBalanceEnv 奖励定义（第 15 节）：task + balance + motion + grasp − energy − collision。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class PickupRewardConfig:
    """奖励权重与任务参数。"""

    w_task: float = 1.0
    w_balance: float = 2.0
    w_motion: float = 0.3
    w_grasp: float = 5.0
    w_energy: float = 1e-5
    w_collision: float = 5.0
    w_step: float = 0.2
    w_fall: float = 50.0
    success_bonus: float = 100.0
    margin_target_m: float = 0.06
    approach_scale: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PickupRewardConfig:
        """从 YAML 映射构造。"""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data).items() if k in known})


def compute_pickup_reward(
    *,
    config: PickupRewardConfig,
    info: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    action_primitive: str,
    dt: float,
) -> tuple[float, dict[str, float]]:
    """计算一步奖励与分量（每个分量可单独消融）。"""
    margin = float(info.get("stability_margin", 0.0))
    predicted = float(info.get("predicted_margin", margin))
    trunk = abs(float(info.get("trunk_pitch", 0.0)))
    hand_distance = float(info.get("hand_distance", 1.0))
    object_height = float(info.get("object_height", 0.0))
    initial_height = float(info.get("initial_object_height", 0.0))
    grasped = bool(info.get("grasped", False))
    success = bool(info.get("success", False))
    fall = bool(info.get("fall", False))
    contacts = int(info.get("self_contacts", 0))
    energy = float(info.get("energy_delta", 0.0))
    steps = int(info.get("steps", 0))
    previous_margin = float(previous.get("stability_margin", margin)) if previous else margin
    task_progress = max(0.0, previous_margin - margin) if previous else 0.0
    components = {
        "task": config.w_task * (config.approach_scale * max(0.0, 1.0 - hand_distance) + 2.0 * int(grasped) + 4.0 * int(success)),
        "balance": config.w_balance * min(margin, max(predicted, 0.0)),
        "motion": config.w_motion * (1.0 - min(trunk / 1.2, 1.0)),
        "grasp": config.w_grasp * (1.0 if grasped else 0.0) + (config.success_bonus if success else 0.0),
        "energy": -config.w_energy * energy,
        "collision": -config.w_collision * contacts,
        "step": -config.w_step * steps,
        "instability": -config.w_fall * (1.0 if fall else 0.0) - 0.5 * task_progress * 0.0,
        "height_progress": 2.0 * max(0.0, object_height - initial_height),
    }
    # 稳定性优先：margin 低于 0 时对任务收益施加 veto 权重
    total = sum(components.values())
    if margin < 0.0:
        total *= 0.5
    return float(total), {k: float(v) for k, v in components.items()}

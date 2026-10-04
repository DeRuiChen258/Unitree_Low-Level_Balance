"""10 个实验场景（第 19 节）+ 域随机化（第 18 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PickupScenario:
    """单个场景定义。"""

    name: str
    object_position: np.ndarray
    object_size: float = 0.035
    object_mass: float = 0.4
    friction: float = 1.0
    perturbation: tuple[float, float, float] | None = None
    target_change_at_s: float | None = None
    target_change_to: np.ndarray | None = None
    post_grasp_impulse: np.ndarray | None = None
    description: str = ""


SCENARIOS: dict[str, PickupScenario] = {
    # 注：G1 肩-腕臂展实测约 0.385 m；「双手抱两侧面」要求箱体侧面落在双臂可达范围，
    # 因此物体中心取 x≈0.34 m（箱体近侧面距足尖约 0.14 m），
    # 抱取点 = 箱体左右两侧面靠上位置，箱体质量 ≈ 机器人质量的一半（见 configs/g1_pickup.yaml）。
    "front": PickupScenario("front", np.array([0.34, 0.0, 0.45]), description="物体正前方（双手分别抱住箱体两侧面）"),
    "left": PickupScenario("left", np.array([0.34, 0.06, 0.45]), description="物体偏左（双手分别抱两侧面）"),
    "right": PickupScenario("right", np.array([0.34, -0.06, 0.45]), description="物体偏右（双手分别抱两侧面）"),
    "far": PickupScenario("far", np.array([0.38, 0.0, 0.45]), description="物体略远（接近双侧抱取可达边界）"),
    "deep": PickupScenario("deep", np.array([0.32, 0.0, 0.45]), object_size=0.32, description="物体更近，需要更大弯腰/屈膝"),
    "push": PickupScenario(
        "push",
        np.array([0.34, 0.0, 0.45]),
        perturbation=(0.35, 0.0, 30.0),
        description="弯腰过程中加入轻微外部扰动",
    ),
    "low_friction": PickupScenario("low_friction", np.array([0.34, 0.0, 0.45]), friction=0.45, description="低摩擦地面"),
    "random": PickupScenario(
        "random",
        np.array([0.34, 0.0, 0.45]),
        description="目标位置随机（由 seed 采样）",
    ),
    "moving": PickupScenario(
        "moving",
        np.array([0.34, 0.0, 0.45]),
        target_change_at_s=1.2,
        target_change_to=np.array([0.36, -0.05, 0.45]),
        description="目标位置发生变化",
    ),
    "post_grasp_push": PickupScenario(
        "post_grasp_push",
        np.array([0.34, 0.0, 0.45]),
        post_grasp_impulse=np.array([0.0, 22.0, 0.0]),
        description="抓取以后再受到轻微侧向扰动",
    ),
}


def sample_scenario(name: str, rng: np.random.Generator) -> PickupScenario:
    """按名称返回场景；random 场景采样目标位置。"""
    scenario = SCENARIOS.get(name)
    if scenario is None:
        raise KeyError(f"unknown pickup scenario {name!r}; available={sorted(SCENARIOS)}")
    if name != "random":
        return PickupScenario(**{**scenario.__dict__})
    position = np.array([rng.uniform(0.48, 0.58), rng.uniform(-0.18, 0.18), 0.31])
    return PickupScenario("random", position, description=f"随机目标 {np.round(position, 3)}")


@dataclass
class RandomizationConfig:
    """域随机化参数（训练 ON / 测试固定标准环境）。"""

    mass_scale: tuple[float, float] = (0.9, 1.1)
    friction_scale: tuple[float, float] = (0.8, 1.2)
    damping_scale: tuple[float, float] = (0.8, 1.2)
    sensor_noise_std: float = 0.002
    imu_noise_std: float = 0.01
    latency_steps: tuple[int, int] = (0, 1)
    object_mass_scale: tuple[float, float] = (0.8, 1.2)
    object_position_noise_m: float = 0.02
    object_size_scale: tuple[float, float] = (0.9, 1.1)
    enabled: bool = False
    seed: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RandomizationConfig:
        """从 YAML 映射构造。"""
        known = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in dict(data).items() if key in known}
        return cls(**payload)

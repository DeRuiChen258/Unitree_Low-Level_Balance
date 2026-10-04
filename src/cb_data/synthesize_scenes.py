"""S5 场景随机化（NOCS 方法论迁移声明见 docs/design/data_synthesis.md）。

注意：NOCS 属物体 6D 位姿领域，本模块只迁移「合成数据覆盖 + 真实数据校准」的方法论，
不声称该论文直接支持人形动作增强。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SceneSpec:
    """单个仿真场景随机化配置。"""

    scene_id: str
    terrain: str
    friction_scale: float
    mass_scale: float
    kp_scale: float
    kd_scale: float
    actuator_delay_steps: int
    obs_noise_std: float
    push_probability: float
    push_force_range_n: list[float] = field(default_factory=list)
    seed: int = 0


def generate_scenes(config: Mapping[str, Any], *, count: int = 12, seed: int = 1) -> list[SceneSpec]:
    """按域随机化范围生成场景列表（可复现）。"""
    rng = np.random.default_rng(seed)
    terrains = config.get("terrain", ["flat"])
    if isinstance(terrains, str):
        terrains = [terrains]
    mass = config.get("mass_scale", [1.0, 1.0])
    friction = config.get("friction_scale", [1.0, 1.0])
    kp = config.get("kp_scale", [1.0, 1.0])
    kd = config.get("kd_scale", [1.0, 1.0])
    delay = config.get("actuator_delay_steps", [0, 0])
    noise = float(config.get("obs_noise_std", 0.0))
    push_prob = float(config.get("push_prob", 0.0))
    push_force = list(config.get("push_force_n", [0.0, 0.0]))
    out: list[SceneSpec] = []
    for i in range(count):
        out.append(
            SceneSpec(
                scene_id=f"scene-{i:03d}",
                terrain=str(terrains[i % len(terrains)]),
                friction_scale=float(rng.uniform(*friction)),
                mass_scale=float(rng.uniform(*mass)),
                kp_scale=float(rng.uniform(*kp)),
                kd_scale=float(rng.uniform(*kd)),
                actuator_delay_steps=int(rng.integers(int(delay[0]), int(delay[1]) + 1)),
                obs_noise_std=noise,
                push_probability=push_prob,
                push_force_range_n=[float(v) for v in push_force],
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return out


def save_scenes(path: str | Path, scenes: list[SceneSpec]) -> Path:
    """写入 scene manifest JSON。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(scene) for scene in scenes], indent=2, ensure_ascii=False), encoding="utf-8")
    return out

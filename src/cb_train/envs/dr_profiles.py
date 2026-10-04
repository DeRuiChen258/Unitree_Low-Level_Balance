"""域随机化：质量/摩擦/PD 增益/延迟/噪声/外力（profile 可 hash）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DomainRandomizer:
    """按 reset 采样的域随机化器。"""

    config: Mapping[str, Any]
    seed: int = 0
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    obs_noise_std: float = 0.0
    obs_delay_steps: int = 0
    action_delay_steps: int = 0
    push_schedule: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=np.random.default_rng, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def profile_hash(self) -> str:
        """随机化 profile hash（进入实验记录）。"""
        payload = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def sample(self, *, episode_steps: int, dt: float) -> dict[str, Any]:
        """采样一次 episode 的随机化参数。"""
        cfg = self.config
        self.mass_scale = float(self.rng.uniform(*cfg.get("mass_scale", [1.0, 1.0])))
        self.friction_scale = float(self.rng.uniform(*cfg.get("friction_scale", [1.0, 1.0])))
        self.kp_scale = float(self.rng.uniform(*cfg.get("kp_scale", [1.0, 1.0])))
        self.kd_scale = float(self.rng.uniform(*cfg.get("kd_scale", [1.0, 1.0])))
        self.obs_noise_std = float(cfg.get("obs_noise_std", 0.0))
        delay = cfg.get("actuator_delay_steps", [0, 0])
        self.action_delay_steps = int(self.rng.integers(int(delay[0]), int(delay[1]) + 1))
        self.obs_delay_steps = int(self.rng.integers(0, max(1, self.action_delay_steps + 1)))
        self.push_schedule = []
        prob = float(cfg.get("push_prob", 0.0))
        force_range = cfg.get("push_force_n", [0.0, 0.0])
        duration_range = cfg.get("push_duration_s", [0.05, 0.1])
        for _ in range(int(episode_steps * prob)):
            start = int(self.rng.integers(20, max(21, episode_steps - 40)))
            duration = int(round(self.rng.uniform(*duration_range) / dt))
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            magnitude = self.rng.uniform(*force_range)
            force = np.array([np.cos(angle), np.sin(angle), 0.0]) * magnitude
            self.push_schedule.append((start, start + max(1, duration), force))
        return {
            "mass_scale": self.mass_scale,
            "friction_scale": self.friction_scale,
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "obs_noise_std": self.obs_noise_std,
            "obs_delay_steps": self.obs_delay_steps,
            "action_delay_steps": self.action_delay_steps,
            "num_pushes": len(self.push_schedule),
            "profile_hash": self.profile_hash,
        }

    def apply_model(self, model: Any, *, base_mass: np.ndarray | None = None, base_friction: np.ndarray | None = None) -> None:
        """把质量/摩擦随机化写入 MjModel（提供 base 时从基准恢复，避免累积放大）。"""
        if base_mass is not None:
            model.body_mass[:] = base_mass * self.mass_scale
            model.body_mass[0] = 0.0
        else:
            model.body_mass[1:] *= self.mass_scale
        if base_friction is not None:
            model.geom_friction[:] = base_friction
            model.geom_friction[:, 0] = np.clip(model.geom_friction[:, 0] * self.friction_scale, 0.05, None)
        else:
            model.geom_friction[:, 0] = np.clip(model.geom_friction[:, 0] * self.friction_scale, 0.05, None)

    def push_at(self, step: int) -> np.ndarray | None:
        """返回该步应施加的外力（无则 None）。"""
        for start, end, force in self.push_schedule:
            if start <= step < end:
                return force
        return None

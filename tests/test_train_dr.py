"""域随机化单测：可复现、范围、质量不累积。"""

from __future__ import annotations

import numpy as np

from cb_train.envs.dr_profiles import DomainRandomizer


class _Model:
    """最小 MjModel 替身。"""

    def __init__(self) -> None:
        self.body_mass = np.ones(4)
        self.geom_friction = np.ones((4, 3))


def test_dr_reproducible() -> None:
    """同 seed 采样一致，范围合法。"""
    config = {"mass_scale": [0.9, 1.1], "friction_scale": [0.8, 1.2], "actuator_delay_steps": [0, 2], "push_prob": 0.01, "push_force_n": [10, 20]}
    first = DomainRandomizer(config, seed=5).sample(episode_steps=500, dt=0.02)
    second = DomainRandomizer(config, seed=5).sample(episode_steps=500, dt=0.02)
    assert first["mass_scale"] == second["mass_scale"]
    assert 0.9 <= first["mass_scale"] <= 1.1


def test_dr_apply_no_accumulation() -> None:
    """提供 base 时反复 apply 不放大质量。"""
    config = {"mass_scale": [1.0, 1.0], "friction_scale": [1.0, 1.0]}
    randomizer = DomainRandomizer(config, seed=1)
    randomizer.sample(episode_steps=10, dt=0.02)
    model = _Model()
    base_mass = model.body_mass.copy()
    base_friction = model.geom_friction.copy()
    for _ in range(10):
        randomizer.apply_model(model, base_mass=base_mass, base_friction=base_friction)
    assert np.allclose(model.body_mass[1:], 1.0)

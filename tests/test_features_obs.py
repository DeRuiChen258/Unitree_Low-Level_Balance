"""观测 spec 单测：维度、hash、组装与缺失策略。"""

from __future__ import annotations

import numpy as np
import pytest

from cb_features.obs_spec import default_obs_spec


def test_spec_dimensions_and_hash() -> None:
    """默认 spec 维度 434，hash 稳定。"""
    spec = default_obs_spec(4, 16)
    assert spec.flat_dim == 131
    assert spec.history_dim == 404
    assert spec.total_dim == 434
    assert spec.hash() == default_obs_spec(4, 16).hash()
    assert spec.hash() != default_obs_spec(2, 16).hash()


def test_assemble_and_missing_zero() -> None:
    """组装形状正确；缺失非历史组按 zero 填充。"""
    spec = default_obs_spec(2, 8)
    frames = {
        "contact": np.array([1.0, 0.0]),
        "phase_clock": np.zeros(4),
        "skill_embedding": np.zeros(8),
        "balance_feats": np.zeros(8),
    }
    history = {
        "ang_vel": np.zeros((2, 3)),
        "gravity": np.zeros((2, 3)),
        "lin_vel": np.zeros((2, 3)),
        "command": np.zeros((2, 5)),
        "joint_pos": np.zeros((2, 29)),
        "joint_vel": np.zeros((2, 29)),
        "last_action": np.zeros((2, 29)),
    }
    obs = spec.assemble(frames, history)
    assert obs.shape == (spec.total_dim,)
    assert np.isfinite(obs).all()
    with pytest.raises(ValueError):
        spec.assemble(frames, {"ang_vel": np.zeros((3, 3))})

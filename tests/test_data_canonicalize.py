"""S2 canonicalize 单测：重采样、根相对、相位。"""

from __future__ import annotations

import numpy as np

from cb_data.canonicalize import canonicalize, estimate_phase, resample_motion, root_relative
from tests.helpers import make_motion


def test_resample_fps() -> None:
    """60 → 30 FPS 重采样后帧数减半。"""
    clip = make_motion(frames=61)
    clip.fps = 60.0
    out = resample_motion(clip, 30.0)
    assert abs(out.frames - 31) <= 1
    assert out.fps == 30.0


def test_root_relative_and_phase() -> None:
    """根相对坐标把根 xz 归零；相位在 [0,1)。"""
    clip = make_motion(frames=20)
    clip.root_pos[:, 0] = np.arange(20) * 0.01
    rel = root_relative(clip)
    assert np.allclose(rel.root_pos[:, 0], 0.0)
    canonical = canonicalize(clip, {"target_fps": 30.0, "root_align": True, "phase_parameterize": True})
    assert canonical.phase.shape == (20,)
    assert canonical.phase.min() >= 0.0 and canonical.phase.max() < 1.0
    assert canonical.phase_sin_cos().shape == (20, 2)


def test_phase_from_contact_switches() -> None:
    """接触切换驱动周期相位。"""
    clip = make_motion(frames=40)
    clip.contacts[:20, 0] = True
    clip.contacts[:20, 1] = False
    clip.contacts[20:, 0] = False
    clip.contacts[20:, 1] = True
    phase = estimate_phase(clip)
    assert phase.shape == (40,)
    assert np.all(np.diff(phase[:20]) >= -1e-9)

"""S6 物理过滤单测：越界拒绝/修复、根高度、原因统计。"""

from __future__ import annotations

import numpy as np

from cb_data.physics_filter import FilterConfig, PhysicsFilter
from tests.helpers import make_segment


def test_filter_accepts_smooth_clip() -> None:
    """平滑片段被接受。"""
    filt = PhysicsFilter(FilterConfig(scene_path=""))
    verdict = filt.check(make_segment(frames=40), joint_limits=(np.full(29, -3.0), np.full(29, 3.0)))
    assert verdict.status in ("accepted", "repaired")
    assert verdict.quality["max_abs_jerk"] >= 0.0


def test_filter_rejects_low_root() -> None:
    """根高度低于安全下限必须拒绝。"""
    segment = make_segment(frames=20)
    segment.root_pos[:, 2] = 0.2
    filt = PhysicsFilter(FilterConfig(scene_path="", root_height_min_m=0.55))
    verdict = filt.check(segment, joint_limits=(np.full(29, -3.0), np.full(29, 3.0)))
    assert verdict.status == "rejected"
    assert "root_height" in verdict.reasons


def test_filter_repairs_small_position_violation() -> None:
    """轻微位置越界可修复并记录。"""
    segment = make_segment(frames=20)
    segment.qpos[0, 0] = 10.0
    filt = PhysicsFilter(FilterConfig(scene_path=""))
    verdict = filt.check(segment, joint_limits=(np.full(29, -1.0), np.full(29, 1.0)))
    assert verdict.status in ("repaired", "rejected")
    if verdict.status == "repaired":
        repaired = filt.repair(segment, (np.full(29, -1.0), np.full(29, 1.0)))
        assert np.max(np.abs(repaired.qpos[:, 0])) <= 1.0

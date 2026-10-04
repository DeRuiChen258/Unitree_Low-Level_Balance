"""组合单测：并行仲裁（躯干优先）、串行混合。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import POLICY_INDEX
from cb_common.types import SkillOutput
from cb_skills.composer import compose_parallel, compose_serial


def test_parallel_priority() -> None:
    """下肢基座优先于上肢叠加。"""
    base = SkillOutput(delta_q=np.zeros(29), skill="run", active_groups=("lower_body", "torso"))
    base.delta_q[POLICY_INDEX["left_knee"]] = 0.5
    overlay = SkillOutput(delta_q=np.zeros(29), skill="wave", active_groups=("right_arm",))
    overlay.delta_q[POLICY_INDEX["right_shoulder_pitch"]] = 0.3
    overlay.delta_q[POLICY_INDEX["left_knee"]] = 9.0
    out = compose_parallel(base, [overlay], weights=[1.0])
    assert abs(out.delta_q[POLICY_INDEX["left_knee"]] - 0.5) < 1e-9
    assert abs(out.delta_q[POLICY_INDEX["right_shoulder_pitch"]] - 0.3) < 1e-9
    assert out.metadata["mode"] == "parallel"


def test_serial_smoke() -> None:
    """串行组合返回最后一个技能的输出形状。"""
    first = SkillOutput(delta_q=np.zeros(29), skill="stand")
    second = SkillOutput(delta_q=np.ones(29), skill="walk")
    out = compose_serial([first, second], blend_frames=4)
    assert out.skill in ("stand", "walk")
    assert np.all(np.isfinite(out.delta_q))

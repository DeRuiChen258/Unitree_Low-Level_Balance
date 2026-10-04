"""挥手技能单测：只写手臂、降幅、周期完成。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import JOINT_GROUPS, POLICY_INDEX
from cb_common.types import SkillCommand, SkillContext
from cb_skills.skill_base import load_skill_specs
from cb_skills.wave import WaveSkill
from tests.helpers import make_state


def _skill(side: str = "right", cycles: float = 2.0) -> WaveSkill:
    spec = load_skill_specs("configs/skills.yaml")["wave"]
    skill = WaveSkill(spec)
    skill.enter(
        SkillContext(
            command=SkillCommand(skill="wave", params={"side": side, "cycles": cycles, "frequency_hz": 1.0}),
            default_pose=np.zeros(29),
        )
    )
    return skill


def test_wave_only_arm_joints() -> None:
    """增量只作用于指定手臂。"""
    skill = _skill()
    out = skill.update(make_state(), 0.1)
    arm = set(POLICY_INDEX[name] for name in JOINT_GROUPS["right_arm"])
    nonzero = set(np.flatnonzero(np.abs(out.delta_q) > 1e-9).tolist())
    assert nonzero <= arm
    assert out.metadata["wave_side"] == 1.0


def test_wave_degrade_and_done() -> None:
    """降幅生效；周期完成。"""
    full_skill = _skill(cycles=1.0)
    reduced_skill = _skill(cycles=1.0)
    reduced_skill.degrade(0.5)
    full = full_skill.update(make_state(), 0.05).delta_q.copy()
    reduced = reduced_skill.update(make_state(), 0.05).delta_q
    assert np.max(np.abs(reduced)) <= np.max(np.abs(full)) + 1e-6
    skill = _skill(cycles=1.0)
    for _ in range(30):
        skill.update(make_state(), 0.05)
    assert skill.done()

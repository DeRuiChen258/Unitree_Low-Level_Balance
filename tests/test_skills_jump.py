"""跳跃技能单测：五段相位、腿部对称、完成判定。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import POLICY_INDEX
from cb_common.types import JumpPhase, SkillCommand, SkillContext
from cb_skills.jump import JumpSkill
from cb_skills.skill_base import load_skill_specs
from tests.helpers import make_state


def _skill() -> JumpSkill:
    spec = load_skill_specs("configs/skills.yaml")["jump"]
    skill = JumpSkill(spec)
    skill.enter(SkillContext(command=SkillCommand(skill="jump", params={"crouch_depth_rad": 0.12}), default_pose=np.zeros(29)))
    return skill


def test_jump_phase_progression() -> None:
    """approach → takeoff → flight → landing → recover。"""
    skill = _skill()
    phases = []
    state = make_state(height=0.79)
    for _ in range(200):
        out = skill.update(state, 0.02)
        phases.append(out.jump_phase)
        state = make_state(height=0.79 + 0.01 * len(phases), vx=0.5)
    assert JumpPhase.TAKEOFF in phases
    assert JumpPhase.NONE in phases[-5:] or skill.done()


def test_jump_legs_symmetric() -> None:
    """蹲伸阶段左右腿增量对称。"""
    skill = _skill()
    state = make_state()
    out = skill.update(state, 0.02)
    left = out.delta_q[POLICY_INDEX["left_knee"]]
    right = out.delta_q[POLICY_INDEX["right_knee"]]
    assert abs(left - right) < 1e-9

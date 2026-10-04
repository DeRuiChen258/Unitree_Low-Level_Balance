"""转弯技能单测：误差方向、容差制动、指令结构。"""

from __future__ import annotations

import numpy as np

from cb_common.types import SkillCommand, SkillContext
from cb_skills.skill_base import load_skill_specs
from cb_skills.turn import TurnSkill
from tests.helpers import make_state


def _skill() -> TurnSkill:
    spec = load_skill_specs("configs/skills.yaml")["turn"]
    skill = TurnSkill(spec)
    skill.enter(
        SkillContext(
            command=SkillCommand(skill="turn", params={"target_yaw_deg": 45.0, "vx": 0.8}),
            default_pose=np.zeros(29),
        )
    )
    return skill


def test_turn_command_direction() -> None:
    """需要正转角时命令为正（或进入容差后的制动命令）。"""
    skill = _skill()
    out = skill.update(make_state(), 0.02)
    yaw_rate = float(out.metadata["command"]["yaw_rate"])
    assert yaw_rate > 0.0 or abs(yaw_rate) <= 0.2
    assert abs(out.metadata["command"]["vx"] - 0.8) < 1e-9


def test_turn_damping_in_tolerance() -> None:
    """接近目标时命令制动而不是继续加速。"""
    skill = _skill()
    skill._yaw_start = 0.0
    skill._yaw_unwrapped = np.radians(44.0)
    skill._last_yaw = 0.0
    state = make_state(yaw_rate=0.3)
    out = skill.update(state, 0.02)
    assert float(out.metadata["command"]["yaw_rate"]) <= 0.2

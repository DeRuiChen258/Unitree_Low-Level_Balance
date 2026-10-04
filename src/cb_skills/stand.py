"""站立保持（默认安全态）。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS
from cb_common.types import RobotState, SkillContext, SkillOutput

from .skill_base import Skill


class StandSkill(Skill):
    """保持默认姿态；漂移由小脑残差策略补偿。"""

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self.spec = spec
        self._t = 0.0

    def enter(self, ctx: SkillContext) -> None:
        """复位计时。"""
        self._t = 0.0

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """返回零增量与相位 0。"""
        self._t += dt
        return SkillOutput(
            delta_q=np.zeros(NUM_JOINTS),
            skill="stand",
            phase=0.0,
            expected_contact=(True, True),
            active_groups=("lower_body", "torso", "arms"),
            metadata={"command": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}},
        )

    def exit(self) -> None:
        """无状态需要清理。"""
        return None

"""恢复技能：小步 / 站立恢复（分级）。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS, POLICY_INDEX
from cb_common.types import RobotState, SkillContext, SkillOutput

from .skill_base import Skill


class RecoverSkill(Skill):
    """失衡后的分级恢复：先降速站立，必要时小步调整。"""

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self.spec = spec
        self.max_time = 1.5
        self._t = 0.0

    def enter(self, ctx: SkillContext) -> None:
        """读取最大恢复时间。"""
        self.max_time = float(ctx.command.params.get("max_recovery_time_s", 1.5))
        self._t = 0.0

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """输出小幅踝/髋校正（站立恢复），并逐步归零。"""
        self._t += dt
        w = float(np.clip(1.0 - self._t / max(self.max_time, 1e-3), 0.0, 1.0))
        roll, pitch, _ = state.rpy()
        delta = np.zeros(NUM_JOINTS)
        ankle_pitch = float(np.clip(0.35 * pitch, -0.15, 0.15)) * w
        ankle_roll = float(np.clip(0.35 * roll, -0.15, 0.15)) * w
        delta[POLICY_INDEX["left_ankle_pitch"]] += ankle_pitch
        delta[POLICY_INDEX["right_ankle_pitch"]] += ankle_pitch
        delta[POLICY_INDEX["left_ankle_roll"]] += ankle_roll
        delta[POLICY_INDEX["right_ankle_roll"]] -= ankle_roll
        delta[POLICY_INDEX["left_hip_pitch"]] += 0.25 * ankle_pitch
        delta[POLICY_INDEX["right_hip_pitch"]] += 0.25 * ankle_pitch
        return SkillOutput(
            delta_q=delta,
            skill="recover",
            phase=min(self._t / max(self.max_time, 1e-3), 1.0),
            expected_contact=(True, True),
            active_groups=("lower_body", "torso"),
            metadata={"command": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}},
        )

    def done(self) -> bool:
        """超时即完成。"""
        return self._t >= self.max_time

    def exit(self) -> None:
        """无状态需要清理。"""
        return None

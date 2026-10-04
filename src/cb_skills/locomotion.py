"""走/跑技能：输出速度指令意图，关节目标由教师/学习策略生成。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS
from cb_common.types import RobotState, SkillContext, SkillOutput

from .skill_base import Skill


class LocomotionSkill(Skill):
    """walk/run：维护 vx/vy/yaw_rate 指令与速度误差。"""

    def __init__(self, spec, *, skill_name: str) -> None:  # type: ignore[no-untyped-def]
        self.spec = spec
        self.skill_name = skill_name
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self._t = 0.0

    def enter(self, ctx: SkillContext) -> None:
        """读取参数中的速度指令。"""
        self.vx = float(ctx.command.params.get("vx", 0.0))
        self.vy = float(ctx.command.params.get("vy", 0.0))
        self.yaw_rate = float(ctx.command.params.get("yaw_rate", 0.0))
        self._t = 0.0

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """返回零增量 + 速度指令意图（直接命令，稳态误差在评测中按稳态窗口统计）。"""
        self._t += dt
        ramp = min(1.0, self._t / 0.8)  # 起步命令渐入，避免站立→跑动的姿态瞬态触发安全层
        command_vx = self.vx * ramp
        command_vy = self.vy * ramp
        command_yaw = self.yaw_rate * ramp
        measured_vx = float(state.base_lin_vel[0])
        measured_vy = float(state.base_lin_vel[1])
        return SkillOutput(
            delta_q=np.zeros(NUM_JOINTS),
            skill=self.skill_name,
            phase=float((self._t % 0.62) / 0.62),
            expected_contact=(False, True) if (self._t % 0.62) < 0.31 else (True, False),
            active_groups=("lower_body", "torso", "arms"),
            metadata={
                "command": {"vx": command_vx, "vy": command_vy, "yaw_rate": command_yaw},
                "speed_error": float(np.hypot(measured_vx - self.vx, measured_vy - self.vy)),
            },
        )

    def exit(self) -> None:
        """无状态需要清理。"""
        return None

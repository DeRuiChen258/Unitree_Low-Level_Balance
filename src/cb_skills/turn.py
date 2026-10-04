"""转弯技能：目标 yaw / yaw_rate → 速度指令意图（由教师策略执行）。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS
from cb_common.types import RobotState, SkillContext, SkillOutput

from .skill_base import Skill


class TurnSkill(Skill):
    """按目标转角闭环输出 yaw_rate 指令。"""

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self.spec = spec
        self.target_yaw = 0.0
        self.yaw_rate_max = 0.6
        self.kp = 1.0
        self.kd = 0.45
        self.vx = 0.8
        self._yaw_start = 0.0
        self._yaw_unwrapped = 0.0
        self._last_yaw: float | None = None
        self._t = 0.0

    def enter(self, ctx: SkillContext) -> None:
        """读取目标转角并记录起始 yaw（由首次 update 填充）。"""
        self.target_yaw = float(np.radians(ctx.command.params.get("target_yaw_deg", 0.0)))
        self.yaw_rate_max = float(ctx.command.params.get("yaw_rate_max_rad_s", 0.35))
        self._yaw_start = float("nan")
        self._yaw_unwrapped = 0.0
        self._last_yaw = None
        self._t = 0.0
        self.vx = float(ctx.command.params.get("vx", 0.8))

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """返回 yaw_rate 指令（增量恒为 0）。"""
        _, _, yaw = state.rpy()
        if self._last_yaw is None:
            self._yaw_unwrapped = yaw
        else:
            delta = ((yaw - self._last_yaw + np.pi) % (2.0 * np.pi)) - np.pi
            self._yaw_unwrapped += delta
        self._last_yaw = yaw
        if np.isnan(self._yaw_start):
            self._yaw_start = self._yaw_unwrapped
        error = (self._yaw_start + self.target_yaw) - self._yaw_unwrapped
        measured = float(state.base_ang_vel[2])
        if abs(error) < np.radians(4.0):
            # 进入容差后主动制动，抵消旋转惯性
            yaw_rate = float(np.clip(-0.6 * measured, -0.2, 0.2))
        else:
            raw = self.kp * error - self.kd * measured
            if abs(raw) < 0.35:
                raw = float(np.copysign(0.35, error))  # 死区补偿：平台低速 yaw 响应弱
            yaw_rate = float(np.clip(raw, -self.yaw_rate_max, self.yaw_rate_max))
        self._t += dt
        return SkillOutput(
            delta_q=np.zeros(NUM_JOINTS),
            skill="turn",
            phase=min(abs(self._yaw_unwrapped - self._yaw_start) / max(abs(self.target_yaw), 1e-3), 1.0),
            expected_contact=state.contact,
            active_groups=("lower_body", "torso"),
            metadata={"command": {"vx": self.vx, "vy": 0.0, "yaw_rate": yaw_rate}, "yaw_error": error},
        )

    def done(self) -> bool:
        """由调用方根据 yaw_error 判断；默认按时间上限。"""
        return self._t >= 3.0

    def exit(self) -> None:
        """无状态需要清理。"""
        return None

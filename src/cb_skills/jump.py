"""跳跃技能：approach → takeoff → flight → landing → recover 五段状态机。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cb_common.joints import NUM_JOINTS, POLICY_INDEX
from cb_common.types import JumpPhase, RobotState, SkillContext, SkillOutput

from .skill_base import Skill


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def _leg_delta(hip: float, knee: float, ankle: float, *, roll: float = 0.0) -> np.ndarray:
    """构造左右腿对称增量（策略序）。"""
    delta = np.zeros(NUM_JOINTS)
    for side in ("left", "right"):
        delta[POLICY_INDEX[f"{side}_hip_pitch"]] = hip
        delta[POLICY_INDEX[f"{side}_knee"]] = knee
        delta[POLICY_INDEX[f"{side}_ankle_pitch"]] = ankle
        delta[POLICY_INDEX[f"{side}_hip_roll"]] = roll if side == "left" else -roll
    return delta


@dataclass
class JumpSkill(Skill):
    """浅蹲 + 快速伸展的可控跳跃（参数经 MuJoCo 实测标定）。"""

    spec: object
    crouch_depth: float = 0.10
    forward_boost: float = 0.10
    phase: JumpPhase = JumpPhase.NONE
    t: float = 0.0
    t_phase: float = 0.0
    apex_h: float = 0.0
    takeoff_h: float = 0.0
    _ctx: SkillContext | None = field(default=None, repr=False)

    def enter(self, ctx: SkillContext) -> None:
        """进入 approach 相位。"""
        self._ctx = ctx
        self.crouch_depth = float(ctx.command.params.get("crouch_depth_rad", self.crouch_depth))
        self.phase = JumpPhase.APPROACH
        self.t = 0.0
        self.t_phase = 0.0
        self.apex_h = 0.0

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """按五段相位返回腿部增量。"""
        self.t += dt
        self.t_phase += dt
        height = float(state.base_pos[2])
        delta = np.zeros(NUM_JOINTS)
        if self.phase == JumpPhase.APPROACH and self.t >= 1.0:
            self._set_phase(JumpPhase.TAKEOFF)
        if self.phase == JumpPhase.TAKEOFF:
            if self.t_phase <= 0.22:  # noqa: SIM102 - 分支语义清晰优先
                w = _smoothstep(self.t_phase / 0.22)
                delta = _leg_delta(-self.crouch_depth * w, 2.0 * self.crouch_depth * w, -self.crouch_depth * w)
            elif self.t_phase <= 0.31:
                delta = _leg_delta(-self.crouch_depth, 2.0 * self.crouch_depth, -self.crouch_depth)
            elif self.t_phase <= 0.41:
                w = _smoothstep((self.t_phase - 0.31) / 0.10)
                extend = _leg_delta(0.0, -0.05, 0.15)
                crouch = _leg_delta(-self.crouch_depth, 2.0 * self.crouch_depth, -self.crouch_depth)
                delta = crouch + w * (extend - crouch)
            else:
                self.takeoff_h = height
                self._set_phase(JumpPhase.FLIGHT)
        if self.phase == JumpPhase.FLIGHT:
            delta = _leg_delta(0.0, -0.05, 0.15)
            self.apex_h = max(self.apex_h, height)
            if (state.base_lin_vel[2] < 0.0 and height < self.apex_h - 0.015) or self.t_phase > 0.8:
                self._set_phase(JumpPhase.LANDING)
        if self.phase == JumpPhase.LANDING:
            w = _smoothstep(self.t_phase / 0.18)
            delta = _leg_delta(-0.30 * w, 0.60 * w, -0.30 * w)
            if self.t_phase >= 0.18:
                self._set_phase(JumpPhase.RECOVER)
        if self.phase == JumpPhase.RECOVER:
            w = 1.0 - _smoothstep(self.t_phase / 0.35)
            delta = _leg_delta(-0.30 * w, 0.60 * w, -0.30 * w)
            if self.t_phase >= 0.35:
                self._set_phase(JumpPhase.NONE)
        return SkillOutput(
            delta_q=delta,
            skill="jump",
            phase=min(self.t / 1.9, 1.0),
            jump_phase=self.phase,
            expected_contact=(False, False) if self.phase == JumpPhase.FLIGHT else (True, True),
            active_groups=("lower_body", "torso"),
            metadata={
                "apex_h": self.apex_h,
                "takeoff_h": self.takeoff_h,
                "forward_boost": self.forward_boost,
            },
        )

    def _set_phase(self, phase: JumpPhase) -> None:
        self.phase = phase
        self.t_phase = 0.0

    def done(self) -> bool:
        """落地恢复完成后返回 True。"""
        return self.phase == JumpPhase.NONE and self.t > 0.1

    def exit(self) -> None:
        """复位相位状态。"""
        self.phase = JumpPhase.NONE

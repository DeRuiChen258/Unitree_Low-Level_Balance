"""挥手技能：单臂轨迹叠加，可并行在站立/行走/跑步之上（复用 G1_Waving 相位思路）。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS, POLICY_INDEX
from cb_common.types import RobotState, SkillContext, SkillOutput

from .skill_base import Skill

ARM_JOINTS: dict[str, tuple[str, ...]] = {
    "left": (
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
    ),
    "right": (
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
    ),
}


class WaveSkill(Skill):
    """挥手：肩抬到位 + 肘/腕正弦摆动；平衡退化时降幅。"""

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self.spec = spec
        self.side = "right"
        self.cycles = 2.0
        self.frequency = 1.2
        self.amplitude_scale = 1.0
        self._t = 0.0
        self._degrade = 0.0

    def enter(self, ctx: SkillContext) -> None:
        """读取参数（side/cycles/frequency/amplitude）并复位。"""
        self.side = str(ctx.command.params.get("side", "right"))
        self.cycles = float(ctx.command.params.get("cycles", 2.0))
        self.frequency = float(ctx.command.params.get("frequency_hz", 1.2))
        self.amplitude_scale = float(ctx.command.params.get("amplitude_scale", 1.0))
        self._t = 0.0
        self._degrade = 0.0

    def degrade(self, factor: float = 0.5) -> None:
        """平衡退化时降幅（fallback 链 wave_reduced）。"""
        self.amplitude_scale = min(self.amplitude_scale, factor)
        self._degrade += 1.0

    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """返回手臂增量；相位为 [0,1) 的技能进度。"""
        self._t += dt
        total = max(self.cycles / max(self.frequency, 1e-3), 1e-3)
        progress = min(self._t / total, 1.0)
        angle = 2.0 * np.pi * self.frequency * self._t
        ramp = min(1.0, self._t / 0.25)
        scale = self.amplitude_scale * ramp
        delta = np.zeros(NUM_JOINTS)
        sign = 1.0 if self.side == "right" else -1.0
        values = {
            "shoulder_pitch": 0.06 * scale,
            "shoulder_roll": -sign * (0.10 + 0.02 * scale),
            "shoulder_yaw": 0.0,
            "elbow": 0.12 + 0.06 * scale * np.sin(angle),
            "wrist_roll": sign * 0.12 * scale * np.sin(angle + 0.3),
            "wrist_pitch": 0.06 * scale * np.sin(angle + 0.6),
        }
        for role, value in values.items():
            name = f"{self.side}_{role}"
            if name in POLICY_INDEX:
                delta[POLICY_INDEX[name]] = float(value)
        return SkillOutput(
            delta_q=delta,
            skill="wave",
            phase=progress,
            expected_contact=state.contact,
            active_groups=(f"{self.side}_arm",),
            metadata={"wave_side": 1.0 if self.side == "right" else -1.0, "degraded": self._degrade > 0},
        )

    def done(self) -> bool:
        """两周期完成后返回 True。"""
        return self._t >= self.cycles / max(self.frequency, 1e-3)

    def exit(self) -> None:
        """无状态需要清理。"""
        return None

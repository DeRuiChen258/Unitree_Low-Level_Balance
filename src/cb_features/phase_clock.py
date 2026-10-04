"""步态相位 / 技能相位 / 跳跃相位（第 5.3 节）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cb_common.types import JumpPhase


@dataclass
class PhaseClock:
    """由接触序列估计步态相位；技能相位由调用方进度更新。"""

    gait_period_s: float = 0.62
    skill_phase: float = 0.0
    jump_phase: JumpPhase = JumpPhase.NONE
    jump_phase_progress: float = 0.0
    _phase: float = field(default=0.0, init=False)
    _last_contact: tuple[bool, bool] = field(default=(True, True), init=False)
    _time_since_switch: float = field(default=0.0, init=False)
    _dt: float = field(default=0.02, init=False)

    def update(self, dt: float, contact: tuple[bool, bool]) -> np.ndarray:
        """推进相位并返回 4 维 [gait_sin, gait_cos, skill_sin, skill_cos]。"""
        self._dt = float(dt)
        self._time_since_switch += dt
        if contact != self._last_contact:
            # 接触切换时对齐相位零点（左脚触地为 0，右腿为半周期）。
            self._phase = 0.0 if contact[0] and not self._last_contact[0] else 0.5
            self._last_contact = contact
        rate = 1.0 / max(self.gait_period_s, 1e-3)
        self._phase = float((self._phase + dt * rate) % 1.0)
        if self.jump_phase != JumpPhase.NONE:
            self.jump_phase_progress = float((self.jump_phase_progress + dt) % 1.0)
        gait_angle = 2.0 * np.pi * self._phase
        skill_angle = 2.0 * np.pi * float(self.skill_phase % 1.0)
        return np.array(
            [np.sin(gait_angle), np.cos(gait_angle), np.sin(skill_angle), np.cos(skill_angle)], dtype=np.float64
        )

    def set_skill_phase(self, phase: float) -> None:
        """设置技能归一化进度 [0,1)。"""
        self.skill_phase = float(phase % 1.0)

    def set_jump_phase(self, phase: JumpPhase, progress: float = 0.0) -> None:
        """设置跳跃五段相位与段内进度。"""
        self.jump_phase = phase
        self.jump_phase_progress = float(np.clip(progress, 0.0, 1.0))

    def jump_one_hot(self) -> np.ndarray:
        """返回 5 维跳跃相位 one-hot（顺序：none/approach/takeoff/flight/landing/recover 取前 5）。"""
        order = [JumpPhase.NONE, JumpPhase.APPROACH, JumpPhase.TAKEOFF, JumpPhase.FLIGHT, JumpPhase.LANDING]
        out = np.zeros(5, dtype=np.float64)
        if self.jump_phase in order:
            out[order.index(self.jump_phase)] = 1.0
        return out

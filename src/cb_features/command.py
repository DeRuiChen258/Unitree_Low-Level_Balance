"""指令向量与技能编码（无魔法数字：技能表来自 configs/skills.yaml）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cb_common.types import SKILL_NAMES, JumpPhase


@dataclass
class CommandVector:
    """调度器指令向量：`vx, vy, yaw_rate, jump_flag, wave_side`。"""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    jump_flag: float = 0.0
    wave_side: float = 0.0     # -1 左 / 0 无 / +1 右

    def to_array(self) -> np.ndarray:
        """返回 5 维向量（单位 m/s, m/s, rad/s, bool, sign）。"""
        return np.array([self.vx, self.vy, self.yaw_rate, self.jump_flag, self.wave_side], dtype=np.float64)

    @classmethod
    def from_array(cls, vec: np.ndarray) -> CommandVector:
        """从 5 维向量构造。"""
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size != 5:
            raise ValueError(f"command vector must be 5, got {arr.size}")
        return cls(*[float(v) for v in arr])


def encode_skill(skill: str, jump_phase: JumpPhase = JumpPhase.NONE, dim: int = 16) -> np.ndarray:
    """返回技能嵌入：8 技能 one-hot + 5 跳跃相位 + 余量（默认 16 维）。"""
    if dim < len(SKILL_NAMES) + 5:
        raise ValueError(f"skill embedding dim {dim} too small; need >= {len(SKILL_NAMES) + 5}")
    out = np.zeros(dim, dtype=np.float64)
    if skill not in SKILL_NAMES:
        raise KeyError(f"unknown skill {skill!r}; expected one of {SKILL_NAMES}")
    out[SKILL_NAMES.index(skill)] = 1.0
    order = [JumpPhase.NONE, JumpPhase.APPROACH, JumpPhase.TAKEOFF, JumpPhase.FLIGHT, JumpPhase.LANDING]
    if jump_phase in order:
        out[len(SKILL_NAMES) + order.index(jump_phase)] = 1.0
    remainder = out[len(SKILL_NAMES) + 5 :]
    if remainder.size:
        remainder[:] = 0.0
    return out

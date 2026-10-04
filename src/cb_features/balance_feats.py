"""平衡专用特征（小脑的「前庭与小脑」类比，第 5.2 节）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cb_common.types import RobotState


def balance_features(
    state: RobotState,
    *,
    target_rpy: tuple[float, float] = (0.0, 0.0),
    com_reference: np.ndarray | None = None,
    contact_history: np.ndarray | None = None,
) -> np.ndarray:
    """返回 8 维平衡特征：姿态误差、CoM/CP 相对量、支撑余量、接触置信度、综合分。"""
    roll, pitch, _ = state.rpy()
    roll_err = float(roll - target_rpy[0])
    pitch_err = float(pitch - target_rpy[1])
    com = np.zeros(3) if state.com is None else np.asarray(state.com, dtype=np.float64)
    cp = np.zeros(2) if state.cp is None else np.asarray(state.cp, dtype=np.float64)
    ref = np.zeros(3) if com_reference is None else np.asarray(com_reference, dtype=np.float64)
    com_rel = com[:2] - ref[:2]
    margin = float(state.support_margin)
    contact_conf = _contact_confidence(state.contact, contact_history)
    score = float(
        np.clip(
            (1.0 - (abs(roll_err) + abs(pitch_err)) / 0.4 - max(0.0, -margin) / 0.05) * (0.5 + 0.5 * contact_conf),
            0.0,
            1.0,
        )
    )
    return np.array(
        [roll_err, pitch_err, float(com_rel[0]), float(com_rel[1]), float(cp[0]), float(cp[1]), margin, score],
        dtype=np.float64,
    )


def _contact_confidence(contact: tuple[bool, bool], history: np.ndarray | None) -> float:
    """接触置信度：双支撑 1.0，单支撑 0.6，腾空 0.2；有历史时按稳定时长加权。"""
    base = 1.0 if all(contact) else (0.6 if any(contact) else 0.2)
    if history is None or history.size == 0:
        return base
    recent = np.asarray(history, dtype=np.float64).reshape(-1)
    stability = float(min(1.0, recent.size / 10.0))
    return float(np.clip(base * (0.8 + 0.2 * stability), 0.0, 1.0))


@dataclass
class BalanceFeatureBuilder:
    """带运动参考的平衡特征构造器（供技能/评测复用）。"""

    com_reference: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_rpy: tuple[float, float] = (0.0, 0.0)
    contact_history: list[float] = field(default_factory=list)
    capacity: int = 10

    def update_reference(self, com_reference: np.ndarray) -> None:
        """更新目标 CoM（默认原点）。"""
        self.com_reference = np.asarray(com_reference, dtype=np.float64)

    def push_contact(self, contact: tuple[bool, bool]) -> None:
        """记录接触历史（用于置信度）。"""
        self.contact_history.append(1.0 if all(contact) else 0.5 if any(contact) else 0.0)
        if len(self.contact_history) > self.capacity:
            self.contact_history.pop(0)

    def build(self, state: RobotState) -> np.ndarray:
        """构造 8 维特征。"""
        return balance_features(
            state,
            target_rpy=self.target_rpy,
            com_reference=self.com_reference,
            contact_history=np.asarray(self.contact_history) if self.contact_history else None,
        )

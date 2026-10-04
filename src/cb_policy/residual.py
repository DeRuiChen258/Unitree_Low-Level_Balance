"""平衡残差补偿（参考 G1_Waving BalanceController 协议；真机默认 passthrough）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from cb_common.joints import NUM_JOINTS, POLICY_INDEX
from cb_common.types import BalanceMode, RobotState


@dataclass
class ResidualCompensator:
    """姿态/CoM 误差 → 踝/髋/腰小补偿（仅仿真；真机需显式开启并限幅）。"""

    mode: BalanceMode = BalanceMode.LOW_LEVEL_SIM_ONLY
    att_gain: float = 0.45
    com_gain: float = 0.35
    gyro_gain: float = 0.03
    integral_gain: float = 0.05
    max_ankle_rad: float = 0.12
    max_hip_rad: float = 0.06
    max_torso_rad: float = 0.05
    deadband_rad: float = 0.005
    filter_hz: float = 5.0
    _ankle_pitch: float = field(default=0.0, init=False)
    _ankle_roll: float = field(default=0.0, init=False)
    _pitch_integral: float = field(default=0.0, init=False)
    _roll_integral: float = field(default=0.0, init=False)

    def reset(self) -> None:
        """清空滤波器与积分。"""
        self._ankle_pitch = self._ankle_roll = 0.0
        self._pitch_integral = self._roll_integral = 0.0

    def update(self, state: RobotState, dt: float, *, com_reference: np.ndarray | None = None) -> np.ndarray:
        """返回 29 维补偿增量（策略序，仿真/限幅使用）。"""
        if self.mode == BalanceMode.PASS_THROUGH:
            return np.zeros(NUM_JOINTS)
        roll, pitch, _ = state.rpy()
        ref = np.zeros(3) if com_reference is None else np.asarray(com_reference, dtype=np.float64)
        com = np.zeros(3) if state.com is None else np.asarray(state.com, dtype=np.float64)
        com_x, com_y = float(com[0] - ref[0]), float(com[1] - ref[1])
        self._pitch_integral = float(np.clip(self._pitch_integral + pitch * dt, -self.max_ankle_rad, self.max_ankle_rad))
        self._roll_integral = float(np.clip(self._roll_integral + roll * dt, -self.max_ankle_rad, self.max_ankle_rad))
        ankle_pitch_raw = -(
            self.att_gain * pitch + self.com_gain * com_x + self.integral_gain * self._pitch_integral + self.gyro_gain * state.base_ang_vel[1]
        )
        ankle_roll_raw = -(
            self.att_gain * roll + self.com_gain * com_y + self.integral_gain * self._roll_integral + self.gyro_gain * state.base_ang_vel[0]
        )
        alpha = float(np.clip(dt * self.filter_hz, 0.0, 1.0))
        self._ankle_pitch += alpha * (_deadband(ankle_pitch_raw, self.deadband_rad) - self._ankle_pitch)
        self._ankle_roll += alpha * (_deadband(ankle_roll_raw, self.deadband_rad) - self._ankle_roll)
        ankle_pitch = float(np.clip(self._ankle_pitch, -self.max_ankle_rad, self.max_ankle_rad))
        ankle_roll = float(np.clip(self._ankle_roll, -self.max_ankle_rad, self.max_ankle_rad))
        hip_pitch = float(np.clip(0.5 * ankle_pitch, -self.max_hip_rad, self.max_hip_rad))
        hip_roll = float(np.clip(0.5 * ankle_roll, -self.max_hip_rad, self.max_hip_rad))
        torso_pitch = float(np.clip(0.25 * ankle_pitch, -self.max_torso_rad, self.max_torso_rad))
        torso_roll = float(np.clip(0.25 * ankle_roll, -self.max_torso_rad, self.max_torso_rad))
        delta = np.zeros(NUM_JOINTS)
        delta[POLICY_INDEX["left_ankle_pitch"]] += ankle_pitch
        delta[POLICY_INDEX["right_ankle_pitch"]] += ankle_pitch
        delta[POLICY_INDEX["left_ankle_roll"]] += ankle_roll
        delta[POLICY_INDEX["right_ankle_roll"]] -= ankle_roll
        delta[POLICY_INDEX["left_hip_pitch"]] += hip_pitch
        delta[POLICY_INDEX["right_hip_pitch"]] += hip_pitch
        delta[POLICY_INDEX["left_hip_roll"]] += hip_roll
        delta[POLICY_INDEX["right_hip_roll"]] += hip_roll
        delta[POLICY_INDEX["waist_pitch"]] += torso_pitch
        delta[POLICY_INDEX["waist_roll"]] += torso_roll
        return delta

    def describe(self) -> Mapping[str, float | str]:
        """返回当前增益（用于日志）。"""
        return {
            "mode": self.mode.value,
            "att_gain": self.att_gain,
            "com_gain": self.com_gain,
            "max_ankle_rad": self.max_ankle_rad,
        }


def _deadband(value: float, threshold: float) -> float:
    """死区：小于阈值的误差置零，避免抖动。"""
    return 0.0 if abs(value) < threshold else value

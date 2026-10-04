"""Mock 适配器：无硬件全链路 demo 与故障注入。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cb_common.errors import RobotError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, NUM_JOINTS
from cb_common.types import JointCommand, RobotMode, RobotState

from .interface import RobotInterface

FaultKind = Literal["nan", "freeze", "estop", "dropout", "jump"]


@dataclass
class MockAdapter(RobotInterface):
    """一阶关节跟踪 + 可注入故障的纯软件机器人。"""

    name: str = "mock"
    dt: float = 0.02
    init_height: float = 0.78
    tracking_tau_s: float = 0.08
    _q: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _dq: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _q_target: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _mode: RobotMode = field(default=RobotMode.IDLE, init=False)
    _time: float = field(default=0.0, init=False)
    _roll: float = field(default=0.0, init=False)
    _pitch: float = field(default=0.0, init=False)
    _height: float = field(default=0.0, init=False)
    _faults: set[str] = field(default_factory=set, init=False)
    _last_command: JointCommand | None = field(default=None, init=False)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0), init=False)

    def connect(self) -> None:
        """初始化到默认站立姿态。"""
        self._q = np.array(DEFAULT_JOINT_POS_POLICY, dtype=np.float64)
        self._dq = np.zeros(NUM_JOINTS)
        self._q_target = self._q.copy()
        self._height = self.init_height
        self.connected = True
        self._mode = RobotMode.STAND

    def close(self) -> None:
        """断开（幂等）。"""
        self.connected = False

    def inject(self, fault: FaultKind) -> None:
        """注入故障（测试专用）。"""
        if fault not in ("nan", "freeze", "estop", "dropout", "jump"):
            raise RobotError("unknown fault", fault=fault)
        self._faults.add(fault)
        if fault == "estop":
            self._q_target = self._q.copy()
        if fault == "jump":
            self._roll = 0.6

    def clear_faults(self) -> None:
        """清除所有注入故障。"""
        self._faults.clear()

    def set_mode(self, mode: RobotMode) -> None:
        """设置模式（mock 只记录）。"""
        self._mode = mode

    def send_command(self, command: JointCommand) -> None:
        """接收命令（未通过安全层时由上层负责拒绝）。"""
        if "dropout" in self._faults:
            return
        self._last_command = command
        self._q_target = np.asarray(command.q_target, dtype=np.float64).copy()

    def advance(self, dt: float) -> None:
        """一阶跟踪推进 + 故障注入。"""
        if not self.connected:
            raise RobotError("mock adapter is not connected")
        self._time += dt
        if "freeze" in self._faults:
            return
        if "nan" in self._faults:
            self._q[3] = np.nan
            return
        alpha = float(np.clip(dt / self.tracking_tau_s, 0.0, 1.0))
        prev = self._q.copy()
        self._q = prev + alpha * (self._q_target - prev)
        self._dq = (self._q - prev) / max(dt, 1e-9)
        self._height += (self.init_height - self._height) * alpha
        self._roll *= 1.0 - alpha
        self._pitch *= 1.0 - alpha
        if "jump" in self._faults:
            self._height = min(1.25, self._height + 0.4 * dt)
            self._roll += 0.02 * dt

    def read_state(self) -> RobotState:
        """返回统一状态。"""
        if not self.connected:
            raise RobotError("mock adapter is not connected")
        quat = _rpy_to_quat(self._roll, self._pitch, 0.0)
        return RobotState(
            timestamp=self._time,
            base_pos=np.array([0.0, 0.0, self._height]),
            base_quat=quat,
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.array([0.0, 0.0, 0.0]),
            joint_pos=self._q.copy(),
            joint_vel=self._dq.copy(),
            joint_torque=None,
            contact=(True, True),
            com=np.array([0.0, 0.0, self._height]),
            com_vel=np.zeros(3),
            cp=np.zeros(2),
            support_margin=0.08,
            frame_id=int(round(self._time / self.dt)),
        )

    @property
    def mode(self) -> RobotMode:
        """当前模式。"""
        return self._mode

    @property
    def last_command(self) -> JointCommand | None:
        """最近一次下发命令。"""
        return self._last_command


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY → wxyz 四元数。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )

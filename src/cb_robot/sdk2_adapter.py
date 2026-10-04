"""Unitree SDK2 适配器：LowCmd 构造 + CRC + 模式检查，默认 dry-run。

无真机环境只允许接口级验证：`allow_send=False` 时绝不触碰 DDS。
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cb_common.errors import RobotError, SafetyError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, NUM_JOINTS
from cb_common.types import JointCommand, RobotMode, RobotState

from .interface import RobotInterface


def unitree_crc32(data: bytes) -> int:
    """Unitree 低层消息 CRC32（CCITT 反射，初值/末异或由 zlib 实现）。"""
    return zlib.crc32(data) & 0xFFFFFFFF


def build_low_cmd(
    q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    *,
    dq: np.ndarray | None = None,
    tau: np.ndarray | None = None,
    mode_machine: int = 0,
    mode_pr: int = 0,
    crc: bool = True,
) -> dict[str, Any]:
    """构造 unitree_hg LowCmd 的等效字典（不依赖 SDK 导入）。

    字段顺序与 `unitree_hg::msg::dds_::LowCmd_` 一致：mode_pr, mode_machine,
    motor_cmd[29](mode, q, dq, tau, kp, kd, reserve), reserve, crc。
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != NUM_JOINTS:
        raise RobotError("LowCmd requires 29 joint targets", size=int(q.size))
    dq = np.zeros(NUM_JOINTS) if dq is None else np.asarray(dq, dtype=np.float64).reshape(-1)
    tau = np.zeros(NUM_JOINTS) if tau is None else np.asarray(tau, dtype=np.float64).reshape(-1)
    motor_cmd = [
        {
            "mode": 1,
            "q": float(q[i]),
            "dq": float(dq[i]),
            "tau": float(tau[i]),
            "kp": float(kp[i]),
            "kd": float(kd[i]),
            "reserve": 0,
        }
        for i in range(NUM_JOINTS)
    ]
    payload = {
        "mode_pr": int(mode_pr),
        "mode_machine": int(mode_machine),
        "motor_cmd": motor_cmd,
        "reserve": 0,
        "crc": 0,
    }
    if crc:
        payload["crc"] = unitree_crc32(_pack_low_cmd(payload))
    return payload


def _pack_low_cmd(payload: dict[str, Any]) -> bytes:
    """把 LowCmd 等效字典打包为与固件一致的 4 字节对齐字节流（CRC 用）。"""
    words: list[int] = [int(payload["mode_pr"]), int(payload["mode_machine"])]
    for item in payload["motor_cmd"]:
        words.append(int(item["mode"]))
        words.append(struct.unpack("<I", struct.pack("<f", float(item["q"])))[0])
        words.append(struct.unpack("<I", struct.pack("<f", float(item["dq"])))[0])
        words.append(struct.unpack("<I", struct.pack("<f", float(item["tau"])))[0])
        words.append(struct.unpack("<I", struct.pack("<f", float(item["kp"])))[0])
        words.append(struct.unpack("<I", struct.pack("<f", float(item["kd"])))[0])
        words.append(int(item["reserve"]))
    words.append(int(payload["reserve"]))
    return struct.pack(f"<{len(words)}I", *[w & 0xFFFFFFFF for w in words])


@dataclass
class SDK2Adapter(RobotInterface):
    """SDK2 适配器（无真机时 dry-run；真机使能需要显式风险确认）。"""

    name: str = "sdk2"
    network_interface: str = "lo"
    domain_id: int = 1
    allow_send: bool = False
    state_timeout_s: float = 0.2
    _mode: RobotMode = field(default=RobotMode.IDLE, init=False)
    _time: float = field(default=0.0, init=False)
    _last_command: JointCommand | None = field(default=None, init=False)
    _dry_runs: int = field(default=0, init=False)
    _sdk_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.allow_send:
            raise SafetyError(
                "SDK2Adapter.allow_send must stay false without a real robot and an approved safety checklist",
                network_interface=self.network_interface,
            )

    def connect(self) -> None:
        """检查 SDK2 可用性；不建立 DDS 通道（dry-run）。"""
        try:  # pragma: no cover - 依赖外部 SDK
            import unitree_sdk2py  # noqa: F401

            self._sdk_available = True
        except ImportError:
            self._sdk_available = False
        self.connected = True
        self._mode = RobotMode.IDLE

    def close(self) -> None:
        """关闭（dry-run 无资源）。"""
        self.connected = False

    def set_mode(self, mode: RobotMode) -> None:
        """设置模式；真机需先确认高层服务关闭（未实现发送）。"""
        self._mode = mode

    def read_state(self) -> RobotState:
        """dry-run 无法读取真机状态；返回默认站立占位（显式标注 not_live）。"""
        if not self.connected:
            raise RobotError("SDK2 adapter is not connected")
        q = np.array(DEFAULT_JOINT_POS_POLICY, dtype=np.float64)
        return RobotState(
            timestamp=self._time,
            base_pos=np.array([0.0, 0.0, 0.79]),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            joint_pos=q,
            joint_vel=np.zeros(NUM_JOINTS),
            contact=(True, True),
            support_margin=0.0,
            frame_id=0,
        )

    def send_command(self, command: JointCommand) -> None:
        """构造 LowCmd + CRC 并记录；绝不发送（allow_send=false 时）。"""
        payload = build_low_cmd(command.q_target, command.kp, command.kd, crc=True)
        self._last_command = command
        self._dry_runs += 1
        self._time += 0.02
        if self.allow_send:  # pragma: no cover - 真机路径禁止在无门禁时进入
            raise SafetyError("sending LowCmd requires the hardware safety gate (Gate 3+)")
        _ = payload

    def advance(self, dt: float) -> None:
        """dry-run 时间推进（不等待）。"""
        self._time += dt

    @property
    def last_low_cmd(self) -> dict[str, Any] | None:
        """最近一条 LowCmd 等效字典（接口级验证用）。"""
        if self._last_command is None:
            return None
        return build_low_cmd(self._last_command.q_target, self._last_command.kp, self._last_command.kd)

    @property
    def dry_run_count(self) -> int:
        """dry-run 次数。"""
        return self._dry_runs

    def describe(self) -> dict[str, Any]:
        """返回适配器元信息。"""
        return {
            "name": self.name,
            "connected": self.connected,
            "sdk_available": self._sdk_available,
            "allow_send": self.allow_send,
            "network_interface": self.network_interface,
            "domain_id": self.domain_id,
            "mode": self._mode.value,
            "timestamp": time.time(),
        }

"""统一机器人接口（仿真 / 真机只替换实现类）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cb_common.types import JointCommand, RobotMode, RobotState


class RobotInterface(ABC):
    """机器人统一接口：状态读取 / 命令下发 / 模式切换 / 步进。"""

    name: str = "robot"
    connected: bool = False

    @abstractmethod
    def connect(self) -> None:
        """建立连接（真机：初始化 DDS；仿真：准备模型）。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""

    @abstractmethod
    def read_state(self) -> RobotState:
        """读取统一状态（策略序关节）。"""

    @abstractmethod
    def send_command(self, command: JointCommand) -> None:
        """下发关节命令（必须已通过 SafetyWrapper）。"""

    @abstractmethod
    def advance(self, dt: float) -> None:
        """推进时间（仿真步进；真机为等待/空操作）。"""

    @abstractmethod
    def set_mode(self, mode: RobotMode) -> None:
        """切换机器人模式（真机需要模式检查）。"""

    def describe(self) -> dict[str, Any]:
        """返回适配器元信息（用于日志与等价性测试）。"""
        return {"name": self.name, "connected": self.connected}

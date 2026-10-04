"""机器人适配层：仿真 / SDK2 / Mock 同一接口。"""

from .interface import RobotInterface
from .mock_adapter import MockAdapter
from .mujoco_adapter import MuJoCoAdapter
from .sdk2_adapter import SDK2Adapter, build_low_cmd, unitree_crc32

__all__ = [
    "MockAdapter",
    "MuJoCoAdapter",
    "RobotInterface",
    "SDK2Adapter",
    "build_low_cmd",
    "unitree_crc32",
]

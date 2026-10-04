"""Balance Monitor 单测：支撑域、CP、ZMP、恢复等级。"""

from __future__ import annotations

import numpy as np

from cb_common.types import RobotState
from cb_pickup.balance_monitor import BalanceMonitor, convex_hull, foot_corners, signed_polygon_margin


def _state(com_xy: tuple[float, float] = (0.0, 0.0), velocity: tuple[float, float] = (0.0, 0.0)) -> RobotState:
    return RobotState(
        timestamp=0.0,
        base_pos=np.array([0.0, 0.0, 0.75]),
        base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        base_lin_vel=np.array([velocity[0], velocity[1], 0.0]),
        base_ang_vel=np.zeros(3),
        joint_pos=np.zeros(29),
        joint_vel=np.zeros(29),
        contact=(True, True),
        com=np.array([com_xy[0], com_xy[1], 0.75]),
    )


def test_polygon_and_margin() -> None:
    """凸包与有符号距离：内部正、外部负。"""
    polygon = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert signed_polygon_margin(np.array([0.5, 0.5]), polygon) > 0
    assert signed_polygon_margin(np.array([1.5, 0.5]), polygon) < 0
    corners = foot_corners(np.zeros(3), np.eye(3))
    assert corners.shape == (4, 2)


def test_monitor_levels_and_capture_point() -> None:
    """前倾/前移时恢复等级升高；CP 含速度项。"""
    monitor = BalanceMonitor()
    left = np.array([0.0, 0.12, 0.04])
    right = np.array([0.0, -0.12, 0.04])
    stable = monitor.update(_state(), left_foot_pos=left, left_foot_rot=np.eye(3), right_foot_pos=right, right_foot_rot=np.eye(3), dt=0.01)
    assert stable.is_stable and not stable.needs_step
    monitor.reset()
    moving = monitor.update(
        _state(com_xy=(0.25, 0.0), velocity=(0.6, 0.0)),
        left_foot_pos=left,
        left_foot_rot=np.eye(3),
        right_foot_pos=right,
        right_foot_rot=np.eye(3),
        dt=0.01,
    )
    assert moving.recovery_level >= 2
    assert moving.needs_step or moving.capture_point[0] > moving.com_projection[0]

"""本体感知特征：关节位置/速度/上一动作与历史关键组。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import NUM_JOINTS
from cb_common.types import ObsHistory, RobotState


def build_proprio(
    state: RobotState,
    *,
    last_action: np.ndarray,
    default_pose: np.ndarray,
    base_ang_vel: np.ndarray | None = None,
    projected_gravity: np.ndarray | None = None,
    base_lin_vel: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """构造单帧本体特征组（策略序，单位 rad / rad/s / m/s）。"""
    last = np.asarray(last_action, dtype=np.float64).reshape(-1)
    if last.size != NUM_JOINTS:
        raise ValueError(f"last_action must be {NUM_JOINTS}, got {last.size}")
    gravity = _projected_gravity(state.base_quat) if projected_gravity is None else np.asarray(projected_gravity, dtype=np.float64)
    ang_vel = state.base_ang_vel if base_ang_vel is None else np.asarray(base_ang_vel, dtype=np.float64)
    lin_vel = state.base_lin_vel if base_lin_vel is None else np.asarray(base_lin_vel, dtype=np.float64)
    return {
        "ang_vel": np.asarray(ang_vel, dtype=np.float64).reshape(3),
        "gravity": np.asarray(gravity, dtype=np.float64).reshape(3),
        "lin_vel": np.asarray(lin_vel, dtype=np.float64).reshape(3),
        "joint_pos": (state.joint_pos - np.asarray(default_pose, dtype=np.float64)).astype(np.float64),
        "joint_vel": state.joint_vel.astype(np.float64),
        "last_action": last,
    }


def history_key_observation(proprio: dict[str, np.ndarray], command: np.ndarray) -> np.ndarray:
    """按 spec 顺序拼接历史关键组的一帧（不含 contact/phase/skill/balance）。"""
    return np.concatenate(
        [
            proprio["ang_vel"],
            proprio["gravity"],
            proprio["lin_vel"],
            np.asarray(command, dtype=np.float64).reshape(-1),
            proprio["joint_pos"],
            proprio["joint_vel"],
            proprio["last_action"],
        ]
    )


def push_history(history: ObsHistory, key_obs: np.ndarray) -> None:
    """兼容命名：向 ObsHistory 追加一帧。"""
    history.append(key_obs)


def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """机体坐标系下的重力方向投影 g_b = R^T * [0,0,-1]。"""
    w, x, y, z = quat_wxyz
    # 旋转矩阵第三列（world z 轴在机体系的分量）
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.array([-r20, -r21, -r22], dtype=np.float64)

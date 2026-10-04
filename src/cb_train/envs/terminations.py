"""终止条件：摔倒/越界/NaN/超时（第 6.3 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def check_termination(
    *,
    base_height: float,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[bool, str]:
    """返回 (terminated, reason)。"""
    if bool(config.get("nan_guard", True)) and (
        not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all() or not np.isfinite(base_quat).all()
    ):
        return True, "nan"
    if base_height < float(config.get("min_height_m", 0.45)):
        return True, "fall_height"
    # 倾倒角只看重力方向（roll/pitch），不能把 yaw 计入
    w, x, y, z = np.asarray(base_quat, dtype=np.float64)
    gravity_z = -(1.0 - 2.0 * (x * x + y * y))
    tilt = float(np.arccos(np.clip(-gravity_z, -1.0, 1.0)))
    if tilt > float(config.get("max_tilt_rad", 1.05)):
        return True, "fall_tilt"
    if float(np.max(np.abs(joint_vel))) > float(config.get("max_joint_velocity_rad_s", 30.0)):
        return True, "joint_velocity"
    return False, ""

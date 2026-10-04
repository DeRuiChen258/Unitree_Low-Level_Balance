"""Balance Monitor：COM / ZMP / 支撑多边形 / Capture Point / 稳定裕度（第 6、7 节）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cb_common.types import RobotState

GRAVITY = 9.81


def signed_polygon_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """点到凸多边形的最小有符号距离（内部为正，逆时针多边形）。"""
    if polygon.shape[0] < 3:
        return 0.0
    inside = True
    best = float("inf")
    for i in range(polygon.shape[0]):
        a, b = polygon[i], polygon[(i + 1) % polygon.shape[0]]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        dist = float(np.dot(point - a, normal))
        if dist < 0.0:
            inside = False
        best = min(best, abs(dist))
    return best if inside else -best


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain 凸包（逆时针）。"""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).round(9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def foot_corners(position: np.ndarray, rotation: np.ndarray, length: float = 0.18, width: float = 0.08) -> np.ndarray:
    """由足体位姿生成四个足底角点（xy）。"""
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pts = []
    for forward, lateral in ((length / 2, width / 2), (length / 2, -width / 2), (-length / 2, width / 2), (-length / 2, -width / 2)):
        offset = rot @ np.array([forward, lateral, 0.0])
        pts.append(np.asarray(position, dtype=np.float64) + offset)
    return np.array([p[[0, 1]] for p in pts])


@dataclass
class BalanceState:
    """Balance Monitor 输出（第 6 节字段齐全）。"""

    com_position: np.ndarray
    com_velocity: np.ndarray
    com_projection: np.ndarray
    support_polygon: np.ndarray
    zmp: np.ndarray
    margin_x: float
    margin_y: float
    stability_margin: float
    trunk_pitch: float
    trunk_roll: float
    capture_point: np.ndarray
    is_stable: bool
    needs_step: bool
    recovery_level: int
    predicted_margin: float = 0.0
    prediction_horizon_s: float = 0.0
    per_foot_margin: tuple[float, float] = (0.0, 0.0)

    def to_features(self) -> np.ndarray:
        """返回决策层/日志用的特征向量（12 维）。"""
        return np.array(
            [
                self.com_projection[0],
                self.com_projection[1],
                self.com_velocity[0],
                self.com_velocity[1],
                self.zmp[0],
                self.zmp[1],
                self.capture_point[0],
                self.capture_point[1],
                self.stability_margin,
                self.predicted_margin,
                self.trunk_pitch,
                self.trunk_roll,
            ],
            dtype=np.float64,
        )


@dataclass
class BalanceMonitor:
    """实时平衡监控与预测（第 7、8 节）。"""

    warn_margin_m: float = 0.08
    step_margin_m: float = 0.04
    critical_margin_m: float = 0.0
    unstable_margin_m: float = -0.04
    prediction_horizon_s: float = 0.25
    com_height_fallback_m: float = 0.75
    use_com: bool = True
    use_zmp: bool = True
    use_capture_point: bool = True
    _previous_com: np.ndarray | None = field(default=None, init=False)
    _previous_com_velocity: np.ndarray | None = field(default=None, init=False)
    _previous_time: float | None = field(default=None, init=False)
    _com_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

    def reset(self) -> None:
        """清空差分状态。"""
        self._previous_com = None
        self._previous_com_velocity = None
        self._previous_time = None
        self._com_acceleration = np.zeros(3)

    def update(
        self,
        state: RobotState,
        *,
        left_foot_pos: np.ndarray,
        left_foot_rot: np.ndarray,
        right_foot_pos: np.ndarray,
        right_foot_rot: np.ndarray,
        dt: float,
        use_com: bool | None = None,
        use_zmp: bool | None = None,
        use_capture_point: bool | None = None,
        trunk_pitch: float | None = None,
        trunk_roll: float | None = None,
    ) -> BalanceState:
        """计算完整 BalanceState；消融开关可关闭 COM/ZMP/CP 信号。"""
        use_com = self.use_com if use_com is None else use_com
        use_zmp = self.use_zmp if use_zmp is None else use_zmp
        use_capture_point = self.use_capture_point if use_capture_point is None else use_capture_point
        com = np.asarray(state.com if state.com is not None else state.base_pos, dtype=np.float64)
        if not use_com:
            com = np.asarray(state.base_pos, dtype=np.float64)
        if self._previous_com is not None and self._previous_time is not None and dt > 0:
            velocity = (com - self._previous_com) / dt
            if self._previous_com_velocity is not None:
                self._com_acceleration = (velocity - self._previous_com_velocity) / dt
            self._previous_com_velocity = velocity
        else:
            velocity = np.asarray(state.base_lin_vel[:3] if state.base_lin_vel is not None else np.zeros(3), dtype=np.float64)
        self._previous_com = com.copy()
        self._previous_time = float(state.timestamp)

        polygon = convex_hull(
            np.vstack(
                [
                    foot_corners(left_foot_pos, left_foot_rot),
                    foot_corners(right_foot_pos, right_foot_rot),
                ]
            )
        )
        com_xy = com[:2]
        margin = signed_polygon_margin(com_xy, polygon)
        # 简化 ZMP：zmp = com_xy - (h/g) * com_acc_xy（准静态近似，真值在仿真中可对照）
        height = max(float(com[2]), 0.1)
        zmp = com_xy - (height / GRAVITY) * self._com_acceleration[:2] if use_zmp else com_xy.copy()
        omega = np.sqrt(GRAVITY / max(height, 0.1))
        capture = com_xy + velocity[:2] / omega if use_capture_point else com_xy.copy()
        # 预测：匀速 COM + 当前支撑域（脚步规划会在新落脚点重算）
        horizon = self.prediction_horizon_s
        future_com = com_xy + velocity[:2] * horizon
        predicted_margin = signed_polygon_margin(future_com, polygon)
        # 前后（x）与左右（y）方向裕度
        margin_x = signed_polygon_margin(np.array([com_xy[0], 0.0]), polygon) if polygon.shape[0] >= 3 else 0.0
        margin_y = signed_polygon_margin(np.array([0.0, com_xy[1]]), polygon) if polygon.shape[0] >= 3 else 0.0
        roll, pitch, _ = state.rpy()
        if trunk_pitch is not None:
            pitch = float(trunk_pitch)
        if trunk_roll is not None:
            roll = float(trunk_roll)
        combined = min(margin, predicted_margin)
        level = 0
        if combined < self.unstable_margin_m:
            level = 4
        elif combined < self.critical_margin_m:
            level = 3
        elif combined < self.step_margin_m:
            level = 2
        elif combined < self.warn_margin_m:
            level = 1
        left_margin = signed_polygon_margin(com_xy, convex_hull(foot_corners(left_foot_pos, left_foot_rot)))
        right_margin = signed_polygon_margin(com_xy, convex_hull(foot_corners(right_foot_pos, right_foot_rot)))
        return BalanceState(
            com_position=com,
            com_velocity=velocity,
            com_projection=com_xy.copy(),
            support_polygon=polygon,
            zmp=zmp,
            margin_x=margin_x,
            margin_y=margin_y,
            stability_margin=float(margin),
            trunk_pitch=float(pitch),
            trunk_roll=float(roll),
            capture_point=capture,
            is_stable=level <= 1,
            needs_step=level >= 2,
            recovery_level=level,
            predicted_margin=float(predicted_margin),
            prediction_horizon_s=horizon,
            per_foot_margin=(float(left_margin), float(right_margin)),
        )

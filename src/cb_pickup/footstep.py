"""Footstep Planner：根据 COM/ZMP/CP 预测与目标位置规划脚步（第 9、8 节）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .balance_monitor import BalanceState, convex_hull, foot_corners, signed_polygon_margin


@dataclass
class FootstepPlan:
    """脚步规划输出（第 9 节字段）。"""

    support_foot: str
    swing_foot: str
    dx: float
    dy: float
    step_duration: float
    target_foot_yaw: float
    landing_xy: np.ndarray
    feasible: bool
    reason: str = ""
    predicted_margin_after: float = 0.0


@dataclass
class FootstepPlanner:
    """平衡感知脚步规划器（含约束与落脚后稳定预测）。"""

    max_step_length_m: float = 0.28
    max_step_width_m: float = 0.16
    min_foot_separation_m: float = 0.12
    lateral_gain: float = 0.8
    forward_gain: float = 0.9
    capture_gain: float = 0.6
    margin_target_m: float = 0.06
    fallback_step_m: float = 0.12
    _last_plan: FootstepPlan | None = field(default=None, init=False)

    def plan(
        self,
        *,
        balance: BalanceState,
        left_foot_pos: np.ndarray,
        left_foot_yaw: float,
        right_foot_pos: np.ndarray,
        right_foot_yaw: float,
        target_position: np.ndarray | None,
        use_capture_point: bool = True,
        use_prediction: bool = True,
        target_bias_m: float = 0.0,
        obstacle_xy: np.ndarray | None = None,
        obstacle_radius: float = 0.0,
    ) -> FootstepPlan:
        """规划下一步；返回可行计划或不可行原因（第 9 节约束）。"""
        com = balance.com_projection
        cp = balance.capture_point if use_capture_point else com
        predicted = balance.predicted_margin if use_prediction else balance.stability_margin
        # 选择摆动脚：偏向裕度更小的一侧 / 目标侧
        left_margin, right_margin = balance.per_foot_margin
        target_dir = np.zeros(2)
        if target_position is not None:
            delta = np.asarray(target_position, dtype=np.float64)[:2] - com
            norm = np.linalg.norm(delta)
            target_dir = delta / norm if norm > 1e-6 else np.zeros(2)
        lateral_bias = (right_margin - left_margin) + target_dir[1] * 0.05
        swing_left = lateral_bias >= 0.0
        support_pos = right_foot_pos if swing_left else left_foot_pos
        swing_pos = left_foot_pos if swing_left else right_foot_pos
        swing_yaw = left_foot_yaw if swing_left else right_foot_yaw

        # 期望落点：把 CP/预测 COM 拉回支撑域内部，并朝目标方向迈步
        # Capture-point 落脚律：落脚点放在 CP 之外 margin_target 处（前后/侧向统一处理）
        away = cp - com
        away_norm = float(np.linalg.norm(away))
        away_dir = away / away_norm if away_norm > 1e-6 else np.zeros(2)
        desired = np.array(cp, dtype=np.float64) + self.margin_target_m * away_dir
        # 回退项：避免落脚点离当前摆动脚过远
        desired = 0.7 * desired + 0.3 * np.array(swing_pos[:2], dtype=np.float64)
        if target_bias_m > 0.0 and target_position is not None and np.linalg.norm(target_dir) > 1e-6:
            # 主动弓步：把摆动脚朝目标方向放置，为伸手预留 CoM 前移空间
            desired = 0.5 * desired + 0.5 * (np.array(swing_pos[:2]) + target_dir * target_bias_m)
        if target_position is not None:
            desired += target_dir * 0.06  # 朝目标方向的小步偏置（弓步方向）
        dx = float(desired[0] - swing_pos[0])
        dy = float(desired[1] - swing_pos[1])
        dx = float(np.clip(dx, -self.max_step_length_m, self.max_step_length_m))
        dy = float(np.clip(dy, -self.max_step_width_m, self.max_step_width_m))
        landing = np.array([swing_pos[0] + dx, swing_pos[1] + dy])
        # 约束 0：不踏入物体占位（避免脚-箱穿模）；把落点投影到障碍圆外
        if obstacle_xy is not None and obstacle_radius > 0.0:
            obstacle = np.asarray(obstacle_xy, dtype=np.float64)[:2]
            delta = landing - obstacle
            distance = float(np.linalg.norm(delta))
            if distance < obstacle_radius:
                direction = delta / distance if distance > 1e-6 else np.array([-1.0, 0.0])
                landing = obstacle + direction * obstacle_radius
                dx = float(np.clip(landing[0] - swing_pos[0], -self.max_step_length_m, self.max_step_length_m))
                dy = float(np.clip(landing[1] - swing_pos[1], -self.max_step_width_m, self.max_step_width_m))
                landing = np.array([swing_pos[0] + dx, swing_pos[1] + dy])

        # 约束 1：双脚最小间距（防自碰撞/交叉）
        separation = float(np.linalg.norm(landing[:2] - support_pos[:2]))
        if separation < self.min_foot_separation_m:
            return FootstepPlan(
                support_foot="right" if swing_left else "left",
                swing_foot="left" if swing_left else "right",
                dx=0.0,
                dy=0.0,
                step_duration=0.0,
                target_foot_yaw=switch_yaw(swing_yaw),
                landing_xy=swing_pos[:2].copy(),
                feasible=False,
                reason="min_foot_separation_violated",
            )
        # 约束 2：支撑脚必须接触且 COM 在支撑脚附近（避免单脚支撑失稳）
        support_margin = signed_polygon_margin(com, convex_hull(foot_corners(support_pos, np.eye(3))))
        if support_margin < -self.max_step_width_m:
            return FootstepPlan(
                support_foot="right" if swing_left else "left",
                swing_foot="left" if swing_left else "right",
                dx=0.0,
                dy=0.0,
                step_duration=0.0,
                target_foot_yaw=switch_yaw(swing_yaw),
                landing_xy=swing_pos[:2].copy(),
                feasible=False,
                reason="support_foot_margin_insufficient",
            )
        # 约束 3：落脚后预测裕度（用新支撑多边形近似）
        new_left = landing if swing_left else left_foot_pos[:2]
        new_right = landing if not swing_left else right_foot_pos[:2]
        new_polygon = convex_hull(
            np.vstack(
                [
                    foot_corners(np.array([new_left[0], new_left[1], 0.0]), np.eye(3)),
                    foot_corners(np.array([new_right[0], new_right[1], 0.0]), np.eye(3)),
                ]
            )
        )
        predicted_after = signed_polygon_margin(cp, new_polygon)
        feasible = predicted_after >= min(0.0, predicted) or abs(dx) + abs(dy) > 1e-6
        if not feasible:
            return FootstepPlan(
                support_foot="right" if swing_left else "left",
                swing_foot="left" if swing_left else "right",
                dx=0.0,
                dy=0.0,
                step_duration=0.0,
                target_foot_yaw=switch_yaw(swing_yaw),
                landing_xy=swing_pos[:2].copy(),
                feasible=False,
                reason="landing_would_not_restore_margin",
            )
        if abs(dx) < 0.02 and abs(dy) < 0.02:
            dx = float(np.copysign(min(self.fallback_step_m, self.max_step_length_m), target_dir[0] if abs(target_dir[0]) > 1e-6 else 1.0))
            landing[0] = swing_pos[0] + dx
        plan = FootstepPlan(
            support_foot="right" if swing_left else "left",
            swing_foot="left" if swing_left else "right",
            dx=dx,
            dy=dy,
            step_duration=0.45,
            target_foot_yaw=float(swing_yaw),
            landing_xy=landing,
            feasible=True,
            reason=f"predicted_margin={predicted:.3f}->{predicted_after:.3f}",
            predicted_margin_after=float(predicted_after),
        )
        self._last_plan = plan
        return plan

    @property
    def last_plan(self) -> FootstepPlan | None:
        """最近一次可行的脚步计划。"""
        return self._last_plan


def switch_yaw(yaw: float) -> float:
    """脚步朝向保持（简化：不改变足端 yaw）。"""
    return float(yaw)

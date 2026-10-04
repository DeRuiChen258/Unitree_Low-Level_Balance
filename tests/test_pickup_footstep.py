"""Footstep Planner 单测：capture-point 落脚、约束、可行性。"""

from __future__ import annotations

import numpy as np

from cb_pickup.balance_monitor import BalanceState
from cb_pickup.footstep import FootstepPlanner


def _balance(cp_x: float, margin: float = 0.02, predicted: float = 0.02) -> BalanceState:
    return BalanceState(
        com_position=np.array([0.1, 0.0, 0.75]),
        com_velocity=np.zeros(3),
        com_projection=np.array([0.1, 0.0]),
        support_polygon=np.array([[-0.15, -0.2], [0.15, -0.2], [0.15, 0.2], [-0.15, 0.2]]),
        zmp=np.array([0.1, 0.0]),
        margin_x=margin,
        margin_y=margin,
        stability_margin=margin,
        trunk_pitch=0.3,
        trunk_roll=0.0,
        capture_point=np.array([cp_x, 0.0]),
        is_stable=margin > 0.02,
        needs_step=margin < 0.04,
        recovery_level=2 if margin < 0.04 else 0,
        predicted_margin=predicted,
        per_foot_margin=(margin, margin),
    )


def test_plan_moves_toward_capture_point() -> None:
    """CP 在前方时摆动脚前移；计划包含约束字段。"""
    planner = FootstepPlanner()
    plan = planner.plan(
        balance=_balance(cp_x=0.35),
        left_foot_pos=np.array([0.0, 0.12, 0.04]),
        left_foot_yaw=0.0,
        right_foot_pos=np.array([0.0, -0.12, 0.04]),
        right_foot_yaw=0.0,
        target_position=np.array([0.4, 0.0, 0.6]),
    )
    assert plan.feasible
    assert abs(plan.dx) <= planner.max_step_length_m + 1e-9
    assert abs(plan.dy) <= planner.max_step_width_m + 1e-9
    assert plan.step_duration > 0
    assert plan.swing_foot in ("left", "right")


def test_plan_rejects_separation_violation() -> None:
    """落脚点与支撑脚过近时拒绝。"""
    planner = FootstepPlanner(min_foot_separation_m=5.0)
    plan = planner.plan(
        balance=_balance(cp_x=0.1),
        left_foot_pos=np.array([0.0, 0.12, 0.04]),
        left_foot_yaw=0.0,
        right_foot_pos=np.array([0.0, -0.12, 0.04]),
        right_foot_yaw=0.0,
        target_position=np.array([0.4, 0.0, 0.6]),
    )
    assert not plan.feasible

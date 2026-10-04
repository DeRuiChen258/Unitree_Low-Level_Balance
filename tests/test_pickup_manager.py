"""Motion Manager 单测：状态机门禁、脚步请求、恢复。"""

from __future__ import annotations

import numpy as np

from cb_pickup.balance_monitor import BalanceState
from cb_pickup.decision import PickupAction, PickupDecision, RuleBasedDecision
from cb_pickup.motion_manager import MotionManager, PickupPhase


class DangerousDecision(RuleBasedDecision):
    """强制返回 LUNGE 的测试决策。"""

    def predict(self, **kwargs):  # type: ignore[override]
        balance = kwargs["balance"]
        return PickupDecision(
            action=PickupAction.LUNGE_LEFT,
            confidence=0.9,
            risk=0.2,
            stability_margin=balance.stability_margin,
            fallback_action=PickupAction.STOP,
            reason="force_lunge",
        )


def _balance(margin: float, predicted: float | None = None, cp_x: float = 0.0) -> BalanceState:
    return BalanceState(
        com_position=np.zeros(3),
        com_velocity=np.zeros(3),
        com_projection=np.zeros(2),
        support_polygon=np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]),
        zmp=np.zeros(2),
        margin_x=margin,
        margin_y=margin,
        stability_margin=margin,
        trunk_pitch=0.0,
        trunk_roll=0.0,
        capture_point=np.array([cp_x, 0.0]),
        is_stable=margin > 0.02,
        needs_step=margin < 0.0,
        recovery_level=2 if margin < 0.0 else 0,
        predicted_margin=margin if predicted is None else predicted,
    )


def _context() -> dict:
    return {
        "left_foot_pos": np.array([0.0, 0.12, 0.04]),
        "left_foot_yaw": 0.0,
        "right_foot_pos": np.array([0.0, -0.12, 0.04]),
        "right_foot_yaw": 0.0,
        "hand_object_distance": 0.5,
        "reach_threshold_m": 0.15,
        "active_primitive": "BEND",
        "lunge_needed": False,
    }


def test_manager_gating_and_lunge() -> None:
    """不稳定时不允许 REACH；主动弓步触发 LUNGE。"""
    manager = MotionManager(planner=__import__("cb_pickup.footstep", fromlist=["FootstepPlanner"]).FootstepPlanner())
    command = manager.update(balance=_balance(0.05), goal={"object_position": np.array([0.4, 0.0, 0.6])}, context=_context(), step_finished=True)
    assert command.primitive in ("BEND", "REACH")
    manager.reset()
    manager.update(
        balance=_balance(0.06, cp_x=0.2),
        goal={"object_position": np.array([0.4, 0.0, 0.6])},
        context=_context(),
        step_finished=True,
    )
    context = _context()
    context["lunge_needed"] = True
    command = manager.update(
        balance=_balance(0.06, cp_x=0.2),
        goal={"object_position": np.array([0.4, 0.0, 0.6])},
        context=context,
        step_finished=True,
    )
    assert command.primitive == "LUNGE"
    assert manager.phase == PickupPhase.LUNGE


def test_manager_recovery_on_critical() -> None:
    """临界裕度触发 RECOVER。"""
    manager = MotionManager()
    command = manager.update(
        balance=_balance(-0.2),
        goal={"object_position": np.array([0.4, 0.0, 0.6])},
        context=_context(),
        step_finished=True,
    )
    assert command.primitive == "RECOVER"
    assert manager.phase == PickupPhase.RECOVERY

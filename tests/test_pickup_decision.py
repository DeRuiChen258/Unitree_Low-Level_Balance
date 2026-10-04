"""JEV-like 决策单测：动作集合、风险/回退、MLP 回退。"""

from __future__ import annotations

import numpy as np

from cb_pickup.balance_monitor import BalanceState
from cb_pickup.decision import MLPDecision, PickupAction, RuleBasedDecision


def _balance(margin: float, pitch: float = 0.3) -> BalanceState:
    return BalanceState(
        com_position=np.zeros(3),
        com_velocity=np.zeros(3),
        com_projection=np.zeros(2),
        support_polygon=np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]),
        zmp=np.zeros(2),
        margin_x=margin,
        margin_y=margin,
        stability_margin=margin,
        trunk_pitch=pitch,
        trunk_roll=0.0,
        capture_point=np.zeros(2),
        is_stable=margin > 0.02,
        needs_step=margin < 0.0,
        recovery_level=3 if margin < -0.05 else 0,
        predicted_margin=margin,
    )


def test_rule_decision_actions() -> None:
    """稳定→BEND；低裕度→缓弯/脚步；危险→RECOVER。"""
    model = RuleBasedDecision()
    candidates = tuple(PickupAction)
    stable = model.predict(balance=_balance(0.06), goal={"object_position": np.array([0.4, 0.0, 0.6])}, phase="BENDING", candidates=candidates)
    assert stable.action in (PickupAction.BEND, PickupAction.BEND_SLOW)
    assert 0.0 <= stable.confidence <= 1.0 and 0.0 <= stable.risk <= 1.0
    dangerous = model.predict(balance=_balance(-0.1), goal={"object_position": np.array([0.4, 0.0, 0.6])}, phase="BENDING", candidates=candidates)
    assert dangerous.action in (PickupAction.RECOVER, PickupAction.LUNGE_LEFT, PickupAction.LUNGE_RIGHT)
    assert dangerous.fallback_action in (PickupAction.RECOVER, PickupAction.STOP)


def test_mlp_falls_back_to_rule() -> None:
    """无 checkpoint 时 MLP 决策显式回退规则。"""
    model = MLPDecision(checkpoint_path="")
    out = model.predict(balance=_balance(0.05), goal={"object_position": np.array([0.4, 0.0, 0.6])}, phase="BENDING", candidates=tuple(PickupAction))
    assert out.reason == "mlp_unavailable_rule_fallback"

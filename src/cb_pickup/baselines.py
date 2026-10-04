"""四个 baseline（第 16 节）：A 固定站姿、B 固定阈值脚步、C 平衡感知、D JEV-like 决策。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .balance_monitor import BalanceState
from .decision import DecisionModel, PickupAction, PickupDecision, RuleBasedDecision
from .footstep import FootstepPlanner
from .motion_manager import MotionManager


@dataclass
class FixedFootDecision(RuleBasedDecision):
    """Baseline A：不允许脚步调整（固定站姿弯腰）。"""

    def predict(self, **kwargs: Any) -> PickupDecision:  # type: ignore[override]
        out = super().predict(**kwargs)
        if out.action in (
            PickupAction.STEP_LEFT,
            PickupAction.STEP_RIGHT,
            PickupAction.LUNGE_LEFT,
            PickupAction.LUNGE_RIGHT,
        ):
            out.action = PickupAction.BEND_SLOW if out.stability_margin > self.critical_margin_m else PickupAction.RECOVER
            out.reason = f"baseline_A_no_step:{out.reason}"
        return out


@dataclass
class ThresholdStepDecision(RuleBasedDecision):
    """Baseline B：固定角度阈值触发脚步（不使用预测/CP）。"""

    pitch_threshold_rad: float = 0.30

    def predict(self, **kwargs: Any) -> PickupDecision:  # type: ignore[override]
        balance: BalanceState = kwargs["balance"]
        goal: Mapping[str, Any] = kwargs["goal"]
        out = super().predict(**kwargs)
        target = np.asarray(goal.get("object_position", balance.com_projection), dtype=np.float64)[:2]
        direction = target - balance.com_projection
        side = "LEFT" if direction[1] >= 0 else "RIGHT"
        if abs(balance.trunk_pitch) > self.pitch_threshold_rad or balance.stability_margin < 0.02:
            out.action = PickupAction[f"LUNGE_{side}"]
            out.reason = "baseline_B_fixed_pitch_threshold"
            out.confidence = 0.85
        return out


def make_decision_model(baseline: str) -> tuple[DecisionModel, dict[str, Any]]:
    """返回（决策模型, manager 配置覆盖）。"""
    key = baseline.strip().upper()
    if key == "A":
        return FixedFootDecision(), {"max_steps": 0}
    if key == "B":
        return ThresholdStepDecision(), {}
    if key == "C":
        return RuleBasedDecision(), {}
    if key == "D":
        from .decision import MLPDecision

        return MLPDecision(checkpoint_path="runs/pickup/decision_mlp.pt", rule_fallback=RuleBasedDecision()), {}
    raise KeyError(f"unknown baseline {baseline!r}; expected A/B/C/D")


def make_manager(baseline: str, *, planner: FootstepPlanner | None = None) -> MotionManager:
    """构造带 baseline 配置的 MotionManager。"""
    model, overrides = make_decision_model(baseline)
    manager = MotionManager(planner=planner or FootstepPlanner(), decision_model=model)
    for key, value in overrides.items():
        setattr(manager, key, value)
    if baseline.strip().upper() == "A":
        manager.max_steps = 0
    return manager


def configure_baseline(controller: Any, baseline: str) -> Any:
    """把 baseline 决策模型/配置应用到已有 PickupController。"""
    model, overrides = make_decision_model(baseline)
    controller.manager.decision_model = model
    for key, value in overrides.items():
        setattr(controller.manager, key, value)
    if baseline.strip().upper() == "A":
        controller.manager.max_steps = 0
    return controller

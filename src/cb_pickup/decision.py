"""JEV-like 行为决策层：只选择行为原语，不输出连续关节角（第 5、12 节）。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .balance_monitor import BalanceState


class PickupAction(StrEnum):
    """行为原语动作集合（第 5 节，至少 13 个）。"""

    STAND = "STAND"
    BEND = "BEND"
    BEND_SLOW = "BEND_SLOW"
    STEP_LEFT = "STEP_LEFT"
    STEP_RIGHT = "STEP_RIGHT"
    LUNGE_LEFT = "LUNGE_LEFT"
    LUNGE_RIGHT = "LUNGE_RIGHT"
    REACH = "REACH"
    GRASP = "GRASP"
    LIFT = "LIFT"
    STAND_UP = "STAND_UP"
    RECOVER = "RECOVER"
    STOP = "STOP"


@dataclass
class PickupDecision:
    """DecisionOutput（第 5 节字段 + 概率与原因）。"""

    action: PickupAction
    confidence: float
    risk: float
    stability_margin: float
    fallback_action: PickupAction
    probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = "ok"
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好输出。"""
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "risk": round(self.risk, 4),
            "stability_margin": round(self.stability_margin, 4),
            "fallback_action": self.fallback_action.value,
            "probabilities": self.probabilities,
            "reason": self.reason,
            "features": self.features,
        }


class DecisionModel(Protocol):
    """JEV-like 决策接口（可替换为规则 / MLP / LayA / Open-Jev）。"""

    def predict(
        self,
        *,
        balance: BalanceState,
        goal: Mapping[str, Any],
        phase: str,
        candidates: Sequence[PickupAction],
    ) -> PickupDecision:  # pragma: no cover - 协议
        """返回行为级决策。"""
        ...


@dataclass
class RuleBasedDecision:
    """规则决策：基于稳定性预测 + 目标位置 + 当前阶段（Baseline C 的决策核心）。"""

    reach_margin_m: float = 0.02
    lunge_margin_m: float = 0.0
    critical_margin_m: float = -0.05
    max_pitch_rad: float = 1.05
    use_com: bool = True
    use_zmp: bool = True
    use_capture_point: bool = True

    def predict(
        self,
        *,
        balance: BalanceState,
        goal: Mapping[str, Any],
        phase: str,
        candidates: Sequence[PickupAction],
    ) -> PickupDecision:
        """规则 + 预测 margin → 行为原语。"""
        margin = balance.stability_margin
        predicted = balance.predicted_margin
        cp = balance.capture_point
        com = balance.com_projection
        target = np.asarray(goal.get("object_position", com + np.array([0.35, 0.0])), dtype=np.float64)[:2]
        direction = target - com
        left_bias = float(direction[1])
        shift_saturated = bool(goal.get("com_shift_saturated", False))
        # 风险评估：裕度、预测裕度、躯干俯仰、CP 越界
        risk = float(
            np.clip(
                0.45 * max(0.0, (self.reach_margin_m - margin) / 0.12)
                + 0.35 * max(0.0, (self.reach_margin_m - predicted) / 0.15)
                + 0.20 * max(0.0, (abs(balance.trunk_pitch) - 0.6) / 0.6),
                0.0,
                1.0,
            )
        )
        action = PickupAction.STAND
        reason = "hold"
        confidence = 0.6
        if phase in ("GRASPING", "LIFTING", "STANDING_UP"):
            action = {
                "GRASPING": PickupAction.GRASP,
                "LIFTING": PickupAction.LIFT,
                "STANDING_UP": PickupAction.STAND_UP,
            }[phase]
            reason = f"phase_{phase.lower()}"
            confidence = 0.95
        elif balance.recovery_level >= 4 or abs(balance.trunk_pitch) > self.max_pitch_rad:
            action, reason, confidence = PickupAction.RECOVER, "unstable_recover", 0.9
        elif balance.recovery_level >= 3 or predicted < self.critical_margin_m:
            action, reason, confidence = PickupAction.LUNGE_LEFT if left_bias >= 0 else PickupAction.LUNGE_RIGHT, "predictive_lunge", 0.8
        elif balance.needs_step or predicted < self.lunge_margin_m or shift_saturated:
            if abs(left_bias) < 0.05:
                action = PickupAction.STEP_LEFT if left_bias >= 0 else PickupAction.STEP_RIGHT
                reason = "threshold_step"
            else:
                action = PickupAction.LUNGE_LEFT if left_bias >= 0 else PickupAction.LUNGE_RIGHT
                reason = "lateral_predictive_step"
            confidence = 0.75
        elif margin < self.reach_margin_m and phase in ("BENDING", "BALANCE_WARNING"):
            action, reason, confidence = PickupAction.BEND_SLOW, "margin_low_slow_bend", 0.8
        elif phase == "BENDING":
            action, reason, confidence = PickupAction.BEND, "bending", 0.85
        elif phase in ("REACHING",):
            action, reason, confidence = PickupAction.REACH, "reaching", 0.9
        elif phase in ("SUCCESS", "FAILURE"):
            action, reason, confidence = PickupAction.STAND, "terminal", 0.9
        if action not in candidates:
            action = candidates[0] if candidates else PickupAction.STAND
            reason = "candidate_filtered"
        probabilities = {action.value: float(confidence)}
        for candidate in candidates:
            probabilities.setdefault(candidate.value, float((1.0 - confidence) / max(1, len(candidates) - 1)))
        fallback = PickupAction.STOP if risk > 0.7 else (PickupAction.RECOVER if risk > 0.4 else PickupAction.STAND)
        return PickupDecision(
            action=action,
            confidence=float(confidence),
            risk=float(risk),
            stability_margin=float(min(margin, predicted)),
            fallback_action=fallback,
            probabilities=probabilities,
            reason=reason,
            features={
                "margin": float(margin),
                "predicted_margin": float(predicted),
                "com_x": float(com[0]),
                "com_y": float(com[1]),
                "cp_x": float(cp[0]),
                "cp_y": float(cp[1]),
                "trunk_pitch": float(balance.trunk_pitch),
                "recovery_level": float(balance.recovery_level),
                "object_dx": float(direction[0]),
                "object_dy": float(direction[1]),
            },
        )


@dataclass
class MLPDecision:
    """轻量 MLP 决策（监督学习自规则 rollout；可回退规则模型）。"""

    checkpoint_path: str = ""
    rule_fallback: RuleBasedDecision = field(default_factory=RuleBasedDecision)
    device: str = "cpu"
    actions: tuple[PickupAction, ...] = tuple(PickupAction)
    model: Any = field(default=None, init=False, repr=False)
    loaded: bool = field(default=False, init=False)

    def load(self) -> bool:
        """加载 torch checkpoint；失败保持规则回退。"""
        if self.loaded:
            return True
        import torch

        path = Path(self.checkpoint_path) if self.checkpoint_path else None
        if path is None or not path.is_file():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        feature_dim = int(payload["feature_dim"])
        hidden = tuple(payload.get("hidden", [128, 128]))
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(feature_dim, hidden[0]),
            nn.ELU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.Linear(hidden[1], len(self.actions)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model
        self.loaded = True
        return True

    @staticmethod
    def feature_vector(balance: BalanceState, goal: Mapping[str, Any], phase: str, phases: Sequence[str]) -> np.ndarray:
        """构造决策特征（balance 12 维 + 目标相对 + 阶段 one-hot）。"""
        object_position = np.asarray(goal.get("object_position", balance.com_projection), dtype=np.float64)[:2]
        delta = object_position - balance.com_projection
        one_hot = np.array([1.0 if phase == name else 0.0 for name in phases], dtype=np.float64)
        return np.concatenate([balance.to_features(), np.array([delta[0], delta[1], np.linalg.norm(delta)]), one_hot])

    def predict(
        self,
        *,
        balance: BalanceState,
        goal: Mapping[str, Any],
        phase: str,
        candidates: Sequence[PickupAction],
    ) -> PickupDecision:
        """MLP 推理；未加载/非法时回退规则。"""
        if not self.loaded and not self.load():
            out = self.rule_fallback.predict(balance=balance, goal=goal, phase=phase, candidates=candidates)
            out.reason = "mlp_unavailable_rule_fallback"
            return out
        import torch

        phases = ("IDLE", "BENDING", "BALANCE_WARNING", "FOOT_ADJUSTMENT", "LUNGE", "REACHING", "GRASPING", "LIFTING", "STANDING_UP", "RECOVERY", "SUCCESS", "FAILURE")
        features = self.feature_vector(balance, goal, phase, phases)
        with torch.no_grad():
            logits = self.model(torch.tensor(features[None, :], dtype=torch.float32, device=self.device))[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        order = np.argsort(-probs)
        action = PickupAction.STAND
        for index in order:
            candidate = self.actions[int(index)]
            if candidate in candidates:
                action = candidate
                break
        confidence = float(probs[self.actions.index(action)])
        rule = self.rule_fallback.predict(balance=balance, goal=goal, phase=phase, candidates=candidates)
        risk = rule.risk
        fallback = rule.fallback_action
        if confidence < 0.55 or risk > 0.7:
            return PickupDecision(
                action=rule.action if risk > 0.7 else action,
                confidence=confidence,
                risk=risk,
                stability_margin=rule.stability_margin,
                fallback_action=fallback,
                probabilities={self.actions[i].value: float(probs[i]) for i in range(len(self.actions))},
                reason="mlp_low_confidence_rule_veto" if risk > 0.7 else "mlp",
                features=rule.features,
            )
        return PickupDecision(
            action=action,
            confidence=confidence,
            risk=risk,
            stability_margin=rule.stability_margin,
            fallback_action=fallback,
            probabilities={self.actions[i].value: float(probs[i]) for i in range(len(self.actions))},
            reason="mlp",
            features=rule.features,
        )

    @staticmethod
    def save_dataset(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
        """保存监督学习数据集（JSONL）。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out

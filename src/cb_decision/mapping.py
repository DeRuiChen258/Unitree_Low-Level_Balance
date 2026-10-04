"""typed answers → DecisionOutput（非法值兜底 + fail-closed）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cb_common.types import DecisionOutput, RobotState, SafetyLevel
from cb_features.text_state import risk_score, safety_level_from_state


def map_answers(
    answers: Mapping[str, Any],
    *,
    state: RobotState,
    candidates: Sequence[str],
    confidence_threshold: float = 0.60,
    risk_threshold: float = 0.35,
    fallback_policy: Mapping[str, str] | None = None,
    rule_veto: bool = False,
    latency_ms: float = 0.0,
    model_name: str = "laya",
    checkpoint: str = "",
    feature_version: str = "1.0",
    state_hash: str = "",
) -> DecisionOutput:
    """把四个头的 typed answers 映射为结构化决策输出。"""
    fallback = dict(fallback_policy or {"low_confidence": "stand", "high_risk": "safe_stop", "model_timeout": "hold_current_skill"})
    heads: dict[str, Any] = {}
    reason = "ok"
    # motion 头
    motion = dict(answers.get("motion_primitive_head", {}))
    action = str(motion.get("choice", candidates[0] if candidates else "stand"))
    probabilities = {str(k): float(v) for k, v in dict(motion.get("probabilities", {})).items()}
    if candidates and action not in candidates:
        reason = "illegal_action_fallback"
        action = str(fallback["low_confidence"])
    confidence = float(motion.get("confidence", max(probabilities.values(), default=0.0)))
    heads["motion_primitive"] = {"choice": action, "confidence": confidence, "probabilities": probabilities}
    # veto 头（noul: p[1] 为 true）
    veto_answer = dict(answers.get("safety_veto_head", {}))
    model_veto = float(veto_answer.get("noul", 0.0)) >= 0.5
    rule_level = safety_level_from_state(state)
    veto = bool(model_veto or rule_veto or rule_level >= SafetyLevel.ABORT)
    heads["safety_veto"] = {"veto": veto, "model_veto": model_veto, "rule_level": rule_level.name}
    # recovery 头
    recovery = dict(answers.get("recovery_head", {}))
    recovery_action = str(recovery.get("choice", "none"))
    heads["recovery"] = {"choice": recovery_action, "confidence": float(recovery.get("confidence", 0.0))}
    # escalation 头
    escalation = dict(answers.get("escalation_head", {}))
    escalate = float(escalation.get("noul", 0.0)) >= 0.5
    heads["escalation"] = {"escalate": escalate, "probability": float(escalation.get("noul", 0.0))}
    uncertainty = 1.0 - confidence
    risk = risk_score(state, model_uncertainty=uncertainty)
    need_review = escalate or confidence < confidence_threshold or risk > risk_threshold
    fallback_action = "hold_current_skill"
    if veto or risk > risk_threshold:
        action = str(fallback["high_risk"])
        fallback_action = str(fallback["high_risk"])
        reason = "safety_veto" if veto else "high_risk"
    elif confidence < confidence_threshold:
        action = str(fallback["low_confidence"])
        fallback_action = str(fallback["low_confidence"])
        reason = "low_confidence"
    output = DecisionOutput(
        action_id=action,
        primitive=action,
        confidence=confidence,
        risk=risk,
        need_human_review=bool(need_review),
        fallback_action=fallback_action,
        probabilities=probabilities,
        reason_code=reason,
        heads=heads,
        meta={
            "model": model_name,
            "checkpoint": checkpoint,
            "feature_version": feature_version,
            "latency_ms": latency_ms,
            "state_hash": state_hash,
        },
    )
    output.validate()
    return output

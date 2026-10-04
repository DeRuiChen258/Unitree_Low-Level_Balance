"""决策映射单测：四头映射、非法值兜底、fail-closed。"""

from __future__ import annotations

from cb_decision.mapping import map_answers
from tests.helpers import make_state


def _answers(action: str = "run", veto: float = 0.0) -> dict:
    return {
        "motion_primitive_head": {
            "type": "choice",
            "choice": action,
            "probabilities": {action: 0.8, "stand": 0.2},
            "confidence": 0.8,
        },
        "safety_veto_head": {"type": "noul", "noul": veto},
        "recovery_head": {"type": "choice", "choice": "none", "probabilities": {"none": 0.9}, "confidence": 0.9},
        "escalation_head": {"type": "noul", "noul": 0.0},
    }


def test_mapping_basic_and_veto() -> None:
    """正常映射；veto 时强制 safe_stop。"""
    candidates = ["continue_current", "stand", "run", "recover", "safe_stop"]
    normal = map_answers(_answers("run"), state=make_state(), candidates=candidates)
    assert normal.action_id == "run"
    assert normal.fallback_action == "hold_current_skill"
    veto = map_answers(_answers("run", veto=1.0), state=make_state(), candidates=candidates)
    assert veto.action_id == "safe_stop"
    assert veto.reason_code == "safety_veto"


def test_mapping_illegal_and_low_confidence() -> None:
    """非法动作与低置信度走 fallback。"""
    candidates = ["continue_current", "stand", "run"]
    illegal = map_answers(_answers("fly"), state=make_state(), candidates=candidates)
    assert illegal.action_id in candidates
    low = _answers("run")
    low["motion_primitive_head"]["confidence"] = 0.1
    low["motion_primitive_head"]["probabilities"] = {"run": 0.1, "stand": 0.09}
    out = map_answers(low, state=make_state(), candidates=candidates)
    assert out.action_id == "stand"
    assert out.reason_code == "low_confidence"

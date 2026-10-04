"""LayA 文本化状态单测：稳定 JSON、字段精度、快照一致。"""

from __future__ import annotations

import numpy as np

from cb_common.types import RobotState, SkillCommand, TaskContext
from cb_features.text_state import TEXT_STATE_TEMPLATE_VERSION, build_text_state, parse_text_state, risk_score


def _state() -> RobotState:
    return RobotState(
        timestamp=1.0,
        base_pos=np.array([0.0, 0.0, 0.8]),
        base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        base_lin_vel=np.array([1.2, 0.1, 0.0]),
        base_ang_vel=np.array([0.0, 0.0, 0.2]),
        joint_pos=np.zeros(29),
        joint_vel=np.zeros(29),
        contact=(True, False),
        com=np.zeros(3),
        cp=np.zeros(2),
        support_margin=0.03,
    )


def test_text_state_stable_and_parseable() -> None:
    """相同输入产生逐字符一致的 JSON，并可解析。"""
    context = TaskContext(task="run_wave", candidates=["continue_current", "stand"])
    first = build_text_state(_state(), SkillCommand(skill="run"), context)
    second = build_text_state(_state(), SkillCommand(skill="run"), context)
    assert first == second
    payload = parse_text_state(first)
    assert payload["state_version"] == TEXT_STATE_TEMPLATE_VERSION
    assert payload["balance"]["contact"] == ["left"]
    assert payload["motion"]["vx"] == 1.2


def test_risk_score_bounds() -> None:
    """风险分在 [0,1]。"""
    value = risk_score(_state())
    assert 0.0 <= value <= 1.0

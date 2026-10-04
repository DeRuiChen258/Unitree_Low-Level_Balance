"""questions 单测：四头齐全、版本 hash、候选集合。"""

from __future__ import annotations

from cb_decision.questions import REQUIRED_HEADS, load_questions


def test_questions_schema_and_hash() -> None:
    """四头定义齐全且 hash 稳定。"""
    questions = load_questions("configs/laya/questions.json")
    assert set(REQUIRED_HEADS) <= set(questions.questions)
    assert questions.version == "1.0"
    assert len(questions.candidates()) >= 5
    assert questions.hash() == load_questions("configs/laya/questions.json").hash()
    agent_shape = questions.for_agent()
    assert agent_shape["safety_veto_head"]["type"] == "noul"

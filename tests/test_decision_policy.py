"""运行时决策策略单测：fail-closed、升级、调度映射。"""

from __future__ import annotations

from cb_common.errors import DecisionError
from cb_decision.policy import DecisionPolicy
from tests.helpers import make_state


class FakeAdapter:
    """可编程决策适配器替身。"""

    checkpoint = "fake"

    def __init__(self, failure: bool = False) -> None:
        self.failure = failure
        self.calls = 0

    def system_one(self, state, questions, timeout_ms=None):  # noqa: ANN001
        self.calls += 1
        if self.failure:
            raise DecisionError("LayA inference timeout")
        return {
            "answers": {
                "motion_primitive_head": {"type": "choice", "choice": "run", "probabilities": {"run": 0.9}, "confidence": 0.9},
                "safety_veto_head": {"type": "noul", "noul": 0.0},
                "recovery_head": {"type": "choice", "choice": "none", "probabilities": {"none": 0.9}, "confidence": 0.9},
                "escalation_head": {"type": "noul", "noul": 0.0},
            }
        }


def test_policy_normal_and_timeout() -> None:
    """正常决策可用；超时 fail-closed 且请求人工复核。"""
    from cb_common.types import TaskContext

    policy = DecisionPolicy(adapter=FakeAdapter())
    output = policy.decide(make_state(), TaskContext(task="run_wave"))
    assert output.action_id == "run"
    assert policy.to_skill_command(output).skill == "run"
    failing = DecisionPolicy(adapter=FakeAdapter(failure=True))
    fallback = failing.decide(make_state(), TaskContext(task="run_wave"))
    assert fallback.need_human_review
    assert fallback.reason_code in ("model_timeout", "model_error")
    assert fallback.action_id in ("hold_current_skill", "safe_stop")


def test_escalation_after_failures() -> None:
    """连续失败达到阈值触发升级。"""
    policy = DecisionPolicy(adapter=FakeAdapter())
    for _ in range(3):
        policy.note_failure()
    assert policy.escalation_needed()

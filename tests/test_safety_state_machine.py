"""状态机单测：合法/非法迁移、锁存、人工复位。"""

from __future__ import annotations

import pytest

from cb_common.errors import SafetyError
from cb_common.types import RobotMode
from cb_safety import SafeStateMachine


def test_state_machine_transitions() -> None:
    """合法迁移成功；非法迁移拒绝。"""
    machine = SafeStateMachine.from_config("configs/safety/state_machine.yaml")
    assert machine.handle("start") == RobotMode.STAND_UP
    assert machine.handle("stand_ready") == RobotMode.STAND
    with pytest.raises(SafetyError):
        machine.handle("recovered")


def test_state_machine_latch_and_reset() -> None:
    """SAFE_STOP 锁存，必须人工复位。"""
    machine = SafeStateMachine.from_config("configs/safety/state_machine.yaml")
    machine.handle("start")
    machine.handle("safety_veto")
    assert machine.mode == RobotMode.SAFE_STOP
    assert machine.latched
    with pytest.raises(SafetyError):
        machine.handle("start")
    machine.handle("manual_reset")
    assert not machine.latched

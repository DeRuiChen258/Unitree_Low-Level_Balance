"""E-Stop 单测：锁存、人工复位、非法复位。"""

from __future__ import annotations

import pytest

from cb_common.errors import SafetyError
from cb_common.types import SafetyLevel
from cb_safety import EStopChannel


def test_estop_trigger_and_reset() -> None:
    """触发锁存；带操作人复位。"""
    estop = EStopChannel(action="damp")
    event = estop.trigger("unit-test")
    assert event.level == SafetyLevel.EMERGENCY
    assert estop.active
    with pytest.raises(SafetyError):
        estop.reset("")
    estop.reset("operator-1")
    assert not estop.active
    assert estop.state.reset_by == "operator-1"


def test_estop_reset_without_trigger_illegal() -> None:
    """未触发时复位非法。"""
    with pytest.raises(SafetyError):
        EStopChannel().reset("operator")

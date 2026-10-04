"""看门狗单测：超时分级与去重。"""

from __future__ import annotations

from cb_common.types import SafetyLevel
from cb_safety.watchdog import Watchdog, WatchdogLimits


def test_watchdog_escalation() -> None:
    """WARN → ABORT → EMERGENCY 分级。"""
    watchdog = Watchdog(WatchdogLimits(0.2, 0.2, 0.05))
    watchdog.kick("state", now=0.0)
    assert not watchdog.check(now=0.1)
    warn = watchdog.check(now=0.25)
    assert warn and warn[0].level == SafetyLevel.WARN
    abort = watchdog.check(now=0.45)
    assert abort and abort[0].level == SafetyLevel.ABORT
    emergency = watchdog.check(now=0.7)
    assert emergency and emergency[0].level == SafetyLevel.EMERGENCY
    assert not watchdog.check(now=0.8)


def test_watchdog_kick_resets() -> None:
    """喂狗后不再触发。"""
    watchdog = Watchdog(WatchdogLimits(0.2, 0.2, 0.05))
    watchdog.kick("loop", now=0.0)
    assert not watchdog.check(now=0.04)
    watchdog.kick("loop", now=0.05)
    assert not watchdog.check(now=0.08)

"""SafetyWrapper：所有运动命令必须经过的唯一出口（fail-closed）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cb_common.errors import SafetyError
from cb_common.logging import JsonlLogger
from cb_common.types import JointCommand, RobotState, SafetyDecision, SafetyLevel

from .e_stop import EStopChannel
from .limits import CommandConditioner, SafetyLimits, check_safety
from .watchdog import Watchdog


@dataclass
class SafetyWrapper:
    """把条件器、限值判定、看门狗与 E-Stop 组合成唯一安全通道。"""

    limits: SafetyLimits
    dt: float
    estop: EStopChannel | None = None
    watchdog: Watchdog | None = None
    logger: JsonlLogger | None = None
    conditioner: CommandConditioner = field(init=False)
    last_decision: SafetyDecision | None = field(default=None, init=False)
    blocked_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.conditioner = CommandConditioner(self.limits, self.dt)

    def filter(self, cmd: JointCommand, state: RobotState, *, now: float | None = None) -> SafetyDecision:
        """安全检查 + 条件化；被拒绝时返回原命令但不放行。"""
        if self.estop is not None and self.estop.active:
            event = self.estop.as_event_if_active()
            decision = SafetyDecision(False, SafetyLevel.EMERGENCY, [event] if event else [], None, "estop_latched")
            self.last_decision = decision
            self.blocked_count += 1
            self._log(decision, cmd, state, now)
            return decision
        if self.watchdog is not None:
            events = self.watchdog.check(now)
            if events:
                level = max(e.level for e in events)
                if level >= SafetyLevel.ABORT:
                    decision = SafetyDecision(False, level, events, None, events[-1].reason)
                    self.last_decision = decision
                    self.blocked_count += 1
                    self._log(decision, cmd, state, now)
                    return decision
        try:
            conditioned = self.conditioner.condition(cmd.q_target, current_q=state.joint_pos)
        except SafetyError as exc:
            event_level = SafetyLevel.EMERGENCY
            from cb_common.types import SafetyEvent

            decision = SafetyDecision(
                False,
                event_level,
                [SafetyEvent("command_check", event_level, exc.message, float(now or state.timestamp))],
                None,
                exc.message,
            )
            self.last_decision = decision
            self.blocked_count += 1
            self._log(decision, cmd, state, now)
            return decision
        safe_cmd = cmd.copy_with(q_target=conditioned)
        decision = check_safety(safe_cmd, state, self.limits, timestamp=now)
        decision.command = safe_cmd if decision.allowed else None
        if not decision.allowed:
            self.blocked_count += 1
        self.last_decision = decision
        self._log(decision, cmd, state, now)
        return decision

    def _log(self, decision: SafetyDecision, cmd: JointCommand, state: RobotState, now: float | None) -> None:
        if self.logger is None:
            return
        for event in decision.events:
            self.logger.log(
                "safety_event",
                level=event.level.name,
                reason=event.reason,
                joint=event.joint,
                value=event.value,
                threshold=event.threshold,
                unit=event.unit,
                action=event.action,
                skill=cmd.skill,
                source=cmd.source,
                sim_time=float(now if now is not None else state.timestamp),
            )

    def reset(self, q: Any) -> None:
        """人工复位后重置条件器状态。"""
        self.conditioner.reset(q)
        self.last_decision = None

"""E-Stop 独立通道：锁存 + 人工复位，不依赖主控制循环。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cb_common.errors import SafetyError
from cb_common.types import SafetyEvent, SafetyLevel


@dataclass
class EStopState:
    """急停状态。"""

    latched: bool = False
    action: str = "damp"
    triggered_at: float | None = None
    trigger_source: str = ""
    reset_by: str = ""
    reset_at: float | None = None


@dataclass
class EStopChannel:
    """软件急停通道（可被硬件急停直接调用，独立于控制循环）。"""

    action: str = "damp"
    latching: bool = True
    requires_manual_reset: bool = True
    _state: EStopState = field(default_factory=EStopState, init=False)

    def __post_init__(self) -> None:
        if self.action not in ("damp", "hold", "zero_torque"):
            raise SafetyError("invalid estop action", action=self.action)
        self._state.action = self.action

    @property
    def state(self) -> EStopState:
        """返回当前急停状态（只读语义）。"""
        return self._state

    @property
    def active(self) -> bool:
        """是否处于急停锁存。"""
        return self._state.latched

    def trigger(self, source: str = "software", now: float | None = None) -> SafetyEvent:
        """触发急停并锁存；返回 EMERGENCY 事件。"""
        if not self._state.latched:
            self._state.latched = True
            self._state.triggered_at = float(time.time() if now is None else now)
            self._state.trigger_source = source
            if not self.latching:
                pass  # 保持字段语义：非锁存模式由调用方在 trigger 后立即 reset
        return SafetyEvent(
            "estop",
            SafetyLevel.EMERGENCY,
            "emergency_stop",
            float(time.time() if now is None else now),
            action=self.action,
        )

    def reset(self, operator: str, now: float | None = None) -> None:
        """人工复位；未锁存时调用是非法操作。"""
        if not self._state.latched:
            raise SafetyError("estop is not latched; reset is illegal")
        if not operator:
            raise SafetyError("manual reset requires operator id")
        self._state.latched = False
        self._state.reset_by = operator
        self._state.reset_at = float(time.time() if now is None else now)

    def as_event_if_active(self) -> SafetyEvent | None:
        """活动状态下返回持续事件，供每帧日志使用。"""
        if not self._state.latched:
            return None
        return SafetyEvent(
            "estop",
            SafetyLevel.EMERGENCY,
            "emergency_stop_latched",
            float(self._state.triggered_at or time.time()),
            action=self.action,
        )

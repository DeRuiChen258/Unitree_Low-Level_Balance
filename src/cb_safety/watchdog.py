"""状态/命令/控制回路看门狗（0.2 / 0.2 / 0.05 s，超时分级升级）。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from cb_common.errors import SafetyError
from cb_common.types import SafetyEvent, SafetyLevel


@dataclass(frozen=True)
class WatchdogLimits:
    """看门狗超时阈值（秒）。"""

    state_timeout_s: float
    command_timeout_s: float
    loop_timeout_s: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> WatchdogLimits:
        """从配置映射构造并校验。"""
        limits = cls(
            state_timeout_s=float(mapping["state_timeout_s"]),
            command_timeout_s=float(mapping["command_timeout_s"]),
            loop_timeout_s=float(mapping["loop_timeout_s"]),
        )
        for name in ("state_timeout_s", "command_timeout_s", "loop_timeout_s"):
            if getattr(limits, name) <= 0.0:
                raise SafetyError(f"watchdog.{name} must be > 0")
        return limits


@dataclass
class Watchdog:
    """三通道看门狗；未按时喂狗 → WARN/ABORT/EMERGENCY 分级事件。"""

    limits: WatchdogLimits
    _last: dict[str, float] = field(default_factory=dict, init=False)
    _fired: dict[str, SafetyLevel] = field(default_factory=dict, init=False)
    clock: object = time.monotonic

    def kick(self, channel: str, now: float | None = None) -> None:
        """喂狗（状态/命令/回路任一通道）。"""
        if channel not in ("state", "command", "loop"):
            raise SafetyError("unknown watchdog channel", channel=channel)
        self._last[channel] = float(self.clock() if now is None else now)
        self._fired.pop(channel, None)

    def elapsed(self, channel: str, now: float | None = None) -> float:
        """返回通道距上次喂狗的时长。"""
        if channel not in self._last:
            return 0.0
        return max(0.0, float(self.clock() if now is None else now) - self._last[channel])

    def check(self, now: float | None = None) -> list[SafetyEvent]:
        """检查所有通道；返回新升级的事件（去重）。"""
        t = float(self.clock() if now is None else now)
        events: list[SafetyEvent] = []
        for channel, timeout in (
            ("state", self.limits.state_timeout_s),
            ("command", self.limits.command_timeout_s),
            ("loop", self.limits.loop_timeout_s),
        ):
            if channel not in self._last:
                continue
            elapsed = self.elapsed(channel, t)
            if elapsed < timeout:
                continue
            if channel == "loop" and elapsed >= 1.5 * timeout or elapsed >= 3.0 * timeout:
                level = SafetyLevel.EMERGENCY
            elif elapsed >= 2.0 * timeout:
                level = SafetyLevel.ABORT
            else:
                level = SafetyLevel.WARN
            if self._fired.get(channel, SafetyLevel.OK) >= level:
                continue
            self._fired[channel] = level
            events.append(
                SafetyEvent(
                    "watchdog",
                    level,
                    f"{channel}_watchdog_timeout",
                    t,
                    value=elapsed,
                    threshold=timeout,
                    unit="s",
                    action="hold" if level < SafetyLevel.EMERGENCY else "safe_stop",
                )
            )
        return events

"""安全状态机：合法迁移表 + 锁存 + 人工复位（第 10.2 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cb_common.config import load_config
from cb_common.errors import SafetyError
from cb_common.types import RobotMode, SafetyEvent


@dataclass
class SafeStateMachine:
    """运行时状态机；非法迁移拒绝并抛 SafetyError。"""

    transitions: Mapping[tuple[RobotMode, str], RobotMode]
    initial: RobotMode = RobotMode.IDLE
    latching: frozenset[RobotMode] = frozenset({RobotMode.SAFE_STOP, RobotMode.ERROR})
    mode: RobotMode = field(init=False)
    history: list[tuple[float, RobotMode, str]] = field(default_factory=list, init=False)
    _latched: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.mode = self.initial
        self.history.append((0.0, self.mode, "init"))

    @classmethod
    def from_config(cls, path: str | Path) -> SafeStateMachine:
        """从 `configs/safety/state_machine.yaml` 构造。"""
        cfg = load_config(path, required=["initial", "states", "transitions", "latching"])
        transitions: dict[tuple[RobotMode, str], RobotMode] = {}
        for item in cfg.get("transitions"):
            src, dst, event = RobotMode(item["from"]), RobotMode(item["to"]), str(item["event"])
            if (src, event) in transitions:
                raise SafetyError("duplicate transition", src=src.value, event=event)
            transitions[(src, event)] = dst
        initial = RobotMode(cfg.get("initial"))
        latching = frozenset(RobotMode(s) for s in cfg.get("latching", []))
        return cls(transitions=transitions, initial=initial, latching=latching)

    @property
    def latched(self) -> bool:
        """是否处于锁存态。"""
        return self._latched

    def handle(self, event: SafetyEvent | str, *, timestamp: float = 0.0) -> RobotMode:
        """处理事件并迁移；返回新状态。"""
        name = event.kind if isinstance(event, SafetyEvent) else str(event)
        if self._latched and name != "manual_reset":
            raise SafetyError("state machine latched; manual_reset required", mode=self.mode.value, event=name)
        key = (self.mode, name)
        if key not in self.transitions:
            raise SafetyError("illegal transition", mode=self.mode.value, event=name)
        target = self.transitions[key]
        self.mode = target
        self.history.append((timestamp, target, name))
        if target in self.latching:
            self._latched = True
        if name == "manual_reset":
            self._latched = False
        return target

    def allowed_events(self) -> list[str]:
        """返回当前状态可触发的事件列表。"""
        if self._latched:
            return ["manual_reset"]
        return sorted(event for (mode, event) in self.transitions if mode == self.mode)

"""技能调度器：技能图 + FSM + 相位时钟 + 仲裁 + 限流/抖动抑制（第 7.3 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cb_common.types import RobotState, SafetyLevel, SkillCommand


@dataclass
class SchedulerLogEntry:
    """一次切换记录（可回放）。"""

    t: float
    from_skill: str
    to_skill: str
    reason_code: str
    source: str
    allowed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillScheduler:
    """规则调度（Tier 1）：安全 > 恢复 > 任务 > 表现力。"""

    skills_cfg: Mapping[str, Any]
    transitions_cfg: Mapping[str, Any]
    min_hold_time: Mapping[str, float] = field(default_factory=dict)
    hyster: Mapping[str, float] = field(default_factory=dict)
    current: str = "stand"
    next_skill: str | None = None
    transition_until: float = 0.0
    last_switch_time: float = -1e9
    history: list[SchedulerLogEntry] = field(default_factory=list)
    _failures: int = field(default=0, init=False)

    @classmethod
    def from_configs(cls, skills_path: str | Path, transitions_path: str | Path) -> SkillScheduler:
        """从 YAML 配置构造。"""
        skills = yaml.safe_load(Path(skills_path).read_text(encoding="utf-8"))
        transitions = yaml.safe_load(Path(transitions_path).read_text(encoding="utf-8"))
        min_hold = {"default": 0.4, **dict(transitions.get("min_hold_time_s", {}))}
        return cls(skills_cfg=skills, transitions_cfg=transitions, min_hold_time=min_hold, hyster=dict(transitions.get("hysteresis", {})))

    @property
    def skill_priority(self) -> dict[str, int]:
        """技能优先级表。"""
        return {item["name"]: int(item["priority"]) for item in self.skills_cfg.get("skills", [])}

    def _allowed(self, from_skill: str, to_skill: str) -> bool:
        """检查迁移是否在 transitions.yaml 白名单中（any 通配）。"""
        for item in self.transitions_cfg.get("transitions", []):
            if (item["from"] in (from_skill, "any")) and item["to"] == to_skill:
                return True
        return from_skill == to_skill

    def request(self, command: SkillCommand, state: RobotState, *, safety: SafetyLevel = SafetyLevel.OK) -> SkillCommand:
        """请求技能切换；返回仲裁后的 SkillCommand（可能保持当前技能）。"""
        now = float(state.timestamp)
        desired = command.skill
        reason = command.reason_code
        # Tier 0/1 仲裁：安全最高，其次恢复，再任务
        if safety >= SafetyLevel.ABORT:
            desired, reason = "safe_stop", f"safety_{safety.name.lower()}"
        elif safety >= SafetyLevel.WARN and self.current not in ("recover", "safe_stop"):
            desired, reason = "recover", "balance_degraded"
        if desired == self.current:
            return SkillCommand(
                skill=self.current,
                params=command.params,
                blend_weight=1.0,
                reason_code="hold",
                source="scheduler",
                timestamp=now,
            )
        if not self._allowed(self.current, desired):
            self.history.append(SchedulerLogEntry(now, self.current, desired, "illegal_transition", command.source, False))
            return SkillCommand(self.current, command.params, 1.0, "illegal_transition_hold", "scheduler", now)
        hold = float(self.min_hold_time.get(desired, self.min_hold_time.get("default", 0.0)))
        if now - self.last_switch_time < hold and self.skill_priority.get(desired, 0) <= self.skill_priority.get(self.current, 0):
            self.history.append(SchedulerLogEntry(now, self.current, desired, "min_hold_active", command.source, False))
            return SkillCommand(self.current, command.params, 1.0, "min_hold", "scheduler", now)
        window = 0.0
        for item in self.transitions_cfg.get("transitions", []):
            if (item["from"] in (self.current, "any")) and item["to"] == desired:
                window = float(item.get("blend_window_s", 0.0))
                break
        self.history.append(SchedulerLogEntry(now, self.current, desired, reason, command.source, True))
        self.last_switch_time = now
        self.transition_until = now + window
        previous, self.current, self.next_skill = self.current, desired, desired
        return SkillCommand(
            skill=desired,
            params=command.params,
            blend_weight=1.0,
            reason_code=reason,
            source=command.source,
            timestamp=now,
            transition_window_s=window,
            from_skill=previous,
        )

    def note_failure(self) -> None:
        """记录技能失败（用于升级/回退链）。"""
        self._failures += 1

    @property
    def consecutive_failures(self) -> int:
        """连续失败次数。"""
        return self._failures

    def reset(self, skill: str = "stand") -> None:
        """复位调度器状态。"""
        self.current = skill
        self.next_skill = None
        self.transition_until = 0.0
        self.last_switch_time = -1e9
        self._failures = 0
        self.history.clear()

    def export_log(self) -> list[dict[str, Any]]:
        """导出切换日志（JSON 友好）。"""
        return [entry.__dict__ for entry in self.history]

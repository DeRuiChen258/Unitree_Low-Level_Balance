"""技能协议与 8 字段定义（第 7.1 节）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cb_common.errors import PolicyError
from cb_common.types import RobotState, SkillContext, SkillOutput


@dataclass(frozen=True)
class SkillSpec:
    """技能静态定义（来自 configs/skills.yaml）。"""

    name: str
    id: int
    priority: int
    interruptible: bool
    enter_condition: str
    exit_condition: str
    params: Mapping[str, Any]
    observation: tuple[str, ...]
    action_source: str
    termination: tuple[str, ...]
    safety: Mapping[str, Any]


class Skill(ABC):
    """技能基类：enter / update / exit 三态。"""

    spec: SkillSpec

    @abstractmethod
    def enter(self, ctx: SkillContext) -> None:
        """进入技能。"""

    @abstractmethod
    def update(self, state: RobotState, dt: float) -> SkillOutput:
        """推进技能并返回输出。"""

    @abstractmethod
    def exit(self) -> None:
        """退出技能。"""

    def done(self) -> bool:
        """是否已完成（默认 False，由子类覆盖）。"""
        return False


def load_skill_specs(path: str | Path) -> dict[str, SkillSpec]:
    """加载技能表并校验 8 个字段。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs: dict[str, SkillSpec] = {}
    for item in data.get("skills", []):
        name = str(item["name"])
        specs[name] = SkillSpec(
            name=name,
            id=int(item["id"]),
            priority=int(item["priority"]),
            interruptible=bool(item["interruptible"]),
            enter_condition=str(item["enter_condition"]),
            exit_condition=str(item["exit_condition"]),
            params=dict(item.get("params", {})),
            observation=tuple(item.get("observation", [])),
            action_source=str(item["action_source"]),
            termination=tuple(item.get("termination", [])),
            safety=dict(item.get("safety", {})),
        )
    if not specs:
        raise PolicyError("skills.yaml contains no skills", path=str(path))
    ids = [spec.id for spec in specs.values()]
    if len(set(ids)) != len(ids):
        raise PolicyError("duplicate skill ids", ids=ids)
    return specs

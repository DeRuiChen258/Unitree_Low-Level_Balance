"""六阶段课程：通过标准与回退（第 6.3 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumStage:
    """单个课程阶段。"""

    name: str
    min_episodes: int
    metrics: dict[str, float]
    allowed_skills: list[str]


@dataclass
class Curriculum:
    """课程调度器；按阶段指标前进/回退。"""

    stages: list[CurriculumStage]
    index: int = 0
    episodes: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    regression_fall_rate: float = 0.2

    @classmethod
    def from_config(cls, reward_cfg: Mapping[str, Any]) -> Curriculum:
        """从 reward.yaml 读取课程阶段。"""
        section = reward_cfg.get("curriculum", {})
        stages = [
            CurriculumStage(
                name=str(item["name"]),
                min_episodes=int(item.get("min_episodes", 0)),
                metrics={k: float(v) for k, v in dict(item.get("metrics", {})).items()},
                allowed_skills=list(item.get("allowed_skills", [])),
            )
            for item in section.get("stages", [])
        ]
        if not stages:
            raise ValueError("curriculum requires at least one stage")
        return cls(
            stages=stages,
            regression_fall_rate=float(section.get("regression_fall_rate", 0.2)),
        )

    @property
    def stage(self) -> CurriculumStage:
        """当前阶段。"""
        return self.stages[min(self.index, len(self.stages) - 1)]

    def update(self, metrics: Mapping[str, float]) -> bool:
        """更新并判断是否切换阶段；返回是否切换。"""
        self.episodes += 1
        self.history.append({"stage": self.stage.name, "episodes": self.episodes, **dict(metrics)})
        if self.episodes < self.stage.min_episodes:
            return False
        ok = all(metrics.get(key, float("inf")) <= value for key, value in self.stage.metrics.items())
        if ok and self.index < len(self.stages) - 1:
            self.index += 1
            self.episodes = 0
            return True
        if not ok and metrics.get("fall_rate", 0.0) > self.regression_fall_rate and self.index > 0:
            self.index -= 1
            self.episodes = 0
            return True
        return False

"""加载 SkillSegment/TransitionSegment，按课程采样参考动作。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cb_common.errors import DataError
from cb_data.pipeline import load_segments
from cb_data.schema import SkillSegment


@dataclass
class MotionSampler:
    """按技能分桶的参考动作采样器（可复现）。"""

    segments: list[SkillSegment]
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not self.segments:
            raise DataError("MotionSampler requires non-empty segments")
        self.rng = np.random.default_rng(self.seed)

    def by_skill(self, skill: str) -> list[SkillSegment]:
        """返回某技能的全部片段。"""
        items = [segment for segment in self.segments if segment.skill == skill]
        if not items:
            raise DataError("no segments for skill", skill=skill)
        return items

    def sample(self, skill: str) -> SkillSegment:
        """随机采样一个片段。"""
        items = self.by_skill(skill)
        return items[int(self.rng.integers(0, len(items)))]

    def distribution(self) -> dict[str, int]:
        """技能分布（训练日志用）。"""
        counts: dict[str, int] = {}
        for segment in self.segments:
            counts[segment.skill] = counts.get(segment.skill, 0) + 1
        return counts


def load_motion_sampler(
    manifest_path: str | Path,
    *,
    split: str = "train",
    seed: int = 0,
    root: str | Path | None = None,
) -> MotionSampler:
    """从 manifest 加载 split 并构造采样器。"""
    segments = load_segments(manifest_path, split, root=root)
    return MotionSampler(segments=segments, seed=seed)


def iter_reference_frames(segment: SkillSegment) -> Iterator[np.ndarray]:
    """逐帧参考关节角（用于 AMP/蒸馏）。"""
    yield from segment.qpos

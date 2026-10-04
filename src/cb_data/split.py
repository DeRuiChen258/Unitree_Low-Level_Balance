"""S7 切分：按源序列聚簇，禁止同源泄漏。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from cb_common.errors import DataError

from .schema import SkillSegment


def split_segments(
    segments: Iterable[SkillSegment],
    *,
    ratios: Mapping[str, float],
    seed: int = 1,
) -> dict[str, list[SkillSegment]]:
    """按 source_clip 聚簇切分；同一源的所有切片只落一个 split。"""
    items = list(segments)
    if not items:
        raise DataError("no segments to split")
    total = sum(float(v) for v in ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise DataError("split ratios must sum to 1", ratios=dict(ratios))
    groups: dict[str, list[SkillSegment]] = {}
    for segment in items:
        groups.setdefault(segment.source_clip or segment.segment_id, []).append(segment)
    keys = sorted(groups)
    order = sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    n_train = int(round(len(order) * float(ratios.get("train", 0.8))))
    n_val = int(round(len(order) * float(ratios.get("val", 0.1))))
    if len(order) >= 3:
        # 小数据集也保证 train/val/test 各至少一个源序列
        n_train = min(max(n_train, 1), len(order) - 2)
        n_val = min(max(n_val, 1), len(order) - n_train - 1)
    if len(order) > 1:
        n_train = min(max(n_train, 1), len(order) - 1)
        n_val = min(n_val, len(order) - n_train)
    assignment = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    out: dict[str, list[SkillSegment]] = {"train": [], "val": [], "test": []}
    for split, group_keys in assignment.items():
        for key in group_keys:
            out[split].extend(groups[key])
    _assert_no_leakage(out)
    return out


def _assert_no_leakage(splits: Mapping[str, list[SkillSegment]]) -> None:
    """断言同源不跨 split。"""
    seen: dict[str, str] = {}
    for split, segments in splits.items():
        for segment in segments:
            source = segment.source_clip or segment.segment_id
            if source in seen and seen[source] != split:
                raise DataError("source leakage across splits", source=source, first=seen[source], second=split)
            seen[source] = split

"""S7 切分单测：无同源泄漏、比例合理、空输入拒绝。"""

from __future__ import annotations

import pytest

from cb_common.errors import DataError
from cb_data.split import split_segments
from tests.helpers import make_segment


def test_split_no_source_leakage() -> None:
    """同一 source_clip 不跨 split。"""
    segments = []
    for source in range(10):
        for index in range(2):
            segment = make_segment(frames=20)
            segment.segment_id = f"s{source}-{index}"
            segment.source_clip = f"clip{source}"
            segments.append(segment)
    splits = split_segments(segments, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=3)
    seen: dict[str, str] = {}
    for name, items in splits.items():
        for segment in items:
            assert segment.source_clip not in seen or seen[segment.source_clip] == name
            seen[segment.source_clip] = name
    assert all(splits[name] for name in ("train", "val", "test"))


def test_split_rejects_empty() -> None:
    """空输入必须拒绝。"""
    with pytest.raises(DataError):
        split_segments([], ratios={"train": 0.8, "val": 0.1, "test": 0.1})

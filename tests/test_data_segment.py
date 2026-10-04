"""S4 segment 单测：技能切片、相位、指令生成。"""

from __future__ import annotations

from cb_data.segment import SegmentConfig, segment_clip
from tests.helpers import make_retargeted


def test_segment_produces_valid_skill_segments() -> None:
    """切片结果字段合法、覆盖完整。"""
    clip = make_retargeted(frames=90)
    config = SegmentConfig(
        min_segment_frames=10,
        boundary_smoothing=3,
        keyword_rules={"walk": ["walk"], "run": ["run"]},
        velocity_thresholds={"stand_max_mps": 0.15, "walk_min_mps": 0.15, "run_min_mps": 1.4, "turn_min_rad_s": 0.6},
    )
    segments = segment_clip(clip, config)
    assert segments
    total = sum(segment.frames for segment in segments)
    assert total == 90
    for segment in segments:
        assert segment.skill in ("stand", "walk", "run", "jump", "wave", "turn", "recover")
        assert segment.phase.shape == (segment.frames,)
        assert "ik_err_mean_m" in segment.quality

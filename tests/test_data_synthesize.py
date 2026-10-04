"""S5 增强单测：镜像、速度、幅度、相位、组合过渡。"""

from __future__ import annotations

import numpy as np

from cb_data.synthesize import augment_segment, mirror_segment, tempo_segment
from cb_data.synthesize_transition import blend_segments
from tests.helpers import make_segment


def test_augment_is_reproducible_and_bounded() -> None:
    """同 seed 复现；增强参数落在配置范围。"""
    segment = make_segment(frames=40, skill="run")
    config = {
        "mirror": True,
        "tempo": [0.9, 1.1],
        "amplitude": [0.95, 1.05],
        "phase_jitter": 0.05,
        "root_xy_noise_m": [0.01, 0.03],
        "augmentation_copies": 3,
    }
    first = augment_segment(segment, config, np.random.default_rng(1))
    second = augment_segment(segment, config, np.random.default_rng(1))
    assert len(first) == 3
    assert first[2].meta["augmentations"] == second[2].meta["augmentations"]
    assert np.allclose(first[2].root_pos, second[2].root_pos)


def test_mirror_swaps_left_right() -> None:
    """镜像交换左右腿关节。"""
    segment = make_segment(frames=20)
    segment.qpos[:, 0] = 0.3
    segment.qpos[:, 1] = -0.1
    mirrored = mirror_segment(segment)
    assert np.allclose(mirrored.qpos[:, 0], -0.1)
    assert np.allclose(mirrored.qpos[:, 1], 0.3)


def test_tempo_and_transition() -> None:
    """时间缩放改变帧数；过渡混合有接触仲裁与 jerk 记录。"""
    segment = make_segment(frames=60, skill="run")
    faster = tempo_segment(segment, 1.5)
    assert faster.frames < segment.frames
    other = make_segment(frames=60, skill="jump")
    transition = blend_segments(segment, other, blend_window_s=0.3, phase_alignment="takeoff_window")
    assert transition.frames == int(0.3 * 30)
    assert "contact_conflict_frames" in transition.quality

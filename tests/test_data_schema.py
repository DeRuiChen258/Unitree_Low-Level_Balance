"""数据 schema 单测：字段冻结、序列化往返、非法输入拒绝。"""

from __future__ import annotations

import numpy as np
import pytest

from cb_common.errors import DataError
from cb_data.schema import DatasetManifest, MotionClip, SkillSegment, TransitionSegment
from tests.helpers import make_motion


def test_motion_roundtrip() -> None:
    """数组往返保持形状与数值。"""
    clip = make_motion()
    arrays = clip.to_arrays()
    restored = MotionClip.from_arrays("test:0", "test", arrays, "standing", "test")
    assert restored.root_pos.shape == (8, 3)
    assert np.allclose(restored.joint_world_pos, clip.joint_world_pos)
    assert restored.summary()["frames"] == 8


def test_motion_rejects_nan() -> None:
    """NaN 输入必须拒绝。"""
    clip = make_motion()
    clip.root_pos[0, 0] = np.nan
    with pytest.raises(DataError):
        MotionClip(
            clip_id=clip.clip_id,
            source=clip.source,
            fps=clip.fps,
            frames=clip.frames,
            root_pos=clip.root_pos,
            root_quat=clip.root_quat,
            joint_pos=clip.joint_pos,
            joint_world_pos=clip.joint_world_pos,
            joint_vel=clip.joint_vel,
            contacts=clip.contacts,
            text=clip.text,
            license=clip.license,
        )


def test_skill_and_transition_validation() -> None:
    """技能/过渡数据结构校验。"""
    segment = SkillSegment(
        segment_id="s0",
        skill="stand",
        t_start=0.0,
        t_end=0.2,
        phase=np.linspace(0, 1, 8, endpoint=False),
        command={},
        quality={},
        qpos=np.zeros((8, 29)),
        root_pos=np.zeros((8, 3)),
        root_quat=np.tile(np.array([1.0, 0, 0, 0]), (8, 1)),
        contacts=np.ones((8, 2), dtype=bool),
        fps=30.0,
    )
    assert segment.frames == 8
    transition = TransitionSegment(
        transition_id="t0",
        from_skill="stand",
        to_skill="walk",
        blend_window=0.3,
        phase_alignment="double_support",
        qpos=np.zeros((8, 29)),
        root_pos=np.zeros((8, 3)),
        contacts=np.ones((8, 2), dtype=bool),
        from_segment="s0",
        to_segment="s1",
    )
    assert transition.summary()["frames"] == 8


def test_manifest_version_gate() -> None:
    """manifest 版本不匹配必须拒绝。"""
    manifest = DatasetManifest.create(
        dataset_version="v1",
        tool_version="t",
        files=[],
        stats={},
        splits={"train": [], "val": [], "test": []},
        aug_profile="a",
        profile="smoke",
    )
    manifest.require_version("v1")
    with pytest.raises(DataError):
        manifest.require_version("v2")

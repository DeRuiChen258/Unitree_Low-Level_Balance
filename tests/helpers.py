"""测试共享构造器（合成数据，不依赖真实数据集）。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import DEFAULT_JOINT_POS_POLICY
from cb_common.types import RobotState
from cb_data.schema import MotionClip, RetargetedClip, SkillSegment


def make_state(*, roll: float = 0.0, pitch: float = 0.0, height: float = 0.79, vx: float = 0.0, yaw_rate: float = 0.0,
               contact: tuple[bool, bool] = (True, True), margin: float = 0.05, joint_pos=None) -> RobotState:
    """构造最小合法 RobotState。"""
    half = roll / 2.0
    half_p = pitch / 2.0
    quat = np.array(
        [
            np.cos(half) * np.cos(half_p),
            np.sin(half) * np.cos(half_p),
            np.cos(half) * np.sin(half_p),
            0.0,
        ]
    )
    q = np.array(DEFAULT_JOINT_POS_POLICY) if joint_pos is None else np.asarray(joint_pos, dtype=np.float64)
    return RobotState(
        timestamp=0.0,
        base_pos=np.array([0.0, 0.0, height]),
        base_quat=quat,
        base_lin_vel=np.array([vx, 0.0, 0.0]),
        base_ang_vel=np.array([0.0, 0.0, yaw_rate]),
        joint_pos=q,
        joint_vel=np.zeros(29),
        contact=contact,
        com=np.array([0.0, 0.0, height]),
        cp=np.zeros(2),
        support_margin=margin,
    )


def make_motion(frames: int = 8, fps: float = 30.0) -> MotionClip:
    """构造最小合法 MotionClip（站立，双脚接触）。"""
    positions = np.zeros((frames, 22, 3))
    positions[:, :, 1] = 0.9
    positions[:, 10, 1] = 0.05
    positions[:, 11, 1] = 0.05
    return MotionClip(
        clip_id="test:0",
        source="test",
        fps=fps,
        frames=frames,
        root_pos=np.tile(np.array([0.0, 0.0, 0.9]), (frames, 1)),
        root_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        joint_pos=np.zeros((frames, 66)),
        joint_world_pos=positions,
        joint_vel=np.zeros((frames, 66)),
        contacts=np.ones((frames, 2), dtype=bool),
        text="standing",
        license="test",
    )


def make_retargeted(frames: int = 8, fps: float = 30.0) -> RetargetedClip:
    """构造最小合法 RetargetedClip。"""
    return RetargetedClip(
        retarget_id="g1:test",
        robot="unitree_g1_29dof",
        joint_names=tuple(f"j{i}" for i in range(29)),
        qpos=np.tile(np.array(DEFAULT_JOINT_POS_POLICY), (frames, 1)),
        qvel=np.zeros((frames, 29)),
        keypoint_pos=np.zeros((frames, 3, 3)),
        retarget_err=np.full(frames, 0.03),
        source_clip="test:0",
        root_pos=np.tile(np.array([0.0, 0.0, 0.79]), (frames, 1)),
        root_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        contacts=np.ones((frames, 2), dtype=bool),
        joint_limit_hits=np.zeros(frames, dtype=np.int32),
        fps=fps,
        meta={"text": "a person walks forward"},
    )


def make_segment(frames: int = 40, skill: str = "walk", fps: float = 30.0) -> SkillSegment:
    """构造最小合法 SkillSegment。"""
    default = np.array(DEFAULT_JOINT_POS_POLICY)
    qpos = np.tile(default, (frames, 1))
    phase = np.linspace(0.0, 1.0, frames, endpoint=False)
    qpos[:, 0] += 0.2 * np.sin(2 * np.pi * phase)
    return SkillSegment(
        segment_id=f"test:{skill}:0",
        skill=skill,
        t_start=0.0,
        t_end=frames / fps,
        phase=phase,
        command={"vx": 0.8},
        quality={"ik_err_mean_m": 0.03},
        qpos=qpos,
        root_pos=np.tile(np.array([0.0, 0.0, 0.79]), (frames, 1)),
        root_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        contacts=np.tile(np.array([True, False]), (frames, 1)),
        fps=fps,
        source_clip="test:0",
    )

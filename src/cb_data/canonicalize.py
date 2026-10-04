"""S2 canonicalize：根相对、朝向对齐、时间重采样、相位参数化。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cb_common.errors import DataError

from .rotations import slerp
from .schema import MotionClip


@dataclass
class CanonicalClip:
    """规范动作空间片段（ANCSH 方法论迁移：根轨迹/相位/幅度/结构分解）。"""

    clip: MotionClip
    phase: np.ndarray                 # (T,) [0,1)
    scale: float
    structure: dict[str, list[int]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        """帧数。"""
        return int(self.clip.frames)

    def phase_sin_cos(self) -> np.ndarray:
        """返回 (T,2) 相位 sin/cos。"""
        angle = 2.0 * np.pi * self.phase
        return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


STRUCTURE_GROUPS: dict[str, list[int]] = {
    "root": [0],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "torso": [3, 6, 9, 12],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "head": [15],
}


def canonicalize(clip: MotionClip, cfg: Mapping[str, Any]) -> CanonicalClip:
    """执行规范化；返回 CanonicalClip（不修改输入）。"""
    target_fps = float(cfg.get("target_fps", 30.0))
    resampled = resample_motion(clip, target_fps) if abs(clip.fps - target_fps) > 1e-9 else clip
    root_aligned = root_relative(resampled) if bool(cfg.get("root_align", True)) else resampled
    scale = estimate_skeleton_scale(root_aligned)
    phase = estimate_phase(root_aligned) if bool(cfg.get("phase_parameterize", True)) else np.linspace(
        0.0, 1.0, root_aligned.frames, endpoint=False
    )
    if not np.isfinite(phase).all():
        raise DataError("phase contains NaN", clip_id=clip.clip_id)
    return CanonicalClip(
        clip=root_aligned,
        phase=phase,
        scale=scale,
        structure=STRUCTURE_GROUPS,
        meta={"target_fps": target_fps, "root_align": True, "scale": scale},
    )


def resample_motion(clip: MotionClip, target_fps: float) -> MotionClip:
    """线性/球面插值重采样到目标 FPS。"""
    if target_fps <= 0:
        raise DataError("target_fps must be > 0", target_fps=target_fps)
    duration = (clip.frames - 1) / clip.fps
    count = int(round(duration * target_fps)) + 1
    t_src = np.arange(clip.frames, dtype=np.float64) / clip.fps
    t_dst = np.arange(count, dtype=np.float64) / target_fps
    root_pos = np.stack([np.interp(t_dst, t_src, clip.root_pos[:, i]) for i in range(3)], axis=-1)
    joint_pos = np.stack([np.interp(t_dst, t_src, clip.joint_pos[:, i]) for i in range(66)], axis=-1)
    joint_vel = np.stack([np.interp(t_dst, t_src, clip.joint_vel[:, i]) for i in range(66)], axis=-1)
    joint_world = np.stack(
        [np.interp(t_dst, t_src, clip.joint_world_pos[:, j, i]) for j in range(clip.joint_world_pos.shape[1]) for i in range(3)],
        axis=-1,
    ).reshape(count, clip.joint_world_pos.shape[1], 3)
    root_quat = _resample_quat(clip.root_quat, t_src, t_dst)
    contacts = np.stack([np.interp(t_dst, t_src, clip.contacts[:, i].astype(np.float64)) > 0.5 for i in range(2)], axis=-1)
    return MotionClip(
        clip_id=clip.clip_id,
        source=clip.source,
        fps=target_fps,
        frames=count,
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        joint_world_pos=joint_world,
        joint_vel=joint_vel,
        contacts=contacts,
        text=clip.text,
        license=clip.license,
        joint_names=clip.joint_names,
        meta={**clip.meta, "resampled_from_fps": clip.fps},
    )


def _resample_quat(quat: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.zeros((len(t_dst), 4), dtype=np.float64)
    for i, t in enumerate(t_dst):
        j = int(np.clip(np.searchsorted(t_src, t) - 1, 0, len(t_src) - 1))
        k = min(j + 1, len(t_src) - 1)
        if k == j:
            out[i] = quat[j]
            continue
        local = (t - t_src[j]) / max(t_src[k] - t_src[j], 1e-9)
        out[i] = slerp(quat[j], quat[k], np.array([local]))[0]
    return out


def root_relative(clip: MotionClip) -> MotionClip:
    """把根 xz 平移到原点（保留高度），关节位置相对根。"""
    root_xy = clip.root_pos[:, [0, 2]]
    world = clip.joint_world_pos.copy()
    world[:, :, 0] -= root_xy[:, 0:1]
    world[:, :, 2] -= root_xy[:, 1:2]
    root_pos = clip.root_pos.copy()
    root_pos[:, 0] = 0.0
    root_pos[:, 2] = 0.0
    return MotionClip(
        clip_id=clip.clip_id,
        source=clip.source,
        fps=clip.fps,
        frames=clip.frames,
        root_pos=root_pos,
        root_quat=clip.root_quat,
        joint_pos=clip.joint_pos,
        joint_world_pos=world,
        joint_vel=clip.joint_vel,
        contacts=clip.contacts,
        text=clip.text,
        license=clip.license,
        joint_names=clip.joint_names,
        meta={**clip.meta, "root_relative": True},
    )


def estimate_skeleton_scale(clip: MotionClip) -> float:
    """用髋-踝距离估计人体尺度（用于重定向归一化）。"""
    left_hip, right_hip = clip.joint_world_pos[:, 1], clip.joint_world_pos[:, 2]
    left_ankle, right_ankle = clip.joint_world_pos[:, 7], clip.joint_world_pos[:, 8]
    hip_center = 0.5 * (left_hip + right_hip)
    ankle_center = 0.5 * (left_ankle + right_ankle)
    leg_length = float(np.median(np.linalg.norm(hip_center - ankle_center, axis=-1)))
    return max(leg_length, 1e-3)


def estimate_phase(clip: MotionClip) -> np.ndarray:
    """由接触切换估计归一化相位（周期性动作），非周期回退为线性进度。"""
    switches = np.flatnonzero(np.any(np.diff(clip.contacts.astype(int), axis=0) != 0, axis=1))
    phase = np.zeros(clip.frames, dtype=np.float64)
    if switches.size >= 2:
        bounds = np.concatenate([[0], switches + 1, [clip.frames]])
        for start, end in zip(bounds[:-1], bounds[1:], strict=False):
            if end <= start:
                continue
            phase[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False)
    else:
        phase = np.linspace(0.0, 1.0, clip.frames, endpoint=False)
    return phase

"""S5 数据增强：镜像 / 幅度 / 速度 / 相位 / 根轨迹，参数全部配置化。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, POLICY_INDEX

from .rotations import matrix_to_quat_wxyz, quat_wxyz_to_matrix, yaw_quat
from .schema import SkillSegment

# 镜像时翻转符号的策略序关节
MIRROR_SIGN_FLIP: tuple[str, ...] = (
    "waist_yaw",
    "waist_roll",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_yaw",
    "right_wrist_yaw",
)
MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_hip_pitch", "right_hip_pitch"),
    ("left_hip_roll", "right_hip_roll"),
    ("left_hip_yaw", "right_hip_yaw"),
    ("left_knee", "right_knee"),
    ("left_ankle_pitch", "right_ankle_pitch"),
    ("left_ankle_roll", "right_ankle_roll"),
    ("left_shoulder_pitch", "right_shoulder_pitch"),
    ("left_shoulder_roll", "right_shoulder_roll"),
    ("left_shoulder_yaw", "right_shoulder_yaw"),
    ("left_elbow", "right_elbow"),
    ("left_wrist_roll", "right_wrist_roll"),
    ("left_wrist_pitch", "right_wrist_pitch"),
    ("left_wrist_yaw", "right_wrist_yaw"),
)


def augment_segment(segment: SkillSegment, config: Mapping[str, Any], rng: np.random.Generator) -> list[SkillSegment]:
    """生成增强副本（镜像/幅度/速度/相位/根轨迹噪声），并记录增强谱系。"""
    copies = int(config.get("augmentation_copies", 1))
    if copies < 1:
        raise DataError("augmentation_copies must be >= 1", copies=copies)
    out: list[SkillSegment] = []
    for i in range(copies):
        current = segment
        meta = dict(current.meta)
        augmentations: list[str] = []
        if bool(config.get("mirror", False)) and i % 2 == 1:
            current = mirror_segment(current)
            augmentations.append("mirror")
        if i > 0:
            tempo = _sample_range(rng, config.get("tempo", [1.0, 1.0]))
            if abs(tempo - 1.0) > 1e-6:
                current = tempo_segment(current, tempo)
                augmentations.append(f"tempo:{tempo:.3f}")
            amplitude = _sample_range(rng, config.get("amplitude", [1.0, 1.0]))
            if abs(amplitude - 1.0) > 1e-6:
                current = amplitude_segment(current, amplitude)
                augmentations.append(f"amplitude:{amplitude:.3f}")
            jitter = float(config.get("phase_jitter", 0.0))
            if jitter > 0.0:
                current = phase_jitter_segment(current, float(rng.uniform(-jitter, jitter)))
                augmentations.append("phase_jitter")
            xy_noise = config.get("root_xy_noise_m", [0.0, 0.0])
            if float(np.max(xy_noise)) > 0.0:
                current = root_noise_segment(current, rng, float(xy_noise[0]), float(xy_noise[1]))
                augmentations.append("root_noise")
        meta["augmentations"] = augmentations
        out.append(
            SkillSegment(
                segment_id=f"{current.segment_id}:aug{i}",
                skill=current.skill,
                t_start=current.t_start,
                t_end=current.t_end,
                phase=current.phase,
                command=dict(current.command),
                quality=dict(current.quality),
                qpos=current.qpos,
                root_pos=current.root_pos,
                root_quat=current.root_quat,
                contacts=current.contacts,
                fps=current.fps,
                source_clip=current.source_clip,
                meta=meta,
            )
        )
    return out


def mirror_segment(segment: SkillSegment) -> SkillSegment:
    """左右镜像：关节交换 + 符号翻转 + 根 y/偏航镜像 + 接触交换。"""
    qpos = np.array(segment.qpos, dtype=np.float64)
    mirrored = qpos.copy()
    for left, right in MIRROR_PAIRS:
        li, ri = POLICY_INDEX[left], POLICY_INDEX[right]
        mirrored[:, li] = qpos[:, ri]
        mirrored[:, ri] = qpos[:, li]
    for name in MIRROR_SIGN_FLIP:
        mirrored[:, POLICY_INDEX[name]] *= -1.0
    root = segment.root_pos.copy()
    root[:, 0] *= -1.0
    mirror = np.diag([1.0, -1.0, 1.0])
    quats = [matrix_to_quat_wxyz(mirror @ quat_wxyz_to_matrix(q) @ mirror) for q in segment.root_quat]
    return SkillSegment(
        segment_id=segment.segment_id,
        skill=segment.skill,
        t_start=segment.t_start,
        t_end=segment.t_end,
        phase=segment.phase,
        command=dict(segment.command),
        quality=dict(segment.quality),
        qpos=mirrored,
        root_pos=root,
        root_quat=np.stack(quats),
        contacts=segment.contacts[:, ::-1].copy(),
        fps=segment.fps,
        source_clip=segment.source_clip,
        meta={**segment.meta, "mirrored": True},
    )


def tempo_segment(segment: SkillSegment, tempo: float) -> SkillSegment:
    """时间缩放（保持 FPS，重采样轨迹）。"""
    if tempo <= 0:
        raise DataError("tempo must be > 0", tempo=tempo)
    t_src = np.arange(segment.frames, dtype=np.float64)
    count = max(2, int(round(segment.frames / tempo)))
    t_dst = np.linspace(0.0, segment.frames - 1, count)

    def interp(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(segment.frames, -1)
        out = np.stack([np.interp(t_dst, t_src, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
        return out.reshape(count, *values.shape[1:])

    contacts = np.stack(
        [np.interp(t_dst, t_src, segment.contacts[:, i].astype(float)) > 0.5 for i in range(2)], axis=-1
    )
    return SkillSegment(
        segment_id=segment.segment_id,
        skill=segment.skill,
        t_start=segment.t_start,
        t_end=segment.t_end,
        phase=np.linspace(0.0, 1.0, count, endpoint=False),
        command=dict(segment.command),
        quality=dict(segment.quality),
        qpos=interp(segment.qpos),
        root_pos=interp(segment.root_pos),
        root_quat=interp(segment.root_quat),
        contacts=contacts,
        fps=segment.fps,
        source_clip=segment.source_clip,
        meta={**segment.meta, "tempo": tempo},
    )


def amplitude_segment(segment: SkillSegment, scale: float) -> SkillSegment:
    """绕默认姿态缩放关节幅度。"""
    default = np.array(DEFAULT_JOINT_POS_POLICY, dtype=np.float64)
    qpos = default + (segment.qpos - default) * float(scale)
    return SkillSegment(
        segment_id=segment.segment_id,
        skill=segment.skill,
        t_start=segment.t_start,
        t_end=segment.t_end,
        phase=segment.phase,
        command=dict(segment.command),
        quality=dict(segment.quality),
        qpos=qpos,
        root_pos=segment.root_pos,
        root_quat=segment.root_quat,
        contacts=segment.contacts,
        fps=segment.fps,
        source_clip=segment.source_clip,
        meta={**segment.meta, "amplitude": scale},
    )


def phase_jitter_segment(segment: SkillSegment, delta: float) -> SkillSegment:
    """相位抖动（只改相位，不移动轨迹）。"""
    return SkillSegment(
        segment_id=segment.segment_id,
        skill=segment.skill,
        t_start=segment.t_start,
        t_end=segment.t_end,
        phase=(segment.phase + delta) % 1.0,
        command=dict(segment.command),
        quality=dict(segment.quality),
        qpos=segment.qpos,
        root_pos=segment.root_pos,
        root_quat=segment.root_quat,
        contacts=segment.contacts,
        fps=segment.fps,
        source_clip=segment.source_clip,
        meta={**segment.meta, "phase_jitter": delta},
    )


def root_noise_segment(segment: SkillSegment, rng: np.random.Generator, xy_min: float, xy_max: float) -> SkillSegment:
    """根 xy 噪声 + 小幅 yaw 扰动（不改关节轨迹）。"""
    root = segment.root_pos.copy()
    root[:, 0] += rng.uniform(xy_min, xy_max, size=segment.frames)
    root[:, 1] += rng.uniform(xy_min, xy_max, size=segment.frames)
    yaw = np.radians(rng.uniform(-2.0, 2.0))
    quats = np.stack([_mul_quat(yaw_quat(yaw), q) for q in segment.root_quat])
    return SkillSegment(
        segment_id=segment.segment_id,
        skill=segment.skill,
        t_start=segment.t_start,
        t_end=segment.t_end,
        phase=segment.phase,
        command=dict(segment.command),
        quality=dict(segment.quality),
        qpos=segment.qpos,
        root_pos=root,
        root_quat=quats,
        contacts=segment.contacts,
        fps=segment.fps,
        source_clip=segment.source_clip,
        meta={**segment.meta, "root_noise": [xy_min, xy_max]},
    )


def _sample_range(rng: np.random.Generator, bounds: Any) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        raise DataError("invalid augmentation range", bounds=list(bounds))
    return float(rng.uniform(lo, hi))


def _mul_quat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )

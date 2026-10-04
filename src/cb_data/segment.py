"""S4 segment：按文本 + 运动学把重定向片段切分为 SkillSegment。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import POLICY_INDEX

from .schema import RetargetedClip, SkillSegment

ALL_SKILLS: tuple[str, ...] = ("stand", "walk", "run", "jump", "wave", "turn", "recover")


@dataclass
class SegmentConfig:
    """切片配置（来自 configs/data.yaml）。"""

    min_segment_frames: int = 24
    boundary_smoothing: int = 5
    keyword_rules: Mapping[str, Sequence[str]] = None  # type: ignore[assignment]
    velocity_thresholds: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.keyword_rules is None:
            self.keyword_rules = {}
        if self.velocity_thresholds is None:
            self.velocity_thresholds = {}


def segment_clip(clip: RetargetedClip, config: SegmentConfig) -> list[SkillSegment]:
    """把一段重定向动作切成技能片段；无法分类的帧归入 stand。"""
    if clip.qpos.shape[0] < config.min_segment_frames:
        raise DataError("clip too short to segment", frames=int(clip.qpos.shape[0]), min=config.min_segment_frames)
    labels = _frame_labels(clip, config)
    labels = _smooth_labels(labels, config.boundary_smoothing)
    segments = _runs(labels, config.min_segment_frames)
    out: list[SkillSegment] = []
    text = str(clip.meta.get("text", ""))
    for idx, (skill, start, end) in enumerate(segments):
        qpos = clip.qpos[start:end]
        root = clip.root_pos[start:end]
        quat = clip.root_quat[start:end]
        contacts = clip.contacts[start:end]
        phase = _phase_for(qpos.shape[0], skill, contacts)
        command = _command_for(skill, qpos, root, quat, text, clip.fps)
        quality = {
            "ik_err_mean_m": float(np.mean(clip.retarget_err[start:end])),
            "ik_err_max_m": float(np.max(clip.retarget_err[start:end])),
            "limit_hits": int(np.sum(clip.joint_limit_hits[start:end])),
            "contact_consistency": float(_contact_consistency(contacts)),
        }
        out.append(
            SkillSegment(
                segment_id=f"{clip.retarget_id}:seg{idx}:{skill}",
                skill=skill,
                t_start=start / clip.fps,
                t_end=end / clip.fps,
                phase=phase,
                command=command,
                quality=quality,
                qpos=qpos,
                root_pos=root,
                root_quat=quat,
                contacts=contacts,
                fps=clip.fps,
                source_clip=clip.source_clip,
                meta={"text": text, "frame_range": [start, end]},
            )
        )
    return out


def _frame_labels(clip: RetargetedClip, config: SegmentConfig) -> list[str]:
    """逐帧技能标签（文本先验 + 运动学证据）。"""
    text = str(clip.meta.get("text", "")).lower()
    prior = np.zeros(len(ALL_SKILLS))
    for skill, words in config.keyword_rules.items():
        if skill not in ALL_SKILLS:
            continue
        prior[ALL_SKILLS.index(skill)] += sum(1.0 for w in words if str(w).lower() in text)
    thresholds = config.velocity_thresholds
    root = clip.root_pos
    dt = 1.0 / clip.fps
    vx = np.gradient(root[:, 0], dt)
    vy = np.gradient(root[:, 1], dt)
    vz = np.gradient(root[:, 2], dt)
    speed = np.sqrt(vx**2 + vy**2)
    yaw_rate = np.gradient(np.unwrap(_yaw(clip.root_quat)), dt)
    shoulder_std = np.std(clip.qpos[:, [POLICY_INDEX["left_shoulder_pitch"], POLICY_INDEX["right_shoulder_pitch"]]], axis=0)
    labels: list[str] = []
    for t in range(clip.qpos.shape[0]):
        scores = np.array(prior, dtype=np.float64)
        s = float(speed[t])
        if s >= float(thresholds.get("run_min_mps", 1.4)):
            scores[ALL_SKILLS.index("run")] += 2.5
        elif s >= float(thresholds.get("walk_min_mps", 0.15)):
            scores[ALL_SKILLS.index("walk")] += 2.0
        elif s <= float(thresholds.get("stand_max_mps", 0.15)):
            scores[ALL_SKILLS.index("stand")] += 1.5
        if abs(float(yaw_rate[t])) >= float(thresholds.get("turn_min_rad_s", 0.6)):
            scores[ALL_SKILLS.index("turn")] += 2.0
        if abs(float(vz[t])) > 0.35 or float(root[t, 2]) > float(np.median(root[:, 2])) + 0.10:
            scores[ALL_SKILLS.index("jump")] += 3.0
        if max(shoulder_std) > 0.25:
            scores[ALL_SKILLS.index("wave")] += 1.5
        if abs(float(vx[t])) > 2.0 or abs(float(vy[t])) > 2.0:
            scores[ALL_SKILLS.index("recover")] += 0.5
        labels.append(ALL_SKILLS[int(np.argmax(scores))])
    return labels


def _smooth_labels(labels: list[str], window: int) -> list[str]:
    """滑动窗口多数投票（边界平滑）。"""
    if window <= 1:
        return labels
    half = window // 2
    out: list[str] = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        window_labels = labels[lo:hi]
        counts = {label: window_labels.count(label) for label in set(window_labels)}
        out.append(max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0])
    return out


def _runs(labels: list[str], min_frames: int) -> list[tuple[str, int, int]]:
    """合并连续标签为片段，过短片段并入邻居。"""
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], start, i))
            start = i
    merged: list[tuple[str, int, int]] = []
    for run in runs:
        if merged and run[2] - run[1] < min_frames:
            prev_skill, prev_start, _ = merged[-1]
            merged[-1] = (prev_skill, prev_start, run[2])
        else:
            merged.append(run)
    if merged and merged[0][2] - merged[0][1] < min_frames and len(merged) > 1:
        merged[1] = (merged[1][0], merged[0][1], merged[1][2])
        merged.pop(0)
    return merged


def _phase_for(frames: int, skill: str, contacts: np.ndarray) -> np.ndarray:
    """周期技能按接触切换给相位；其余线性。"""
    if skill in ("walk", "run") and contacts.shape[0] > 1:
        switches = np.flatnonzero(np.any(np.diff(contacts.astype(int), axis=0) != 0, axis=1))
        if switches.size >= 2:
            bounds = np.concatenate([[0], switches + 1, [frames]])
            phase = np.zeros(frames)
            for a, b in zip(bounds[:-1], bounds[1:], strict=False):
                phase[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
            return phase
    return np.linspace(0.0, 1.0, frames, endpoint=False)


def _command_for(
    skill: str, qpos: np.ndarray, root: np.ndarray, quat: np.ndarray, text: str, fps: float
) -> dict[str, Any]:
    """从片段统计生成技能指令。"""
    dt = 1.0 / fps
    vx = float(np.mean(np.gradient(root[:, 0], dt)))
    vy = float(np.mean(np.gradient(root[:, 1], dt)))
    yaw = np.unwrap(_yaw(quat)) if quat.shape[0] > 1 else np.zeros(1)
    yaw_rate = float(np.mean(np.gradient(yaw, dt))) if yaw.size > 1 else 0.0
    command: dict[str, Any] = {"vx": round(vx, 3), "vy": round(vy, 3), "yaw_rate": round(yaw_rate, 3)}
    if skill == "jump":
        command.update(
            {
                "height_m": round(float(np.max(root[:, 2]) - np.median(root[:, 2])), 3),
                "forward_m": round(float(root[-1, 0] - root[0, 0]), 3),
            }
        )
    if skill == "wave":
        lowered = text.lower()
        command["side"] = "left" if "left" in lowered else "right"
    if skill == "turn":
        command["target_yaw_deg"] = round(float(np.degrees(yaw[-1] - yaw[0])), 1)
    return command


def _yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _contact_consistency(contacts: np.ndarray) -> float:
    """接触一致性：至少一脚接触的帧占比（0–1）。"""
    if contacts.size == 0:
        return 0.0
    return float(np.mean(np.any(contacts, axis=1)))

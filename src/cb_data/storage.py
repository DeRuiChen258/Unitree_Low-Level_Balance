"""片段集合落盘/读取：每片段一个 npz + index.jsonl（可复现、可抽检）。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import MUJOCO_JOINT_NAMES

from .schema import MotionClip, RetargetedClip, SkillSegment, TransitionSegment


def _write_index(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    """np.ndarray / np.generic / Path → JSON 可序列化。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _read_index(directory: Path) -> list[dict[str, Any]]:
    path = directory / "index.jsonl"
    if not path.is_file():
        raise DataError("collection index missing", path=str(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def save_motion_clips(directory: str | Path, clips: Iterable[MotionClip]) -> list[dict[str, Any]]:
    """保存 MotionClip 集合。"""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for clip in clips:
        name = _safe_name(clip.clip_id)
        np.savez_compressed(base / f"{name}.npz", **clip.to_arrays())
        row = clip.summary()
        row["file"] = f"{name}.npz"
        rows.append(row)
    _write_index(base, rows)
    return rows


def load_motion_clips(directory: str | Path) -> list[MotionClip]:
    """读取 MotionClip 集合。"""
    base = Path(directory)
    out: list[MotionClip] = []
    for row in _read_index(base):
        with np.load(base / row["file"]) as arrays:
            out.append(
                MotionClip.from_arrays(
                    clip_id=row["clip_id"],
                    source=row["source"],
                    arrays={key: arrays[key] for key in arrays.files},
                    text=row["text"],
                    license_=row["license"],
                    meta=row.get("meta", {}),
                )
            )
    return out


def save_retargeted(directory: str | Path, clips: Iterable[RetargetedClip]) -> list[dict[str, Any]]:
    """保存 RetargetedClip 集合。"""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for clip in clips:
        name = _safe_name(clip.retarget_id)
        np.savez_compressed(base / f"{name}.npz", **clip.to_arrays())
        row = clip.summary()
        row["file"] = f"{name}.npz"
        rows.append(row)
    _write_index(base, rows)
    return rows


def load_retargeted(directory: str | Path) -> list[RetargetedClip]:
    """读取 RetargetedClip 集合。"""
    base = Path(directory)
    out: list[RetargetedClip] = []
    for row in _read_index(base):
        with np.load(base / row["file"]) as arrays:
            out.append(
                RetargetedClip(
                    retarget_id=row["retarget_id"],
                    robot=row["robot"],
                    joint_names=MUJOCO_JOINT_NAMES,
                    qpos=arrays["qpos"],
                    qvel=arrays["qvel"],
                    keypoint_pos=arrays["keypoint_pos"],
                    retarget_err=arrays["retarget_err"],
                    source_clip=row["source_clip"],
                    root_pos=arrays["root_pos"],
                    root_quat=arrays["root_quat"],
                    contacts=arrays["contacts"].astype(bool),
                    joint_limit_hits=arrays["joint_limit_hits"],
                    fps=float(arrays["fps"][0]),
                    meta=dict(row.get("meta", {})),
                )
            )
    return out


def save_segments(directory: str | Path, segments: Iterable[SkillSegment]) -> list[dict[str, Any]]:
    """保存 SkillSegment 集合。"""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for segment in segments:
        name = _safe_name(segment.segment_id)
        np.savez_compressed(base / f"{name}.npz", **segment.to_arrays())
        row = segment.summary()
        row["file"] = f"{name}.npz"
        rows.append(row)
    _write_index(base, rows)
    return rows


def load_segments_from_dir(directory: str | Path) -> list[SkillSegment]:
    """读取 SkillSegment 集合。"""
    base = Path(directory)
    out: list[SkillSegment] = []
    for row in _read_index(base):
        with np.load(base / row["file"]) as arrays:
            out.append(
                SkillSegment(
                    segment_id=row["segment_id"],
                    skill=row["skill"],
                    t_start=float(row["t_start"]),
                    t_end=float(row["t_end"]),
                    phase=arrays["phase"],
                    command=dict(row.get("command", {})),
                    quality=dict(row.get("quality", {})),
                    qpos=arrays["qpos"],
                    root_pos=arrays["root_pos"],
                    root_quat=arrays["root_quat"],
                    contacts=arrays["contacts"].astype(bool),
                    fps=float(arrays["fps"][0]),
                    source_clip=row.get("source_clip", ""),
                    meta=dict(row.get("meta", {})),
                )
            )
    return out


def save_transitions(directory: str | Path, transitions: Iterable[TransitionSegment]) -> list[dict[str, Any]]:
    """保存 TransitionSegment 集合。"""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in transitions:
        name = _safe_name(item.transition_id)
        np.savez_compressed(base / f"{name}.npz", **item.to_arrays())
        row = item.summary()
        row["file"] = f"{name}.npz"
        rows.append(row)
    _write_index(base, rows)
    return rows


def load_transitions(directory: str | Path) -> list[TransitionSegment]:
    """读取 TransitionSegment 集合。"""
    base = Path(directory)
    out: list[TransitionSegment] = []
    for row in _read_index(base):
        with np.load(base / row["file"]) as arrays:
            out.append(
                TransitionSegment(
                    transition_id=row["transition_id"],
                    from_skill=row["from_skill"],
                    to_skill=row["to_skill"],
                    blend_window=float(row["blend_window"]),
                    phase_alignment=row["phase_alignment"],
                    qpos=arrays["qpos"],
                    root_pos=arrays["root_pos"],
                    contacts=arrays["contacts"].astype(bool),
                    from_segment=row["from_segment"],
                    to_segment=row["to_segment"],
                    quality=dict(row.get("quality", {})),
                    fps=float(arrays["fps"][0]),
                    meta=dict(row.get("meta", {})),
                )
            )
    return out

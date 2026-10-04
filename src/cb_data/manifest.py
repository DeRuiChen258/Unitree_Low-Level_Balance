"""S8 manifest：统计 + sha256 + 版本 + 质量报告。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, POLICY_JOINT_NAMES

from .schema import DatasetManifest, SkillSegment, TransitionSegment

TOOL_VERSION = "cb_data-1.0.0"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """文件 sha256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    dataset_version: str,
    root: Path,
    segments: list[SkillSegment],
    transitions: list[TransitionSegment],
    splits: Mapping[str, list[SkillSegment]],
    profile: str,
    aug_profile: str,
    feature_version: str,
    obs_spec_hash: str,
    reject_stats: Mapping[str, int] | None = None,
    source_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    """构造 manifest（包含文件哈希与统计量）。"""
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    stats = segment_statistics(segments, transitions)
    if source_counts:
        stats["count_by_source"] = dict(source_counts)
    return DatasetManifest.create(
        dataset_version=dataset_version,
        tool_version=TOOL_VERSION,
        files=files,
        stats=stats,
        splits={name: [s.segment_id for s in items] for name, items in splits.items()},
        aug_profile=aug_profile,
        profile=profile,
        obs_spec_hash=obs_spec_hash,
        feature_version=feature_version,
        reject_stats=dict(reject_stats or {}),
    )


def segment_statistics(segments: list[SkillSegment], transitions: list[TransitionSegment]) -> dict[str, Any]:
    """统计：计数、时长、qpos 均值/标准差、IK 误差。"""
    if not segments:
        raise DataError("no accepted segments; refusing to write empty manifest")
    qpos = np.concatenate([s.qpos for s in segments], axis=0)
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    for segment in segments:
        counts[segment.skill] = counts.get(segment.skill, 0) + 1
        durations[segment.skill] = durations.get(segment.skill, 0.0) + float(segment.frames / segment.fps)
    ik_err = np.array([s.quality.get("ik_err_mean_m", 0.0) for s in segments], dtype=np.float64)
    return {
        "num_segments": len(segments),
        "num_transitions": len(transitions),
        "num_frames": int(qpos.shape[0]),
        "total_duration_s": float(sum(durations.values())),
        "count_by_skill": counts,
        "duration_by_skill_s": {key: round(value, 3) for key, value in durations.items()},
        "qpos_mean": {name: float(v) for name, v in zip(POLICY_JOINT_NAMES, qpos.mean(axis=0), strict=False)},
        "qpos_std": {name: float(v) for name, v in zip(POLICY_JOINT_NAMES, qpos.std(axis=0), strict=False)},
        "default_pose_deviation_mean_rad": float(np.mean(np.abs(qpos - np.array(DEFAULT_JOINT_POS_POLICY)))),
        "ik_err_mean_m": float(np.mean(ik_err)),
        "ik_err_p95_m": float(np.percentile(ik_err, 95)),
    }


def write_quality_report(
    root: Path,
    *,
    manifest: DatasetManifest,
    verdicts: Iterable[Any],
    rejected: list[Any],
    command: str,
) -> Path:
    """写 report.md 与 stats.csv。"""
    root.mkdir(parents=True, exist_ok=True)
    _ = verdicts
    report = root / "report.md"
    lines = [
        f"# Data Quality Report `{manifest.dataset_version}`",
        "",
        f"- profile: `{manifest.profile}`",
        f"- created_at: `{manifest.created_at}`",
        f"- tool_version: `{manifest.tool_version}`",
        f"- command: `{command}`",
        f"- obs_spec_hash: `{manifest.obs_spec_hash}`",
        f"- feature_version: `{manifest.feature_version}`",
        "",
        "## 统计",
        "",
        f"- segments: {manifest.stats['num_segments']}",
        f"- transitions: {manifest.stats['num_transitions']}",
        f"- frames: {manifest.stats['num_frames']}",
        f"- IK err mean: {manifest.stats['ik_err_mean_m']:.4f} m",
        f"- IK err P95: {manifest.stats['ik_err_p95_m']:.4f} m",
        f"- skill counts: {json.dumps(manifest.stats['count_by_skill'], ensure_ascii=False)}",
        f"- splits: train={len(manifest.splits['train'])} val={len(manifest.splits['val'])} test={len(manifest.splits['test'])}",
        "",
        "## 拒绝 / 修复",
        "",
        f"- reject_stats: `{json.dumps(manifest.reject_stats, ensure_ascii=False)}`",
        f"- rejected segments: {len(rejected)}",
        "",
        "## 领域 gap 声明",
        "",
        "- NOCS / ANCSH 属物体位姿领域，本项目只迁移「合成覆盖 + 真实校准」「规范空间 + 结构分解」",
        "  方法论，不声称其直接支持人形动作增强。",
        "- KIT-ML 未解压、AMASS 为空：未参与统计，脚本在启用时显式报错。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_csv = root / "stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["skill", "segments", "duration_s"])
        for skill, count in sorted(manifest.stats["count_by_skill"].items()):
            writer.writerow([skill, count, manifest.stats["duration_by_skill_s"].get(skill, 0.0)])
    return report


def load_manifest(path: str | Path) -> DatasetManifest:
    """读取 manifest。"""
    return DatasetManifest.load(path)

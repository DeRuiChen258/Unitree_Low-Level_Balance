"""第 4 节 8 步数据管线编排（S1 → S8）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config, resolve_path
from cb_common.errors import DataError
from cb_common.joints import mujoco_to_isaac
from cb_common.logging import JsonlLogger
from cb_common.seeding import set_global_seed
from cb_safety import load_joint_limits_mjcf

from .canonicalize import canonicalize
from .ingest import iter_motion_clips, resolve_source_specs, skill_from_text
from .manifest import build_manifest, load_manifest, write_quality_report
from .physics_filter import FilterConfig, PhysicsFilter
from .retarget import G1IKSolver, IKConfig, retarget_clip
from .schema import DatasetManifest, SkillSegment, TransitionSegment
from .segment import SegmentConfig, segment_clip
from .split import split_segments
from .storage import load_segments_from_dir, save_motion_clips, save_retargeted, save_segments, save_transitions
from .synthesize import augment_segment
from .synthesize_scenes import generate_scenes, save_scenes
from .synthesize_transition import synthesize_transitions
from .visualize import plot_segment


@dataclass
class PipelineResult:
    """数据管线结果（路径 + manifest）。"""

    manifest: DatasetManifest
    root: Path
    accepted: list[SkillSegment]
    rejected: list[SkillSegment]
    transitions: list[TransitionSegment]


def run_data_pipeline(
    data_cfg: Config,
    system_cfg: Config,
    *,
    output_root: str | Path | None = None,
    logger: JsonlLogger | None = None,
) -> PipelineResult:
    """执行 S1–S8 全流程并落盘。"""
    profile = str(data_cfg.get("profile", "smoke"))
    profile_cfg = dict(data_cfg.get(f"profiles.{profile}", {}) or {})
    seed = int(data_cfg.get("seed", 1))
    seed_record = set_global_seed(seed)
    dataset_version = str(data_cfg.get("manifest.dataset_version"))
    if not dataset_version:
        raise DataError("manifest.dataset_version is required")
    runs_dir = resolve_path(data_cfg, "manifest.output_dir")
    root = Path(output_root) if output_root else runs_dir / dataset_version
    root.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.log("data_pipeline", stage="start", profile=profile, dataset_version=dataset_version, **seed_record)

    # S1 ingest
    sources = resolve_source_specs(data_cfg.to_dict())
    clips = []
    for _name, spec in sources.items():
        if not spec.enabled:
            continue
        candidate_cap = max(int(profile_cfg.get("max_clips", 0)) * 4, 200)
        for clip in iter_motion_clips(
            spec,
            max_clips=candidate_cap,
            min_frames=int(profile_cfg.get("min_frames", 20)),
            max_frames=int(profile_cfg.get("max_frames_per_clip", 0)),
        ):
            clips.append(clip)
        if clips:
            break
    if not clips:
        raise DataError("no motion clips ingested; check data config and dataset root")
    clips, selection_stats = _select_clips_by_skill(
        clips,
        max_clips=int(profile_cfg.get("max_clips", 0)),
        clips_per_skill=int(profile_cfg.get("clips_per_skill", 0)),
        keyword_rules=dict(data_cfg.get("segment.keyword_rules", {}) or {}),
    )
    not_upright = [clip for clip in clips if not bool(clip.meta.get("upright", True))]
    clips = [clip for clip in clips if bool(clip.meta.get("upright", True))]
    if not clips:
        raise DataError("all ingested clips failed the upright quality gate", skipped=len(not_upright))
    save_motion_clips(root / "motion", clips)
    if logger:
        logger.log("data_pipeline", stage="ingest", clips=len(clips))

    # S2 canonicalize
    canonical_cfg = data_cfg.section("canonicalize").to_dict()
    canonical_clips = [canonicalize(clip, canonical_cfg) for clip in clips]

    # S3 retarget
    scene_path = str(system_cfg.get("limits.mjcf_scene"))
    retarget_cfg = dict(data_cfg.get("retarget", {}) or {})
    ik_config = IKConfig(
        scene_path=scene_path,
        ik_iters=int(retarget_cfg.get("ik_iters", 40)),
        ik_lr=float(retarget_cfg.get("ik_lr", 0.65)),
        ik_damping=float(retarget_cfg.get("ik_damping", 1e-3)),
        ik_tolerance_m=float(retarget_cfg.get("ik_tolerance_m", 0.06)),
        ik_reject_m=float(retarget_cfg.get("ik_reject_m", 0.16)),
        joint_limit_margin_rad=float(retarget_cfg.get("joint_limit_margin_rad", 0.02)),
    )
    solver = G1IKSolver(ik_config)
    retargeted = [retarget_clip(clip, ik_config, solver=solver) for clip in canonical_clips]
    for item in retargeted:
        item.meta["text"] = clips[0].text if len(retargeted) == 1 else item.meta.get("text", "")
    retargeted = [item for item in retargeted if float(np.mean(item.retarget_err)) <= ik_config.ik_reject_m]
    if not retargeted:
        raise DataError("all retargeted clips were rejected; check IK thresholds")
    save_retargeted(root / "retargeted", retargeted)
    if logger:
        logger.log(
            "data_pipeline",
            stage="retarget",
            clips=len(retargeted),
            ik_err_mean=float(np.mean([np.mean(item.retarget_err) for item in retargeted])),
        )

    # S4 segment
    segment_config = SegmentConfig(
        min_segment_frames=int(data_cfg.get("segment.min_segment_frames", 24)),
        boundary_smoothing=int(data_cfg.get("segment.boundary_smoothing", 5)),
        keyword_rules=dict(data_cfg.get("segment.keyword_rules", {}) or {}),
        velocity_thresholds=dict(data_cfg.get("segment.velocity_thresholds", {}) or {}),
    )
    base_segments: list[SkillSegment] = []
    for item in retargeted:
        base_segments.extend(segment_clip(item, segment_config))
    if not base_segments:
        raise DataError("segmentation produced no segments")

    # S5 augment + transitions + scenes
    augment_cfg = dict(data_cfg.get("augment", {}) or {})
    rng = np.random.default_rng(seed + 17)
    augmented: list[SkillSegment] = []
    for segment in base_segments:
        augmented.extend(augment_segment(segment, augment_cfg, rng))
    transitions = synthesize_transitions(augmented, augment_cfg)
    scenes = generate_scenes(augment_cfg, count=max(12, len(augmented)), seed=seed)
    save_scenes(root / "scenes.json", scenes)

    # S6 physics filter
    lo_mj, hi_mj = load_joint_limits_mjcf(scene_path, policy_order=False)
    limits_policy = (mujoco_to_isaac(lo_mj), mujoco_to_isaac(hi_mj))
    filter_cfg = FilterConfig(
        position_margin_rad=float(data_cfg.get("physics_filter.position_margin_rad", 0.02)),
        velocity_limit_scale=float(data_cfg.get("physics_filter.velocity_limit_scale", 1.0)),
        acceleration_limit_rad_s2=float(data_cfg.get("physics_filter.acceleration_limit_rad_s2", 120.0)),
        jerk_limit_rad_s3=float(data_cfg.get("physics_filter.jerk_limit_rad_s3", 4000.0)),
        foot_slip_max_m_s=float(data_cfg.get("physics_filter.foot_slip_max_m_s", 0.35)),
        foot_penetration_max_m=float(data_cfg.get("physics_filter.foot_penetration_max_m", 1e-3)),
        root_height_min_m=float(data_cfg.get("physics_filter.root_height_min_m", 0.55)),
        com_margin_min_m=float(data_cfg.get("physics_filter.com_margin_min_m", -0.02)),
        allow_dynamic_violation_frames=int(data_cfg.get("physics_filter.allow_dynamic_violation_frames", 12)),
        scene_path=scene_path,
    )
    physics = PhysicsFilter(filter_cfg)
    accepted: list[SkillSegment] = []
    rejected: list[SkillSegment] = []
    verdicts = []
    reject_stats: dict[str, int] = {}
    for segment in augmented:
        verdict = physics.check(segment, joint_limits=limits_policy)
        verdicts.append(verdict)
        if verdict.status == "rejected":
            rejected.append(segment)
            for key in verdict.reasons:
                reject_stats[key] = reject_stats.get(key, 0) + 1
        elif verdict.status == "repaired":
            accepted.append(physics.repair(segment, limits_policy))
            reject_stats["repaired"] = reject_stats.get("repaired", 0) + 1
        else:
            accepted.append(segment)
    if not accepted:
        raise DataError("physics filter rejected all segments")
    save_segments(root / "segments", accepted)
    if transitions:
        save_transitions(root / "transitions", transitions)

    # S7 split
    splits = split_segments(
        accepted,
        ratios=dict(data_cfg.get("split.ratios", {}) or {}),
        seed=int(data_cfg.get("split.seed", 1)),
    )

    # S8 manifest + report + visualization
    from cb_features.obs_spec import spec_from_config

    spec = spec_from_config({**system_cfg.to_dict(), "skill_embedding_dim": 16})
    manifest = build_manifest(
        dataset_version=dataset_version,
        root=root,
        segments=accepted,
        transitions=transitions,
        splits=splits,
        profile=profile,
        aug_profile=json.dumps({k: v for k, v in augment_cfg.items() if k != "transitions"}, sort_keys=True),
        feature_version=str(system_cfg.get("feature_version", "1.0")),
        obs_spec_hash=spec.hash(),
        reject_stats={**reject_stats, **({"not_upright": len(not_upright)} if not_upright else {})},
        source_counts=_source_counts(clips),
    )
    manifest.stats["selection_by_skill"] = selection_stats
    manifest.save(root / "manifest.json")
    command = f"python scripts/data_build.py --config configs/data.yaml --profile {profile}"
    write_quality_report(root, manifest=manifest, verdicts=verdicts, rejected=rejected, command=command)
    for segment in accepted[:5]:
        plot_segment(segment, root / "viz" / f"{_safe(segment.segment_id)}.png")
    if logger:
        logger.log(
            "data_pipeline",
            stage="done",
            accepted=len(accepted),
            rejected=len(rejected),
            transitions=len(transitions),
            manifest=str(root / "manifest.json"),
        )
    return PipelineResult(manifest=manifest, root=root, accepted=accepted, rejected=rejected, transitions=transitions)


def load_segments(manifest_path: str | Path, split: str, *, root: str | Path | None = None) -> list[SkillSegment]:
    """按 manifest 与 split 加载技能片段（训练入口使用）。"""
    manifest = load_manifest(manifest_path)
    base = Path(root) if root else Path(manifest_path).parent
    segments = load_segments_from_dir(base / "segments")
    ids = set(manifest.splits.get(split, []))
    if not ids:
        raise DataError("split is empty or unknown", split=split, available=sorted(manifest.splits))
    return [segment for segment in segments if segment.segment_id in ids]


def _source_counts(clips: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip.source] = counts.get(clip.source, 0) + 1
    return counts


def _select_clips_by_skill(
    clips: list[Any],
    *,
    max_clips: int,
    clips_per_skill: int,
    keyword_rules: dict[str, Any],
) -> tuple[list[Any], dict[str, int]]:
    """按文本技能桶做均衡采样：每技能最多 clips_per_skill 个，再按顺序补齐 max_clips。

    这样避免顺序读文件时只拿到单一动作类型（首个 split 文件是随机排列的）。
    """
    if not clips:
        return clips, {}
    if clips_per_skill <= 0:
        selected = clips[:max_clips] if max_clips else clips
        return selected, _bucket_counts(selected, keyword_rules)
    buckets: dict[str, list[Any]] = {}
    for clip in clips:
        bucket = skill_from_text(str(clip.text), keyword_rules)
        buckets.setdefault(bucket, []).append(clip)
    selected: list[Any] = []
    for bucket in sorted(buckets):
        selected.extend(buckets[bucket][:clips_per_skill])
    if max_clips:
        selected_ids = {id(clip) for clip in selected}
        for clip in clips:
            if len(selected) >= max_clips:
                break
            if id(clip) not in selected_ids:
                selected.append(clip)
        selected = selected[:max_clips]
    order = {id(clip): index for index, clip in enumerate(clips)}
    selected.sort(key=lambda clip: order.get(id(clip), 0))
    return selected, _bucket_counts(selected, keyword_rules)


def _bucket_counts(clips: list[Any], keyword_rules: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in clips:
        bucket = skill_from_text(str(clip.text), keyword_rules)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)

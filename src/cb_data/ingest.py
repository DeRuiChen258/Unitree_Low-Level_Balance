"""S1 ingest：读取 HumanML3D_272d / 263d / KIT-ML，产出 MotionClip + 文本。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cb_common.errors import DataError

from .rotations import matrix_to_axis_angle, rotation_6d_to_matrix, yaw_quat
from .schema import SMPL_JOINT_NAMES, MotionClip

FOOT_JOINTS = (10, 11)
CONTACT_HEIGHT_THRESHOLD_M = 0.06
CONTACT_VELOCITY_THRESHOLD_MPS = 0.12


@dataclass(frozen=True)
class SourceSpec:
    """数据源配置（只读）。"""

    name: str
    root: Path
    fps: float
    license: str
    enabled: bool = True


def resolve_source_specs(data_cfg: Mapping) -> dict[str, SourceSpec]:
    """把 `configs/data.yaml` 的 sources 解析为强类型对象。"""
    out: dict[str, SourceSpec] = {}
    for name, item in dict(data_cfg.get("sources", {})).items():
        item = dict(item)
        out[name] = SourceSpec(
            name=name,
            root=Path(str(item["root"])).expanduser(),
            fps=float(item.get("fps", 30.0)),
            license=str(item.get("license", "unknown")),
            enabled=bool(item.get("enabled", False)),
        )
    return out


def split_ids(root: Path, split: str) -> list[str]:
    """读取 HumanML3D 划分文件，返回 clip id（按文件顺序）。"""
    path = root / "split" / f"{split}.txt"
    if not path.is_file():
        raise DataError("split file missing", path=str(path))
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise DataError("split file is empty", path=str(path))
    return ids


def parse_text_file(path: Path) -> str:
    """解析 HumanML3D 文本文件：取第一条描述（`#` 前）。"""
    if not path.is_file():
        raise DataError("text file missing", path=str(path))
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not first:
        raise DataError("text file empty", path=str(path))
    text = first[0].split("#")[0].strip()
    if not text:
        raise DataError("text is empty before '#'", path=str(path))
    return text


def skill_from_text(text: str, keyword_rules: Mapping[str, Sequence[str]]) -> str:
    """按关键词把片段文本归入技能桶（无匹配返回 stand）。"""
    lowered = text.lower()
    best = "stand"
    best_score = 0
    for skill, words in keyword_rules.items():
        score = sum(1 for word in words if str(word).lower() in lowered)
        if score > best_score:
            best, best_score = str(skill), score
    return best


def parse_272d(array: np.ndarray, *, fps: float = 30.0) -> dict[str, np.ndarray]:
    """解析 MotionStreamer 272 维表示（已用官方脚本核对，布局见模块 docstring）。

    布局（T,272）：
      0:2    root xz 速度（heading-free）
      2:8    heading 增量 6D（t=0 为单位阵）
      8:74   22 关节局部位置（x,z 相对原点；y=高度）
      74:140 22 关节局部速度
      140:272 22 关节 6D 旋转（heading-free）
    """
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 272:
        raise DataError("272d motion must be (T,272)", shape=arr.shape)
    t = arr.shape[0]
    positions = arr[:, 8:74].reshape(t, 22, 3).copy()
    velocities = arr[:, 74:140].reshape(t, 22, 3).copy()
    rot6d = arr[:, 140:272].reshape(t, 22, 6).copy()
    rotations = rotation_6d_to_matrix(rot6d)  # (T,22,3,3)
    joint_axis_angle = matrix_to_axis_angle(rotations).reshape(t, 66)
    # 关节轴角速度：对轴角做时间差分（272 维局部速度字段是位置差分，用于接触判定）
    joint_vel = np.gradient(joint_axis_angle, 1.0 / fps, axis=0)
    # 接触：足端（SMPL 10/11）高度接近最低点且速度小
    foot_height = positions[:, FOOT_JOINTS, 1]
    floor = float(np.min(positions[:, :, 1]))
    foot_speed = np.linalg.norm(velocities[:, FOOT_JOINTS, :], axis=-1) * fps
    contacts = (foot_height <= floor + CONTACT_HEIGHT_THRESHOLD_M) & (foot_speed <= CONTACT_VELOCITY_THRESHOLD_MPS)
    root_pos = positions[:, 0, :].copy()
    # root 朝向：heading 累积（单位阵 → 增量 6D）
    heading = np.zeros(t, dtype=np.float64)
    for i in range(1, t):
        delta = rotation_6d_to_matrix(arr[i, 2:8])
        heading[i] = heading[i - 1] + float(np.arctan2(delta[0, 2], delta[2, 2]))
    root_quat = np.stack([yaw_quat(float(y)) for y in heading])
    return {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_axis_angle": joint_axis_angle,
        "joint_world_pos": positions,
        "joint_vel": joint_vel,
        "joint_world_vel": velocities,
        "contacts": contacts,
        "heading": heading,
        "root_lin_vel_xz": arr[:, 0:2].copy(),
    }


def load_272d_clip(root: Path, clip_id: str, *, license_: str = "academic") -> MotionClip:
    """读取单个 272d 片段 + 文本，返回 MotionClip。"""
    motion_path = root / "motion_data" / f"{clip_id}.npy"
    text_path = root / "texts" / f"{clip_id}.txt"
    if not motion_path.is_file():
        raise DataError("motion file missing", path=str(motion_path))
    text = parse_text_file(text_path)
    parsed = parse_272d(np.load(motion_path), fps=30.0)
    positions = parsed["joint_world_pos"]
    pelvis_height = float(np.median(positions[:, 0, 1]))
    spine_height = float(np.median(positions[:, 12, 1] - positions[:, 0, 1]))
    foot_below = float(np.mean(positions[:, FOOT_JOINTS, 1] < positions[:, 0:1, 1]))
    upright = pelvis_height >= 0.60 and spine_height >= 0.25 and foot_below >= 0.6
    return MotionClip(
        clip_id=f"humanml3d_272d:{clip_id}:0-{parsed['root_pos'].shape[0]}",
        source="humanml3d_272d",
        fps=30.0,
        frames=int(parsed["root_pos"].shape[0]),
        root_pos=parsed["root_pos"],
        root_quat=parsed["root_quat"],
        joint_pos=parsed["joint_axis_angle"],
        joint_world_pos=parsed["joint_world_pos"],
        joint_vel=parsed["joint_vel"],
        contacts=parsed["contacts"],
        text=text,
        license=license_,
        joint_names=SMPL_JOINT_NAMES,
        meta={
            "source_file": str(motion_path),
            "source_sha256": sha256_file(motion_path),
            "heading_rad": parsed["heading"],
            "root_lin_vel_xz": parsed["root_lin_vel_xz"],
            "upright": upright,
            "pelvis_height_median_m": pelvis_height,
            "spine_height_median_m": spine_height,
            "foot_below_ratio": foot_below,
        },
    )


def iter_motion_clips(
    spec: SourceSpec,
    *,
    max_clips: int = 0,
    min_frames: int = 20,
    max_frames: int = 0,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Iterator[MotionClip]:
    """遍历数据源片段；缺失的可选数据源显式报错（禁止静默跳过）。"""
    if not spec.enabled:
        return
    if not spec.root.exists():
        raise DataError("dataset root does not exist", source=spec.name, path=str(spec.root))
    if spec.name != "humanml3d_272d":
        raise DataError(
            "source is not implemented for automatic ingest on this machine",
            source=spec.name,
            hint="KIT-ML 需先解压 .rar；HumanML3D 263d 本机不完整；AMASS 为空目录",
        )
    seen: set[str] = set()
    count = 0
    for split in splits:
        for clip_id in split_ids(spec.root, split):
            if clip_id in seen:
                continue
            seen.add(clip_id)
            motion_path = spec.root / "motion_data" / f"{clip_id}.npy"
            if not motion_path.is_file():
                continue
            frames = int(np.load(motion_path, mmap_mode="r").shape[0])
            if frames < min_frames or (max_frames and frames > max_frames):
                continue
            clip = load_272d_clip(spec.root, clip_id, license_=spec.license)
            clip.meta["split_source"] = split
            yield clip
            count += 1
            if max_clips and count >= max_clips:
                return


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """文件 sha256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()

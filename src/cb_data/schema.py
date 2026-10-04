"""冻结的数据结构（第 4.1 节，SCHEMA_VERSION=1.0）。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import DataError

SCHEMA_VERSION = "1.0"
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def _check_shape(name: str, array: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if array.ndim != len(shape):
        raise DataError(f"{name} ndim mismatch", expected=shape, got=array.shape)
    for got, want in zip(array.shape, shape, strict=False):
        if want is not None and int(got) != int(want):
            raise DataError(f"{name} shape mismatch", expected=shape, got=array.shape)
    if not np.isfinite(array).all():
        raise DataError(f"{name} contains NaN/Inf", shape=array.shape)


@dataclass
class MotionClip:
    """原始动捕片段（人类骨架空间）。"""

    clip_id: str
    source: str
    fps: float
    frames: int
    root_pos: np.ndarray              # (T,3) m，heading-free 坐标系
    root_quat: np.ndarray             # (T,4) wxyz（heading-free 根朝向）
    joint_pos: np.ndarray             # (T,66) 每关节轴角 (rad)
    joint_world_pos: np.ndarray       # (T,22,3) m，heading-free 关节位置
    joint_vel: np.ndarray             # (T,66) 轴角速度 (rad/s)，缺失时差分估计
    contacts: np.ndarray              # (T,2) bool
    text: str
    license: str
    joint_names: Sequence[str] = SMPL_JOINT_NAMES
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float64)
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float64)
        self.joint_world_pos = np.asarray(self.joint_world_pos, dtype=np.float64)
        self.joint_vel = np.asarray(self.joint_vel, dtype=np.float64)
        self.contacts = np.asarray(self.contacts, dtype=bool)
        t = int(self.frames)
        _check_shape("root_pos", self.root_pos, (t, 3))
        _check_shape("root_quat", self.root_quat, (t, 4))
        _check_shape("joint_pos", self.joint_pos, (t, 66))
        _check_shape("joint_world_pos", self.joint_world_pos, (t, len(self.joint_names), 3))
        _check_shape("joint_vel", self.joint_vel, (t, 66))
        if self.contacts.shape != (t, 2):
            raise DataError("contacts shape mismatch", expected=(t, 2), got=self.contacts.shape)
        if self.fps <= 0:
            raise DataError("fps must be > 0", fps=self.fps)

    def summary(self) -> dict[str, Any]:
        """返回元数据摘要（不包含大数组）。"""
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": round(self.frames / self.fps, 3),
            "text": self.text,
            "license": self.license,
            "schema_version": self.schema_version,
            "meta": self.meta,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """导出为 npz 友好的数组字典。"""
        return {
            "root_pos": self.root_pos,
            "root_quat": self.root_quat,
            "joint_pos": self.joint_pos,
            "joint_world_pos": self.joint_world_pos,
            "joint_vel": self.joint_vel,
            "contacts": self.contacts.astype(np.uint8),
            "fps": np.array([self.fps]),
            "frames": np.array([self.frames]),
        }

    @classmethod
    def from_arrays(cls, clip_id: str, source: str, arrays: Mapping[str, np.ndarray], text: str, license_: str,
                    meta: Mapping[str, Any] | None = None) -> MotionClip:
        """从 npz 数组恢复。"""
        return cls(
            clip_id=clip_id,
            source=source,
            fps=float(np.asarray(arrays["fps"]).reshape(-1)[0]),
            frames=int(np.asarray(arrays["frames"]).reshape(-1)[0]),
            root_pos=arrays["root_pos"],
            root_quat=arrays["root_quat"],
            joint_pos=arrays["joint_pos"],
            joint_world_pos=arrays["joint_world_pos"],
            joint_vel=arrays["joint_vel"],
            contacts=np.asarray(arrays["contacts"]).astype(bool),
            text=text,
            license=license_,
            meta=dict(meta or {}),
        )


@dataclass
class RetargetedClip:
    """重定向到 G1 的片段（qpos 为策略序）。"""

    retarget_id: str
    robot: str
    joint_names: Sequence[str]
    qpos: np.ndarray                  # (T,29) rad
    qvel: np.ndarray                  # (T,29) rad/s
    keypoint_pos: np.ndarray          # (T,K,3) m
    retarget_err: np.ndarray          # (T,) m
    source_clip: str
    root_pos: np.ndarray              # (T,3) m
    root_quat: np.ndarray             # (T,4) wxyz
    contacts: np.ndarray              # (T,2) bool
    joint_limit_hits: np.ndarray      # (T,) int
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        self.qvel = np.asarray(self.qvel, dtype=np.float64)
        self.keypoint_pos = np.asarray(self.keypoint_pos, dtype=np.float64)
        self.retarget_err = np.asarray(self.retarget_err, dtype=np.float64)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float64)
        self.contacts = np.asarray(self.contacts, dtype=bool)
        t = self.qpos.shape[0]
        if len(self.joint_names) != self.qpos.shape[1] or self.qpos.shape[1] != 29:
            raise DataError("G1 retarget must have 29 joints", names=len(self.joint_names), qpos=self.qpos.shape)
        _check_shape("qpos", self.qpos, (t, 29))
        _check_shape("qvel", self.qvel, (t, 29))
        _check_shape("root_pos", self.root_pos, (t, 3))
        _check_shape("root_quat", self.root_quat, (t, 4))
        if self.contacts.shape != (t, 2) or self.retarget_err.shape != (t,):
            raise DataError("retarget contacts/err shape mismatch", contacts=self.contacts.shape, err=self.retarget_err.shape)
        if self.joint_limit_hits.shape != (t,):
            raise DataError("joint_limit_hits shape mismatch", got=self.joint_limit_hits.shape)

    def summary(self) -> dict[str, Any]:
        """元数据摘要 + IK 误差统计。"""
        return {
            "retarget_id": self.retarget_id,
            "robot": self.robot,
            "source_clip": self.source_clip,
            "frames": int(self.qpos.shape[0]),
            "fps": self.fps,
            "ik_err_mean_m": float(np.mean(self.retarget_err)),
            "ik_err_max_m": float(np.max(self.retarget_err)),
            "joint_limit_hits": int(np.sum(self.joint_limit_hits)),
            "schema_version": self.schema_version,
            "meta": self.meta,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """导出 npz 数组。"""
        return {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "keypoint_pos": self.keypoint_pos,
            "retarget_err": self.retarget_err,
            "root_pos": self.root_pos,
            "root_quat": self.root_quat,
            "contacts": self.contacts.astype(np.uint8),
            "joint_limit_hits": self.joint_limit_hits,
            "fps": np.array([self.fps]),
        }


@dataclass
class SkillSegment:
    """单一技能切片（训练/组合基本单位，扩展了可训练轨迹数组）。"""

    segment_id: str
    skill: str
    t_start: float
    t_end: float
    phase: np.ndarray                 # (T,) [0,1)
    command: dict[str, Any]
    quality: dict[str, Any]
    qpos: np.ndarray                  # (T,29)
    root_pos: np.ndarray              # (T,3)
    root_quat: np.ndarray             # (T,4)
    contacts: np.ndarray              # (T,2)
    fps: float
    source_clip: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.phase = np.asarray(self.phase, dtype=np.float64)
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float64)
        self.contacts = np.asarray(self.contacts, dtype=bool)
        t = self.qpos.shape[0]
        if self.phase.shape != (t,) or self.root_pos.shape != (t, 3) or self.root_quat.shape != (t, 4):
            raise DataError("skill segment shape mismatch", phase=self.phase.shape, qpos=self.qpos.shape)
        if self.contacts.shape != (t, 2):
            raise DataError("skill segment contacts shape mismatch", got=self.contacts.shape)
        if self.t_end <= self.t_start:
            raise DataError("t_end must be > t_start", t_start=self.t_start, t_end=self.t_end)

    @property
    def frames(self) -> int:
        """帧数。"""
        return int(self.qpos.shape[0])

    def summary(self) -> dict[str, Any]:
        """元数据摘要。"""
        return {
            "segment_id": self.segment_id,
            "skill": self.skill,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "frames": self.frames,
            "fps": self.fps,
            "command": self.command,
            "quality": self.quality,
            "source_clip": self.source_clip,
            "meta": self.meta,
            "schema_version": self.schema_version,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """导出 npz 数组。"""
        return {
            "phase": self.phase,
            "qpos": self.qpos,
            "root_pos": self.root_pos,
            "root_quat": self.root_quat,
            "contacts": self.contacts.astype(np.uint8),
            "fps": np.array([self.fps]),
        }


@dataclass
class TransitionSegment:
    """技能间过渡片段（合成轨迹 + 溯源）。"""

    transition_id: str
    from_skill: str
    to_skill: str
    blend_window: float
    phase_alignment: str
    qpos: np.ndarray
    root_pos: np.ndarray
    contacts: np.ndarray
    from_segment: str
    to_segment: str
    quality: dict[str, Any] = field(default_factory=dict)
    fps: float = 30.0
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def frames(self) -> int:
        """帧数。"""
        return int(self.qpos.shape[0])

    def __post_init__(self) -> None:
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.contacts = np.asarray(self.contacts, dtype=bool)
        t = self.qpos.shape[0]
        if self.root_pos.shape != (t, 3) or self.contacts.shape != (t, 2):
            raise DataError("transition shape mismatch", qpos=self.qpos.shape, root=self.root_pos.shape)
        if self.blend_window <= 0:
            raise DataError("blend_window must be > 0")

    def summary(self) -> dict[str, Any]:
        """元数据摘要。"""
        return {
            "transition_id": self.transition_id,
            "from_skill": self.from_skill,
            "to_skill": self.to_skill,
            "blend_window": self.blend_window,
            "phase_alignment": self.phase_alignment,
            "frames": int(self.qpos.shape[0]),
            "fps": self.fps,
            "from_segment": self.from_segment,
            "to_segment": self.to_segment,
            "quality": self.quality,
            "schema_version": self.schema_version,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """导出 npz 数组。"""
        return {
            "qpos": self.qpos,
            "root_pos": self.root_pos,
            "contacts": self.contacts.astype(np.uint8),
            "fps": np.array([self.fps]),
        }


@dataclass
class DatasetManifest:
    """数据集版本与溯源（第 4.1 节）。"""

    dataset_version: str
    created_at: str
    tool_version: str
    files: list[dict[str, Any]]
    stats: dict[str, Any]
    splits: dict[str, list[str]]
    aug_profile: str
    profile: str
    obs_spec_hash: str = ""
    feature_version: str = "1.0"
    reject_stats: dict[str, int] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(cls, dataset_version: str, tool_version: str, **kwargs: Any) -> DatasetManifest:
        """带 UTC 时间戳创建 manifest。"""
        return cls(
            dataset_version=dataset_version,
            created_at=datetime.now(UTC).isoformat(),
            tool_version=tool_version,
            **kwargs,
        )

    def save(self, path: str | Path) -> Path:
        """写入 JSON 文件。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        """读取 JSON 文件并校验 schema 版本。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise DataError("manifest schema mismatch", found=data.get("schema_version"), expected=SCHEMA_VERSION)
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise DataError("manifest has unknown fields", unknown=sorted(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """转字典（asdict 深拷贝）。"""
        return asdict(self)

    def require_version(self, expected: str) -> None:
        """训练端版本校验（不匹配直接拒绝）。"""
        if self.dataset_version != expected:
            raise DataError("dataset version mismatch", expected=expected, found=self.dataset_version)

    def verify_files(self, root: str | Path) -> list[str]:
        """校验文件 sha256；返回不匹配清单。"""
        import hashlib

        problems: list[str] = []
        base = Path(root)
        for item in self.files:
            path = base / item["path"]
            if not path.is_file():
                problems.append(f"missing:{item['path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                problems.append(f"sha256:{item['path']}")
        return problems

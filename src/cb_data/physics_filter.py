"""S6 物理可行性过滤：限位/速度/加速度/jerk/接触/支撑域（硬门禁）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import MUJOCO_JOINT_NAMES, NUM_JOINTS, isaac_to_mujoco

from .schema import SkillSegment


@dataclass
class FilterConfig:
    """过滤阈值（来自 configs/data.yaml）。"""

    position_margin_rad: float = 0.02
    velocity_limit_scale: float = 1.0
    acceleration_limit_rad_s2: float = 120.0
    jerk_limit_rad_s3: float = 4000.0
    foot_slip_max_m_s: float = 0.35
    foot_penetration_max_m: float = 1e-3
    root_height_min_m: float = 0.55
    com_margin_min_m: float = -0.02
    allow_dynamic_violation_frames: int = 12
    scene_path: str = ""


@dataclass
class SegmentVerdict:
    """单片段过滤结果。"""

    segment_id: str
    status: str
    reasons: dict[str, int] = field(default_factory=dict)
    repaired: bool = False
    quality: dict[str, Any] = field(default_factory=dict)


class PhysicsFilter:
    """物理可行性过滤器（MuJoCo 用于 CoM/支撑域真值）。"""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._mujoco = None
        self._model = None
        self._data = None
        self._qpos_addr: np.ndarray | None = None
        if config.scene_path:
            import mujoco

            path = Path(config.scene_path)
            if not path.is_file():
                raise DataError("physics filter scene missing", path=str(path))
            self._mujoco = mujoco
            self._model = mujoco.MjModel.from_xml_path(str(path))
            self._data = mujoco.MjData(self._model)
            self._qpos_addr = np.array(
                [self._model.jnt_qposadr[self._model.joint(name).id] for name in MUJOCO_JOINT_NAMES]
            )

    def check(self, segment: SkillSegment, *, joint_limits: tuple[np.ndarray, np.ndarray] | None = None) -> SegmentVerdict:
        """检查片段；超硬阈值 rejected，轻微越界返回 repaired。"""
        reasons: dict[str, int] = {}
        qpos = segment.qpos
        if qpos.shape[1] != NUM_JOINTS:
            return SegmentVerdict(segment.segment_id, "rejected", {"joint_count": 1})
        lo, hi = joint_limits if joint_limits is not None else (np.full(NUM_JOINTS, -np.inf), np.full(NUM_JOINTS, np.inf))
        margin = self.config.position_margin_rad
        out_of_range = (qpos < lo + margin) | (qpos > hi - margin)
        if out_of_range.any():
            reasons["position_out_of_range"] = int(np.count_nonzero(out_of_range))
        dt = 1.0 / segment.fps
        vel = np.gradient(qpos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        if np.max(np.abs(acc)) > self.config.acceleration_limit_rad_s2:
            reasons["acceleration"] = int(np.count_nonzero(np.abs(acc) > self.config.acceleration_limit_rad_s2))
        if np.max(np.abs(jerk)) > self.config.jerk_limit_rad_s3:
            reasons["jerk"] = int(np.count_nonzero(np.abs(jerk) > self.config.jerk_limit_rad_s3))
        if np.min(segment.root_pos[:, 2]) < self.config.root_height_min_m:
            reasons["root_height"] = int(np.count_nonzero(segment.root_pos[:, 2] < self.config.root_height_min_m))
        slip = self._foot_slip(segment, vel)
        if slip > self.config.foot_slip_max_m_s:
            reasons["foot_slip"] = 1
        margin_value = self._com_margin(segment)
        if margin_value is not None and margin_value < self.config.com_margin_min_m:
            reasons["com_margin"] = 1
        hard = {"acceleration", "jerk", "root_height", "foot_slip", "com_margin"}
        if any(key in reasons for key in hard) and not self._repairable(reasons):
            return SegmentVerdict(segment.segment_id, "rejected", reasons)
        quality = {
            "max_abs_vel": float(np.max(np.abs(vel))),
            "max_abs_acc": float(np.max(np.abs(acc))),
            "max_abs_jerk": float(np.max(np.abs(jerk))),
            "foot_slip_max": float(slip),
            "com_margin_min": None if margin_value is None else float(margin_value),
            "contact_consistency": float(np.mean(np.any(segment.contacts, axis=1))),
        }
        status = "repaired" if reasons else "accepted"
        return SegmentVerdict(segment.segment_id, status, reasons, repaired=bool(reasons), quality=quality)

    def repair(self, segment: SkillSegment, limits: tuple[np.ndarray, np.ndarray]) -> SkillSegment:
        """修复：限位裁剪 + 迭代平滑，直到加速度/jerk 落入阈值或达到上限。"""
        lo, hi = limits
        qpos = np.clip(segment.qpos, lo + self.config.position_margin_rad, hi - self.config.position_margin_rad)
        for _ in range(5):
            dt = 1.0 / segment.fps
            acc = np.gradient(np.gradient(qpos, dt, axis=0), dt, axis=0)
            jerk = np.gradient(acc, dt, axis=0)
            if (
                np.max(np.abs(acc)) <= self.config.acceleration_limit_rad_s2
                and np.max(np.abs(jerk)) <= self.config.jerk_limit_rad_s3
            ):
                break
            qpos = _smooth(qpos, window=3)
        return SkillSegment(
            segment_id=segment.segment_id,
            skill=segment.skill,
            t_start=segment.t_start,
            t_end=segment.t_end,
            phase=segment.phase,
            command=dict(segment.command),
            quality={**segment.quality, "repaired": True},
            qpos=qpos,
            root_pos=segment.root_pos,
            root_quat=segment.root_quat,
            contacts=segment.contacts,
            fps=segment.fps,
            source_clip=segment.source_clip,
            meta={**segment.meta, "physics_repair": "joint_clip"},
        )

    def _repairable(self, reasons: Mapping[str, int]) -> bool:
        """位置越界 / 加速度 / jerk / 脚滑可尝试修复；根高度与 CoM 不可修复。"""
        if set(reasons) - {"position_out_of_range", "acceleration", "jerk", "foot_slip"}:
            return False
        return (
            reasons.get("position_out_of_range", 0) <= self.config.allow_dynamic_violation_frames
            and reasons.get("acceleration", 0) <= 10 * self.config.allow_dynamic_violation_frames
            and reasons.get("jerk", 0) <= 10 * self.config.allow_dynamic_violation_frames
        )

    def _foot_slip(self, segment: SkillSegment, joint_vel: np.ndarray) -> float:
        """接触帧踝关节速度作为脚滑近似（乘腿长换算 m/s）。"""
        contact_any = np.any(segment.contacts, axis=1)
        if not np.any(contact_any):
            return 0.0
        ankle_cols = [4, 5, 10, 11]
        ankle_speed = np.linalg.norm(joint_vel[:, ankle_cols], axis=1) * 0.12
        return float(np.max(ankle_speed[contact_any]))

    def _com_margin(self, segment: SkillSegment) -> float | None:
        """MuJoCo 逐帧 CoM 到支撑多边形的有符号余量（取最小值）。"""
        if self._mujoco is None or self._model is None or self._data is None or self._qpos_addr is None:
            return None
        margins: list[float] = []
        for t in range(segment.frames):
            self._data.qpos[:] = 0.0
            self._data.qpos[0:3] = segment.root_pos[t]
            self._data.qpos[3:7] = segment.root_quat[t]
            self._data.qpos[self._qpos_addr] = isaac_to_mujoco(segment.qpos[t])
            self._mujoco.mj_forward(self._model, self._data)
            com = np.array(self._data.subtree_com[0], dtype=np.float64)
            feet = []
            for body_id in range(self._model.nbody):
                name = self._mujoco.mj_id2name(self._model, self._mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                if name.endswith("ankle_roll_link"):
                    feet.append(np.array(self._data.xpos[body_id], dtype=np.float64)[[0, 1]])
            if len(feet) >= 2:
                margins.append(_polygon_margin(com[:2], _convex_hull(np.array(feet))))
            else:
                margins.append(0.0)
        return float(min(margins)) if margins else None


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """二维凸包（Andrew monotone chain）。"""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).round(9))))
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o: Any, a: Any, b: Any) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """滑动平均（边缘常数填充）。"""
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])], axis=-1)


def _polygon_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """点到凸多边形的最小有符号距离（内部为正）。"""
    if polygon.shape[0] < 3:
        return 0.0
    inside = True
    best = float("inf")
    for i in range(polygon.shape[0]):
        a, b = polygon[i], polygon[(i + 1) % polygon.shape[0]]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        dist = float(np.dot(point - a, normal))
        if dist < 0.0:
            inside = False
        best = min(best, abs(dist))
    return best if inside else -best

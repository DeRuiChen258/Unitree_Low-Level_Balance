"""观测 spec：分组、维度、单位、归一化、缺失策略与稳定 hash。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cb_common.joints import NUM_JOINTS

FEATURE_VERSION = "1.0"


@dataclass(frozen=True)
class ObsGroup:
    """单个观测组定义。"""

    name: str
    dim: int
    unit: str
    normalize: bool = True
    missing: str = "zero"
    history: bool = False


@dataclass(frozen=True)
class ObsSpec:
    """观测 spec（训练/部署同源；hash 写入 checkpoint 与 ONNX metadata）。"""

    groups: tuple[ObsGroup, ...]
    history_length: int = 4
    version: str = FEATURE_VERSION
    clip: float = 5.0

    @property
    def history_groups(self) -> tuple[ObsGroup, ...]:
        """需要历史堆叠的组。"""
        return tuple(g for g in self.groups if g.history)

    @property
    def flat_dim(self) -> int:
        """单帧（不含历史）维度。"""
        return sum(g.dim for g in self.groups)

    @property
    def history_dim(self) -> int:
        """历史组按组堆叠后的维度。"""
        return sum(g.dim * self.history_length for g in self.history_groups)

    @property
    def total_dim(self) -> int:
        """最终观测维度。"""
        return self.history_dim + sum(g.dim for g in self.groups if not g.history)

    def hash(self) -> str:
        """spec 内容 hash（含版本/裁剪/历史长度）。"""
        payload = {
            "version": self.version,
            "history_length": self.history_length,
            "clip": self.clip,
            "groups": [g.__dict__ for g in self.groups],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def slice_of(self, name: str) -> slice:
        """返回单帧展开后某组的索引区间（仅非历史部分或单帧视图）。"""
        offset = 0
        for group in self.groups:
            if group.name == name:
                return slice(offset, offset + group.dim)
            offset += group.dim
        raise KeyError(name)

    def assemble(self, frames: Mapping[str, np.ndarray], history: Mapping[str, np.ndarray]) -> np.ndarray:
        """把单帧组值与历史缓冲组装为最终观测向量。

        `history[name]` 的形状为 `(history_length, dim)`（旧 → 新），仅历史组需要。
        """
        parts: list[np.ndarray] = []
        for group in self.history_groups:
            if group.name not in history:
                raise KeyError(f"missing history group {group.name!r}")
            block = np.asarray(history[group.name], dtype=np.float64)
            if block.shape != (self.history_length, group.dim):
                raise ValueError(f"history[{group.name}] must be {(self.history_length, group.dim)}, got {block.shape}")
            parts.append(block.reshape(-1))
        for group in self.groups:
            if group.history:
                continue
            if group.name not in frames:
                if group.missing == "zero":
                    parts.append(np.zeros(group.dim))
                    continue
                raise KeyError(f"missing observation group {group.name!r}")
            value = np.asarray(frames[group.name], dtype=np.float64).reshape(-1)
            if value.size != group.dim:
                raise ValueError(f"group {group.name} expects {group.dim}, got {value.size}")
            parts.append(value)
        out = np.concatenate(parts)
        if out.size != self.total_dim:
            raise ValueError(f"assembled obs dim {out.size} != spec {self.total_dim}")
        return np.nan_to_num(out, nan=0.0, posinf=self.clip, neginf=-self.clip)


def default_obs_spec(history_length: int = 4, skill_embedding_dim: int = 16) -> ObsSpec:
    """构造默认 29 DOF 观测 spec（第 5.1 节）。"""
    groups = (
        ObsGroup("ang_vel", 3, "rad/s", normalize=True, history=True),
        ObsGroup("gravity", 3, "unit", normalize=False, history=True),
        ObsGroup("lin_vel", 3, "m/s", normalize=True, history=True),
        ObsGroup("command", 5, "mixed", normalize=True, history=True),
        ObsGroup("joint_pos", NUM_JOINTS, "rad", normalize=True, history=True),
        ObsGroup("joint_vel", NUM_JOINTS, "rad/s", normalize=True, history=True),
        ObsGroup("last_action", NUM_JOINTS, "rad", normalize=False, history=True),
        ObsGroup("contact", 2, "bool", normalize=False, history=False),
        ObsGroup("phase_clock", 4, "unit", normalize=False, history=False),
        ObsGroup("skill_embedding", skill_embedding_dim, "unit", normalize=False, history=False),
        ObsGroup("balance_feats", 8, "mixed", normalize=True, history=False),
    )
    return ObsSpec(groups=groups, history_length=history_length)


OBS_SPEC = default_obs_spec()


def spec_from_config(system_config: Mapping[str, object]) -> ObsSpec:
    """按 `system.yaml` 的 history_length 与 skills.yaml 的 embedding 维度构造 spec。"""
    history = int(system_config.get("history_length", 4))
    embed = int(system_config.get("skill_embedding_dim", 16))
    return default_obs_spec(history_length=history, skill_embedding_dim=embed)


def groups_by_name(spec: ObsSpec) -> dict[str, ObsGroup]:
    """返回 group 字典。"""
    return {g.name: g for g in spec.groups}


def group_slices(spec: ObsSpec) -> dict[str, Sequence[tuple[int, int]]]:
    """返回最终观测中每个 group 的 (块序号, 起止) 位置，便于测试/调试。"""
    out: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for group in spec.history_groups:
        for block in range(spec.history_length):
            out.setdefault(group.name, []).append((block, offset))
            offset += group.dim
    for group in spec.groups:
        if group.history:
            continue
        out.setdefault(group.name, []).append((-1, offset))
        offset += group.dim
    return {k: v for k, v in out.items()}

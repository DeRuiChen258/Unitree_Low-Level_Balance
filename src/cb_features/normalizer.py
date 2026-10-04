"""Running mean/std 归一化与裁剪命中率统计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunningMeanStd:
    """增量均值/方差（Welford），与 checkpoint 一起保存。"""

    shape: tuple[int, ...]
    epsilon: float = 1e-4
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: float = field(default=0.0, init=False)
    clip_hits: int = field(default=0, init=False)
    total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        """用一批样本更新统计量。"""
        x = np.asarray(batch, dtype=np.float64)
        if x.shape[-len(self.shape) :] != self.shape:
            raise ValueError(f"expected trailing shape {self.shape}, got {x.shape}")
        x = x.reshape(-1, *self.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = float(total)

    def normalize(self, obs: np.ndarray, *, clip: float = 5.0) -> np.ndarray:
        """归一化并裁剪；记录裁剪命中率。"""
        arr = np.asarray(obs, dtype=np.float64)
        out = (arr - self.mean) / np.sqrt(self.var + self.epsilon)
        hits = int(np.sum(np.abs(out) > clip))
        self.clip_hits += hits
        self.total += out.size
        return np.clip(out, -clip, clip)

    def clip_ratio(self) -> float:
        """裁剪命中率。"""
        return float(self.clip_hits / max(1, self.total))

    def state_dict(self) -> dict[str, np.ndarray | float]:
        """序列化统计量。"""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        """恢复统计量。"""
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def save(self, path: str | Path) -> None:
        """保存为 npz。"""
        np.savez(Path(path), mean=self.mean, var=self.var, count=self.count)

    @classmethod
    def load(cls, path: str | Path) -> RunningMeanStd:
        """从 npz 恢复。"""
        data = np.load(Path(path))
        obj = cls(shape=tuple(np.asarray(data["mean"]).shape))
        obj.load_state_dict({"mean": data["mean"], "var": data["var"], "count": float(data["count"])})
        return obj

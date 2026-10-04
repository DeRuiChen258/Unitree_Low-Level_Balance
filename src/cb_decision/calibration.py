"""温度分桶校准、阈值搜索、可靠性曲线、ECE（第 8.3 节）。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cb_common.errors import DecisionError


def bucket_for(qtype: str, options: int) -> str:
    """按题目类型与选项数返回温度桶（与 rl_agent_config.json 对齐）。"""
    if qtype == "noul":
        return "noul:2"
    if qtype == "score":
        if options <= 2:
            return "score:2"
        if options <= 5:
            return "score:3-5"
        return "score:6-10"
    if options <= 2:
        return "choice:2"
    if options <= 5:
        return "choice:3-5"
    if options <= 10:
        return "choice:6-10"
    return "choice:11+"


@dataclass
class TemperatureScaler:
    """分桶温度 + 类别偏置（在留出集上拟合，最后写入 calibration.yaml）。"""

    temperatures: dict[str, float] = field(default_factory=dict)
    default_temperature: float = 1.0
    biases: dict[str, float] = field(default_factory=dict)

    def apply(self, probabilities: Sequence[float] | np.ndarray, bucket: str) -> np.ndarray:
        """应用温度与偏置后重新归一化。"""
        p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if p.size == 0:
            return p
        p = np.clip(p, 1e-9, 1.0)
        temperature = float(self.temperatures.get(bucket, self.default_temperature))
        logits = np.log(p) / max(temperature, 1e-6) + float(self.biases.get(bucket, 0.0))
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[Sequence[float]],
        labels: Sequence[int],
        buckets: Sequence[str],
        *,
        grid: Sequence[float] | None = None,
    ) -> TemperatureScaler:
        """按桶网格搜索最小 NLL 的温度。"""
        grid = list(grid or np.linspace(0.5, 3.0, 26))
        per_bucket: dict[str, list[tuple[np.ndarray, int]]] = {}
        for probs, label, bucket in zip(probabilities, labels, buckets, strict=False):
            per_bucket.setdefault(str(bucket), []).append((np.asarray(probs, dtype=np.float64), int(label)))
        scaler = cls()
        for bucket, items in per_bucket.items():
            best_t, best_nll = 1.0, float("inf")
            for temperature in grid:
                nll = 0.0
                for probs, label in items:
                    p = np.clip(probs, 1e-9, 1.0)
                    logits = np.log(p) / temperature
                    logits -= logits.max()
                    q = np.exp(logits) / np.exp(logits).sum()
                    nll -= float(np.log(max(q[min(label, len(q) - 1)], 1e-9)))
                nll /= max(1, len(items))
                if nll < best_nll:
                    best_t, best_nll = float(temperature), nll
            scaler.temperatures[bucket] = best_t
        return scaler

    def save(self, path: str | Path) -> Path:
        """写出校准参数 JSON。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"temperatures": self.temperatures, "biases": self.biases, "default_temperature": self.default_temperature},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        """读取校准参数。"""
        file = Path(path)
        if not file.is_file():
            raise DecisionError("calibration file not found", path=str(file))
        data = json.loads(file.read_text(encoding="utf-8"))
        return cls(
            temperatures={str(k): float(v) for k, v in data.get("temperatures", {}).items()},
            default_temperature=float(data.get("default_temperature", 1.0)),
            biases={str(k): float(v) for k, v in data.get("biases", {}).items()},
        )


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10) -> float:
    """ECE（标量）。"""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def reliability_curve(probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10) -> list[dict[str, float]]:
    """可靠性曲线数据点。"""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[dict[str, float]] = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if np.any(mask):
            curve.append({"bin": i, "confidence": float(p[mask].mean()), "accuracy": float(y[mask].mean()), "count": int(mask.sum())})
    return curve

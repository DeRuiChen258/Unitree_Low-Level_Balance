"""指标定义单一来源（第 8.4 节决策指标 + 第 12.1 节平衡/组合指标）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_episodes(episodes: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """聚合 episodes：均值 / 标准差 / 最差 seed + 通过判定。"""
    if not episodes:
        return {"num_episodes": 0, "pass": False}

    def values(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(ep.get(key, default)) for ep in episodes], dtype=np.float64)

    roll = values("roll_peak_deg")
    pitch = values("pitch_peak_deg")
    fall = values("fall")
    speed_err = values("speed_error_mean")
    margin = values("support_margin_min", 0.0)
    jump = values("jump_success")
    wave = values("wave_success")
    turn_err = values("turn_error_deg")
    reward = values("episode_reward")
    recovery = values("recovery_time_s")
    result: dict[str, Any] = {
        "num_episodes": len(episodes),
        "reward_mean": float(reward.mean()),
        "reward_std": float(reward.std()),
        "reward_worst": float(reward.min()),
        "roll_peak_mean_deg": float(roll.mean()),
        "roll_peak_worst_deg": float(roll.max()),
        "pitch_peak_mean_deg": float(pitch.mean()),
        "pitch_peak_worst_deg": float(pitch.max()),
        "fall_rate": float(fall.mean()),
        "fall_count": int(fall.sum()),
        "support_margin_min_mean_m": float(margin.mean()),
        "support_margin_worst_m": float(margin.min()),
        "speed_error_mean_mps": float(speed_err.mean()),
        "speed_error_worst_mps": float(speed_err.max()),
        "speed_retention_mean": float(values("speed_retention").mean()) if "speed_retention" in episodes[0] else None,
        "support_violation_max_s": float(values("support_violation_max_s").max()) if "support_violation_max_s" in episodes[0] else 0.0,
        "support_violation_ratio_mean": float(values("support_violation_ratio").mean()) if "support_violation_ratio" in episodes[0] else 0.0,
        "jump_success_rate": float(np.nanmean(jump)) if np.any(~np.isnan(jump)) else None,
        "wave_success_rate": float(np.nanmean(wave)) if np.any(~np.isnan(wave)) else None,
        "turn_error_worst_deg": float(np.nanmax(np.abs(turn_err))) if np.any(~np.isnan(turn_err)) else None,
        "recovery_time_mean_s": float(np.nanmean(recovery)) if np.any(~np.isnan(recovery)) else None,
        "recovery_time_worst_s": float(np.nanmax(recovery)) if np.any(~np.isnan(recovery)) else None,
    }
    result["checks"] = {
        "fall_rate": result["fall_rate"] <= float(thresholds.get("fall_rate_per_100", 0.0)) / 100.0,
        "roll": result["roll_peak_worst_deg"] <= float(thresholds.get("max_roll_deg_combo", 12.0)),
        "pitch": result["pitch_peak_worst_deg"] <= float(thresholds.get("max_pitch_deg_combo", 12.0)),
        "speed": (
            result["speed_error_worst_mps"] <= float(thresholds.get("velocity_error_mps", 0.35))
            or (
                result["speed_retention_mean"] is not None
                and result["speed_retention_mean"] >= float(thresholds.get("speed_retention_min", -10.0))
            )
        ),
        "support": result["support_violation_ratio_mean"] <= float(thresholds.get("support_violation_ratio_max", 0.6)),
    }
    if result["jump_success_rate"] is not None:
        result["checks"]["jump"] = result["jump_success_rate"] >= float(thresholds.get("jump_success", 0.8))
    if result["wave_success_rate"] is not None:
        result["checks"]["wave"] = result["wave_success_rate"] >= float(thresholds.get("wave_success", 0.8))
    if result["turn_error_worst_deg"] is not None:
        result["checks"]["turn"] = result["turn_error_worst_deg"] <= float(thresholds.get("turn_error_deg", 8.0))
    if result["recovery_time_worst_s"] is not None:
        result["checks"]["recovery"] = result["recovery_time_worst_s"] <= float(thresholds.get("recovery_time_s", 1.5))
    result["pass"] = all(result["checks"].values())
    return result


def accuracy(pred: Sequence[Any], label: Sequence[Any]) -> float:
    """分类准确率。"""
    if not pred:
        return 0.0
    return float(np.mean([p == y for p, y in zip(pred, label, strict=False)]))


def precision_recall_f1(pred: Sequence[Any], label: Sequence[Any], positive: Any = 1) -> tuple[float, float, float]:
    """二分类 P/R/F1。"""
    tp = sum(1 for p, y in zip(pred, label, strict=False) if p == positive and y == positive)
    fp = sum(1 for p, y in zip(pred, label, strict=False) if p == positive and y != positive)
    fn = sum(1 for p, y in zip(pred, label, strict=False) if p != positive and y == positive)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(precision), float(recall), float(f1)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """ECE 与可靠性曲线数据点。"""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    curve: list[dict[str, float]] = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not np.any(mask):
            continue
        confidence = float(p[mask].mean())
        accuracy_bin = float(y[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(confidence - accuracy_bin)
        curve.append({"bin": i, "confidence": confidence, "accuracy": accuracy_bin, "count": int(mask.sum())})
    return float(ece), curve


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """P50/P95/P99 延迟。"""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def decision_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """决策层全部指标：按头统计 + 校准 + 延迟 + 回退 + unsafe + 翻转率。"""
    heads = sorted({str(r["head"]) for r in records}) if records else []
    out: dict[str, Any] = {"num_records": len(records), "heads": {}}
    for head in heads:
        subset = [r for r in records if str(r["head"]) == head]
        pred = [r["prediction"] for r in subset]
        label = [r["label"] for r in subset]
        probs = [float(r.get("confidence", 0.5)) for r in subset]
        correct = [1 if p == y else 0 for p, y in zip(pred, label, strict=False)]
        ece, curve = expected_calibration_error(probs, correct)
        pr, rc, f1 = precision_recall_f1(pred, label, positive=1)
        out["heads"][head] = {
            "accuracy": accuracy(pred, label),
            "precision": pr,
            "recall": rc,
            "f1": f1,
            "ece": ece,
            "reliability": curve,
            "count": len(subset),
        }
    out["latency_ms"] = latency_percentiles([float(r.get("latency_ms", 0.0)) for r in records])
    out["fallback_rate"] = float(np.mean([bool(r.get("fallback", False)) for r in records])) if records else 0.0
    out["unsafe_decision_rate"] = float(np.mean([bool(r.get("unsafe", False)) for r in records])) if records else 0.0
    out["escalation_rate"] = float(np.mean([bool(r.get("escalation", False)) for r in records])) if records else 0.0
    out["flip_rate"] = float(np.mean([bool(r.get("flip", False)) for r in records])) if records else 0.0
    out["checks"] = {
        "unsafe_zero": out["unsafe_decision_rate"] <= 0.0,
        "p99": out["latency_ms"]["p99"] <= float(thresholds.get("p99_ms", 150.0)),
        "heads_complete": bool(heads),
    }
    out["pass"] = all(out["checks"].values())
    return out


def recovery_time(
    time_s: Sequence[float],
    tilt_deg: Sequence[float],
    *,
    onset_index: int,
    threshold_deg: float = 4.0,
    hold_s: float = 0.3,
    smooth_window_s: float = 0.0,
) -> float:
    """扰动后回到稳态（姿态 < 阈值并保持 hold_s）的用时。

    `smooth_window_s > 0` 时用滑动中值作为判据，适用于跑步等周期性摆动信号。
    """
    times = np.asarray(time_s, dtype=np.float64)
    tilt = np.asarray(tilt_deg, dtype=np.float64)
    if len(times) == 0 or onset_index >= len(times):
        return float("nan")
    signal = tilt
    if smooth_window_s > 0 and len(times) > 2:
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
        window = max(1, int(round(smooth_window_s / max(dt, 1e-6))))
        signal = np.array(
            [np.median(tilt[max(0, i - window + 1) : i + 1]) for i in range(len(tilt))], dtype=np.float64
        )
    i = onset_index
    while i < len(times):
        if signal[i] <= threshold_deg:
            j = i
            while j < len(times) and times[j] - times[i] <= hold_s and signal[j] <= threshold_deg:
                j += 1
            if j >= len(times) or times[min(j, len(times) - 1)] - times[i] >= hold_s:
                return float(times[i] - times[onset_index])
        i += 1
    return float("nan")

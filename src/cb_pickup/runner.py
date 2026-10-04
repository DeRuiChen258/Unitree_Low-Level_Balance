"""实验运行器：执行单次 episode 并汇总指标（供 baseline/ablation/eval 复用）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .controller import PickupController, PickupStepInfo
from .scenarios import PickupScenario


@dataclass
class EpisodeResult:
    """单次 episode 指标与轨迹。"""

    scenario: str
    seed: int
    success: bool
    fall: bool
    time_s: float
    steps: int
    min_margin: float
    mean_margin: float
    final_margin: float
    energy: float
    smoothness: float
    decision_latency_ms: float
    primitive_durations: dict[str, float]
    recovery_success: bool
    trajectory: dict[str, list[Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好输出（不含轨迹）。"""
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "success": self.success,
            "fall": self.fall,
            "time_s": round(self.time_s, 3),
            "steps": self.steps,
            "min_margin": round(self.min_margin, 4),
            "mean_margin": round(self.mean_margin, 4),
            "final_margin": round(self.final_margin, 4),
            "energy": round(self.energy, 1),
            "smoothness": round(self.smoothness, 3),
            "decision_latency_ms": round(self.decision_latency_ms, 3),
            "primitive_durations": {k: round(v, 2) for k, v in self.primitive_durations.items()},
            "recovery_success": self.recovery_success,
        }


def run_episode(
    controller: PickupController,
    scenario: PickupScenario,
    *,
    seed: int = 0,
    max_time_s: float = 35.0,
    randomize: bool = False,
    record: bool = False,
) -> EpisodeResult:
    """执行一次完整动作；返回指标（可选轨迹）。"""
    controller.reset(scenario, seed=seed, randomize=randomize)
    margins: list[float] = []
    latencies: list[float] = []
    primitive_time: dict[str, float] = {}
    trajectory: dict[str, list[Any]] = {}
    info: PickupStepInfo | None = None
    while controller.time < max_time_s:
        info = controller.step()
        margins.append(float(info.stability_margin))
        latencies.append(float(info.decision_latency_ms))
        primitive_time[info.primitive] = primitive_time.get(info.primitive, 0.0) + controller.config.control_dt
        if record:
            for key, value in (
                ("time", info.time),
                ("phase", info.phase),
                ("primitive", info.primitive),
                ("margin", info.stability_margin),
                ("predicted", info.balance.predicted_margin),
                ("trunk_pitch", info.trunk_pitch),
                ("com", np.asarray(info.com, dtype=np.float64)),
                ("zmp", np.asarray(info.zmp, dtype=np.float64)),
                ("left_foot", np.asarray(info.left_foot_pos, dtype=np.float64)),
                ("right_foot", np.asarray(info.right_foot_pos, dtype=np.float64)),
                ("hand", np.asarray(info.hand_position, dtype=np.float64)),
                ("object", np.asarray(info.object_position, dtype=np.float64)),
                ("support_polygon", np.asarray(info.support_polygon, dtype=np.float64)),
                ("energy", info.energy),
            ):
                trajectory.setdefault(key, []).append(value)
        if info.success or info.fall:
            break
    assert info is not None
    margin_array = np.asarray(margins) if margins else np.zeros(1)
    return EpisodeResult(
        scenario=scenario.name,
        seed=seed,
        success=bool(info.success),
        fall=bool(info.fall),
        time_s=float(info.time),
        steps=int(info.steps),
        min_margin=float(margin_array.min()),
        mean_margin=float(margin_array.mean()),
        final_margin=float(info.stability_margin),
        energy=float(info.energy),
        smoothness=float(info.smoothness),
        decision_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        primitive_durations=primitive_time,
        recovery_success=bool(info.success and not info.fall),
        trajectory=trajectory,
    )


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """聚合多次 episode 的论文式指标。"""
    if not results:
        return {"episodes": 0}
    success = np.array([r.success for r in results], dtype=np.float64)
    fall = np.array([r.fall for r in results], dtype=np.float64)
    margins = np.array([r.mean_margin for r in results], dtype=np.float64)
    min_margins = np.array([r.min_margin for r in results], dtype=np.float64)
    steps = np.array([r.steps for r in results], dtype=np.float64)
    latency = np.array([r.decision_latency_ms for r in results], dtype=np.float64)
    energy = np.array([r.energy for r in results], dtype=np.float64)
    duration = np.array([r.time_s for r in results], dtype=np.float64)
    return {
        "episodes": len(results),
        "pickup_success_rate": float(success.mean()),
        "fall_rate": float(fall.mean()),
        "recovery_success_rate": float(np.mean([r.recovery_success for r in results])),
        "mean_stability_margin": float(margins.mean()),
        "min_stability_margin": float(min_margins.min()),
        "mean_steps": float(steps.mean()),
        "mean_decision_latency_ms": float(latency.mean()),
        "mean_episode_duration_s": float(duration.mean()),
        "mean_energy": float(energy.mean()),
        "success_std": float(success.std()),
    }

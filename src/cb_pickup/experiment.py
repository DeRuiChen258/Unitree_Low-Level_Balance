"""实验记录：experiments/<timestamp>/ 全量产物（第 21 节）。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .visualization import plot_pickup_dashboard


def _git_commit() -> str:
    """返回 git commit；非 git 仓库时返回 source-tree hash。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, cwd=Path(__file__).resolve().parents[2]
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - 无 git 时降级
        pass
    digest = hashlib.sha256()
    src = Path(__file__).resolve().parents[1]
    for path in sorted(src.rglob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()[:16]}"


@dataclass
class ExperimentLogger:
    """采集轨迹/决策/事件并生成 metrics.json、trajectory.npz、plots/、README.md。"""

    root: str | Path = "experiments"
    tag: str = "pickup"
    config: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    scenario: str = "front"
    versions: Mapping[str, Any] = field(default_factory=dict)
    run_dir: Path = field(init=False)
    trajectories: dict[str, list[Any]] = field(default_factory=dict, init=False)
    decisions: list[dict[str, Any]] = field(default_factory=list, init=False)
    events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _start: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.root) / f"{stamp}_{self.tag}"
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "video").mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": self.tag,
            "seed": self.seed,
            "scenario": self.scenario,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "versions": dict(self.versions),
            "config": dict(self.config),
        }
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(_yaml_safe(payload), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def log_step(self, info: Mapping[str, Any], *, decision: Mapping[str, Any] | None = None, event: Mapping[str, Any] | None = None) -> None:
        """记录一帧轨迹 +（可选）决策/事件。"""
        for key in (
            "time",
            "phase",
            "primitive",
            "stability_margin",
            "predicted_margin",
            "trunk_pitch",
            "trunk_roll",
            "hand_distance",
            "object_height",
            "energy",
            "steps",
            "decision_latency_ms",
        ):
            if key in info:
                self.trajectories.setdefault(key, []).append(info[key])
        for key in ("com", "zmp", "left_foot", "right_foot", "hand_position", "object_position"):
            if key in info:
                self.trajectories.setdefault(key, []).append(np.asarray(info[key], dtype=np.float64))
        if "support_polygon" in info:
            self.trajectories.setdefault("support_polygon", []).append(np.asarray(info["support_polygon"], dtype=np.float64))
        if decision is not None:
            record = {"time": float(info.get("time", 0.0)), **{k: v for k, v in decision.items()}}
            self.decisions.append(record)
        if event is not None:
            self.events.append({"time": float(info.get("time", 0.0)), **{k: v for k, v in event.items()}})

    def finalize(self, metrics: Mapping[str, Any], *, make_plots: bool = True) -> Path:
        """落盘 metrics/trajectory/decision/events/plots/README。"""
        payload = dict(metrics)
        payload.setdefault("seed", self.seed)
        payload.setdefault("scenario", self.scenario)
        payload.setdefault("wall_s", round(time.time() - self._start, 2))
        payload.setdefault("git_commit", _git_commit())
        payload.setdefault("versions", dict(self.versions))
        (self.run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        arrays: dict[str, np.ndarray] = {}
        for key, values in self.trajectories.items():
            try:
                arrays[key] = np.asarray(values)
            except ValueError:
                arrays[key] = np.asarray(values, dtype=object)
        np.savez_compressed(self.run_dir / "trajectory.npz", **arrays)
        with (self.run_dir / "decision.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.decisions:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        with (self.run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.events:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        if make_plots:
            try:
                plot_pickup_dashboard(arrays, self.run_dir / "plots")
            except Exception as exc:  # noqa: BLE001 - 绘图失败不阻塞实验
                (self.run_dir / "plots" / "error.txt").write_text(str(exc), encoding="utf-8")
        readme = [
            f"# Pickup Experiment `{self.run_dir.name}`",
            "",
            f"- scenario: `{self.scenario}`",
            f"- seed: `{self.seed}`",
            f"- git: `{payload.get('git_commit')}`",
            f"- metrics: `{json.dumps({k: payload.get(k) for k in ('success', 'fall', 'steps', 'min_stability_margin', 'recovery_success')}, ensure_ascii=False)}`",
            "",
            "文件：config.yaml / metrics.json / trajectory.npz / decision.jsonl / events.jsonl / plots/ / video/",
        ]
        (self.run_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        return self.run_dir


def _json_default(value: Any) -> Any:
    """numpy → JSON。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def _yaml_safe(value: Any) -> Any:
    """numpy/Path → YAML 可序列化类型（递归）。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value

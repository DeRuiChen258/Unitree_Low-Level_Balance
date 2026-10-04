#!/usr/bin/env python3
"""运行 baseline A/B/C/D 并输出对比表（新增需求第 16 节）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_pickup.baselines import configure_baseline
from cb_pickup.controller import PickupConfig, PickupController
from cb_pickup.runner import run_episode, summarize
from cb_pickup.scenarios import SCENARIOS


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="experiments/baselines")
    parser.add_argument("--max-time", type=float, default=35.0)
    args = parser.parse_args()
    pickup_cfg = load_config("configs/g1_pickup.yaml")
    scene = str(pickup_cfg.get("scene_path"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    rows: list[dict] = []
    for baseline in args.baselines:
        results = []
        for scenario_name in args.scenarios:
            for seed in range(args.seeds):
                controller = PickupController(
                    PickupConfig.from_mapping(pickup_cfg, scene_path=scene, max_episode_s=args.max_time),
                )
                configure_baseline(controller, baseline)
                result = run_episode(controller, SCENARIOS[scenario_name], seed=seed, max_time_s=args.max_time)
                results.append(result)
        summary = summarize(results)
        table[baseline] = summary
        for result in results:
            rows.append({"baseline": baseline, **result.to_dict()})
    (out / "baselines.json").write_text(json.dumps({"summary": table, "episodes": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "| Baseline | Pickup Success | Fall Rate | Mean Margin | Min Margin | Steps | Latency(ms) | Duration(s) | Energy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for baseline, summary in table.items():
        lines.append(
            f"| {baseline} | {summary['pickup_success_rate']:.2f} | {summary['fall_rate']:.2f} | "
            f"{summary['mean_stability_margin']:.3f} | {summary['min_stability_margin']:.3f} | "
            f"{summary['mean_steps']:.1f} | {summary['mean_decision_latency_ms']:.2f} | "
            f"{summary['mean_episode_duration_s']:.1f} | {summary['mean_energy']:.0f} |"
        )
    (out / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

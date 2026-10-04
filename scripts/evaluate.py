#!/usr/bin/env python3
"""G1 拾取系统总评测：4 baseline × 10 场景 × seeds → 论文式表格。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_pickup.baselines import configure_baseline
from cb_pickup.controller import PickupConfig, PickupController
from cb_pickup.runner import run_episode, summarize
from cb_pickup.scenarios import SCENARIOS, sample_scenario


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="experiments/evaluation")
    parser.add_argument("--max-time", type=float, default=35.0)
    args = parser.parse_args()
    pickup_cfg = load_config("configs/g1_pickup.yaml")
    scene = str(pickup_cfg.get("scene_path"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for baseline in args.baselines:
        results = []
        for scenario_name in args.scenarios:
            for seed in range(args.seeds):
                controller = PickupController(PickupConfig.from_mapping(pickup_cfg, scene_path=scene, max_episode_s=args.max_time))
                configure_baseline(controller, baseline)
                rng = __import__("numpy").random.default_rng(seed)
                scenario = sample_scenario(scenario_name, rng)
                result = run_episode(controller, scenario, seed=seed, max_time_s=args.max_time)
                results.append(result)
                rows.append({"baseline": baseline, "scenario": scenario_name, **result.to_dict()})
        summary[baseline] = summarize(results)
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Baseline | Pickup Success Rate | Fall Rate | Recovery Success | Mean Margin | Min Margin | Steps | Decision Latency (ms) | Duration (s) | Energy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for baseline, item in summary.items():
        lines.append(
            f"| {baseline} | {item['pickup_success_rate']:.2f} | {item['fall_rate']:.2f} | "
            f"{item['recovery_success_rate']:.2f} | {item['mean_stability_margin']:.3f} | "
            f"{item['min_stability_margin']:.3f} | {item['mean_steps']:.1f} | "
            f"{item['mean_decision_latency_ms']:.2f} | {item['mean_episode_duration_s']:.1f} | {item['mean_energy']:.0f} |"
        )
    (out / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

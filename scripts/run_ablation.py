#!/usr/bin/env python3
"""运行消融 1–10 并输出对比表（新增需求第 17 节）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_pickup.ablation import ABLATIONS, apply_ablation
from cb_pickup.controller import PickupConfig, PickupController
from cb_pickup.runner import run_episode, summarize
from cb_pickup.scenarios import SCENARIOS


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablations", nargs="+", default=[str(i) for i in range(1, 11)])
    parser.add_argument("--scenarios", nargs="+", default=["front", "left", "right", "far", "deep"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="experiments/ablation")
    args = parser.parse_args()
    pickup_cfg = load_config("configs/g1_pickup.yaml")
    scene = str(pickup_cfg.get("scene_path"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    for name in args.ablations:
        config = ABLATIONS[str(name)]
        results = []
        for scenario_name in args.scenarios:
            for seed in range(args.seeds):
                controller = PickupController(PickupConfig.from_mapping(pickup_cfg, scene_path=scene))
                apply_ablation(controller, config)
                results.append(run_episode(controller, SCENARIOS[scenario_name], seed=seed))
        table[str(name)] = {**summarize(results), "description": config.description}
    (out / "ablation.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "| Ablation | Description | Success | Fall | Mean Margin | Min Margin | Steps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in table.items():
        lines.append(
            f"| {name} | {summary['description']} | {summary['pickup_success_rate']:.2f} | "
            f"{summary['fall_rate']:.2f} | {summary['mean_stability_margin']:.3f} | "
            f"{summary['min_stability_margin']:.3f} | {summary['mean_steps']:.1f} |"
        )
    (out / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

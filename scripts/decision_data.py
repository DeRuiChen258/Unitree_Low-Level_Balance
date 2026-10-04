#!/usr/bin/env python3
"""构建决策数据集（仿真 rollout + 规则专家标签，薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_decision.dataset import build_decision_dataset


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--out", default="runs/decision/data_v1")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--decision-every", type=int, default=4)
    args = parser.parse_args()
    summary = build_decision_dataset(
        load_config(args.config),
        load_config(args.system_config),
        load_config(args.reward_config),
        output_dir=args.out,
        seeds=args.seeds,
        decision_every=args.decision_every,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""训练入口（PPO/AMP/蒸馏，薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_train.distill import distill
from cb_train.train_ppo import train


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--mode", choices=["ppo", "distill"], default="ppo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="distill 模式的教师/残差策略 checkpoint")
    args = parser.parse_args()
    train_cfg = load_config(args.config)
    system_cfg = load_config(args.system_config)
    reward_cfg = load_config(args.reward_config)
    if args.mode == "ppo":
        result = train(
            args.config,
            system_config_path=args.system_config,
            reward_config_path=args.reward_config,
            seed=args.seed,
            updates=args.updates,
            num_envs=args.num_envs,
            run_dir=args.run_dir,
        )
        print(json.dumps(result.summary, indent=2, ensure_ascii=False))
        return 0
    policy = None
    if args.checkpoint:
        from cb_policy.balance_policy import TorchPolicy

        policy = TorchPolicy(args.checkpoint, device="cpu")
    out_dir = Path(args.run_dir or (Path(str(system_cfg.get("paths.runs_dir", "runs"))) / "exp" / "distill"))
    result = distill(
        train_cfg,
        system_cfg,
        reward_cfg,
        output_dir=out_dir,
        seed=int(args.seed or 1),
        policy=policy,
    )
    print(json.dumps(result.report, indent=2, ensure_ascii=False))
    return 0 if result.report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

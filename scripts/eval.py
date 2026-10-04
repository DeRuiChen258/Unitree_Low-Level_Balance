#!/usr/bin/env python3
"""策略评测（多场景/多扰动/多 seed，薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_common.logging import JsonlLogger, new_run_id
from cb_eval.rollout_eval import EvalConfig, evaluate_policy
from cb_policy.balance_policy import OnnxPolicy, TorchPolicy


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--policy", default=None, help="onnx/ckpt 路径；缺省为零残差基线")
    parser.add_argument("--policy-mode", choices=["residual", "absolute"], default="residual")
    parser.add_argument("--scenarios", nargs="+", default=["run_jump", "run_wave", "turn_wave"])
    parser.add_argument("--profiles", nargs="+", default=["none", "push", "noise", "delay"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/reports")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    train_cfg = load_config(args.config)
    system_cfg = load_config(args.system_config)
    reward_cfg = load_config(args.reward_config)
    policy = None
    label = args.label or "baseline"
    if args.policy:
        policy = OnnxPolicy(args.policy) if args.policy.endswith(".onnx") else TorchPolicy(args.policy)
        label = args.label or Path(args.policy).stem
    cfg = EvalConfig(
        scenarios=tuple(args.scenarios),
        profiles=tuple(args.profiles),
        seeds=int(args.seeds),
        thresholds=dict(train_cfg.get("thresholds", {})),
        output_dir=args.output_dir,
        policy_mode=args.policy_mode,
    )
    logger = JsonlLogger(
        Path(args.output_dir) / f"eval-{new_run_id('log')}.jsonl",
        event_types=["eval_scenario"],
        module="scripts.eval",
    )
    report = evaluate_policy(
        train_cfg=train_cfg,
        system_cfg=system_cfg,
        reward_cfg=reward_cfg,
        config=cfg,
        policy=policy,
        label=label,
        logger=logger,
    )
    logger.close()
    print(json.dumps({"run_id": report["run_id"], "pass": report["pass"], "overall": report["overall"]}, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

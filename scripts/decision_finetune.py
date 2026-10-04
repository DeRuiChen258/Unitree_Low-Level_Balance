#!/usr/bin/env python3
"""决策头校准/微调 + 四头指标（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_decision.finetune import finetune_decisions


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="runs/decision/data_v1")
    parser.add_argument("--out", default="runs/decision/v1")
    parser.add_argument("--model-config", default="configs/laya/model.yaml")
    parser.add_argument("--questions", default="configs/laya/questions.json")
    parser.add_argument("--calibration-config", default="configs/laya/calibration.yaml")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    result = finetune_decisions(
        data_dir=args.data,
        out_dir=args.out,
        model_config_path=args.model_config,
        questions_path=args.questions,
        calibration_path=args.calibration_config,
        max_samples=args.max_samples,
        use_model=not args.no_model,
    )
    print(json.dumps(result.report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

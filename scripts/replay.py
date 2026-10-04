#!/usr/bin/env python3
"""轨迹/决策回放（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_eval.replay import replay_jsonl


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="包含 state/target 字段的 JSONL")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = replay_jsonl(args.log)
    payload = {
        "frames": result.frames,
        "max_target_diff": result.max_target_diff,
        "max_state_diff": result.max_state_diff,
        "consistent": result.consistent,
        "mismatches": result.mismatches[:10],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())

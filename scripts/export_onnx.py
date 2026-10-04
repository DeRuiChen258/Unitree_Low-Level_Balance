#!/usr/bin/env python3
"""导出 ONNX + 一致性校验（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_train.export_onnx import export_checkpoint


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.out)
    print(json.dumps({"onnx": result.onnx_path, "max_abs_diff": result.max_abs_diff, "metadata": result.metadata}, indent=2, ensure_ascii=False))
    return 0 if result.max_abs_diff <= 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())

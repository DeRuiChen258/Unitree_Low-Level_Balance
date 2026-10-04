#!/usr/bin/env python3
"""执行第 4 节 8 步数据管线（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_common.logging import JsonlLogger
from cb_data.pipeline import run_data_pipeline


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--profile", default=None, help="覆盖 data.yaml 的 profile")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()
    data_cfg = load_config(args.config, required=["profile", "sources", "manifest.dataset_version"])
    system_cfg = load_config(args.system_config, required=["limits.mjcf_scene", "feature_version"])
    overrides = {}
    if args.profile:
        overrides["profile"] = args.profile
    if args.max_clips:
        overrides.setdefault("profiles", {})
    if overrides:
        data_cfg = data_cfg.with_overrides(overrides)
    if args.max_clips:
        profile = str(data_cfg.get("profile"))
        data_cfg = data_cfg.with_overrides({"profiles": {profile: {**dict(data_cfg.get(f"profiles.{profile}", {})), "max_clips": args.max_clips}}})
    logger = JsonlLogger(
        Path(str(system_cfg.get("paths.runs_dir", "runs"))) / "logs" / "data_build.jsonl",
        event_types=["data_pipeline"],
        module="scripts.data_build",
    )
    result = run_data_pipeline(data_cfg, system_cfg, output_root=args.output_dir, logger=logger)
    logger.close()
    print(json.dumps({"manifest": str(result.root / "manifest.json"), "segments": len(result.accepted), "rejected": len(result.rejected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

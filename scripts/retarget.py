#!/usr/bin/env python3
"""单文件重定向调试：输出 IK 误差与限位触碰（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_data.canonicalize import canonicalize
from cb_data.ingest import load_272d_clip
from cb_data.retarget import G1IKSolver, IKConfig, retarget_clip


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_cfg = load_config(args.data_config)
    system_cfg = load_config(args.system_config)
    root = Path(str(data_cfg.get("sources.humanml3d_272d.root")))
    clip = load_272d_clip(root, args.clip_id)
    canonical = canonicalize(clip, data_cfg.section("canonicalize").to_dict())
    ik_cfg = IKConfig(
        scene_path=str(system_cfg.get("limits.mjcf_scene")),
        ik_iters=int(data_cfg.get("retarget.ik_iters", 40)),
        ik_tolerance_m=float(data_cfg.get("retarget.ik_tolerance_m", 0.06)),
        ik_reject_m=float(data_cfg.get("retarget.ik_reject_m", 0.16)),
    )
    result = retarget_clip(canonical, ik_cfg, solver=G1IKSolver(ik_cfg))
    output = {
        "retarget_id": result.retarget_id,
        "frames": int(result.qpos.shape[0]),
        "ik_err_mean_m": float(np.mean(result.retarget_err)),
        "ik_err_p95_m": float(np.percentile(result.retarget_err, 95)),
        "ik_err_max_m": float(np.max(result.retarget_err)),
        "joint_limit_hits": int(np.sum(result.joint_limit_hits)),
    }
    if args.out:
        np.savez_compressed(args.out, **result.to_arrays())
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["ik_err_mean_m"] <= ik_cfg.ik_reject_m else 1


if __name__ == "__main__":
    raise SystemExit(main())

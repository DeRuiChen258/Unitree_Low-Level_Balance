#!/usr/bin/env python3
"""数据质量报告与抽检（薄 CLI）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_data.manifest import load_manifest
from cb_data.storage import load_segments_from_dir
from cb_data.visualize import plot_segment


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-plots", type=int, default=5)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    problems = manifest.verify_files(root)
    segments = load_segments_from_dir(root / "segments") if (root / "segments").exists() else []
    for segment in segments[: args.num_plots]:
        plot_segment(segment, root / "viz" / f"{segment.segment_id.replace(':', '_')}.png")
    summary = {
        "dataset_version": manifest.dataset_version,
        "stats": manifest.stats,
        "reject_stats": manifest.reject_stats,
        "file_problems": problems,
        "plots": min(args.num_plots, len(segments)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""启动决策服务（薄 CLI）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_serving.api import create_app


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--local", action="store_true", help="使用本地规则后端（不加载 LayA）")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local:
        from cb_serving.api import LocalCalibratedBackend

        app = create_app(args.config, backend=LocalCalibratedBackend())
    else:
        app = create_app(args.config)
    import uvicorn

    uvicorn.run(app, host=args.host or str(cfg.get("host", "127.0.0.1")), port=int(args.port or cfg.get("port", 8765)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

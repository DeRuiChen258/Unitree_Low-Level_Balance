"""JSON + Markdown 报告生成（模板稳定、可 diff）。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_report(path: str | Path, title: str, payload: Mapping[str, Any], *, summary_keys: tuple[str, ...] = ()) -> Path:
    """写 JSON + Markdown 双份报告。"""
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in summary_keys:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines += ["", "```json", json.dumps(payload, indent=2, ensure_ascii=False)[:8000], "```"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path

"""轨迹/决策回放：离线逐帧复现线上行为（第 8.4 节）。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReplayResult:
    """回放结果。"""

    frames: int
    max_target_diff: float
    max_state_diff: float
    consistent: bool
    mismatches: list[dict[str, Any]] = field(default_factory=list)


def replay_jsonl(path: str | Path, *, target_tolerance: float = 1e-6, state_tolerance: float = 1e-6) -> ReplayResult:
    """回放 JSONL：对相邻记录的 target/state 做一致性核对。

    线上日志应包含 `target`（关节目标）与 `state`（观测/状态摘要）字段；
    本函数验证「同一 state 的重复记录给出相同 target」，并统计差异。
    """
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return ReplayResult(0, 0.0, 0.0, False, [{"reason": "empty_log"}])
    max_target_diff = 0.0
    max_state_diff = 0.0
    mismatches: list[dict[str, Any]] = []
    by_hash: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        digest = str(record.get("state_hash", ""))
        if digest and digest in by_hash:
            previous = by_hash[digest]
            diff = _array_diff(record.get("target"), previous.get("target"))
            if diff > target_tolerance:
                mismatches.append({"index": index, "state_hash": digest, "target_diff": diff})
        by_hash[digest] = record
        if index > 0:
            max_state_diff = max(max_state_diff, _array_diff(record.get("state"), records[index - 1].get("state")))
    return ReplayResult(
        frames=len(records),
        max_target_diff=max_target_diff,
        max_state_diff=max_state_diff,
        consistent=not mismatches,
        mismatches=mismatches,
    )


def _array_diff(a: Any, b: Any) -> float:
    """数组最大绝对差；缺失返回 0。"""
    if a is None or b is None:
        return 0.0
    arr_a, arr_b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        return float("inf")
    return float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0

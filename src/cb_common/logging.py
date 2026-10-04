"""结构化 JSONL 日志（事件类型固定，字段齐全，可回放）。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

REQUIRED_FIELDS: tuple[str, ...] = ("ts", "level", "event", "module", "run_id")


def new_run_id(prefix: str = "run") -> str:
    """生成可排序的运行 id：`<prefix>-<utc>-<rand>`。"""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class JsonlLogger:
    """线程安全的 JSONL 事件日志；未注册事件类型直接拒绝。"""

    path: Path
    event_types: Iterable[str]
    module: str
    run_id: str = field(default_factory=new_run_id)
    level: str = "INFO"
    _allowed: set[str] = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _fh: TextIO | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._allowed = set(self.event_types)
        if not self._allowed:
            raise ValueError("event_types must not be empty")
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: str, /, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        """写入一条事件；返回落盘记录（便于测试断言）。"""
        if event not in self._allowed:
            raise ValueError(f"unregistered event type {event!r}; allowed={sorted(self._allowed)}")
        record = {
            "ts": round(time.time(), 6),
            "level": level.upper(),
            "event": event,
            "module": self.module,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self._lock:
            assert self._fh is not None
            self._fh.write(line + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        """关闭文件句柄（可重复调用）。"""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover - numpy 是硬依赖
        pass
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def configure_root_logger(level: str = "INFO") -> None:
    """配置标准 logging（供第三方库与 CLI 使用）。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

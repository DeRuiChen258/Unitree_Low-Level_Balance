"""LayA 适配器：RLAgent.system_one / Router 封装，超时与 fail-closed。"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cb_common.config import Config, load_config
from cb_common.errors import DecisionError

_AGENT_CACHE: dict[tuple[str, str, str], Any] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class LayAAdapter:
    """LayA 决策模型适配器（惰性加载 + 单飞锁 + 超时）。"""

    model_config_path: str = "configs/laya/model.yaml"
    device: str | None = None
    config: Config = field(init=False)
    agent: Any = field(default=None, init=False, repr=False)
    available: bool = field(default=False, init=False)
    failure_reason: str = field(default="", init=False)
    load_time_s: float = field(default=0.0, init=False)
    warmup_ms: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = load_config(self.model_config_path)

    @property
    def checkpoint(self) -> str:
        """当前主 checkpoint 名称。"""
        return str(self.config.get("primary", "typed-decisions"))

    def _checkpoint_dir(self, name: str) -> Path:
        entry = self.config.get(f"checkpoints.{name}")
        if not isinstance(entry, Mapping):
            raise DecisionError("checkpoint config missing", checkpoint=name)
        return Path(str(entry["dir"]))

    def load(self) -> bool:
        """加载 checkpoint；失败时记录原因并返回 False（显式降级）。"""
        if self.available:
            return True
        with self._lock:
            if self.available:
                return True
            start = time.time()
            try:
                from safetensors.torch import load_file  # noqa: F401  # 依赖检查
                from transformers import AutoTokenizer  # noqa: F401

                root = self._checkpoint_dir(self.checkpoint)
                if not root.is_dir():
                    raise DecisionError("checkpoint directory not found", path=str(root))
                module_path = root / "rl_agent_api.py"
                if not module_path.is_file():
                    # LayA 仓库布局：模型代码在 checkpoint 的上一级目录
                    parent = root.parent
                    candidate = parent / "rl_agent_api.py"
                    if candidate.is_file():
                        module_path = candidate
                        if str(parent) not in sys.path:
                            sys.path.insert(0, str(parent))
                    else:
                        raise DecisionError("rl_agent_api.py not found", path=str(root))
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                key = (str(root), str(root / "model.safetensors"), str(self.device or self.config.get("device", "cuda")))
                agent = _AGENT_CACHE.get(key)
                if agent is None:
                    spec = importlib.util.spec_from_file_location(f"cb_laya_agent_{abs(hash(key))}", module_path)
                    if spec is None or spec.loader is None:
                        raise DecisionError("failed to load rl_agent_api.py", path=str(module_path))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    old_cwd = Path.cwd()
                    try:
                        import os

                        os.chdir(root)
                        agent = module.RLAgent(str(root), device=self.device or str(self.config.get("device", "cuda")))
                    finally:
                        os.chdir(old_cwd)
                    _AGENT_CACHE[key] = agent
                self.agent = agent
                # 预热：首次前向包含 CUDA 内核/注意力后端初始化，必须排除在服务延迟之外
                warmup_start = time.time()
                dummy = {
                    "warmup_head": {
                        "type": "noul",
                        "instructions": "warmup",
                        "criteria": {"false": "no", "true": "yes"},
                    }
                }
                for _ in range(3):
                    agent.system_one({"state_version": "1.0"}, dummy)
                self.warmup_ms = (time.time() - warmup_start) * 1000.0 / 3.0
                self.available = True
            except Exception as exc:  # noqa: BLE001 - 明确记录降级原因
                self.failure_reason = f"{type(exc).__name__}: {exc}"
                self.available = False
            finally:
                self.load_time_s = time.time() - start
        return self.available

    def system_one(self, state: str | Mapping[str, Any], questions: Mapping[str, Any], *, timeout_ms: float | None = None) -> dict[str, Any]:
        """调用 LayA；未加载/超时/异常一律抛 DecisionError（由调用方 fail-closed）。"""
        if not self.available and not self.load():
            raise DecisionError("LayA model unavailable", reason=self.failure_reason)
        timeout = float(timeout_ms if timeout_ms is not None else self.config.get("timeout_ms", 500)) / 1000.0
        result: dict[str, Any] = {}
        error: list[BaseException] = []

        def _run() -> None:
            try:
                result.update(self.agent.system_one(state, questions))  # type: ignore[union-attr]
            except BaseException as exc:  # noqa: BLE001 - 线程内异常转交
                error.append(exc)

        thread = threading.Thread(target=_run, name="laya-infer", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # 超时后禁用适配器，避免线程堆积；由上层 fail-closed 到规则后端
            self.available = False
            self.failure_reason = "inference_timeout"
            raise DecisionError("LayA inference timeout", timeout_ms=timeout * 1000)
        if error:
            raise DecisionError("LayA inference failed", error=str(error[0]))
        return result


def adapter_status(adapter: LayAAdapter | None = None) -> dict[str, Any]:
    """返回适配器状态（服务 /v1/version 与报告使用）。"""
    adapter = adapter or LayAAdapter()
    return {
        "checkpoint": adapter.checkpoint,
        "available": adapter.available,
        "failure_reason": adapter.failure_reason,
        "load_time_s": adapter.load_time_s,
    }

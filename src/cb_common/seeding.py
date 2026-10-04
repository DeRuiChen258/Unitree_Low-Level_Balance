"""全局随机种子与确定性开关。"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """设置 python/numpy/torch 随机种子，返回记录（可写入实验 meta）。"""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    record: dict[str, Any] = {"seed": seed, "deterministic": deterministic, "torch": False, "cuda": False}
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        record["torch"] = True
        record["cuda"] = bool(torch.cuda.is_available())
        record["torch_version"] = torch.__version__
    except ImportError:  # pragma: no cover - torch 为硬依赖
        pass
    return record


def rng(seed: int) -> np.random.Generator:
    """返回独立 numpy 生成器（避免污染全局状态）。"""
    return np.random.default_rng(seed)

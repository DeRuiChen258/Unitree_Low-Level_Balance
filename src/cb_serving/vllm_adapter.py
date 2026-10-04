"""vLLM 适配器占位（仅生成式模型时启用，默认不导入 vLLM）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cb_common.errors import DecisionError


@dataclass
class VllmAdapter:
    """统一决策后端形状的 vLLM 适配器接口。

    说明：LayA 是编码器分类模型，vLLM 面向自回归生成调度；默认实现不启用 vLLM。
    仅当后续接入生成式动作模型或统一 LLM 网关时，才在部署配置中启用本适配器。
    """

    endpoint: str = ""
    enabled: bool = False
    name: str = "vllm"

    def ready(self) -> bool:
        """未启用时明确返回 False。"""
        return bool(self.enabled and self.endpoint)

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
        """未启用/未实现时抛错，禁止静默返回默认动作。"""
        if not self.ready():
            raise DecisionError("vLLM adapter is disabled; use FastAPI+ONNX/LayA backend")
        raise DecisionError("vLLM adapter is an interface placeholder; implement for generative action models")

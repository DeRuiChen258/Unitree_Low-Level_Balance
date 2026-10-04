"""DecisionClient 抽象：HTTP / 本地，上层不感知后端。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DecisionClient(Protocol):
    """决策客户端协议。"""

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> dict[str, Any]:  # pragma: no cover
        """返回 typed answers。"""
        ...


@dataclass
class HttpDecisionClient:
    """HTTP 客户端（requests 风格，使用标准库 urllib 避免额外依赖）。"""

    base_url: str = "http://127.0.0.1:8765"
    timeout_s: float = 1.0

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """POST /v1/decision。"""
        import json
        import urllib.request

        payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/decision",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass
class LocalDecisionClient:
    """本地后端客户端（同进程，无网络）。"""

    backend: Any

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """直接调用后端。"""
        from cb_decision.questions import load_questions

        question_set = questions or load_questions("configs/laya/questions.json").for_agent()
        return self.backend.decide(state, question_set)


class RecordingPolicy:
    """把任意 policy 包成带记录函数（demo/replay 使用）。"""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.records: list[dict[str, Any]] = []

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """调用并记录动作。"""
        action = self.policy(obs) if callable(self.policy) else self.policy.act(obs)
        self.records.append({"obs_hash": int(np.sum(obs) * 1e6), "action": np.asarray(action).tolist()})
        return np.asarray(action, dtype=np.float64)

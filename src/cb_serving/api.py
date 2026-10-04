"""FastAPI 决策服务（第 9.1 节：单条/批量/health/version，fail-closed）。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from cb_common.config import Config, load_config
from cb_common.errors import DecisionError
from cb_decision.adapter import LayAAdapter
from cb_decision.questions import load_questions


class DecisionRequest(BaseModel):
    """单条决策请求。"""

    state: dict[str, Any] | str = Field(..., description="文本化状态或状态 JSON")
    questions: dict[str, Any] | None = None


class BatchRequest(BaseModel):
    """批量决策请求。"""

    items: list[DecisionRequest] = Field(..., min_length=1)


class DecisionBackend(Protocol):
    """决策后端协议（LayA / ONNX / 本地校准 / vLLM 统一形状）。"""

    name: str

    def ready(self) -> bool:  # pragma: no cover - 协议
        """模型是否可用。"""
        ...

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
        """返回 typed answers 结构。"""
        ...


@dataclass
class LayABackend:
    """LayA 真模型后端。"""

    model_config_path: str = "configs/laya/model.yaml"
    name: str = "laya"
    adapter: LayAAdapter | None = None

    def __post_init__(self) -> None:
        self.adapter = self.adapter or LayAAdapter(self.model_config_path)

    def ready(self) -> bool:
        """惰性加载并返回可用性。"""
        assert self.adapter is not None
        return self.adapter.available or self.adapter.load()

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
        """调用 LayA；异常向上抛，由 API 转 5xx（不返回默认动作）。"""
        assert self.adapter is not None
        return self.adapter.system_one(state, questions)


@dataclass
class LocalCalibratedBackend:
    """规则/校准后端：模型不可用时的显式降级路径（记录 fallback_reason）。"""

    name: str = "local_calibrated"
    calibration_path: str = "configs/laya/calibration.yaml"
    fallback_reason: str = ""
    _config: Config | None = None

    def __post_init__(self) -> None:
        self._config = load_config(self.calibration_path)

    def ready(self) -> bool:
        """规则后端始终可用。"""
        return True

    def decide(self, state: str | Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
        """基于文本状态中的 balance 字段给出保守答案。"""
        payload = json.loads(state) if isinstance(state, str) else dict(state)
        balance = dict(payload.get("balance", {}))
        roll = abs(float(balance.get("roll_deg", 0.0)))
        pitch = abs(float(balance.get("pitch_deg", 0.0)))
        margin = float(balance.get("com_margin_m", 0.0))
        rules = dict((self._config or Config({})).get("rules", {}))
        veto = roll > float(rules.get("roll_abort_deg", 8.0)) or pitch > float(rules.get("pitch_abort_deg", 8.0)) or margin < float(
            rules.get("com_margin_min_m", -0.02)
        )
        motion = dict(questions.get("motion_primitive_head", {}))
        candidates = list(dict(motion.get("criteria", {})).keys())
        choice = "safe_stop" if veto and "safe_stop" in candidates else (candidates[0] if candidates else "continue_current")
        confidence = 0.9 if veto else 0.6
        return {
            "model": "local_calibrated",
            "answers": {
                "motion_primitive_head": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {choice: confidence},
                    "confidence": confidence,
                },
                "safety_veto_head": {"type": "noul", "noul": 1.0 if veto else 0.05},
                "recovery_head": {
                    "type": "choice",
                    "choice": "stand_recover" if veto else "none",
                    "probabilities": {"stand_recover" if veto else "none": 0.8},
                    "confidence": 0.8,
                },
                "escalation_head": {"type": "noul", "noul": 1.0 if veto else 0.0},
            },
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "fallback_reason": self.fallback_reason,
        }


def create_app(
    config_path: str = "configs/serving.yaml",
    *,
    backend: DecisionBackend | None = None,
) -> Any:
    """构造 FastAPI app（测试可注入 backend）。"""
    from fastapi import FastAPI, HTTPException

    cfg = load_config(config_path, required=["host", "port", "timeout_ms"])
    questions = load_questions(str(cfg.get("questions_file", "configs/laya/questions.json")))
    if backend is None:
        model_backend = str(cfg.get("model_backend", "laya"))
        backend = LayABackend() if model_backend == "laya" else LocalCalibratedBackend()

    app = FastAPI(title="G1 cerebellum decision service", version="1.0.0")
    version_info = {
        "model": str(cfg.get("version.model", "laya-typed-decisions")),
        "checkpoint": str(cfg.get("version.checkpoint", "typed-decisions")),
        "feature_version": str(cfg.get("version.feature_version", "1.0")),
        "questions_version": questions.version,
        "backend": backend.name,
    }

    def _decide_one(item: DecisionRequest) -> dict[str, Any]:
        request_questions = item.questions or questions.for_agent()
        start = time.time()
        if not backend.ready():  # pragma: no cover - 依赖模型可用性
            raise HTTPException(status_code=503, detail="decision backend unavailable")
        try:
            result = backend.decide(item.state, request_questions)
        except DecisionError as exc:
            raise HTTPException(status_code=504 if "timeout" in str(exc).lower() else 503, detail=exc.message) from exc
        result.setdefault("meta", {})
        result["meta"].update(
            {
                "latency_ms": (time.time() - start) * 1000.0,
                "backend": backend.name,
                "questions_hash": questions.hash(),
                **version_info,
            }
        )
        return result

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        """健康检查：模型是否加载。"""
        return {"status": "ok", "model_loaded": bool(backend.ready()), "backend": backend.name}

    @app.get("/v1/version")
    def version() -> dict[str, Any]:
        """版本信息（模型/checkpoint/feature/questions）。"""
        return dict(version_info)

    @app.post("/v1/decision")
    def decision(item: DecisionRequest) -> dict[str, Any]:
        """单条决策。"""
        return _decide_one(item)

    @app.post("/v1/decision:batch")
    def decision_batch(batch: BatchRequest) -> dict[str, Any]:
        """批量决策（顺序执行，保证延迟上界；生产可替换为批处理池）。"""
        started = time.time()
        results = [_decide_one(item) for item in batch.items]
        return {"results": results, "meta": {"batch_size": len(results), "latency_ms": (time.time() - started) * 1000.0}}

    return app

"""服务化：FastAPI 决策服务、客户端抽象、vLLM 适配器占位。"""

from .api import DecisionBackend, LayABackend, LocalCalibratedBackend, create_app
from .client import DecisionClient, HttpDecisionClient, LocalDecisionClient
from .vllm_adapter import VllmAdapter

__all__ = [
    "DecisionBackend",
    "DecisionClient",
    "HttpDecisionClient",
    "LayABackend",
    "LocalCalibratedBackend",
    "LocalDecisionClient",
    "VllmAdapter",
    "create_app",
]

"""领域异常：任何异常必须携带足够上下文，禁止无信息 pass。"""

from __future__ import annotations

from typing import Any


class CBError(RuntimeError):
    """所有领域异常的基类，携带结构化上下文。"""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:  # pragma: no cover - 展示用
        if not self.context:
            return self.message
        fields = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({fields})"


class ConfigError(CBError):
    """配置缺失、类型错误或非法取值（fail-fast / fail-closed）。"""


class DataError(CBError):
    """数据缺失、格式不合法、版本不匹配或统计口径冲突。"""


class SafetyError(CBError):
    """安全层拒绝执行或触发锁存。"""


class PolicyError(CBError):
    """策略加载、观测一致性或动作后处理失败。"""


class DecisionError(CBError):
    """决策模型加载、推理、超时或输出映射失败。"""


class RobotError(CBError):
    """机器人适配层（仿真 / SDK2）错误。"""

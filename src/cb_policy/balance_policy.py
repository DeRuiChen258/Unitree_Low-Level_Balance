"""策略推理封装：PyTorch checkpoint / ONNX，含 obs spec 一致性校验（fail-closed）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cb_common.errors import PolicyError
from cb_features.obs_spec import FEATURE_VERSION


class BalancePolicy(Protocol):
    """小脑策略协议：观测 → 关节目标增量（29）。"""

    obs_dim: int

    def act(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - 协议
        """返回 29 维动作。"""
        ...

    def reset(self) -> None:  # pragma: no cover - 协议
        """清空内部状态。"""
        ...


@dataclass
class PolicyMetadata:
    """ONNX/checkpoint 元数据（训练/部署一致性契约）。"""

    feature_version: str = FEATURE_VERSION
    obs_spec_hash: str = ""
    obs_dim: int = 0
    action_dim: int = 29
    action_kind: str = "delta"          # delta（相对默认姿态）| absolute
    output_scale: float = 1.0
    dataset_version: str = ""
    model_version: str = "1.0.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON（写入 ONNX metadata）。"""
        return json.dumps(self.__dict__, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> PolicyMetadata:
        """从 JSON 恢复。"""
        data = json.loads(text)
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise PolicyError("unknown policy metadata fields", fields=sorted(unknown))
        return cls(**data)


def validate_metadata(
    metadata: PolicyMetadata,
    *,
    obs_dim: int | None = None,
    obs_spec_hash: str | None = None,
    feature_version: str | None = None,
    allow_empty_hash: bool = False,
) -> None:
    """校验部署端与训练端一致性；不匹配直接拒绝加载。"""
    if obs_dim is not None and metadata.obs_dim not in (0, obs_dim):
        raise PolicyError("obs dim mismatch", expected=obs_dim, found=metadata.obs_dim)
    if feature_version is not None and metadata.feature_version != feature_version:
        raise PolicyError("feature version mismatch", expected=feature_version, found=metadata.feature_version)
    if obs_spec_hash is not None and metadata.obs_spec_hash not in ("", obs_spec_hash):
        raise PolicyError("obs spec hash mismatch", expected=obs_spec_hash, found=metadata.obs_spec_hash)
    if not allow_empty_hash and obs_spec_hash is not None and not metadata.obs_spec_hash:
        raise PolicyError("policy metadata missing obs_spec_hash (fail-closed)")


@dataclass
class OnnxPolicy:
    """ONNX Runtime 策略（CPU 推理，避免与训练争抢 GPU）。"""

    path: str
    metadata: PolicyMetadata = field(init=False)
    obs_dim: int = field(init=False)
    session: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import onnxruntime as ort

        file = Path(self.path)
        if not file.is_file():
            raise PolicyError("ONNX policy not found", path=str(file))
        self.session = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise PolicyError("policy must have exactly one input", inputs=len(inputs))
        self.obs_dim = int(inputs[0].shape[-1])
        meta = self.session.get_modelmeta().custom_metadata_map
        raw = meta.get("policy_metadata")
        self.metadata = PolicyMetadata.from_json(raw) if raw else PolicyMetadata(obs_dim=self.obs_dim)
        self.input_name = inputs[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def act(self, obs: np.ndarray) -> np.ndarray:
        """推理；返回 29 维动作。"""
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self.obs_dim:
            raise PolicyError("obs dim mismatch", expected=self.obs_dim, got=int(arr.shape[1]))
        out = self.session.run(self.output_names, {self.input_name: arr})[0]
        # 返回网络原始输出；residual_scale 由运行时（env/scheduler）按 metadata 决定
        action = np.asarray(out, dtype=np.float64).reshape(-1)
        if action.size != 29:
            raise PolicyError("policy output dim mismatch", size=int(action.size))
        return action

    def reset(self) -> None:
        """无内部状态（保持协议）。"""
        return None


@dataclass
class TorchPolicy:
    """PyTorch checkpoint 策略（训练/调试用）。"""

    checkpoint: str
    device: str = "cpu"
    metadata: PolicyMetadata = field(init=False)
    obs_dim: int = field(init=False)
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import torch

        data = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        payload = data.get("model_state_dict", data.get("model", None))
        if payload is None:
            raise PolicyError("checkpoint missing model weights", path=str(self.checkpoint))
        cfg = data.get("metadata", {})
        self.metadata = PolicyMetadata.from_json(cfg) if isinstance(cfg, str) else PolicyMetadata(**cfg)
        self.obs_dim = int(data.get("obs_dim", self.metadata.obs_dim))
        hidden = tuple(data.get("hidden_sizes", (256, 256)))
        self.model = PolicyMLP(self.obs_dim, 29, hidden)
        self.model.load_state_dict(payload)
        self.model.to(self.device).eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """推理（无梯度）。"""
        import torch

        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(self.device)
            out = self.model(tensor).cpu().numpy().reshape(-1)
        return out.astype(np.float64)

    def reset(self) -> None:
        """无内部状态（保持协议）。"""
        return None


def policy_mlp(obs_dim: int, action_dim: int = 29, hidden: tuple[int, ...] = (256, 256)) -> Any:
    """构造策略 MLP（torch.nn.Module）。"""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = obs_dim
    for size in hidden:
        layers += [nn.Linear(prev, size), nn.ELU()]
        prev = size
    layers.append(nn.Linear(prev, action_dim))
    return nn.Sequential(*layers)


PolicyMLP = None  # 延迟绑定：见 _ensure_policy_mlp()


def _ensure_policy_mlp() -> None:
    """延迟创建 torch 模型类，避免导入 torch 的副作用。"""
    global PolicyMLP
    if PolicyMLP is not None:
        return
    import torch.nn as nn

    class _PolicyMLP(nn.Module):
        """策略 MLP（ELU 激活，与训练网络一致）。"""

        def __init__(self, obs_dim: int, action_dim: int = 29, hidden: tuple[int, ...] = (256, 256)) -> None:
            super().__init__()
            self.net = policy_mlp(obs_dim, action_dim, hidden)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    PolicyMLP = _PolicyMLP


_ensure_policy_mlp()

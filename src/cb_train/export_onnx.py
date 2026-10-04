"""checkpoint → ONNX 导出与数值一致性校验（误差 ≤ 1e-5）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import PolicyError

from .networks import ActorCritic, StudentPolicy


@dataclass
class ExportResult:
    """导出结果。"""

    onnx_path: str
    max_abs_diff: float
    obs_dim: int
    metadata: dict[str, Any]


class _ActorOnly:
    """把 ActorCritic 包装为只输出确定性动作的模块（torch.onnx 友好）。"""

    def __new__(cls, policy: ActorCritic):  # type: ignore[no-untyped-def]
        import torch

        class ActorOnly(torch.nn.Module):
            def __init__(self, inner: ActorCritic) -> None:
                super().__init__()
                self.actor = inner.actor

            def forward(self, obs):  # type: ignore[no-untyped-def]
                return torch.tanh(self.actor(obs))

        return ActorOnly(policy)


def export_checkpoint(checkpoint: str | Path, out_path: str | Path, *, opset: int = 17) -> ExportResult:
    """导出 checkpoint 为 ONNX 并验证一致性。"""
    import onnx
    import onnxruntime as ort
    import torch

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = data.get("model_state_dict")
    if state is None:
        raise PolicyError("checkpoint missing model_state_dict", path=str(checkpoint))
    obs_dim = int(data.get("obs_dim", state.get("actor.0.weight", torch.zeros(0, 0)).shape[1] if "actor.0.weight" in state else 0))
    metadata = dict(data.get("metadata", {}))
    if obs_dim <= 0:
        raise PolicyError("cannot infer obs_dim from checkpoint", path=str(checkpoint))
    if any(key.startswith("actor.") for key in state):
        critic_hidden: list[int] = []
        index = 0
        while f"critic.{index}.weight" in state:
            critic_hidden.append(int(state[f"critic.{index}.weight"].shape[0]))
            index += 2
        critic_hidden = critic_hidden[:-1]  # 去掉输出层（1 维）
        policy = ActorCritic(
            obs_dim,
            29,
            actor_hidden=tuple(data.get("hidden_sizes", [256, 256])),
            critic_hidden=tuple(critic_hidden or [256, 256]),
        )
        policy.load_state_dict(state)
        policy.eval()
        module: torch.nn.Module = _ActorOnly(policy)  # type: ignore[assignment]
    else:
        student = StudentPolicy(obs_dim, 29, tuple(data.get("hidden_sizes", [256, 256])))
        student.load_state_dict(state)
        student.eval()
        module = student
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        dummy,
        str(out_file),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    model = onnx.load(str(out_file))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = json.dumps(value) if not isinstance(value, str) else value
    entry = model.metadata_props.add()
    entry.key = "policy_metadata"
    entry.value = json.dumps(metadata, sort_keys=True)
    onnx.save(model, str(out_file))
    # 一致性校验
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, size=(4, obs_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = module(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["action"], {"obs": sample})[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff > 1e-5:
        raise PolicyError("ONNX consistency check failed", max_abs_diff=max_diff)
    return ExportResult(str(out_file), max_diff, obs_dim, metadata)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.out)
    print(json.dumps({"onnx": result.onnx_path, "max_abs_diff": result.max_abs_diff, "metadata": result.metadata}, indent=2))

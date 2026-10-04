"""教师 → 学生蒸馏：观测 → 绝对关节目标增量（第 6.2 节方案 C）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config
from cb_common.errors import PolicyError
from cb_common.seeding import set_global_seed

from .networks import StudentPolicy
from .train_ppo import _build_env_factory


@dataclass
class DistillResult:
    """蒸馏结果。"""

    checkpoint: str
    samples: int
    final_mse: float
    epochs: int
    report: dict[str, Any]


def collect_distill_data(
    train_cfg: Config,
    system_cfg: Config,
    reward_cfg: Config,
    *,
    samples: int,
    seed: int,
    scenario_names: list[str] | None = None,
    policy_fn: Any = None,
    absolute: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """采集蒸馏数据。

    absolute=True：目标 = 教师+技能叠加后的关节目标增量（方案 A 的统一条件策略）；
    absolute=False：目标 = 残差动作本身（残差策略蒸馏）。
    """
    factory, env_cfg, scenarios, _teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed)
    names = scenario_names or list(scenarios)
    env = factory(0, names[0])
    obs, _ = env.reset(seed=seed, randomize=True)
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    while len(observations) < samples:
        action = np.zeros(29) if policy_fn is None else np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        next_obs, _reward, terminated, truncated, info = env.step(action)
        observations.append(obs.astype(np.float32))
        if absolute:
            target = np.asarray(info["target"], dtype=np.float64) - np.asarray(env.core.default_qpos_policy)
            targets.append(target.astype(np.float32))
        else:
            targets.append(action.astype(np.float32))
        obs = next_obs
        if terminated or truncated:
            env.scenario = scenarios[names[len(observations) % len(names)]]
            obs, _ = env.reset(seed=seed + len(observations), randomize=True)
    return np.stack(observations), np.stack(targets), {"obs_dim": env_cfg.obs_spec.total_dim, "obs_spec_hash": env_cfg.obs_spec.hash()}


def distill(
    train_cfg: Config,
    system_cfg: Config,
    reward_cfg: Config,
    *,
    output_dir: str | Path,
    seed: int = 1,
    samples: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    obs: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    policy: Any = None,
    absolute: bool = True,
) -> DistillResult:
    """训练学生策略（若未提供数据则先采集）。"""
    import torch

    seed_record = set_global_seed(seed)
    distill_cfg = dict(train_cfg.get("distill", {}))
    samples = int(distill_cfg.get("samples", 24000) if samples is None else samples)
    epochs = int(distill_cfg.get("epochs", 12) if epochs is None else epochs)
    batch_size = int(distill_cfg.get("batch_size", 1024) if batch_size is None else batch_size)
    lr = float(distill_cfg.get("lr", 1e-3) if lr is None else lr)
    if obs is None or targets is None:
        obs, targets, meta = collect_distill_data(
            train_cfg, system_cfg, reward_cfg, samples=samples, seed=seed, policy_fn=policy, absolute=absolute
        )
    else:
        meta = {"obs_dim": int(obs.shape[1]), "obs_spec_hash": ""}
    if obs.shape[0] < 32:
        raise PolicyError("not enough distillation samples", samples=int(obs.shape[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentPolicy(int(obs.shape[1]), 29).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(obs_t, target_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    history: list[dict[str, float]] = []
    start = time.time()
    final_mse = float("inf")
    for epoch in range(epochs):
        losses: list[float] = []
        for batch_obs, batch_target in loader:
            prediction = student(batch_obs)
            loss = torch.nn.functional.mse_loss(prediction, batch_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_mse = float(np.mean(losses))
        history.append({"epoch": epoch, "mse": final_mse})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "student.pt"
    metadata = {
        "feature_version": str(system_cfg.get("feature_version", "1.0")),
        "obs_spec_hash": meta.get("obs_spec_hash", ""),
        "obs_dim": int(obs.shape[1]),
        "action_dim": 29,
        "action_kind": "absolute_delta" if absolute else "residual_delta",
        "output_scale": 1.0,
        "dataset_version": "n/a",
        "model_version": "1.0.0",
    }
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "obs_dim": int(obs.shape[1]),
            "action_dim": 29,
            "hidden_sizes": [256, 256],
            "metadata": metadata,
            "seed": seed,
        },
        checkpoint,
    )
    report = {
        "samples": int(obs.shape[0]),
        "epochs": epochs,
        "final_mse": final_mse,
        "max_mse_threshold": float(distill_cfg.get("max_mse", 0.0025)),
        "pass": bool(final_mse <= float(distill_cfg.get("max_mse", 0.0025))),
        "history": history,
        "wall_s": round(time.time() - start, 2),
        "seed_record": seed_record,
    }
    (out_dir / "distill_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return DistillResult(str(checkpoint), int(obs.shape[0]), final_mse, epochs, report)

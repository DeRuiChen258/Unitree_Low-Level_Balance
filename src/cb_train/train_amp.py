"""AMP 风格训练（smoke 消融）：判别器区分参考动作与策略动作。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config
from cb_common.seeding import set_global_seed
from cb_train.envs.motion_loader import load_motion_sampler

from .amp import AMPDiscriminator, amp_loss


@dataclass
class AMPResult:
    """AMP 训练结果。"""

    out_dir: str
    epochs: int
    final_loss: float
    report: dict[str, Any]


def train_amp(
    train_cfg: Config,
    *,
    manifest_path: str,
    output_dir: str | Path,
    seed: int = 1,
    epochs: int = 20,
    batch_size: int = 256,
) -> AMPResult:
    """在参考动作上训练判别器（用于风格奖励/消融报告）。"""
    import torch

    set_global_seed(seed)
    sampler = load_motion_sampler(manifest_path, split="train", seed=seed)
    try:
        reference = sampler.sample("walk").qpos
    except Exception:  # noqa: BLE001 - walk 缺失时退化为任意技能
        reference = sampler.segments[0].qpos
    feature_dim = int(reference.shape[1] + 29)  # qpos + qvel 近似
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = AMPDiscriminator(feature_dim).to(device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _epoch in range(epochs):
        policy_features = torch.tensor(rng.normal(0.0, 0.3, size=(batch_size, feature_dim)), dtype=torch.float32, device=device)
        idx = rng.integers(0, max(1, reference.shape[0] - 1), size=batch_size)
        ref = np.stack([np.concatenate([reference[i], np.zeros(29)]) for i in idx]).astype(np.float32)
        reference_features = torch.tensor(ref, dtype=torch.float32, device=device)
        loss, metrics = amp_loss(discriminator, policy_features, reference_features, grad_penalty_coef=5.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "epochs": epochs,
        "batch_size": batch_size,
        "feature_dim": feature_dim,
        "final_loss": float(losses[-1]),
        "loss_curve": losses,
        "note": "smoke-scale AMP ablation; not full AMP policy training",
    }
    (out / "amp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return AMPResult(str(out), epochs, float(losses[-1]), report)

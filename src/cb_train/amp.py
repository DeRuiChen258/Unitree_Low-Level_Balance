"""AMP 判别器与风格奖励（可开关，用于消融）。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .networks import mlp


class AMPDiscriminator(nn.Module):
    """判别「策略运动 vs 参考运动」的判别器。"""

    def __init__(self, feature_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.net = mlp(feature_dim, 1, hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回 logit。"""
        return self.net(features).squeeze(-1)


def amp_loss(
    discriminator: AMPDiscriminator,
    policy_features: torch.Tensor,
    reference_features: torch.Tensor,
    *,
    grad_penalty_coef: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """最小二乘 GAN 判别损失 + 梯度惩罚（WGAN-GP 近似）。"""
    policy_logits = discriminator(policy_features)
    ref_logits = discriminator(reference_features)
    loss = 0.5 * ((policy_logits + 1.0) ** 2).mean() + 0.5 * ((ref_logits - 1.0) ** 2).mean()
    alpha = torch.rand(policy_features.shape[0], 1, device=policy_features.device)
    mixed = (alpha * policy_features + (1.0 - alpha) * reference_features).requires_grad_(True)
    logits = discriminator(mixed)
    grad = torch.autograd.grad(logits.sum(), mixed, create_graph=True)[0]
    penalty = grad_penalty_coef * (grad.norm(dim=-1) - 1.0).pow(2).mean()
    return loss + penalty, {"amp_loss": float(loss.detach()), "grad_penalty": float(penalty.detach())}


def style_reward(discriminator: AMPDiscriminator, features: torch.Tensor) -> torch.Tensor:
    """风格奖励：判别器输出越高越像参考动作（0.5 为中性）。"""
    with torch.no_grad():
        return 0.5 * (discriminator(features) + 1.0)

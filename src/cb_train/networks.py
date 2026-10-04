"""策略/价值网络与蒸馏学生网络（CUDA/CPU 通用）。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim: int, output_dim: int, hidden: Sequence[int], activation: str = "elu") -> nn.Sequential:
    """构造 MLP。"""
    act: nn.Module = nn.ELU() if activation == "elu" else nn.ReLU()
    layers: list[nn.Module] = []
    prev = input_dim
    for size in hidden:
        layers += [nn.Linear(prev, int(size)), act]
        prev = int(size)
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """PPO Actor-Critic；actor 输出 tanh 后位于 [-1,1]。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        actor_hidden: Sequence[int] = (256, 256),
        critic_hidden: Sequence[int] = (256, 256),
        init_noise_std: float = 0.35,
        log_std_min: float = -3.0,
        log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, action_dim, actor_hidden)
        self.critic = mlp(obs_dim, 1, critic_hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_noise_std)).log())
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回动作均值（tanh）与状态价值。"""
        mean = torch.tanh(self.actor(obs))
        value = self.critic(obs).squeeze(-1)
        return mean, value

    def distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """返回动作分布、均值与价值。"""
        mean, value = self.forward(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp()), mean, value

    @staticmethod
    def log_prob_of(dist: torch.distributions.Normal, raw_action: torch.Tensor, squashed: torch.Tensor) -> torch.Tensor:
        """tanh 变换后的 log 概率（含 Jacobian 修正）。"""
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(-1)
        return log_prob


class StudentPolicy(nn.Module):
    """蒸馏学生：观测 → 关节目标增量（29）。"""

    def __init__(
        self, obs_dim: int, action_dim: int = 29, hidden: Sequence[int] = (256, 256), activation: str = "elu"
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向。"""
        return self.net(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak 软更新（AMP/教师平滑使用）。"""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters(), strict=False):
            t_param.data.mul_(1.0 - tau).add_(s_param.data, alpha=tau)


def mse_action_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """动作空间 MSE（蒸馏指标与损失同源）。"""
    return F.mse_loss(pred, target)

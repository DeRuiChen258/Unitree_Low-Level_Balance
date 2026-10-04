"""小脑策略训练（MuJoCo 降级栈 + 教师复用 + PPO/蒸馏）。"""

from .networks import ActorCritic, StudentPolicy

__all__ = ["ActorCritic", "StudentPolicy"]

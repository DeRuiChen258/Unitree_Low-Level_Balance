"""训练环境：MuJoCo + 教师策略 + 技能叠加 + 域随机化。"""

from .g1_balance_env import EnvConfig, G1BalanceEnv, ScenarioScript

__all__ = ["EnvConfig", "G1BalanceEnv", "ScenarioScript"]

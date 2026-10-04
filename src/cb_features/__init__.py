"""特征工程：训练与部署共用同一 obs spec / 文本化模板。"""

from .balance_feats import BalanceFeatureBuilder, balance_features
from .command import CommandVector, encode_skill
from .normalizer import RunningMeanStd
from .obs_spec import OBS_SPEC, ObsSpec
from .phase_clock import PhaseClock
from .proprio import build_proprio, history_key_observation
from .text_state import TEXT_STATE_TEMPLATE_VERSION, build_text_state

__all__ = [
    "OBS_SPEC",
    "BalanceFeatureBuilder",
    "CommandVector",
    "ObsSpec",
    "PhaseClock",
    "RunningMeanStd",
    "TEXT_STATE_TEMPLATE_VERSION",
    "balance_features",
    "build_proprio",
    "build_text_state",
    "encode_skill",
    "history_key_observation",
]

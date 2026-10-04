"""平衡特征单测：维度、物理含义、综合分界。"""

from __future__ import annotations

import numpy as np

from cb_common.types import RobotState
from cb_features.balance_feats import BalanceFeatureBuilder, balance_features


def _state(roll: float = 0.0, margin: float = 0.05) -> RobotState:
    half = roll / 2.0
    return RobotState(
        timestamp=0.0,
        base_pos=np.array([0.0, 0.0, 0.8]),
        base_quat=np.array([np.cos(half), np.sin(half), 0.0, 0.0]),
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.zeros(3),
        joint_pos=np.zeros(29),
        joint_vel=np.zeros(29),
        contact=(True, True),
        com=np.zeros(3),
        cp=np.zeros(2),
        support_margin=margin,
    )


def test_balance_features_shape_and_margin() -> None:
    """8 维特征；负余量降低综合分。"""
    good = balance_features(_state(margin=0.08))
    bad = balance_features(_state(margin=-0.05))
    assert good.shape == (8,)
    assert good[6] > bad[6]
    assert 0.0 <= good[7] <= 1.0


def test_builder_history() -> None:
    """构造器记录接触历史并保持容量。"""
    builder = BalanceFeatureBuilder(capacity=3)
    for _ in range(5):
        builder.push_contact((True, True))
    feature = builder.build(_state())
    assert len(builder.contact_history) == 3
    assert feature.shape == (8,)

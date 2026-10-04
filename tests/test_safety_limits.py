"""安全限幅单测：配置校验、条件器、越界/NaN 拒绝。"""

from __future__ import annotations

import numpy as np
import pytest

from cb_common import load_config
from cb_common.errors import SafetyError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, KD_POLICY, KP_POLICY
from cb_common.types import JointCommand, SafetyLevel
from cb_safety import CommandConditioner, check_safety, load_safety_limits
from tests.helpers import make_state


def _limits():
    scene = load_config("configs/system.yaml").get("limits.mjcf_scene")
    return load_safety_limits("configs/safety/limits.yaml", scene_path=scene)


def test_limits_load_and_check() -> None:
    """阈值加载成功；越界命令被拒绝。"""
    limits = _limits()
    state = make_state()
    command = JointCommand(np.array(DEFAULT_JOINT_POS_POLICY), np.array(KP_POLICY), np.array(KD_POLICY), 0.0)
    assert check_safety(command, state, limits).allowed
    bad = command.copy_with(q_target=np.full(29, 100.0))
    decision = check_safety(bad, state, limits)
    assert not decision.allowed
    assert decision.level >= SafetyLevel.ABORT


def test_conditioner_respects_limits() -> None:
    """条件器输出在限位内且逐步逼近。"""
    limits = _limits()
    conditioner = CommandConditioner(limits, 0.02)
    conditioner.reset(np.array(DEFAULT_JOINT_POS_POLICY))
    target = np.array(DEFAULT_JOINT_POS_POLICY) + 5.0
    out = conditioner.condition(target)
    assert np.all(out <= limits.joints.position_max - limits.position_margin_rad + 1e-9)
    with pytest.raises(SafetyError):
        conditioner.condition(np.full(29, np.nan))

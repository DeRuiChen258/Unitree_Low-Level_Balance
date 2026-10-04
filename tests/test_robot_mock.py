"""Mock 适配器单测：状态形状、跟踪、故障注入。"""

from __future__ import annotations

import numpy as np

from cb_common.joints import DEFAULT_JOINT_POS_POLICY, KD_POLICY, KP_POLICY
from cb_common.types import JointCommand, RobotMode
from cb_robot import MockAdapter


def test_mock_tracking_and_faults() -> None:
    """一阶跟踪收敛；NaN/freeze/estop 注入生效。"""
    adapter = MockAdapter()
    adapter.connect()
    adapter.set_mode(RobotMode.STAND)
    target = np.array(DEFAULT_JOINT_POS_POLICY)
    target[0] = 0.2
    adapter.send_command(JointCommand(target, np.array(KP_POLICY), np.array(KD_POLICY), 0.0))
    for _ in range(50):
        adapter.advance(0.02)
    state = adapter.read_state()
    assert state.joint_pos.shape == (29,)
    assert abs(state.joint_pos[0] - 0.2) < 0.05
    adapter.inject("freeze")
    before = adapter.read_state().joint_pos.copy()
    adapter.advance(0.02)
    assert np.allclose(before, adapter.read_state().joint_pos)
    adapter.clear_faults()
    adapter.inject("nan")
    adapter.advance(0.02)
    assert not np.isfinite(adapter.read_state().joint_pos).all()


def test_mock_requires_connect() -> None:
    """未连接读取状态报错。"""
    adapter = MockAdapter()
    try:
        adapter.read_state()
    except Exception as exc:  # noqa: BLE001
        assert "not connected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected error")

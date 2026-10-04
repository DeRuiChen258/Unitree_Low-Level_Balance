"""适配层等价性单测：mock / MuJoCo / SDK2 dry-run 使用同一命令与关节序。"""

from __future__ import annotations

import numpy as np
import pytest

from cb_common import load_config
from cb_common.errors import SafetyError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, KD_POLICY, KP_POLICY, NUM_JOINTS
from cb_common.types import JointCommand
from cb_robot import MockAdapter, MuJoCoAdapter, SDK2Adapter
from cb_robot.sdk2_adapter import build_low_cmd, unitree_crc32


def _command() -> JointCommand:
    return JointCommand(np.array(DEFAULT_JOINT_POS_POLICY), np.array(KP_POLICY), np.array(KD_POLICY), 0.0, skill="stand")


def test_mock_and_mujoco_same_interface() -> None:
    """两种适配器返回同形状状态，可执行同一命令。"""
    mock = MockAdapter()
    mock.connect()
    mock.send_command(_command())
    mock.advance(0.02)
    scene = str(load_config("configs/system.yaml").get("paths.menagerie_scene"))
    mujoco = MuJoCoAdapter(scene, headless=True)
    mujoco.connect()
    mujoco.send_command(_command())
    for _ in range(20):
        mujoco.advance(0.02)
    for adapter in (mock, mujoco):
        state = adapter.read_state()
        assert state.joint_pos.shape == (NUM_JOINTS,)
        assert state.joint_vel.shape == (NUM_JOINTS,)
        assert len(state.contact) == 2
        assert tuple(state.joint_names) == tuple(__import__("cb_common").POLICY_JOINT_NAMES)
    assert np.isfinite(mujoco.read_state().joint_pos).all()
    mujoco.close()


def test_sdk2_dry_run_only() -> None:
    """SDK2 适配器只做 dry-run，构造 LowCmd+CRC，不发送。"""
    adapter = SDK2Adapter()
    adapter.connect()
    adapter.send_command(_command())
    adapter.advance(0.02)
    low_cmd = adapter.last_low_cmd
    assert low_cmd is not None
    assert len(low_cmd["motor_cmd"]) == 29
    assert low_cmd["crc"] != 0
    assert adapter.dry_run_count == 1
    assert build_low_cmd(np.zeros(29), np.ones(29), np.ones(29))["crc"] != 0
    assert unitree_crc32(b"abc") == unitree_crc32(b"abc")
    with pytest.raises(SafetyError):
        SDK2Adapter(allow_send=True)

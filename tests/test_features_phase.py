"""相位时钟单测：步态/技能/跳跃相位。"""

from __future__ import annotations

import numpy as np

from cb_common.types import JumpPhase
from cb_features.phase_clock import PhaseClock


def test_phase_clock_range_and_jump_one_hot() -> None:
    """相位输出在单位圆上；跳跃 one-hot 正确。"""
    clock = PhaseClock()
    out = clock.update(0.02, (True, False))
    assert out.shape == (4,)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)
    clock.set_jump_phase(JumpPhase.TAKEOFF)
    hot = clock.jump_one_hot()
    assert hot.shape == (5,)
    assert hot[2] == 1.0


def test_phase_advances_and_switches() -> None:
    """接触切换重置步态相位。"""
    clock = PhaseClock(gait_period_s=0.5)
    values = [clock.update(0.02, (True, False)) for _ in range(10)]
    first = values[0]
    clock.update(0.02, (False, True))
    switched = clock.update(0.02, (False, True))
    assert not np.allclose(first, switched)

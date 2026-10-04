"""过渡单测：混合权重、接触仲裁、最小 jerk。"""

from __future__ import annotations

import numpy as np

from cb_common.types import SkillOutput
from cb_skills.transition import blend_outputs, min_jerk_weights


def test_blend_and_contacts() -> None:
    """混合端点值正确、接触阈值离散。"""
    a = SkillOutput(delta_q=np.zeros(29), skill="stand", expected_contact=(True, True))
    b = SkillOutput(delta_q=np.ones(29), skill="walk", expected_contact=(False, True))
    start = blend_outputs(a, b, 0.0)
    end = blend_outputs(a, b, 1.0)
    assert np.allclose(start.delta_q, 0.0)
    assert np.allclose(end.delta_q, 1.0)
    assert start.expected_contact == (True, True)
    assert end.expected_contact == (False, True)


def test_min_jerk_weights() -> None:
    """权重单调 0→1 且端点精确。"""
    weights = min_jerk_weights(21)
    assert weights[0] == 0.0 and abs(weights[-1] - 1.0) < 1e-9
    assert np.all(np.diff(weights) >= -1e-9)

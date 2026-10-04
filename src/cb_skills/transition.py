"""技能间过渡：最小 jerk 混合 + 接触仲裁。"""

from __future__ import annotations

import numpy as np

from cb_common.types import SkillOutput


def blend_outputs(
    source: SkillOutput,
    target: SkillOutput,
    weight: float,
    *,
    contact_source: float = 0.5,
) -> SkillOutput:
    """按权重混合两个技能输出（接触按阈值离散仲裁）。"""
    w = float(np.clip(weight, 0.0, 1.0))
    delta = (1.0 - w) * source.delta_q + w * target.delta_q
    contacts = source.expected_contact if w < contact_source else target.expected_contact
    return SkillOutput(
        delta_q=delta.astype(np.float64),
        skill=target.skill if w >= 0.5 else source.skill,
        phase=float((1.0 - w) * source.phase + w * target.phase),
        jump_phase=target.jump_phase if w >= 0.5 else source.jump_phase,
        expected_contact=contacts,
        active_groups=tuple(sorted(set(source.active_groups) | set(target.active_groups))),
        metadata={"blend_weight": w, "from": source.skill, "to": target.skill},
    )


def min_jerk_weights(count: int) -> np.ndarray:
    """返回最小 jerk 平滑权重序列 [0,1]。"""
    s = np.linspace(0.0, 1.0, max(2, int(count)))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

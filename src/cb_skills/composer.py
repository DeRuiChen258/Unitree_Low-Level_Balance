"""组合编排：串行 / 并行叠加 / 相位同步（第 7.2 节）。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from cb_common.joints import JOINT_GROUPS, NUM_JOINTS, POLICY_INDEX
from cb_common.types import SkillOutput

from .transition import blend_outputs

# 组仲裁优先级：躯干/平衡通道优先于上肢叠加
GROUP_PRIORITY: dict[str, int] = {
    "lower_body": 100,
    "torso": 90,
    "left_leg": 100,
    "right_leg": 100,
    "left_arm": 50,
    "right_arm": 50,
    "arms": 50,
}


def compose(current: SkillOutput, overlay: SkillOutput, w: float) -> SkillOutput:
    """通用叠加：按组分配控制权，躯干/下肢优先，上肢使用剩余余量。"""
    return compose_parallel(current, [overlay], weights=[w])


def compose_parallel(
    base: SkillOutput,
    overlays: Iterable[SkillOutput],
    *,
    weights: Iterable[float] | None = None,
) -> SkillOutput:
    """并行叠加：基座（下肢/平衡）优先，overlay 只写各自组。"""
    overlay_list = list(overlays)
    weight_list = list(weights) if weights is not None else [1.0] * len(overlay_list)
    if len(weight_list) != len(overlay_list):
        raise ValueError("weights length must match overlays")
    delta = np.zeros(NUM_JOINTS)
    claimed: dict[int, int] = {}
    # 基座先占位
    for group in base.active_groups:
        for name in JOINT_GROUPS.get(group, ()):
            claimed[POLICY_INDEX[name]] = GROUP_PRIORITY.get(group, 0)
    for name in JOINT_GROUPS.get("lower_body", ()):
        claimed.setdefault(POLICY_INDEX[name], 100)
    for name in JOINT_GROUPS.get("torso", ()):
        claimed.setdefault(POLICY_INDEX[name], 90)
    for overlay, weight in zip(overlay_list, weight_list, strict=False):
        for group in overlay.active_groups:
            priority = GROUP_PRIORITY.get(group, 0)
            for name in JOINT_GROUPS.get(group, ()):
                index = POLICY_INDEX[name]
                if priority < claimed.get(index, -1):
                    continue
                delta[index] += float(weight) * overlay.delta_q[index]
    delta += base.delta_q
    metadata = dict(base.metadata)
    metadata.update({"mode": "parallel", "overlays": [o.skill for o in overlay_list], "weights": weight_list})
    return SkillOutput(
        delta_q=delta,
        skill=base.skill,
        phase=base.phase,
        jump_phase=base.jump_phase,
        expected_contact=base.expected_contact,
        active_groups=base.active_groups,
        metadata=metadata,
    )


def compose_serial(
    outputs: Iterable[SkillOutput],
    *,
    blend_frames: int = 10,
) -> SkillOutput:
    """串行组合：相邻输出之间用最小 jerk 混合（返回最后一段的元数据）。"""
    from .transition import min_jerk_weights

    items = list(outputs)
    if not items:
        raise ValueError("compose_serial requires at least one output")
    result = items[0]
    for nxt in items[1:]:
        for w in min_jerk_weights(blend_frames):
            result = blend_outputs(result, nxt, float(w))
    return result


def phase_sync_ok(state_contact: tuple[bool, bool], alignment: str) -> bool:
    """相位同步检查：起跳相位必须落在支撑窗口，转弯切换落在双支撑。"""
    if alignment in ("any", ""):
        return True
    if alignment == "double_support":
        return all(state_contact)
    if alignment == "takeoff_window":
        return any(state_contact)
    if alignment == "landing":
        return all(state_contact)
    return True

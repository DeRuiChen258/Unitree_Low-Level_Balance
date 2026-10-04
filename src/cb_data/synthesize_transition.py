"""S5 过渡合成：相位对齐 + 最小 jerk 混合 + 接触仲裁。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cb_common.errors import DataError

from .schema import SkillSegment, TransitionSegment


def blend_segments(
    from_segment: SkillSegment,
    to_segment: SkillSegment,
    *,
    blend_window_s: float,
    phase_alignment: str = "double_support",
) -> TransitionSegment:
    """按最小 jerk 权重混合两段轨迹；返回可训练 TransitionSegment。"""
    fps = from_segment.fps
    window = max(4, int(round(blend_window_s * fps)))
    if from_segment.frames < window or to_segment.frames < window:
        raise DataError(
            "segments too short for blend window",
            from_frames=from_segment.frames,
            to_frames=to_segment.frames,
            window=window,
        )
    from_tail = from_segment.qpos[-window:]
    to_head = to_segment.qpos[:window]
    s = np.linspace(0.0, 1.0, window)
    weight = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    qpos = (1.0 - weight)[:, None] * from_tail + weight[:, None] * to_head
    root = (1.0 - weight)[:, None] * from_segment.root_pos[-window:] + weight[:, None] * to_segment.root_pos[:window]
    contacts = np.where(weight[:, None] < 0.5, from_segment.contacts[-window:], to_segment.contacts[:window])
    conflict = int(np.count_nonzero(np.all(~contacts, axis=1)))
    jerk = float(np.max(np.abs(np.diff(qpos, n=3, axis=0)))) if window > 3 else 0.0
    quality = {
        "contact_conflict_frames": conflict,
        "jerk_max": jerk,
        "blend_window_s": blend_window_s,
        "phase_alignment": phase_alignment,
    }
    return TransitionSegment(
        transition_id=f"transition:{from_segment.segment_id}->{to_segment.segment_id}",
        from_skill=from_segment.skill,
        to_skill=to_segment.skill,
        blend_window=blend_window_s,
        phase_alignment=phase_alignment,
        qpos=qpos,
        root_pos=root,
        contacts=contacts,
        from_segment=from_segment.segment_id,
        to_segment=to_segment.segment_id,
        quality=quality,
        fps=fps,
        meta={"min_jerk": True},
    )


def synthesize_transitions(segments: list[SkillSegment], config: Mapping) -> list[TransitionSegment]:
    """按配置中的技能对生成过渡片段（每对取第一个匹配片段）。"""
    pairs = [tuple(pair) for pair in config.get("transitions", [])]
    window_bounds = config.get("blend_window_s", [0.3, 0.6])
    out: list[TransitionSegment] = []
    for from_skill, to_skill in pairs:
        source = next((s for s in segments if s.skill == from_skill), None)
        target = next((s for s in segments if s.skill == to_skill), None)
        if source is None or target is None:
            continue
        window = 0.5 * (float(window_bounds[0]) + float(window_bounds[1]))
        try:
            out.append(
                blend_segments(
                    source,
                    target,
                    blend_window_s=window,
                    phase_alignment="takeoff_window" if to_skill == "jump" else "double_support",
                )
            )
        except DataError:
            continue
    return out

"""动作/接触/相位可视化抽检（PNG）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cb_common.joints import POLICY_JOINT_NAMES

from .schema import SkillSegment


def plot_segment(segment: SkillSegment, path: str | Path, *, joint_subset: tuple[str, ...] | None = None) -> Path:
    """绘制关节轨迹 + 根高度 + 接触 + 相位，保存 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset = joint_subset or (
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "right_hip_pitch",
        "left_shoulder_pitch",
    )
    indices = [POLICY_JOINT_NAMES.index(name) for name in subset]
    t = np.arange(segment.frames) / segment.fps
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for index, name in zip(indices, subset, strict=False):
        axes[0].plot(t, segment.qpos[:, index], label=name)
    axes[0].set_ylabel("joint (rad)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(t, segment.root_pos[:, 2], color="tab:blue", label="root height (m)")
    axes[1].plot(t, segment.phase, color="tab:orange", label="phase")
    axes[1].legend(fontsize=7)
    axes[1].set_ylabel("m / phase")
    axes[2].step(t, segment.contacts[:, 0].astype(float), label="left contact")
    axes[2].step(t, segment.contacts[:, 1].astype(float) + 0.05, label="right contact (+0.05)")
    axes[2].set_ylim(-0.1, 1.3)
    axes[2].set_ylabel("contact")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{segment.segment_id} [{segment.skill}]")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

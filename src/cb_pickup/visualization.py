"""可视化：COM/支撑域/ZMP/裕度/俯仰/脚步/关节/决策/置信度/风险 + 关键图（第 22 节）。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _mpl():  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pickup_dashboard(trajectory: Mapping[str, np.ndarray], out_dir: str | Path) -> list[Path]:
    """生成 10 张图；返回文件列表。"""
    plt = _mpl()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    t = np.asarray(trajectory.get("time", np.arange(len(trajectory.get("stability_margin", [])))), dtype=np.float64)
    com = np.asarray(trajectory.get("com", np.zeros((len(t), 3))))
    zmp = np.asarray(trajectory.get("zmp", np.zeros((len(t), 2))))
    margin = np.asarray(trajectory.get("stability_margin", np.zeros(len(t))), dtype=np.float64)
    predicted = np.asarray(trajectory.get("predicted_margin", margin), dtype=np.float64)
    pitch = np.asarray(trajectory.get("trunk_pitch", np.zeros(len(t))), dtype=np.float64)
    roll = np.asarray(trajectory.get("trunk_roll", np.zeros(len(t))), dtype=np.float64)
    left = np.asarray(trajectory.get("left_foot", np.zeros((len(t), 3))))
    right = np.asarray(trajectory.get("right_foot", np.zeros((len(t), 3))))

    def save(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(com[:, 0], com[:, 1], label="COM")
    ax.plot(zmp[:, 0], zmp[:, 1], "--", label="ZMP")
    ax.plot(left[:, 0], left[:, 1], ".", ms=2, label="L foot")
    ax.plot(right[:, 0], right[:, 1], ".", ms=2, label="R foot")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("COM / ZMP / feet trajectory")
    ax.legend(fontsize=7)
    save(fig, "01_com_zmp_feet.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, com[:, 0], label="COM x")
    ax.plot(t, zmp[:, 0], label="ZMP x")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (m)")
    ax.legend(fontsize=7)
    save(fig, "02_com_x_timeline.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, margin, label="margin")
    ax.plot(t, predicted, "--", label="predicted")
    ax.axhline(0.0, color="r", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("stability margin (m)")
    ax.legend(fontsize=7)
    save(fig, "03_stability_margin.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, np.degrees(pitch), label="trunk pitch")
    ax.plot(t, np.degrees(roll), label="trunk roll")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.legend(fontsize=7)
    save(fig, "04_trunk_pitch_roll.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    polygons = trajectory.get("support_polygon")
    if polygons is not None:
        for index in np.linspace(0, len(polygons) - 1, min(6, len(polygons)), dtype=int):
            poly = np.asarray(polygons[index], dtype=np.float64)
            if poly.ndim == 2 and poly.shape[0] >= 3:
                closed = np.vstack([poly, poly[0]])
                ax.plot(closed[:, 0], closed[:, 1], alpha=0.5)
    ax.plot(com[:, 0], com[:, 1], "k-", label="COM")
    ax.set_title("Support polygon evolution")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7)
    save(fig, "05_support_polygon.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, left[:, 0], label="L foot x")
    ax.plot(t, right[:, 0], label="R foot x")
    ax.plot(t, left[:, 1], "--", label="L foot y")
    ax.plot(t, right[:, 1], "--", label="R foot y")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m")
    ax.set_title("Footstep trajectory")
    ax.legend(fontsize=7)
    save(fig, "06_footsteps.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    if "hand_distance" in trajectory:
        ax.plot(t, np.asarray(trajectory["hand_distance"], dtype=np.float64), label="hand-object distance")
    if "object_height" in trajectory:
        ax.plot(t, np.asarray(trajectory["object_height"], dtype=np.float64), label="object height")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7)
    save(fig, "07_reach_grasp.png")

    fig, ax = plt.subplots(figsize=(7, 3))
    if "energy" in trajectory:
        energy = np.asarray(trajectory["energy"], dtype=np.float64)
        ax.plot(t[: len(energy)], energy)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sum torque^2 dt")
    save(fig, "08_energy.png")

    primitives = trajectory.get("primitive")
    phases = trajectory.get("phase")
    if primitives is not None:
        unique = sorted({str(p) for p in primitives})
        mapping = {name: index for index, name in enumerate(unique)}
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t[: len(primitives)], [mapping[str(p)] for p in primitives], drawstyle="steps-post")
        ax.set_yticks(range(len(unique)))
        ax.set_yticklabels(unique, fontsize=7)
        ax.set_xlabel("time (s)")
        ax.set_title("Primitive timeline")
        save(fig, "09_primitives.png")
    if phases is not None:
        unique = sorted({str(p) for p in phases})
        mapping = {name: index for index, name in enumerate(unique)}
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t[: len(phases)], [mapping[str(p)] for p in phases], drawstyle="steps-post")
        ax.set_yticks(range(len(unique)))
        ax.set_yticklabels(unique, fontsize=7)
        ax.set_xlabel("time (s)")
        ax.set_title("MotionManager phase timeline")
        save(fig, "10_phases.png")
    return files

"""G1 29 DOF 关节序事实源（策略序 / MuJoCo 序 / 增益 / 限幅）。

数据来源（只读复用，2026-10-04 实测）：
`Action/G1_run/deploy/g1_deploy/config/g1_amp.yaml` 与
`Action/G1_run/deploy/g1_deploy/assets/g1_29dof.xml`。

策略序（Isaac 序，部署事实标准）：
  [ 0] left_hip_pitch   [ 1] right_hip_pitch  [ 2] waist_yaw
  [ 3] left_hip_roll    [ 4] right_hip_roll   [ 5] waist_roll
  [ 6] left_hip_yaw     [ 7] right_hip_yaw    [ 8] waist_pitch
  [ 9] left_knee        [10] right_knee       [11] left_shoulder_pitch
  [12] right_shoulder_pitch [13] left_ankle_pitch [14] right_ankle_pitch
  [15] left_shoulder_roll [16] right_shoulder_roll [17] left_ankle_roll
  [18] right_ankle_roll  [19] left_shoulder_yaw [20] right_shoulder_yaw
  [21] left_elbow       [22] right_elbow      [23] left_wrist_roll
  [24] right_wrist_roll  [25] left_wrist_pitch [26] right_wrist_pitch
  [27] left_wrist_yaw   [28] right_wrist_yaw
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 29

POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)

MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# g1_amp.yaml: mujoco_to_isaac_map / isaac_to_mujoco_map
MUJOCO_TO_ISAAC: tuple[int, ...] = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)
ISAAC_TO_MUJOCO: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)

# g1_amp.yaml（策略序）
DEFAULT_JOINT_POS_POLICY: tuple[float, ...] = (
    -0.1, -0.1, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.3, 0.3, 0.3, 0.3,
    -0.2, -0.2, 0.25, -0.25,
    0.0, 0.0, 0.0, 0.0,
    0.97, 0.97, 0.15, -0.15,
    0.0, 0.0, 0.0, 0.0,
)

ACTION_SCALE_POLICY: tuple[float, ...] = (
    0.25, 0.25, 0.25,
    0.25, 0.25, 0.25,
    0.25, 0.25, 0.25,
    0.25, 0.25, 0.08, 0.08,
    0.25, 0.25, 0.08, 0.08,
    0.25, 0.25, 0.08, 0.08,
    0.08, 0.08,
    0.05, 0.05, 0.05, 0.05,
    0.05, 0.05,
)

KP_POLICY: tuple[float, ...] = (
    100.0, 100.0, 200.0,
    100.0, 100.0, 40.0,
    100.0, 100.0, 40.0,
    150.0, 150.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0,
    40.0, 40.0,
    40.0, 40.0, 40.0, 40.0,
    40.0, 40.0,
)

KD_POLICY: tuple[float, ...] = (
    2.0, 2.0, 5.0,
    2.0, 2.0, 5.0,
    2.0, 2.0, 5.0,
    4.0, 4.0, 1.0, 1.0,
    2.0, 2.0, 1.0, 1.0,
    2.0, 2.0, 1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0,
)

JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "left_leg": (
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    ),
    "right_leg": (
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    ),
    "lower_body": (
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    ),
    "torso": ("waist_yaw", "waist_roll", "waist_pitch"),
    "left_arm": (
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    ),
    "right_arm": (
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ),
    "arms": (
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ),
}

POLICY_INDEX: dict[str, int] = {name: i for i, name in enumerate(POLICY_JOINT_NAMES)}
MUJOCO_INDEX: dict[str, int] = {name: i for i, name in enumerate(MUJOCO_JOINT_NAMES)}


def isaac_to_mujoco(vec: np.ndarray) -> np.ndarray:
    """策略序向量 → MuJoCo 关节序向量（长度 29）。"""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape[-1] != NUM_JOINTS:
        raise ValueError(f"expected last dim {NUM_JOINTS}, got {arr.shape}")
    return arr[..., list(ISAAC_TO_MUJOCO)]


def mujoco_to_isaac(vec: np.ndarray) -> np.ndarray:
    """MuJoCo 关节序向量 → 策略序向量（长度 29）。"""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape[-1] != NUM_JOINTS:
        raise ValueError(f"expected last dim {NUM_JOINTS}, got {arr.shape}")
    return arr[..., list(MUJOCO_TO_ISAAC)]

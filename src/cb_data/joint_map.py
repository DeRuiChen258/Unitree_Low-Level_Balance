"""G1 23/29 DOF 关节序、方向、零位与部署栈映射（事实源校验）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cb_common.errors import DataError
from cb_common.joints import (
    ACTION_SCALE_POLICY,
    DEFAULT_JOINT_POS_POLICY,
    ISAAC_TO_MUJOCO,
    KD_POLICY,
    KP_POLICY,
    MUJOCO_JOINT_NAMES,
    NUM_JOINTS,
    POLICY_JOINT_NAMES,
    isaac_to_mujoco,
    mujoco_to_isaac,
)

SUPPORTED_ROBOTS: dict[str, int] = {"unitree_g1_29dof": 29, "unitree_g1_23dof": 23}

SMPL_TO_G1_BODY: dict[str, str] = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_pitch_link",
    "right_hip": "right_hip_pitch_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}


@dataclass(frozen=True)
class DeploymentJointSpec:
    """部署栈关节规格（只读复用 g1_amp.yaml）。"""

    robot: str
    policy_names: tuple[str, ...]
    mujoco_names: tuple[str, ...]
    default_pos: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    action_scale: tuple[float, ...]
    isaac_to_mujoco: tuple[int, ...]
    mujoco_to_isaac: tuple[int, ...]
    raw: Mapping[str, Any]


def load_deployment_spec(config_path: str | Path) -> DeploymentJointSpec:
    """加载 g1_amp.yaml 并与代码事实源逐项比对，不一致立即报错。"""
    path = Path(config_path)
    if not path.is_file():
        raise DataError("deployment config not found", path=str(path))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("joint_names_mujoco", "kps", "kds", "default_joint_pos", "action_scale",
                "isaac_to_mujoco_map", "mujoco_to_isaac_map"):
        if key not in raw:
            raise DataError("deployment config missing key", key=key, path=str(path))
    mujoco_names = tuple(str(v) for v in raw["joint_names_mujoco"])
    if len(mujoco_names) != NUM_JOINTS:
        raise DataError("expected 29 DOF deployment config", found=len(mujoco_names), path=str(path))
    spec = DeploymentJointSpec(
        robot="unitree_g1_29dof",
        policy_names=POLICY_JOINT_NAMES,
        mujoco_names=mujoco_names,
        default_pos=tuple(float(v) for v in raw["default_joint_pos"]),
        kp=tuple(float(v) for v in raw["kps"]),
        kd=tuple(float(v) for v in raw["kds"]),
        action_scale=tuple(float(v) for v in raw["action_scale"]),
        isaac_to_mujoco=tuple(int(v) for v in raw["isaac_to_mujoco_map"]),
        mujoco_to_isaac=tuple(int(v) for v in raw["mujoco_to_isaac_map"]),
        raw=raw,
    )
    _assert_close("default_joint_pos", np.array(spec.default_pos), np.array(DEFAULT_JOINT_POS_POLICY), 1e-6)
    _assert_close("kps", np.array(spec.kp), np.array(KP_POLICY), 1e-6)
    _assert_close("kds", np.array(spec.kd), np.array(KD_POLICY), 1e-6)
    _assert_close("action_scale", np.array(spec.action_scale), np.array(ACTION_SCALE_POLICY), 1e-6)
    if spec.isaac_to_mujoco != ISAAC_TO_MUJOCO or spec.mujoco_to_isaac != tuple(
        int(np.argsort(np.array(ISAAC_TO_MUJOCO)))
    ):
        raise DataError("joint map mismatch with cb_common.joints", path=str(path))
    if spec.mujoco_names != MUJOCO_JOINT_NAMES:
        raise DataError("MuJoCo joint order mismatch", path=str(path))
    return spec


def _assert_close(name: str, got: np.ndarray, want: np.ndarray, tol: float) -> None:
    if got.shape != want.shape or not np.allclose(got, want, atol=tol, rtol=0.0):
        raise DataError(f"{name} mismatch with cb_common.joints", got=got.tolist(), want=want.tolist())


def policy_qpos_from_mujoco(qpos_mj: np.ndarray) -> np.ndarray:
    """MuJoCo 关节序 → 策略序。"""
    return mujoco_to_isaac(qpos_mj)


def mujoco_qpos_from_policy(qpos_policy: np.ndarray) -> np.ndarray:
    """策略序 → MuJoCo 关节序。"""
    return isaac_to_mujoco(qpos_policy)

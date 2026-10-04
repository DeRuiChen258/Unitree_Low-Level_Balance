"""S3 retarget：SMPL-22 → G1 29 DOF，MuJoCo 阻尼最小二乘 IK + 误差记录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cb_common.errors import DataError
from cb_common.joints import MUJOCO_JOINT_NAMES, NUM_JOINTS, isaac_to_mujoco, mujoco_to_isaac

from .joint_map import SMPL_TO_G1_BODY
from .schema import SMPL_JOINT_NAMES, MotionClip, RetargetedClip


@dataclass
class IKConfig:
    """IK 参数（全部来自 configs/data.yaml）。"""

    scene_path: str
    ik_iters: int = 40
    ik_lr: float = 0.65
    ik_damping: float = 1.0e-3
    ik_tolerance_m: float = 0.06
    ik_reject_m: float = 0.16
    joint_limit_margin_rad: float = 0.02
    max_step_rad: float = 0.25


class G1IKSolver:
    """基于真实 G1 MJCF 的位置 IK（阻尼最小二乘，关节限位硬约束）。"""

    def __init__(self, config: IKConfig) -> None:
        import mujoco

        path = Path(config.scene_path)
        if not path.is_file():
            raise DataError("IK scene not found", path=str(path))
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        if self.model.njnt - 1 != NUM_JOINTS:
            raise DataError("IK model must expose 29 actuated joints", found=int(self.model.njnt - 1))
        self.joint_names = MUJOCO_JOINT_NAMES
        self.qpos_addr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names])
        self.dof_addr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in self.joint_names])
        self.lo = np.array([self.model.jnt_range[i + 1][0] for i in range(NUM_JOINTS)])
        self.hi = np.array([self.model.jnt_range[i + 1][1] for i in range(NUM_JOINTS)])
        self.config = config
        self.keypoints: list[tuple[str, int]] = []
        for smpl_name, body_name in SMPL_TO_G1_BODY.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise DataError("G1 body missing for retarget", body=body_name, smpl=smpl_name)
            smpl_index = SMPL_JOINT_NAMES.index(smpl_name)
            self.keypoints.append((smpl_name, smpl_index))
            setattr(self, f"_body_{smpl_name}", body_id)
        self.body_ids = np.array([getattr(self, f"_body_{name}") for name, _ in self.keypoints], dtype=int)
        self.body_by_name = {name: int(getattr(self, f"_body_{name}")) for name, _ in self.keypoints}
        self.default_base_height = self._default_base_height()
        self.g1_leg_length = self._leg_length(np.zeros(NUM_JOINTS))

    def _default_base_height(self) -> float:
        """G1 默认姿态下的骨盆高度（用于相对高度对齐）。"""
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 0.79
        self.data.qpos[3] = 1.0
        self.mujoco.mj_forward(self.model, self.data)
        return 0.79

    def _leg_length(self, q_mj: np.ndarray) -> float:
        """G1 髋中心到踝中心的距离（默认姿态）。"""
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 0.79
        self.data.qpos[3] = 1.0
        self.data.qpos[self.qpos_addr] = q_mj
        self.mujoco.mj_forward(self.model, self.data)
        left_hip = self.data.xpos[self.body_by_name["left_hip"]]
        right_hip = self.data.xpos[self.body_by_name["right_hip"]]
        left_ankle = self.data.xpos[self.body_by_name["left_ankle"]]
        right_ankle = self.data.xpos[self.body_by_name["right_ankle"]]
        hip = 0.5 * (left_hip + right_hip)
        ankle = 0.5 * (left_ankle + right_ankle)
        return float(np.linalg.norm(hip - ankle))

    def solve(
        self,
        target_positions: np.ndarray,
        *,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, int]:
        """求解单帧 IK：返回 (q_policy, mean_err_m, limit_hits)。"""
        targets = np.asarray(target_positions, dtype=np.float64)
        if targets.shape != (len(self.keypoints), 3):
            raise DataError("IK targets shape mismatch", expected=(len(self.keypoints), 3), got=targets.shape)
        q = np.zeros(NUM_JOINTS) if q_init is None else isaac_to_mujoco(q_init).copy()
        limit_hits = 0
        mean_err = float("inf")
        jacp = np.zeros((3, self.model.nv))
        for _ in range(int(self.config.ik_iters)):
            self.data.qpos[:] = 0.0
            self.data.qpos[0:3] = base_pos
            self.data.qpos[3:7] = base_quat
            self.data.qpos[self.qpos_addr] = q
            self.mujoco.mj_forward(self.model, self.data)
            errors = np.zeros((len(self.keypoints), 3), dtype=np.float64)
            jacobians = np.zeros((3 * len(self.keypoints), NUM_JOINTS), dtype=np.float64)
            for row, body_id in enumerate(self.body_ids):
                errors[row] = targets[row] - self.data.xpos[body_id]
                self.mujoco.mj_jacBody(self.model, self.data, jacp, None, int(body_id))
                jacobians[3 * row : 3 * row + 3, :] = jacp[:, self.dof_addr]
            mean_err = float(np.mean(np.linalg.norm(errors, axis=-1)))
            if mean_err <= self.config.ik_tolerance_m:
                break
            e = errors.reshape(-1)
            jj_t = jacobians @ jacobians.T + self.config.ik_damping * np.eye(3 * len(self.keypoints))
            dq = jacobians.T @ np.linalg.solve(jj_t, e)
            dq = np.clip(dq * self.config.ik_lr, -self.config.max_step_rad, self.config.max_step_rad)
            q = q + dq
            lo = self.lo + self.config.joint_limit_margin_rad
            hi = self.hi - self.config.joint_limit_margin_rad
            hit = np.count_nonzero((q < lo) | (q > hi))
            if hit:
                limit_hits += int(hit)
            q = np.clip(q, lo, hi)
        return mujoco_to_isaac(q), mean_err, limit_hits


def retarget_clip(
    canonical,
    config: IKConfig,
    *,
    robot: str = "unitree_g1_29dof",
    solver: G1IKSolver | None = None,
) -> RetargetedClip:
    """把 CanonicalClip 重定向到 G1，记录每帧 IK 误差与限位触碰。"""
    clip: MotionClip = canonical.clip
    solver = solver or G1IKSolver(config)
    human_positions = clip.joint_world_pos  # (T,22,3) heading-free
    human = _human_leg_length(clip)
    scale = float(solver.g1_leg_length / max(human, 1e-3))
    pelvis = human_positions[:, 0, :]
    pelvis_height = pelvis[:, 1]
    median_height = float(np.median(pelvis_height))
    # G1 默认骨盆高度 + 人体骨盆高度的缩放偏差（保留蹲起/跳跃的竖直动态）
    base_z = float(solver.default_base_height) + scale * (pelvis_height - median_height)
    # 根水平位移：优先使用 meta 中的 heading-free 速度积分
    velocity = clip.meta.get("root_lin_vel_xz")
    if velocity is not None and np.asarray(velocity).shape == (clip.frames, 2):
        velocity = np.asarray(velocity, dtype=np.float64)
        # 272 维 root 速度字段是「每帧位移」，不是 m/s
        # human (x=lateral, z=forward) → mujoco (x=forward, y=lateral)
        x = np.concatenate([[0.0], np.cumsum(velocity[1:, 1])])
        y = np.concatenate([[0.0], np.cumsum(velocity[1:, 0])])
    else:
        x = np.zeros(clip.frames)
        y = np.zeros(clip.frames)
    base_positions = np.stack([x, y, base_z], axis=-1)  # MuJoCo 系
    base_quats = np.array(clip.root_quat, dtype=np.float64).copy()
    relative = (human_positions - pelvis[:, None, :]) * scale
    targets = base_positions[:, None, :] + relative[..., [2, 0, 1]]  # human (x,y,z) → mujoco (z,x,y)
    target_indices = [index for _, index in solver.keypoints]
    targets = targets[:, target_indices, :]

    qpos = np.zeros((clip.frames, NUM_JOINTS), dtype=np.float64)
    errs = np.zeros(clip.frames, dtype=np.float64)
    hits = np.zeros(clip.frames, dtype=np.int32)
    q_prev = np.array([0.0] * NUM_JOINTS)
    for t in range(clip.frames):
        q, err, hit = solver.solve(
            targets[t],
            base_pos=base_positions[t],
            base_quat=base_quats[t],
            q_init=q_prev,
        )
        qpos[t], errs[t], hits[t] = q, err, hit
        q_prev = q
    qpos = _smooth_trajectory(qpos, window=5)
    qvel = np.gradient(qpos, 1.0 / clip.fps, axis=0)
    clipped = qpos.copy()
    clipped[errs > config.ik_reject_m] = np.array([0.0] * NUM_JOINTS)
    return RetargetedClip(
        retarget_id=f"g1:{clip.clip_id}",
        robot=robot,
        joint_names=MUJOCO_JOINT_NAMES,
        qpos=qpos,
        qvel=qvel,
        keypoint_pos=targets,
        retarget_err=errs,
        source_clip=clip.clip_id,
        root_pos=base_positions,
        root_quat=base_quats,
        contacts=clip.contacts,
        joint_limit_hits=hits,
        fps=clip.fps,
        meta={
            "scale": scale,
            "ik_tolerance_m": config.ik_tolerance_m,
            "ik_reject_m": config.ik_reject_m,
            "text": clip.text,
            "rejected_frames": int(np.count_nonzero(errs > config.ik_reject_m)),
            "smoothed": True,
            "smoothing_window": 5,
        },
    )


def _human_leg_length(clip: MotionClip) -> float:
    hip = 0.5 * (clip.joint_world_pos[:, 1] + clip.joint_world_pos[:, 2])
    ankle = 0.5 * (clip.joint_world_pos[:, 7] + clip.joint_world_pos[:, 8])
    return float(np.median(np.linalg.norm(hip - ankle, axis=-1)))


def _smooth_trajectory(qpos: np.ndarray, window: int = 5) -> np.ndarray:
    """滑动平均平滑 IK 解（抑制 DLS 解跳变），窗口边缘做常数填充。"""
    if window <= 1:
        return qpos
    pad = window // 2
    padded = np.pad(qpos, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(qpos.shape[1])], axis=-1)
    return out

"""全身 IK + 关节 PD 控制器（第一阶段 joint-space，第二阶段 IK；第 28 节）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cb_common.errors import PolicyError
from cb_common.joints import KD_POLICY, KP_POLICY, NUM_JOINTS


@dataclass
class BodyTask:
    """单个笛卡尔任务：位置/朝向目标（世界系），可选权重。"""

    body: str
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    pos_weight: float = 1.0
    rot_weight: float = 0.6
    task_type: str = "body"          # body | com（com 使用 mj_jacSubtreeCom，只约束 xy）


@dataclass
class WholeBodyIK:
    """多任务阻尼最小二乘 IK（29 DOF，基座给定，解腿/躯干/手臂关节角）。"""

    scene_path: str
    max_iters: int = 12
    damping: float = 1e-2
    max_step_rad: float = 0.15
    position_margin_rad: float = 0.02
    _mujoco: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _data: object = field(default=None, init=False, repr=False)
    _body_ids: dict[str, int] = field(default_factory=dict, init=False)
    _qpos_addr: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _dof_addr: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _lo: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _hi: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    joint_names: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        import mujoco

        path = Path(self.scene_path)
        if not path.is_file():
            raise PolicyError("IK scene not found", path=str(path))
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        model = self._model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
        if len(names) != NUM_JOINTS:
            raise PolicyError("expected 29 actuated joints", found=len(names))
        self.joint_names = tuple(names)
        self._qpos_addr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
        self._dof_addr = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
        self._lo = np.array([model.jnt_range[i + 1][0] for i in range(NUM_JOINTS)])
        self._hi = np.array([model.jnt_range[i + 1][1] for i in range(NUM_JOINTS)])
        for body in (
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "torso_link",
            "left_elbow_link",
            "right_elbow_link",
            "head_link",
        ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id >= 0:
                self._body_ids[body] = int(body_id)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        """底层 MjModel（只读）。"""
        return self._model

    def body_id(self, name: str) -> int:
        """返回 body id；缺失时报错。"""
        if name not in self._body_ids:
            raise PolicyError("body not found in G1 model", body=name)
        return self._body_ids[name]

    def solve(
        self,
        *,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        tasks: Sequence[BodyTask],
        q_init: np.ndarray,
        frozen_joints: Sequence[str] = (),
        posture_reference: Mapping[str, float] | None = None,
        posture_weight: float = 0.06,
    ) -> tuple[np.ndarray, float, int]:
        """求解关节角（MuJoCo 关节序）：返回 (q, 任务误差均值, 限位触碰)。

        `posture_reference` 为低权重关节姿态正则（零空间偏好），用于把腿部拉向
        「微屈膝」构型，避免直腿奇异位形下 IK 无法下蹲（进而被地面反推、双脚外滑）。
        """
        mj = self._mujoco
        model, data = self._model, self._data
        q = np.asarray(q_init, dtype=np.float64).copy()
        frozen = np.array([self.joint_names.index(name) for name in frozen_joints if name in self.joint_names], dtype=int)
        posture_names = [name for name in (posture_reference or {}) if name in self.joint_names]
        posture_idx = np.array([self.joint_names.index(name) for name in posture_names], dtype=int)
        posture_ref = np.array([float(posture_reference[name]) for name in posture_names], dtype=np.float64)
        limit_hits = 0
        mean_err = float("inf")
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        prev_q = q.copy()
        prev_penetration = self._robot_penetration(q, base_pos, base_quat)
        last_free_q: np.ndarray | None = None
        last_free_err = float("inf")
        for _ in range(int(self.max_iters)):
            data.qpos[:] = 0.0
            data.qpos[0:3] = base_pos
            data.qpos[3:7] = base_quat
            data.qpos[self._qpos_addr] = q
            mj.mj_forward(model, data)
            errors: list[np.ndarray] = []
            jac_blocks: list[np.ndarray] = []
            weights: list[float] = []
            for task in tasks:
                if task.task_type == "com":
                    if task.position is None:
                        continue
                    com = np.array(data.subtree_com[0], dtype=np.float64)
                    err = (np.asarray(task.position, dtype=np.float64)[:2] - com[:2]) * np.sqrt(task.pos_weight)
                    mj.mj_jacSubtreeCom(model, data, jacp, 0)
                    errors.append(err)
                    jac_blocks.append(jacp[:2, self._dof_addr] * np.sqrt(task.pos_weight))
                    weights.append(task.pos_weight)
                    continue
                body_id = self.body_id(task.body)
                if task.position is not None:
                    err = np.asarray(task.position, dtype=np.float64) - np.array(data.xpos[body_id])
                    mj.mj_jacBody(model, data, jacp, jacr, body_id)
                    errors.append(err * np.sqrt(task.pos_weight))
                    jac_blocks.append(jacp[:, self._dof_addr] * np.sqrt(task.pos_weight))
                    weights.append(task.pos_weight)
                if task.rotation is not None:
                    target = np.asarray(task.rotation, dtype=np.float64).reshape(3, 3)
                    current = np.array(data.xmat[body_id]).reshape(3, 3)
                    rot_err = _rotation_error(current, target)
                    mj.mj_jacBody(model, data, jacp, jacr, body_id)
                    errors.append(rot_err * np.sqrt(task.rot_weight))
                    jac_blocks.append(jacr[:, self._dof_addr] * np.sqrt(task.rot_weight))
                    weights.append(task.rot_weight)
            if posture_idx.size:
                # 低权重身份块：仅在主任务零空间内把关节拉向参考姿态
                weight = float(np.sqrt(max(posture_weight, 0.0)))
                block = np.zeros((posture_idx.size, len(self._dof_addr)), dtype=np.float64)
                block[np.arange(posture_idx.size), posture_idx] = weight
                errors.append(weight * (posture_ref - q[posture_idx]))
                jac_blocks.append(block)
                weights.append(max(posture_weight, 0.0))
            if not errors:
                break
            error_vec = np.concatenate(errors)
            jac = np.vstack(jac_blocks)
            if frozen.size:
                jac[:, frozen] = 0.0
            mean_err = float(np.linalg.norm(error_vec) / max(1, len(errors)))
            if mean_err <= 1e-4:
                break
            jjt = jac @ jac.T + self.damping * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, error_vec)
            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)
            if frozen.size:
                dq[frozen] = 0.0
            q = q + dq
            lo = self._lo + self.position_margin_rad
            hi = self._hi - self.position_margin_rad
            hits = int(np.count_nonzero((q < lo) | (q > hi)))
            limit_hits += hits
            q = np.clip(q, lo, hi)
            # 自穿模线搜索：按穿透深度而非接触点计数拒绝解，避免计数抖动导致 IK 冻结
            penetration = self._robot_penetration(q, base_pos, base_quat)
            if penetration > prev_penetration:
                best_q, best_penetration = prev_q, prev_penetration
                for alpha in (0.5, 0.25, 0.1):
                    candidate = prev_q + alpha * (q - prev_q)
                    candidate_penetration = self._robot_penetration(candidate, base_pos, base_quat)
                    if candidate_penetration <= prev_penetration:
                        best_q, best_penetration = candidate, candidate_penetration
                        break
                q = best_q
                penetration = best_penetration
            prev_q, prev_penetration = q.copy(), penetration
            if penetration <= 1e-4 and mean_err < last_free_err:
                last_free_q = q.copy()
                last_free_err = mean_err
        # 最终解若仍有自碰撞，回退到迭代过程中最优的无碰撞构型（避免执行器顶穿躯干/手臂）
        if self._robot_penetration(q, base_pos, base_quat) > 1e-4 and last_free_q is not None:
            q = last_free_q
            mean_err = last_free_err
        return q, mean_err, limit_hits

    def _robot_contacts(self, q: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> int:
        """统计机器人体段之间的接触数（排除地面/足底接触）。"""
        mj = self._mujoco
        model, data = self._model, self._data
        data.qpos[:] = 0.0
        data.qpos[0:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qpos[self._qpos_addr] = q
        mj.mj_forward(model, data)
        count = 0
        for i in range(data.ncon):
            b1 = int(model.geom_bodyid[data.contact[i].geom1])
            b2 = int(model.geom_bodyid[data.contact[i].geom2])
            if b1 > 0 and b2 > 0:
                count += 1
        return count

    def _robot_penetration(self, q: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> float:
        """机器人体段之间的自穿模深度之和（m）。

        比「接触点计数」更稳健：几何体间的多个接触点不会被重复计数放大，
        且允许在同等深度下调整构型（穿模计数在迭代中抖动会导致 IK 被反复否决）。
        """
        mj = self._mujoco
        model, data = self._model, self._data
        data.qpos[:] = 0.0
        data.qpos[0:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qpos[self._qpos_addr] = q
        mj.mj_forward(model, data)
        total = 0.0
        for i in range(data.ncon):
            b1 = int(model.geom_bodyid[data.contact[i].geom1])
            b2 = int(model.geom_bodyid[data.contact[i].geom2])
            if b1 > 0 and b2 > 0:
                total += max(0.0, -float(data.contact[i].dist))
        return total


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """旋转误差 → 前两行叉积近似（世界系旋转向量）。"""
    rot = target @ current.T
    angle = np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(3)
    return axis / norm * angle


@dataclass
class JointPDController:
    """关节空间 PD（MuJoCo position 执行器直接跟踪，motor 执行器用增益计算力矩）。"""

    kp: np.ndarray = field(default_factory=lambda: np.array(KP_POLICY, dtype=np.float64))
    kd: np.ndarray = field(default_factory=lambda: np.array(KD_POLICY, dtype=np.float64))
    target: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    previous_target: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    max_delta_per_step: float = 0.06

    def set_target(self, target: np.ndarray, *, rate_limit: bool = True) -> np.ndarray:
        """设置目标（可选每步变化限幅，保证命令连续、类人平滑）。"""
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if target.size != NUM_JOINTS:
            raise PolicyError("joint target dim mismatch", size=int(target.size))
        if rate_limit:
            delta = np.clip(target - self.previous_target, -self.max_delta_per_step, self.max_delta_per_step)
            target = self.previous_target + delta
        self.previous_target = target.copy()
        self.target = target
        return target

    def reset(self, target: np.ndarray) -> None:
        """复位到给定目标。"""
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.previous_target = self.target.copy()

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """计算 PD 力矩（用于 motor 执行器）。"""
        return self.kp * (self.target - q) - self.kd * dq

    def describe(self) -> Mapping[str, float]:
        """控制器摘要。"""
        return {"max_delta_per_step": self.max_delta_per_step, "steps": float(self.previous_target.sum())}

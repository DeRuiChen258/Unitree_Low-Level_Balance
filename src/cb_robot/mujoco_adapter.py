"""MuJoCo 适配器：真实 G1 MJCF + PD 执行 + 状态读取（与真机同接口）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cb_common.errors import RobotError
from cb_common.joints import (
    DEFAULT_JOINT_POS_POLICY,
    MUJOCO_JOINT_NAMES,
    NUM_JOINTS,
    isaac_to_mujoco,
    mujoco_to_isaac,
)
from cb_common.types import JointCommand, RobotMode, RobotState

from .interface import RobotInterface


def _polygon_signed_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """点到凸多边形的最小有符号距离（内部为正）。"""
    if polygon.shape[0] < 3:
        return 0.0
    n = polygon.shape[0]
    inside = True
    min_dist = float("inf")
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        dist = float(np.dot(point - a, normal))
        if dist < 0.0:
            inside = False
        min_dist = min(min_dist, abs(dist))
    return min_dist if inside else -min_dist


@dataclass
class MuJoCoAdapter(RobotInterface):
    """G1 MuJoCo 适配器：支持 position 与 motor 两类执行器。"""

    scene_path: str
    sim_dt: float = 0.005
    control_dt: float = 0.02
    init_height: float = 0.79
    name: str = "mujoco"
    headless: bool = True
    _mujoco: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _data: object = field(default=None, init=False, repr=False)
    _qpos_addr: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _qvel_addr: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _actuator_type: str = field(default="position", init=False)
    _q_target_mj: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _kp_mj: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _kd_mj: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _ctrlrange: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _foot_geom_ids: tuple[int, ...] = field(default=(), init=False)
    _foot_body_ids: tuple[int, ...] = field(default=(), init=False)
    _mode: RobotMode = field(default=RobotMode.IDLE, init=False)
    _time: float = field(default=0.0, init=False)
    _renderer: object | None = field(default=None, init=False, repr=False)

    def connect(self) -> None:
        """加载 MJCF、校验 29 DOF 并重置到站立。"""
        import mujoco

        path = Path(self.scene_path)
        if not path.is_file():
            raise RobotError("MuJoCo scene not found", path=str(path))
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        model = self._model
        if model.njnt - 1 != NUM_JOINTS:
            raise RobotError("expected 29 actuated joints", found=int(model.njnt - 1), path=str(path))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i + 1) for i in range(NUM_JOINTS)]
        if tuple(names) != MUJOCO_JOINT_NAMES:
            raise RobotError("joint order mismatch with deployment fact source", found=names)
        self._qpos_addr = np.array([model.jnt_qposadr[model.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self._qvel_addr = np.array([model.jnt_dofadr[model.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        # 执行器类型：position（gainprm[0]!=0 且 biasprm[1]!=0）或 motor（bias 全 0）
        gain = float(model.actuator_gainprm[0, 0])
        bias = float(model.actuator_biasprm[0, 1])
        self._actuator_type = "position" if abs(bias) > 1e-9 and abs(gain) > 1e-9 else "motor"
        self._ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float64)
        model.opt.timestep = float(self.sim_dt)
        self._q_target_mj = isaac_to_mujoco(np.array(DEFAULT_JOINT_POS_POLICY))
        self._kp_mj = np.full(NUM_JOINTS, 100.0)
        self._kd_mj = np.full(NUM_JOINTS, 2.0)
        # 足端几何：按 body 名匹配 ankle_roll_link；同时记录足底 body 用于支撑多边形
        foot_geoms: list[int] = []
        foot_bodies: list[int] = []
        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.endswith("ankle_roll_link"):
                foot_bodies.append(body_id)
                for geom_id in range(model.ngeom):
                    if model.geom_bodyid[geom_id] == body_id and model.geom_contype[geom_id] != 0:
                        foot_geoms.append(geom_id)
        self._foot_geom_ids = tuple(foot_geoms)
        self._foot_body_ids = tuple(foot_bodies)
        self.reset()
        self.connected = True
        self._mode = RobotMode.STAND

    def reset(self, q_policy: np.ndarray | None = None, *, base_height: float | None = None) -> None:
        """重置模型到默认/指定姿态。"""
        assert self._data is not None and self._model is not None
        q = np.array(DEFAULT_JOINT_POS_POLICY if q_policy is None else q_policy, dtype=np.float64)
        if q.shape != (NUM_JOINTS,):
            raise RobotError("q_policy shape invalid", shape=str(q.shape))
        data = self._data
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[2] = float(self.init_height if base_height is None else base_height)
        data.qpos[3] = 1.0  # wxyz identity
        data.qpos[self._qpos_addr] = q
        self._q_target_mj = q.copy()
        self._time = 0.0
        self._mujoco.mj_forward(self._model, data)  # type: ignore[union-attr]

    def close(self) -> None:
        """释放渲染器与模型引用。"""
        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if callable(close):
                close()
            self._renderer = None
        self.connected = False

    def set_mode(self, mode: RobotMode) -> None:
        """记录模式（仿真无模式握手）。"""
        self._mode = mode

    @property
    def mode(self) -> RobotMode:
        """当前模式。"""
        return self._mode

    @property
    def model(self) -> object:
        """底层 MjModel（可视化/评测使用）。"""
        return self._model

    @property
    def data(self) -> object:
        """底层 MjData（可视化/评测使用）。"""
        return self._data

    def send_command(self, command: JointCommand) -> None:
        """保存目标（策略序 → MuJoCo 序）与增益。"""
        self._q_target_mj = isaac_to_mujoco(command.q_target)
        self._kp_mj = isaac_to_mujoco(command.kp)
        self._kd_mj = isaac_to_mujoco(command.kd)

    def _apply_control(self) -> None:
        assert self._model is not None and self._data is not None
        data = self._data
        if self._actuator_type == "position":
            ctrl = np.clip(self._q_target_mj, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        else:
            q = data.qpos[self._qpos_addr]
            dq = data.qvel[self._qvel_addr]
            tau = self._kp_mj * (self._q_target_mj - q) - self._kd_mj * dq
            ctrl = np.clip(tau, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        data.ctrl[:] = ctrl

    def advance(self, dt: float) -> None:
        """按仿真步长推进（内部按 sim_dt 取整）。"""
        if not self.connected:
            raise RobotError("MuJoCo adapter is not connected")
        assert self._mujoco is not None and self._model is not None and self._data is not None
        steps = max(1, int(round(dt / self.sim_dt)))
        for _ in range(steps):
            self._apply_control()
            self._mujoco.mj_step(self._model, self._data)
            self._time += self.sim_dt

    def step_physics(self, n: int = 1) -> None:
        """直接推进 n 个物理步（不重复应用控制）。"""
        assert self._mujoco is not None and self._model is not None and self._data is not None
        for _ in range(n):
            self._apply_control()
            self._mujoco.mj_step(self._model, self._data)
            self._time += self.sim_dt

    def read_state(self) -> RobotState:
        """读取统一状态（含 CoM / CP / 支撑域余量）。"""
        if not self.connected:
            raise RobotError("MuJoCo adapter is not connected")
        assert self._mujoco is not None and self._model is not None and self._data is not None
        mj, model, data = self._mujoco, self._model, self._data
        mj.mj_forward(model, data)
        base_quat = np.array(data.qpos[3:7], dtype=np.float64)  # wxyz
        base_pos = np.array(data.qpos[0:3], dtype=np.float64)
        base_lin = np.array(data.qvel[0:3], dtype=np.float64)
        base_ang = np.array(data.qvel[3:6], dtype=np.float64)
        q_mj = np.array(data.qpos[self._qpos_addr], dtype=np.float64)
        dq_mj = np.array(data.qvel[self._qvel_addr], dtype=np.float64)
        joint_pos = mujoco_to_isaac(q_mj)
        joint_vel = mujoco_to_isaac(dq_mj)
        torque = None
        if self._actuator_type == "motor":
            torque = mujoco_to_isaac(np.array(data.actuator_force[:NUM_JOINTS], dtype=np.float64))
        contact = self._read_contacts()
        com = np.array(data.subtree_com[0], dtype=np.float64)
        com_vel = np.array(data.cvel[0][3:6], dtype=np.float64) if data.cvel.shape[0] else np.zeros(3)
        height = float(max(1e-6, com[2]))
        omega = np.sqrt(9.81 / height)
        cp = np.array([com[0] + com_vel[0] / omega, com[1] + com_vel[1] / omega])
        polygon = self._support_polygon()
        margin = _polygon_signed_margin(com[:2], polygon) if polygon.shape[0] >= 3 else 0.0
        return RobotState(
            timestamp=self._time,
            base_pos=base_pos,
            base_quat=base_quat,
            base_lin_vel=base_lin,
            base_ang_vel=base_ang,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torque=torque,
            contact=contact,
            com=com,
            com_vel=com_vel,
            cp=cp,
            support_margin=margin,
            frame_id=int(round(self._time / self.sim_dt)),
        )

    def _read_contacts(self) -> tuple[bool, bool]:
        """按左右足端几何判定接触。"""
        assert self._model is not None and self._data is not None
        left = right = False
        left_geoms = set(self._foot_geom_ids[: len(self._foot_geom_ids) // 2]) if self._foot_geom_ids else set()
        for i in range(self._data.ncon):
            con = self._data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in self._foot_geom_ids or g2 in self._foot_geom_ids:
                is_left = g1 in left_geoms or g2 in left_geoms
                if is_left:
                    left = True
                else:
                    right = True
        return left, right

    def _support_polygon(self) -> np.ndarray:
        """由足端几何位置构造支撑多边形（xy，双支撑/单支撑自适应）。"""
        assert self._model is not None and self._data is not None
        points: list[np.ndarray] = []
        for geom_id in self._foot_geom_ids:
            pos = np.array(self._data.geom_xpos[geom_id], dtype=np.float64)
            points.append(pos[[0, 1]])
        if not points:
            return np.zeros((0, 2))
        pts = np.array(points)
        if pts.shape[0] >= 3:
            center = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            order = np.argsort(angles)
            pts = pts[order]
        return pts

    def render(self, width: int = 640, height: int = 480, camera: str | None = None) -> np.ndarray:
        """离屏渲染一帧 RGB 图像（用于演示视频）。"""
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        renderer = self._renderer
        if camera:
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            renderer.update_scene(self._data, camera=cam_id if cam_id >= 0 else camera)
        else:
            renderer.update_scene(self._data)
        return renderer.render().copy()

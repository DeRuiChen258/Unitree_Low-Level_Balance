"""教师策略封装：复用 G1_run/G1_Walk 的 Sim2Sim ONNX 策略（只读复用）。

设计要点：
- `TeacherCore` 加载配置 + ONNX 会话（可被多个环境共享，CPU 推理）；
- `TeacherRuntime` 每个环境持有独立 MjModel/MjData（支持域随机化与并行）；
- 观测构造与上游 `sim2sim_walk.py` 完全一致（96 维 × 4 历史，group-major 堆叠）。
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cb_common.errors import PolicyError
from cb_common.joints import NUM_JOINTS, isaac_to_mujoco, mujoco_to_isaac


def _load_upstream_module(deploy_dir: Path):
    """按路径加载上游 sim2sim_walk 模块（不修改上游仓库）。"""
    module_path = deploy_dir / "g1_python" / "sim2sim_walk.py"
    if not module_path.is_file():
        raise PolicyError("upstream sim2sim_walk.py not found", path=str(module_path))
    spec = importlib.util.spec_from_file_location("cb_upstream_sim2sim_walk", module_path)
    if spec is None or spec.loader is None:
        raise PolicyError("failed to create module spec", path=str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["cb_upstream_sim2sim_walk"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class TeacherCore:
    """共享的教师策略核心：配置、ONNX 会话与关节事实源。"""

    deploy_dir: str
    config_path: str
    onnx_name: str
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    module: Any = field(init=False, repr=False)
    config: Any = field(init=False)
    session: Any = field(init=False, repr=False)
    xml_path: Path = field(init=False)
    joint_qpos_addrs: np.ndarray = field(init=False)
    joint_qvel_addrs: np.ndarray = field(init=False)
    actuator_ids: np.ndarray = field(init=False)
    kp_mj: np.ndarray = field(init=False)
    kd_mj: np.ndarray = field(init=False)
    default_qpos_mj: np.ndarray = field(init=False)
    default_qpos_policy: np.ndarray = field(init=False)
    action_scale_policy: np.ndarray = field(init=False)
    init_height: float = field(init=False)
    policy_decimation: int = field(init=False)
    sim_dt: float = field(init=False)
    control_dt: float = field(init=False)

    def __post_init__(self) -> None:
        import onnxruntime as ort
        import yaml

        deploy = Path(self.deploy_dir)
        config_file = Path(self.config_path)
        if not config_file.is_file():
            raise PolicyError("teacher config not found", path=str(config_file))
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        self.config = SimpleNamespace(**raw)
        self.xml_path = deploy / "assets" / Path(str(raw["xml_path"])).name
        if not self.xml_path.is_file():
            raise PolicyError("teacher MJCF not found", path=str(self.xml_path))
        self.module = _load_upstream_module(deploy)
        model = self.module.mujoco.MjModel.from_xml_path(str(self.xml_path))
        names = list(raw["joint_names_mujoco"])
        self.joint_qpos_addrs = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
        self.joint_qvel_addrs = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
        self.actuator_ids = np.array([model.actuator(n).id for n in raw["actuator_names_mujoco"]])
        self.kp_mj = np.asarray(raw["kps"], dtype=np.float64)[np.asarray(raw["isaac_to_mujoco_map"], dtype=int)]
        self.kd_mj = np.asarray(raw["kds"], dtype=np.float64)[np.asarray(raw["isaac_to_mujoco_map"], dtype=int)]
        self.default_qpos_policy = np.asarray(raw["default_joint_pos"], dtype=np.float64)
        self.default_qpos_mj = self.default_qpos_policy[np.asarray(raw["isaac_to_mujoco_map"], dtype=int)]
        self.action_scale_policy = np.asarray(raw["action_scale"], dtype=np.float64)
        self.init_height = float(raw.get("init_height", 0.90))
        self.sim_dt = float(raw["sim_dt"])
        self.control_dt = float(raw["control_dt"])
        self.policy_decimation = int(round(self.control_dt / self.sim_dt))
        onnx_path = deploy / "exported_policy" / self.onnx_name
        if not onnx_path.is_file():
            raise PolicyError("teacher ONNX not found", path=str(onnx_path))
        self.session = ort.InferenceSession(str(onnx_path), providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def new_runtime(self) -> TeacherRuntime:
        """创建独立运行时（每个环境一个 MjModel/MjData）。"""
        return TeacherRuntime(core=self)


@dataclass
class TeacherRuntime:
    """单环境教师运行态：独立 MuJoCo 数据、观测历史与上一动作。"""

    core: TeacherCore
    mujoco: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    data: Any = field(init=False, repr=False)
    obs_history: deque = field(init=False, repr=False)
    last_action: np.ndarray = field(init=False)
    last_target_policy: np.ndarray = field(init=False)
    base_body_mass: np.ndarray = field(init=False)
    base_geom_friction: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        mujoco = self.core.module.mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.core.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.core.sim_dt
        self.obs_history = deque(maxlen=4)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.last_target_policy = np.array(self.core.default_qpos_policy, dtype=np.float64)
        self.base_body_mass = np.array(self.model.body_mass, dtype=np.float64)
        self.base_geom_friction = np.array(self.model.geom_friction, dtype=np.float64)
        self.reset()

    def reset(self, *, randomize: np.random.Generator | None = None, qpos: np.ndarray | None = None) -> None:
        """重置到默认姿态（可选域随机化）。"""
        mj, model, data = self.mujoco, self.model, self.data
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[2] = self.core.init_height
        data.qpos[3] = 1.0
        joints = np.array(self.core.default_qpos_policy if qpos is None else qpos, dtype=np.float64)
        data.qpos[self.core.joint_qpos_addrs] = isaac_to_mujoco(joints)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mj.mj_forward(model, data)
        self.last_action[:] = 0.0
        self.last_target_policy = joints.copy()
        self.obs_history.clear()
        if randomize is not None:
            self._apply_domain_randomization(randomize)
        for _ in range(4):
            obs = self.observation(np.zeros(3))
            self.obs_history.append(obs)

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
        """质量/摩擦/PD 增益域随机化（范围由调用方保证）。"""
        self.model.body_mass[:] *= 1.0
        mass_scale = float(rng.uniform(0.9, 1.1))
        self.model.body_mass[1:] *= mass_scale
        friction_scale = float(rng.uniform(0.8, 1.2))
        self.model.geom_friction[:, 0] = np.clip(self.model.geom_friction[:, 0] * friction_scale, 0.05, None)

    def observation(self, command: np.ndarray) -> np.ndarray:
        """构造上游 96 维单帧观测。"""
        c = self.core
        data = self.data
        quat_wxyz = data.qpos[3:7]
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        proj_grav = c.module.compute_projected_gravity(quat_xyzw)
        base_ang = data.qvel[3:6]
        qpos_mj = data.qpos[c.joint_qpos_addrs]
        qvel_mj = data.qvel[c.joint_qvel_addrs]
        qpos_rel = mujoco_to_isaac(qpos_mj) - c.default_qpos_policy
        qvel_isaac = mujoco_to_isaac(qvel_mj)
        return c.module.build_obs(
            base_ang,
            proj_grav,
            np.asarray(command, dtype=np.float32).reshape(3),
            qpos_rel,
            qvel_isaac,
            self.last_action,
            c.config,
        )

    def stacked_observation(self, command: np.ndarray) -> np.ndarray:
        """历史堆叠（group-major，与训练一致）→ 384 维。"""
        obs = self.observation(command)
        self.obs_history.append(obs)
        frames = np.array(list(self.obs_history), dtype=np.float64)
        n = NUM_JOINTS
        indices = [(0, 3), (3, 6), (6, 9), (9, 9 + n), (9 + n, 9 + 2 * n), (9 + 2 * n, 9 + 3 * n)]
        return np.concatenate([frames[:, start:end].reshape(-1) for start, end in indices])

    def act(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """运行教师 ONNX：返回 (raw_action 29, target_qpos_policy 29)。"""
        obs = self.stacked_observation(command)
        outputs = self.core.session.run(self.core.output_names, {self.core.input_name: obs[None, :].astype(np.float32)})
        action = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if action.size != NUM_JOINTS:
            raise PolicyError("teacher action dim mismatch", size=int(action.size))
        self.last_action = action.astype(np.float64)
        target = action * self.core.action_scale_policy + self.core.default_qpos_policy
        self.last_target_policy = target
        return action, target

    def apply_pd(
        self,
        target_policy: np.ndarray,
        target_vel_policy: np.ndarray | None = None,
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> None:
        """PD 力矩下发（MuJoCo 执行器为 motor）。"""
        target_mj = isaac_to_mujoco(np.asarray(target_policy, dtype=np.float64))
        target_vel_mj = (
            np.zeros(NUM_JOINTS)
            if target_vel_policy is None
            else isaac_to_mujoco(np.asarray(target_vel_policy, dtype=np.float64))
        )
        q = self.data.qpos[self.core.joint_qpos_addrs]
        dq = self.data.qvel[self.core.joint_qvel_addrs]
        tau = (self.core.kp_mj * kp_scale) * (target_mj - q) + (self.core.kd_mj * kd_scale) * (target_vel_mj - dq)
        self.data.ctrl[self.core.actuator_ids] = tau

    @property
    def base_position(self) -> np.ndarray:
        """基座位置。"""
        return np.array(self.data.qpos[0:3], dtype=np.float64)

    @property
    def base_quaternion(self) -> np.ndarray:
        """基座姿态（wxyz）。"""
        return np.array(self.data.qpos[3:7], dtype=np.float64)

    @property
    def joint_pos_policy(self) -> np.ndarray:
        """策略序关节角。"""
        return mujoco_to_isaac(self.data.qpos[self.core.joint_qpos_addrs])

    @property
    def joint_vel_policy(self) -> np.ndarray:
        """策略序关节速度。"""
        return mujoco_to_isaac(self.data.qvel[self.core.joint_qvel_addrs])

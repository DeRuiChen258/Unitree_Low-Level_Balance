"""PickupController：原语执行 + 全身 IK + 关节 PD + 抓取 + 平衡监控（500Hz 仿真/100Hz 控制）。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import PolicyError
from cb_common.joints import DEFAULT_JOINT_POS_POLICY, MUJOCO_JOINT_NAMES, NUM_JOINTS, isaac_to_mujoco

from .balance_monitor import BalanceMonitor, BalanceState, signed_polygon_margin
from .decision import DecisionModel, PickupDecision
from .footstep import FootstepPlanner
from .grasp import GraspMock
from .motion_manager import ManagerCommand, MotionManager, PickupPhase
from .primitives import (
    BendPrimitive,
    BodyTarget,
    GraspPrimitive,
    LiftPrimitive,
    LungePrimitive,
    Primitive,
    ReachPrimitive,
    RecoveryPrimitive,
    StandPrimitive,
    StandUpPrimitive,
    StepPrimitive,
    embrace_points,
    quintic,
    yaw_matrix,
)
from .scenarios import PickupScenario, RandomizationConfig
from .whole_body import BodyTask, JointPDController, WholeBodyIK


@dataclass
class PickupConfig:
    """拾取系统配置（configs/g1_pickup.yaml）。"""

    scene_path: str
    control_dt: float = 0.01
    sim_dt: float = 0.002
    decision_dt: float = 0.05
    footstep_dt: float = 0.1
    trunk_pitch_target: float = 0.50
    stance_width: float = 0.24
    reach_side: str = "right"
    lift_height: float = 0.18
    grasp_distance: float = 0.12
    reach_threshold: float = 0.15
    max_steps: int = 3
    reach_margin: float = 0.015
    step_margin: float = 0.0
    critical_margin: float = -0.05
    max_episode_s: float = 20.0
    two_handed: bool = True
    foot_anchor: bool = True
    bend_crouch_depth: float = 0.17
    bend_hip_shift: float = 0.05
    posture_weight: float = 0.06
    reach_com_shift_m: float = 0.0
    com_margin_target: float = 0.045
    com_margin_gain: float = 0.8
    carry_pull_duration_s: float = 1.2
    object_mass_ratio: float = 0.50
    object_half_height: float = 0.31
    object_half_depth: float = 0.14
    object_half_width: float = 0.14
    #: 物体台面高度（m）：G1 手臂只能下探到约 0.56 m，地面箱体只能抓在近顶边；
    #: 抬高 0.25~0.30 m 后台面箱体的握持点落在侧面中部（符合人抱箱子的姿态）
    object_base_height_m: float = 0.0
    embrace_clearance_m: float = 0.055
    embrace_side_drop_m: float = 0.06
    wrist_flush_face: bool = True
    wrist_rot_weight: float = 0.35
    base_assist: dict[str, float] = field(default_factory=dict)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], **overrides: Any) -> PickupConfig:
        """从 `configs/g1_pickup.yaml` 映射构造配置（CLI 脚本统一入口，避免逐字段重复）。"""

        def value(key: str, default: Any) -> Any:
            raw = config.get(key, default)
            return default if raw is None else raw

        def mapping(key: str) -> dict[str, Any]:
            """取子配置为普通 dict（兼容 Config / Mapping 两种来源）。"""
            raw = config.get(key, {}) or {}
            if isinstance(raw, Mapping):
                return dict(raw)
            return {name: getattr(raw, name) for name in getattr(raw, "__dict__", {}).get("data", {})}

        fields: dict[str, Any] = {
            "scene_path": str(value("scene_path", "")),
            "control_dt": float(value("control_dt", 0.01)),
            "sim_dt": float(value("sim_dt", 0.002)),
            "decision_dt": float(value("decision_dt", 0.05)),
            "footstep_dt": float(value("footstep_dt", 0.1)),
            "trunk_pitch_target": float(value("trunk_pitch_target", 0.50)),
            "stance_width": float(value("stance_width", 0.24)),
            "reach_side": str(value("reach_side", "right")),
            "lift_height": float(value("lift_height", 0.18)),
            "grasp_distance": float(value("grasp_distance", 0.12)),
            "reach_threshold": float(value("reach_threshold", 0.15)),
            "max_steps": int(value("max_steps", 3)),
            "reach_margin": float(value("reach_margin", 0.015)),
            "step_margin": float(value("step_margin", 0.0)),
            "critical_margin": float(value("critical_margin", -0.05)),
            "max_episode_s": float(value("max_episode_s", 20.0)),
            "two_handed": bool(value("two_handed", True)),
            "foot_anchor": bool(value("foot_anchor", True)),
            "bend_crouch_depth": float(value("bend_crouch_depth", 0.17)),
            "bend_hip_shift": float(value("bend_hip_shift", 0.05)),
            "posture_weight": float(value("posture_weight", 0.06)),
            "reach_com_shift_m": float(value("reach_com_shift_m", 0.0)),
            "com_margin_target": float(value("com_margin_target", 0.045)),
            "com_margin_gain": float(value("com_margin_gain", 0.8)),
            "carry_pull_duration_s": float(value("carry_pull_duration_s", 1.2)),
            "object_mass_ratio": float(value("object_mass_ratio", 0.50)),
            "object_half_height": float(value("object_half_height", 0.31)),
            "object_half_depth": float(value("object_half_depth", 0.14)),
            "object_half_width": float(value("object_half_width", 0.14)),
            "object_base_height_m": float(value("object_base_height_m", 0.0)),
            "embrace_clearance_m": float(value("embrace_clearance_m", 0.055)),
            "embrace_side_drop_m": float(value("embrace_side_drop_m", 0.06)),
            "wrist_flush_face": bool(value("wrist_flush_face", True)),
            "wrist_rot_weight": float(value("wrist_rot_weight", 0.35)),
            "base_assist": mapping("base_assist"),
        }
        fields.update(overrides)
        return cls(**fields)


@dataclass
class PickupStepInfo:
    """单控制步输出（env / 日志 / 可视化）。"""

    time: float
    phase: str
    primitive: str
    decision: PickupDecision | None
    balance: BalanceState
    com: np.ndarray
    zmp: np.ndarray
    support_polygon: np.ndarray
    left_foot_pos: np.ndarray
    right_foot_pos: np.ndarray
    hand_position: np.ndarray
    object_position: np.ndarray
    trunk_pitch: float
    stability_margin: float
    steps: int
    grasped: bool
    success: bool
    fall: bool
    energy: float
    smoothness: float
    decision_latency_ms: float


class PickupController:
    """完整拾取闭环控制器（仿真 500 Hz，控制 100 Hz，决策 20 Hz，脚步 10 Hz）。"""

    #: Phase-1 基座辅助默认参数（可被 configs/g1_pickup.yaml 的 base_assist 覆盖）
    DEFAULT_BASE_ASSIST: dict[str, float] = {
        "pos_kp": 5000.0,
        "pos_kd": 120.0,
        "max_force_n": 1000.0,
        "rot_kp": 900.0,
        "rot_kd": 70.0,
        "max_torque_nm": 380.0,
    }
    #: 搬运位姿（相对骨盆，世界系、箱体保持直立）：抓住箱体上沿抱在身前，
    #: 与躯干/大腿留出间隙（避免箱体与胸腹、膝关节接触）
    CARRY_OFFSET_WORLD = np.array([0.50, 0.0, -0.16])

    def __init__(self, config: PickupConfig, *, decision_model: DecisionModel | None = None) -> None:
        import mujoco

        self.mujoco = mujoco
        self.config = config
        path = Path(config.scene_path)
        if not path.is_file():
            raise PolicyError("pickup scene not found", path=str(path))
        spec = mujoco.MjSpec.from_file(str(path))
        half_height = float(config.object_half_height)
        half_depth = float(config.object_half_depth)
        half_width = float(config.object_half_width)
        base_height = float(config.object_base_height_m)
        object_center_z = base_height + half_height
        if base_height > 0.0:
            pedestal = spec.worldbody.add_body(name="pickup_pedestal", pos=[0.55, 0.0, base_height / 2.0])
            pedestal.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.24, 0.24, base_height / 2.0],
                rgba=[0.55, 0.55, 0.58, 1.0],
                mass=0.0,
                name="pickup_pedestal_geom",
            )
        object_body = spec.worldbody.add_body(name="pickup_object", pos=[0.55, 0.0, object_center_z])
        object_body.add_freejoint(name="pickup_object_free")
        object_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half_depth, half_width, half_height],
            rgba=[0.85, 0.25, 0.2, 1.0],
            mass=0.4,
            name="pickup_object_geom",
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # 载荷的接触由「躯干等效力旋」承担：关闭箱体碰撞，避免运动学负载与接触求解
        # 互相冲突（会把机器人躯干顶成前倾，并让手臂抖动）
        self.model.geom_contype[self.model.geom("pickup_object_geom").id] = 0
        self.model.geom_conaffinity[self.model.geom("pickup_object_geom").id] = 0
        # 载荷质量：按需求取机器人总质量的约一半（真实质量参与 MuJoCo 动力学）
        robot_mass = float(self.model.body_mass.sum()) - float(self.model.body_mass[self.model.body("pickup_object").id])
        self.robot_mass = robot_mass
        self.object_mass = float(config.object_mass_ratio) * robot_mass
        self.model.body_mass[self.model.body("pickup_object").id] = self.object_mass
        self.model.opt.timestep = float(config.sim_dt)
        # 拾取场景使用更高摩擦（仿真研究设定；真机需按地面实测重新标定）
        self.model.geom_friction[:, 0] = np.maximum(self.model.geom_friction[:, 0], 1.2)
        self.decimation = max(1, int(round(config.control_dt / config.sim_dt)))
        self.ik = WholeBodyIK(config.scene_path)
        self.joint_names = tuple(MUJOCO_JOINT_NAMES)
        self.qpos_addr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names])
        self.qvel_addr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in self.joint_names])
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.model.joint("pickup_object_free").id])
        self.object_body_id = int(self.model.body("pickup_object").id)
        self.object_geom_id = int(self.model.geom("pickup_object_geom").id)
        self._robot_geom_ids = [
            g for g in range(self.model.ngeom) if g != self.object_geom_id and self.model.geom_bodyid[g] != 0
        ]
        self.monitor = BalanceMonitor(
            warn_margin_m=0.08,
            step_margin_m=config.step_margin,
            critical_margin_m=config.critical_margin,
        )
        self.planner = FootstepPlanner()
        self.joint_pd = JointPDController(max_delta_per_step=0.02)
        self.grasp = GraspMock(grasp_distance_m=config.grasp_distance)
        self.manager = MotionManager(
            planner=self.planner,
            decision_model=decision_model,
            reach_margin_m=config.reach_margin,
            step_margin_m=config.step_margin,
            critical_margin_m=config.critical_margin,
            target_bend_pitch=config.trunk_pitch_target,
            crouch_depth=config.bend_crouch_depth,
            hip_shift=config.bend_hip_shift,
            max_steps=config.max_steps,
        )
        self.primitives: dict[str, Primitive] = {
            "STAND": StandPrimitive(),
            "BEND": BendPrimitive(),
            "STEP": StepPrimitive(),
            "LUNGE": LungePrimitive(),
            "REACH": ReachPrimitive(),
            "GRASP": GraspPrimitive(),
            "LIFT": LiftPrimitive(),
            "STAND_UP": StandUpPrimitive(),
            "RECOVER": RecoveryPrimitive(),
        }
        self.active: Primitive = self.primitives["STAND"]
        self.body_target: BodyTarget = self._initial_target()
        self.primitives["STAND"].reset(self.body_target, {})
        self.time = 0.0
        self.steps = 0
        self.energy = 0.0
        self.smoothness_sum = 0.0
        self.previous_q_target = self.body_target.base_pos.copy()
        self.success = False
        self.fall = False
        self.scenario: PickupScenario | None = None
        self.randomization = config.randomization
        self.foot_anchor_enabled = bool(config.foot_anchor)
        self.base_assist_enabled = True
        self.base_assist = dict(self.DEFAULT_BASE_ASSIST)
        self.base_assist.update({key: float(value) for key, value in dict(config.base_assist or {}).items() if key in self.base_assist})
        self.carry_pull_duration_s = float(config.carry_pull_duration_s)
        self._grasp_time = 0.0
        self._com_offset = np.zeros(2)
        self._last_box_clearance = float("inf")
        self._rng = np.random.default_rng(0)
        self._next_decision = 0.0
        self._last_command: ManagerCommand | None = None
        self._last_info: PickupStepInfo | None = None
        self.object_half_height = half_height
        self.object_half_depth = half_depth
        self.object_half_width = half_width
        self._object_base_height = base_height
        self._initial_object_z = object_center_z
        self._initial_grasp_z = object_center_z + half_height
        self._object_home = np.array([0.55, 0.0, object_center_z], dtype=np.float64)
        self._stand_base_z = 0.79
        self._grasp_box_pos = np.zeros(3)
        self._grasp_attached = False
        self._post_grasp_pushed = False
        self._com_shift_saturated = False
        self._base_mass = np.array(self.model.body_mass, dtype=np.float64)
        self._base_friction = np.array(self.model.geom_friction, dtype=np.float64)
        self._base_damping = np.array(self.model.dof_damping, dtype=np.float64)

    # ------------------------------------------------------------------ setup
    def _initial_target(self) -> BodyTarget:
        """默认站立目标（从模型 keyframe / FK 得到双脚位姿）。"""
        self._reset_pose()
        return self._body_target_from_data()

    def _reset_pose(self) -> None:
        """重置为微屈膝站姿（避免直腿奇异位形，保证 IK 可解）。"""
        mj = self.mujoco
        data = self.data
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[2] = 0.79
        data.qpos[3] = 1.0
        data.qpos[self.qpos_addr] = isaac_to_mujoco(np.array(DEFAULT_JOINT_POS_POLICY))
        mj.mj_forward(self.model, data)

    def _set_stance(self, width: float) -> None:
        """把双脚调整到指定总间距（更宽支撑多边形，提升平衡裕度）。"""
        half = max(0.10, float(width) / 2.0)
        data = self.data
        base_pos = np.array(data.qpos[0:3], dtype=np.float64)
        base_quat = np.array(data.qpos[3:7], dtype=np.float64)
        left = np.array(data.xpos[self.model.body("left_ankle_roll_link").id], dtype=np.float64)
        right = np.array(data.xpos[self.model.body("right_ankle_roll_link").id], dtype=np.float64)
        com = np.array(data.subtree_com[self.model.body("pelvis").id], dtype=np.float64)
        tasks = [
            BodyTask("pelvis", position=np.array([com[0], com[1], base_pos[2]]), pos_weight=1.2, task_type="com"),
            BodyTask(
                "left_ankle_roll_link",
                position=np.array([left[0], half, left[2]]),
                rotation=yaw_matrix(0.0),
                pos_weight=1.0,
                rot_weight=1.0,
            ),
            BodyTask(
                "right_ankle_roll_link",
                position=np.array([right[0], -half, right[2]]),
                rotation=yaw_matrix(0.0),
                pos_weight=1.0,
                rot_weight=1.0,
            ),
        ]
        q, _err, _hits = self.ik.solve(
            base_pos=base_pos, base_quat=base_quat, tasks=tasks, q_init=np.array(data.qpos[self.qpos_addr])
        )
        data.qpos[self.qpos_addr] = q
        self.mujoco.mj_forward(self.model, data)

    def _body_target_from_data(self) -> BodyTarget:
        data = self.data
        base_pos = np.array(data.qpos[0:3], dtype=np.float64)
        base_quat = np.array(data.qpos[3:7], dtype=np.float64)
        _, _, yaw = _rpy(base_quat)
        left_pos = np.array(data.xpos[self.model.body("left_ankle_roll_link").id], dtype=np.float64)
        right_pos = np.array(data.xpos[self.model.body("right_ankle_roll_link").id], dtype=np.float64)
        left_yaw = float(_rpy(np.array(data.xquat[self.model.body("left_ankle_roll_link").id]))[2])
        right_yaw = float(_rpy(np.array(data.xquat[self.model.body("right_ankle_roll_link").id]))[2])
        return BodyTarget(
            base_pos=base_pos,
            base_yaw=float(yaw),
            left_foot_pos=left_pos,
            left_foot_yaw=left_yaw,
            right_foot_pos=right_pos,
            right_foot_yaw=right_yaw,
            trunk_pitch=0.0,
            hand_target=None,
        )

    def reset(self, scenario: PickupScenario, *, seed: int = 0, randomize: bool | None = None) -> None:
        """重置模型、场景、监控器与状态机。"""
        mj = self.mujoco
        self._rng = np.random.default_rng(seed)
        self._reset_pose()
        self._set_stance(self.config.stance_width)
        # 物体位置/尺寸/质量
        self.scenario = scenario
        rng = self._rng
        use_dr = self.randomization.enabled if randomize is None else randomize
        position = np.array(scenario.object_position, dtype=np.float64)
        # 类人选择：目标在左侧用左手，右侧/正前方用右手
        self.reach_side = "left" if float(position[1]) > 0.05 else "right"
        if use_dr:
            position = position + rng.normal(0, self.randomization.object_position_noise_m, size=3)
            position[2] = max(0.02, position[2])
        # 有台面时物体中心 = 台面高度 + 半高（场景只给平面位置与尺寸）
        if self._object_base_height > 0.0:
            position[2] = self._object_base_height + self.object_half_height
        qpos = self.data.qpos
        qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = position
        qpos[self.object_qpos_addr + 3 : self.object_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self._initial_object_z = float(position[2])
        self._object_home = position.copy()
        if use_dr:
            self.model.body_mass[:] = self._base_mass * float(rng.uniform(*self.randomization.mass_scale))
            self.model.body_mass[0] = 0.0
            self.model.geom_friction[:] = self._base_friction
            self.model.geom_friction[:, 0] = np.clip(
                self.model.geom_friction[:, 0] * float(rng.uniform(*self.randomization.friction_scale)), 0.05, None
            )
            self.model.dof_damping[:] = self._base_damping * float(rng.uniform(*self.randomization.damping_scale))
        else:
            self.model.body_mass[:] = self._base_mass
            self.model.geom_friction[:] = self._base_friction
            self.model.dof_damping[:] = self._base_damping
        self.model.geom_friction[:, 0] = np.clip(
            self.model.geom_friction[:, 0] * float(scenario.friction), 0.05, None
        )
        mj.mj_forward(self.model, self.data)
        self.time = 0.0
        self.steps = 0
        self.energy = 0.0
        self.smoothness_sum = 0.0
        self.success = False
        self.fall = False
        self._grasp_attached = False
        self._grasp_box_pos = np.zeros(3)
        self._post_grasp_pushed = False
        self._com_offset = np.zeros(2)
        self._next_decision = 0.0
        self.monitor.reset()
        self.manager.reset()
        self.grasp.reset()
        self.active = self.primitives["STAND"]
        self.body_target = self._body_target_from_data()
        self.primitives["STAND"].reset(self.body_target, {})
        self.previous_q_target = self.data.qpos[self.qpos_addr].copy()
        self.joint_pd.reset(self.data.qpos[self.qpos_addr].copy())

    # ------------------------------------------------------------------- step
    def step(self) -> PickupStepInfo:
        """执行一个控制周期（含 decimation 个物理步），返回完整状态。"""
        mj = self.mujoco
        data = self.data
        data.xfrc_applied[:] = 0.0
        balance = self._balance()
        left_pos, left_rot = self._foot_pose("left")
        right_pos, right_rot = self._foot_pose("right")
        hand_position = self._body_position(self._hand_body())
        offhand_position = self._body_position(self._offhand_body())
        object_position = np.array(data.xpos[self.object_body_id], dtype=np.float64)
        # 抓取点 = 箱体「近侧顶边」：手从前方触顶，避免手进入箱体且保证臂展可达
        grasp_point = object_position + np.array([-(self.object_half_depth + 0.02), 0.0, self.object_half_height])
        embrace = self._embrace_points(grasp_point)
        hand_targets = self._hand_targets_for_side(hand_position, offhand_position, embrace, grasp_point)
        hand_distance = max(float(np.linalg.norm(hand - goal)) for hand, goal in hand_targets.values())
        command = self._last_command
        decision_latency_ms = 0.0
        if self.time >= self._next_decision:
            self._next_decision += self.config.decision_dt
            start = time.perf_counter()
            context = {
                "left_foot_pos": left_pos,
                "left_foot_yaw": float(_rpy(np.array(data.xquat[self.model.body("left_ankle_roll_link").id]))[2]),
                "right_foot_pos": right_pos,
                "right_foot_yaw": float(_rpy(np.array(data.xquat[self.model.body("right_ankle_roll_link").id]))[2]),
                "object_position": grasp_point,
                "hand_object_distance": hand_distance,
                "reach_threshold_m": self.config.reach_threshold,
                "reach_side": self.reach_side,
                "grasp_available": hand_distance <= self.config.grasp_distance,
                "grasped": self.grasp.grasped,
                "lift_complete": self._lift_complete(object_position),
                "stand_complete": self._stand_complete(),
                # 恢复门禁：RECOVER 完成且裕度回到安全阈值以上才算恢复，避免卡死在恢复态
                # （负重阶段以 carry 阈值为准，与 manager 的触发条件保持一致）
                "recovered": bool(
                    self.active.name == "RECOVER"
                    and self.active.is_finished()
                    and balance.stability_margin
                    >= (self.manager.carry_critical_margin_m + 0.05 if self.grasp.grasped else self.config.reach_margin)
                    and abs(self.body_target.trunk_pitch) <= 0.35
                ),
                "stance_width": self.config.stance_width,
                "stand_base_z": self._stand_base_z,
                "lift_height": self.config.lift_height,
                "use_capture_point": True,
                "use_prediction": True,
                "active_primitive": self.active.name,
            }
            bend_progress = float(self.body_target.metadata.get("bend_progress", 0.0))
            context["lunge_needed"] = bool(
                self.active.name == "BEND"
                and hand_distance > 0.35
                and bend_progress > 0.90          # 弯腰基本完成后才考虑脚步（不要过早动腿）
                and float(object_position[0]) > 0.56   # 仅当目标明显超出臂展时才主动弓步
                and not self.manager._step_in_progress
            )
            context["hand_position"] = hand_position
            context["offhand_position"] = self._body_position(self._offhand_body())
            context["two_handed"] = bool(self.config.two_handed)
            context["object_half_width"] = float(self.object_half_width)
            context["object_half_height"] = float(self.object_half_height)
            context["embrace_clearance_m"] = float(self.config.embrace_clearance_m)
            context["embrace_side_drop_m"] = float(self.config.embrace_side_drop_m)
            context["object_half_depth"] = float(self.object_half_depth)
            context["obstacle_xy"] = object_position[:2]
            context["obstacle_radius"] = float(np.hypot(self.object_half_depth, self.object_half_width) + 0.16)
            step_finished = self.active.is_finished()
            command = self.manager.update(
                balance=balance,
                goal={"object_position": grasp_point, "com_shift_saturated": self._com_shift_saturated},
                context=context,
                step_finished=step_finished,
            )
            decision_latency_ms = (time.perf_counter() - start) * 1000.0
            # 只在原语切换时重入：REACH 完成后保持终点姿态继续逼近，
            # 反复重入会把手臂拉回 waving 准备位（表现为「手先后甩」）
            if command.primitive != self.active.name:
                self._switch_primitive(command, context)
            elif hasattr(self.active, "object_position") and isinstance(command.params.get("object_position"), np.ndarray):
                # 目标可能移动：运行中的 REACH/GRASP 跟踪最新抓取点
                self.active.object_position = np.asarray(command.params["object_position"], dtype=np.float64)
            self._last_command = command
        self.body_target = self.active.update(self.body_target, self.config.control_dt)
        if self.grasp.grasped:
            # 贴身抱持：箱体位姿由躯干决定（等价于把载荷压在胸前），双手触点再跟踪箱体
            self.body_target = self._attach_carry_target(self.body_target)
        self.body_target = self._regulate_com(self.body_target, balance)
        q_target, ik_err, _hits = self._solve_ik(balance)
        self._last_ik_err = float(ik_err)
        self._last_ik_hits = int(_hits)
        q_target = self.joint_pd.set_target(q_target, rate_limit=True)
        self.smoothness_sum += float(np.mean(np.abs(q_target - self.previous_q_target)))
        self.previous_q_target = q_target.copy()
        # 物理仿真
        for _ in range(self.decimation):
            # 每个物理子步都重新施加外力（xfrc_applied 会保留到下一步，
            # 若只做累加会被放大 decimation 倍：载荷/gantry/足端锚定都受影响）
            data.xfrc_applied[:] = 0.0
            self._apply_base_assist()
            self._apply_foot_anchors()
            self._apply_carry_load()
            data.ctrl[:] = q_target
            mj.mj_step(self.model, data)
        self.time += self.config.control_dt
        self.steps += 1
        # 抓取/扰动/目标变化
        hand_position = self._body_position(self._hand_body())
        object_position = np.array(data.xpos[self.object_body_id], dtype=np.float64)
        grasp_point = object_position + np.array([-(self.object_half_depth + 0.02), 0.0, self.object_half_height])
        offhand_position = self._body_position(self._offhand_body())
        embrace = self._embrace_points(grasp_point)
        hand_targets = self._hand_targets_for_side(hand_position, offhand_position, embrace, grasp_point)
        # 新抓取只允许在 GRASPING 阶段发生（恢复/停止等状态不得接下重载）
        request_grasp = bool(self.body_target.grasp) and self.manager.phase == PickupPhase.GRASPING
        self.grasp.update(
            hand_positions=[hand for hand, _goal in hand_targets.values()],
            object_positions=[goal for _hand, goal in hand_targets.values()],
            request_grasp=request_grasp,
            time_s=self.time,
        )
        if self.grasp.grasped:
            if not self._grasp_attached:
                self._grasp_box_pos = np.array(object_position, dtype=np.float64).copy()
                self._grasp_attached = True
                self._grasp_time = self.time
            # 贴身抱持：箱体跟随躯干（相对位姿在抓取瞬间记录），保持直立
            attached = self._carry_pose()
            data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7] = attached
            dof_attached = int(np.asarray(self.model.joint("pickup_object_free").dofadr).reshape(-1)[0])
            data.qvel[dof_attached : dof_attached + 6] = 0.0
        else:
            # 未抓取时物体固定在初始位置（排除被脚踢动的混杂因素；抓取后由 mocap 附着）
            data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = self._object_home
            data.qpos[self.object_qpos_addr + 3 : self.object_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            dof = int(np.asarray(self.model.joint("pickup_object_free").dofadr).reshape(-1)[0])
            data.qvel[dof : dof + 6] = 0.0
        self._apply_scenario_events()
        self._last_box_clearance = self._box_clearance()
        force = np.array(data.actuator_force[:NUM_JOINTS], dtype=np.float64)
        self.energy += float(np.sum(force**2) * self.config.control_dt)
        self._check_success_failure()
        info = PickupStepInfo(
            time=self.time,
            phase=self.manager.phase.value,
            primitive=self.active.name,
            decision=self.manager.last_decision,
            balance=balance,
            com=self._load_com(),
            zmp=balance.zmp,
            support_polygon=balance.support_polygon,
            left_foot_pos=left_pos,
            right_foot_pos=right_pos,
            hand_position=hand_position,
            object_position=object_position,
            trunk_pitch=float(balance.trunk_pitch),
            stability_margin=float(balance.stability_margin),
            steps=self.manager.step_count,
            grasped=self.grasp.grasped,
            success=self.success,
            fall=self.fall,
            energy=self.energy,
            smoothness=self.smoothness_sum,
            decision_latency_ms=decision_latency_ms,
        )
        self._last_info = info
        return info

    # -------------------------------------------------------------- internals
    def _switch_primitive(self, command: ManagerCommand, context: Mapping[str, Any]) -> None:
        """切换原语并以当前 body target 为起点。"""
        primitive = self.primitives.get(command.primitive)
        if primitive is None:
            primitive = self.primitives["STAND"]
        merged = {**dict(context), **dict(command.params)}
        if command.primitive in ("STEP", "LUNGE") and merged.get("plan") is None:
            # 没有可行脚步计划时不切换（保持当前原语，由下一决策周期重新规划）
            return
        primitive.reset(self.body_target, merged)
        self.active = primitive

    def _solve_ik(self, balance: BalanceState) -> tuple[np.ndarray, float, int]:
        """全身 IK：双脚 + 躯干 + 手（可选）。"""
        target = self.body_target
        support_center = 0.5 * (target.left_foot_pos[:2] + target.right_foot_pos[:2])
        # 单脚支撑相：CoM 目标必须落在支撑脚上方（而不是双脚中点）
        if isinstance(self.active, StepPrimitive) and hasattr(self.active, "support"):
            stance_xy = (
                target.right_foot_pos[:2] if self.active.support == "right" else target.left_foot_pos[:2]
            )
            swing_xy = target.left_foot_pos[:2] if self.active.swing == "left" else target.right_foot_pos[:2]
            support_center = stance_xy + 0.35 * (swing_xy - stance_xy)
        object_position = np.array(self.data.xpos[self.object_body_id], dtype=np.float64)
        grasp_xy = object_position[:2] + np.array([0.0, 0.0])
        torso_yaw = 0.0
        if target.hand_target is not None:
            # 类人伸手：整机（骨盆+躯干）向目标方向旋转，扩大侧向可达范围；双脚朝向保持
            bearing = float(np.arctan2(grasp_xy[1] - support_center[1], grasp_xy[0] - support_center[0]))
            torso_yaw = 0.0 if abs(bearing) < 0.25 else float(np.clip(0.6 * bearing, -0.5, 0.5))
        base_quat = _quat_from_rpy(0.0, 0.0, target.base_yaw + torso_yaw)
        # CoM 目标 = 支撑中心 + 前后配平偏移（偏移由 `_regulate_com` 按裕度误差生成）
        direction = grasp_xy - support_center
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.zeros(2)
        com_target = support_center + self._com_offset * direction
        polygon = balance.support_polygon
        if polygon.shape[0] >= 3:
            self._com_shift_saturated = signed_polygon_margin(com_target, polygon) < 0.0
        self._last_com_target = np.array(com_target, dtype=np.float64)
        tasks = [
            BodyTask("pelvis", position=np.array([com_target[0], com_target[1], target.base_pos[2]]), pos_weight=1.5, task_type="com"),
            BodyTask("left_ankle_roll_link", position=target.left_foot_pos, rotation=yaw_matrix(target.left_foot_yaw), pos_weight=1.0, rot_weight=0.8),
            BodyTask("right_ankle_roll_link", position=target.right_foot_pos, rotation=yaw_matrix(target.right_foot_yaw), pos_weight=1.0, rot_weight=0.8),
            BodyTask(
                "torso_link",
                rotation=_quat_matrix(_quat_from_rpy(target.trunk_roll, target.trunk_pitch, target.base_yaw + torso_yaw)),
                pos_weight=0.0,
                rot_weight=0.5,
            ),
        ]
        hand_tasks = {
            target.reach_side: target.hand_target,
            target.offhand_side: target.offhand_target,
        }
        for side, hand_target in hand_tasks.items():
            if hand_target is not None:
                weight = 1.4 if side == target.reach_side else 1.2
                rotation = None
                rot_weight = 0.0
                if self.config.wrist_flush_face:
                    rotation = self._wrist_rotation_target(side, np.asarray(hand_target))
                    rot_weight = float(self.config.wrist_rot_weight)
                tasks.append(
                    BodyTask(
                        f"{side}_wrist_yaw_link",
                        position=np.asarray(hand_target),
                        rotation=rotation,
                        pos_weight=weight,
                        rot_weight=rot_weight,
                    )
                )
        q_init = self.data.qpos[self.qpos_addr].copy()
        index = {name: i for i, name in enumerate(self.joint_names)}
        arm_roles = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
        active_hands = {side for side, hand in hand_tasks.items() if hand is not None}
        # 腿部姿态参考：随基座下沉深度给出「微屈膝」构型，避免直腿奇异导致 IK 无法下蹲
        crouch = max(0.0, self._stand_base_z - float(target.base_pos[2]))
        knee_ref = float(np.clip(0.21 + 8.0 * crouch, 0.0, 1.7))
        posture_reference = {
            f"{side}_{role}_joint": value
            for side in ("left", "right")
            for role, value in (
                ("knee", knee_ref),
                ("hip_pitch", -0.46 * knee_ref),
                ("ankle_pitch", -0.54 * knee_ref),
            )
        }
        if active_hands:
            # 抱取/抱持期：肩部微外展，让大臂离开躯干（否则 IK 会为够到箱体侧面上角
            # 内收大臂，造成肩-躯干自穿模，线搜索会整段否决解并冻结 IK）
            posture_reference["left_shoulder_roll_joint"] = 0.20
            posture_reference["right_shoulder_roll_joint"] = -0.20

        def arm_joint_names(side: str) -> list[str]:
            return [f"{side}_{role}_joint" for role in arm_roles]

        reaching = isinstance(self.active, ReachPrimitive)
        prepare_w = self.active.prepare_weight() if reaching else 0.0
        frozen: list[str] = []
        for side in ("left", "right"):
            joints = arm_joint_names(side)
            if side not in active_hands:
                # 未参与抱取：冻结为护臂姿态（肩外展 + 肘微屈），避免手腕插入髋部
                sign = 1.0 if side == "left" else -1.0
                q_init[index[f"{side}_shoulder_roll_joint"]] = sign * 0.45
                q_init[index[f"{side}_shoulder_pitch_joint"]] = -0.10
                q_init[index[f"{side}_elbow_joint"]] = 0.25
                frozen.extend(joints)
            elif reaching:
                # 第一阶段（waving prepare）：手臂关节从当前姿态插值到准备位，IK 不参与
                prepare = ReachPrimitive.PREPARE_POSE[side]
                for role, value in prepare.items():
                    joint = f"{side}_{role}_joint"
                    q_init[index[joint]] = (1.0 - prepare_w) * q_init[index[joint]] + prepare_w * float(value)
                if prepare_w < 1.0:
                    frozen.extend(joints)
        # 仅 REACH 准备阶段把手部 IK 权重降为 0（先抬臂到位，再进入接近阶段）；
        # 其它原语（GRASP/LIFT/STAND_UP/RECOVER）必须全程保持手部跟踪，否则抱持会松手
        if reaching:
            for task in tasks:
                if task.body.endswith("_wrist_yaw_link"):
                    task.pos_weight *= max(prepare_w, 0.0)
        q, err, hits = self.ik.solve(
            base_pos=target.base_pos,
            base_quat=base_quat,
            tasks=tasks,
            q_init=q_init,
            frozen_joints=frozen,
            posture_reference=posture_reference,
            posture_weight=float(self.config.posture_weight),
        )
        # 手臂穿模修复：若 IK 解仍有机器人体段接触，双臂向外/向前微调直到无碰撞
        index = {name: i for i, name in enumerate(self.joint_names)}
        for _ in range(15):
            if self.ik._robot_penetration(q, target.base_pos, base_quat) <= 1e-4:
                break
            for side, sign in (("left", 1.0), ("right", -1.0)):
                q[index[f"{side}_shoulder_roll_joint"]] += sign * 0.12
                q[index[f"{side}_shoulder_pitch_joint"]] -= 0.06
                q[index[f"{side}_elbow_joint"]] -= 0.06
            q = np.clip(q, self.ik._lo + self.ik.position_margin_rad, self.ik._hi - self.ik.position_margin_rad)
        return q, err, hits

    def _balance(self) -> BalanceState:
        """构造 RobotState 并调用 BalanceMonitor。"""
        from cb_common.types import RobotState

        data = self.data
        left_pos, left_rot = self._foot_pose("left")
        right_pos, right_rot = self._foot_pose("right")
        state = RobotState(
            timestamp=self.time,
            base_pos=np.array(data.qpos[0:3], dtype=np.float64),
            base_quat=np.array(data.qpos[3:7], dtype=np.float64),
            base_lin_vel=np.array(data.qvel[0:3], dtype=np.float64),
            base_ang_vel=np.array(data.qvel[3:6], dtype=np.float64),
            joint_pos=np.array(data.qpos[self.qpos_addr], dtype=np.float64),
            joint_vel=np.array(data.qvel[self.qvel_addr], dtype=np.float64),
            contact=self._contacts(),
            com=self._load_com(),
        )
        return self.monitor.update(
            state,
            left_foot_pos=left_pos,
            left_foot_rot=left_rot,
            right_foot_pos=right_pos,
            right_foot_rot=right_rot,
            dt=self.config.control_dt,
            trunk_pitch=float(_rpy(np.array(data.xquat[self.model.body("torso_link").id]))[1]),
            trunk_roll=float(_rpy(np.array(data.xquat[self.model.body("torso_link").id]))[0]),
        )

    def _foot_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """返回足端位置与姿态矩阵。"""
        body_id = self.model.body(f"{side}_ankle_roll_link").id
        pos = np.array(self.data.xpos[body_id], dtype=np.float64)
        rot = np.array(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return pos, rot

    def _body_position(self, name: str) -> np.ndarray:
        """返回 body 世界位置。"""
        return np.array(self.data.xpos[self.model.body(name).id], dtype=np.float64)

    def _hand_body(self) -> str:
        """当前伸手侧的手腕 body 名。"""
        return f"{self.reach_side}_wrist_yaw_link"

    def _offhand_body(self) -> str:
        """副手（抱取另一侧）的手腕 body 名。"""
        return f"{self._offhand_side()}_wrist_yaw_link"

    def _offhand_side(self) -> str:
        """副手侧：与主手相反。"""
        return "left" if self.reach_side == "right" else "right"

    def _hand_targets_for_side(
        self,
        hand_position: np.ndarray,
        offhand_position: np.ndarray,
        embrace: Mapping[str, np.ndarray],
        grasp_point: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """按侧给出 (实测手位, 目标触点)：单手模式只用主手与近侧顶边中点。"""
        if not self.config.two_handed:
            return {self.reach_side: (np.asarray(hand_position), np.asarray(grasp_point))}
        measured = {self.reach_side: np.asarray(hand_position), self._offhand_side(): np.asarray(offhand_position)}
        return {side: (measured[side], embrace[side]) for side in (self.reach_side, self._offhand_side())}

    def _embrace_points(self, grasp_point: np.ndarray) -> dict[str, np.ndarray]:
        """抱取触点（箱体左右两侧面）：统一由配置尺寸生成。"""
        return embrace_points(
            grasp_point,
            self.object_half_width,
            self.config.embrace_clearance_m,
            self.object_half_depth,
            self.config.embrace_side_drop_m,
        )

    def _palm_axes(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 (掌心法向, 手指方向) 在 `{side}_wrist_yaw_link` 坐标系中的单位向量。

        手部几何体（G1 手板）在腕部连杆下带非单位 geom_quat，且尺寸为
        (薄, 宽, 长) = (0.0258, 0.0616, 0.0742)：薄轴 = 掌心法向，长轴 = 手指方向。
        这里按几何四元数标定，避免用固定关节角猜姿态（实测腕外翻 ±53° 就是这么来的）。
        """
        body = self.model.body(f"{side}_wrist_yaw_link").id
        candidates = [g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == body]
        if not candidates:
            raise PolicyError("hand geom not found", side=side)
        geom = max(candidates, key=lambda g: float(np.prod(self.model.geom_size[g])))
        w, x, y, z = (float(v) for v in self.model.geom_quat[geom])
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        normal = rot[:, 0] / np.linalg.norm(rot[:, 0])
        fingers = rot[:, 2] / np.linalg.norm(rot[:, 2])
        return normal, fingers

    def _wrist_rotation_target(self, side: str, hand_target: np.ndarray) -> np.ndarray:
        """腕部目标姿态：掌心与箱体侧面平齐，手指沿箱面向前下方。

        标定结论（MuJoCo 实测，模型自带手板几何）：
        - 左右手的手板几何共用同一法向约定（都取世界 +y 时掌面与箱体侧面平齐）；
          若按「镜像目标」给左手 −y，左手腕 roll 会顶到 ±1.972 rad 限位且仍差 15°。
        - 手指方向取世界 +x（沿箱体侧面朝前），此时两侧掌面偏差 0.7°、
          腕 roll 仅 −0.08/+0.06 rad，箱体最小间隙 +0.025 m（无穿模）。
        目标基与连杆基保持同手性，避免给出反射矩阵让 IK 用极端关节角去凑。
        """
        normal_link, fingers_link = self._palm_axes(side)
        normal_world = np.array([0.0, 1.0, 0.0])
        fingers_world = np.array([1.0, 0.0, 0.0])
        frame_link, hand_link = self._orthonormal_frame(normal_link, fingers_link)
        frame_world, hand_world = self._orthonormal_frame(normal_world, fingers_world)
        if hand_link * hand_world < 0.0:
            frame_world[:, 1] *= -1.0
            frame_world[:, 2] = np.cross(frame_world[:, 0], frame_world[:, 1])
        target = frame_world @ frame_link.T
        _ = hand_target  # 位置约束由调用方给出；此处只返回姿态目标
        return target

    @staticmethod
    def _orthonormal_frame(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
        """由两个近似正交方向构造规范正交基，并返回其手性（±1）。"""
        a = np.asarray(first, dtype=np.float64)
        a = a / np.linalg.norm(a)
        b = np.asarray(second, dtype=np.float64)
        b = b - a * float(np.dot(b, a))
        b = b / np.linalg.norm(b)
        c = np.cross(a, b)
        return np.column_stack([a, b, c]), float(np.linalg.det(np.column_stack([a, b, c])))

    def _carry_pose(self) -> np.ndarray:
        """抱持中的箱体位姿（7 维）：相对骨盆贴身搬运，箱体保持直立。

        抓取后 `carry_pull_duration_s` 内把载荷从地面位置收进腹前（减小重力矩臂，
        并让双手始终落在臂展可达的箱体近侧上角），随后随基座一起升降。
        收拢后再做一次「碰撞感知外推」：若箱体与髋/膝/躯干的安全距离不足，
        沿远离机器人方向平移，避免身体与箱体穿模。
        """
        world_target = np.asarray(self.body_target.base_pos, dtype=np.float64) + self.CARRY_OFFSET_WORLD
        # 搬运高度取「站立高度 + 偏移」的绝对高度：收拢过程先在原地把箱体抬到胸腹高度，
        # 再水平贴身，避免箱体沿直线穿过髋/膝关节（造成穿模）
        world_target[2] = self._stand_base_z + self.CARRY_OFFSET_WORLD[2]
        world_target = self._push_out_of_body(world_target)
        pull = quintic(min(1.0, max(0.0, self.time - self._grasp_time) / max(self.carry_pull_duration_s, 1e-6)))
        box_pos = (1.0 - pull) * self._grasp_box_pos + pull * world_target
        return np.concatenate([box_pos, np.array([1.0, 0.0, 0.0, 0.0])])

    def _push_out_of_body(self, box_pos: np.ndarray, margin_m: float = 0.025) -> np.ndarray:
        """把箱体目标沿 +x 外推，直到与髋/膝/躯干安全距离满足 margin（最多 0.2 m）。"""
        half = np.array([self.object_half_depth, self.object_half_width, self.object_half_height], dtype=np.float64)
        key_bodies = ("pelvis", "torso_link", "left_hip_yaw_link", "right_hip_yaw_link", "left_knee_link", "right_knee_link")
        result = np.array(box_pos, dtype=np.float64)
        for _ in range(12):
            worst = float("inf")
            for name in key_bodies:
                point = self._body_position(name)
                q = np.abs(point - result) - half
                sdf = float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))
                worst = min(worst, sdf)
            if worst >= margin_m:
                break
            result[0] += 0.02
        return result

    def _attach_carry_target(self, target: BodyTarget) -> BodyTarget:
        """把箱体放到躯干相对位姿，并把双手触点重定向到箱体近侧上角。"""
        pose = self._carry_pose()
        center = pose[:3]
        grasp_point = center + np.array([-(self.object_half_depth + 0.02), 0.0, self.object_half_height])
        points = self._embrace_points(grasp_point)
        return target.copy(
            hand_target=points[target.reach_side],
            offhand_target=points[target.offhand_side] if self.config.two_handed else None,
            grasp=True,
        )

    def _regulate_com(self, target: BodyTarget, balance: BalanceState) -> BodyTarget:
        """重心前后调节：按稳定裕度误差生成骨盆前后配平偏移（人类负重时的「前后找平衡」）。

        前移 δ 近似等量消耗前方裕度，故取 shift ≈ gain × (margin_now − target)：
        裕度富余时前移（帮助够到物体），负重导致裕度不足时后移配平。
        偏移量做速率限制后交给 `_solve_ik` 的 CoM 任务执行（不直接改写原语基座目标，
        否则会被原语链反复累加而失稳）。
        """
        polygon = balance.support_polygon
        if polygon.shape[0] < 3:
            return target
        support_center = 0.5 * (target.left_foot_pos[:2] + target.right_foot_pos[:2])
        margin_now = signed_polygon_margin(support_center, polygon)
        shift = float(np.clip(self.config.com_margin_gain * (margin_now - self.config.com_margin_target), -0.15, 0.15))
        # 速率限制：偏移用一阶跟踪，避免大步长激励出振荡
        self._com_offset += np.clip(shift - self._com_offset, -0.01, 0.01)
        return target

    def _contacts(self) -> tuple[bool, bool]:
        """足底接触（几何高度阈值，与现有适配层一致）。"""
        left, right = self._foot_pose("left")[0], self._foot_pose("right")[0]
        return bool(left[2] < 0.06), bool(right[2] < 0.06)

    def _lift_complete(self, object_position: np.ndarray) -> bool:
        """物体是否已被抬起到目标高度。"""
        return bool(
            self.grasp.grasped
            and (object_position[2] > self._initial_grasp_z + 0.6 * self.config.lift_height or self.active.is_finished())
        )

    def _stand_complete(self) -> bool:
        """是否已回到直立站姿。"""
        return bool(
            self.active.name == "STAND_UP"
            and self.active.is_finished()
            and abs(self.body_target.trunk_pitch) < 0.12
        )

    def _apply_scenario_events(self) -> None:
        """目标变化 / 抓取后扰动等场景事件。"""
        scenario = self.scenario
        if scenario is None:
            return
        if scenario.target_change_at_s is not None and scenario.target_change_to is not None and self.time >= scenario.target_change_at_s:
            target = np.asarray(scenario.target_change_to, dtype=np.float64)
            self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = target
            scenario.target_change_at_s = None
        if scenario.post_grasp_impulse is not None and self.grasp.grasped and not self._post_grasp_pushed and self.time > 0.5:
            self.data.xfrc_applied[self.model.body("pelvis").id, :3] += scenario.post_grasp_impulse
            self._post_grasp_pushed = True
        if scenario.perturbation is not None:
            start, duration, force_n = scenario.perturbation
            if start <= self.time < start + duration:
                self.data.xfrc_applied[self.model.body("pelvis").id, 0] += 0.5 * force_n

    def _check_success_failure(self) -> None:
        """成功/失败判定。"""
        com = self._robot_com()
        _, pitch, _ = _rpy(np.array(self.data.qpos[3:7], dtype=np.float64))
        if com[2] < 0.45 or abs(pitch) > 1.05:
            self.fall = True
            self.manager.phase = PickupPhase.FAILURE
        if self.manager.phase == PickupPhase.SUCCESS:
            self.success = True

    def close(self) -> None:
        """无外部资源需要释放（保持接口）。"""
        return None

    def _apply_foot_anchors(self) -> None:
        """支撑脚锚定（研究用 Phase-1 简化：消除脚滑混杂因素，保留 CoM/支撑域动力学）。

        摆动脚不锚定；支撑脚用世界系弹簧-阻尼力钉在目标位置/朝向上，
        使「CoM 是否越界」成为唯一的失稳来源，便于验证脚步规划逻辑。
        """
        if not getattr(self, "foot_anchor_enabled", False):
            return
        swing = None
        swing_active = False
        if isinstance(self.active, StepPrimitive) and hasattr(self.active, "swing"):
            progress = self.active.t / max(self.active.duration_s, 1e-6)
            swing_active = 0.15 <= progress <= 0.85
            swing = self.active.swing
        for side in ("left", "right"):
            if swing_active and swing == side:
                continue
            body_name = f"{side}_ankle_roll_link"
            body_id = self.model.body(body_name).id
            pos = np.array(self.data.xpos[body_id], dtype=np.float64)
            vel = np.array(self.data.cvel[body_id][3:6], dtype=np.float64)
            target = self.body_target.left_foot_pos if side == "left" else self.body_target.right_foot_pos
            force = 3000.0 * (np.asarray(target, dtype=np.float64) - pos) - 80.0 * vel
            force = np.clip(force, -400.0, 400.0)
            rot = np.array(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            yaw = self.body_target.left_foot_yaw if side == "left" else self.body_target.right_foot_yaw
            rot_error = _rotation_error(rot, yaw_matrix(yaw))
            torque = 300.0 * rot_error - 20.0 * np.array(self.data.cvel[body_id][0:3], dtype=np.float64)
            torque = np.clip(torque, -120.0, 120.0)
            self.data.xfrc_applied[body_id, :3] += force
            self.data.xfrc_applied[body_id, 3:6] += torque

    def _apply_base_assist(self) -> None:
        """Phase-1 基座稳定辅助（虚拟 gantry）：把骨盆拉向规划位姿。

        说明：完整 WBC/RL 平衡控制器待后续替换；本辅助只稳定基座，
        CoM/支撑域/脚步规划/决策闭环仍基于 MuJoCo 真实状态计算。
        负重（约半体重）时需更大辅助力矩，参数来自 configs/g1_pickup.yaml 的 base_assist 段。
        """
        if not getattr(self, "base_assist_enabled", True):
            return
        assist = self.base_assist
        pelvis = self.model.body("pelvis").id
        pos = np.array(self.data.qpos[0:3], dtype=np.float64)
        vel = np.array(self.data.qvel[0:3], dtype=np.float64)
        target = np.asarray(self.body_target.base_pos, dtype=np.float64)
        force = assist["pos_kp"] * (target - pos) - assist["pos_kd"] * vel
        force = np.clip(force, -assist["max_force_n"], assist["max_force_n"])
        roll, pitch, yaw = _rpy(np.array(self.data.qpos[3:7], dtype=np.float64))
        omega = np.array(self.data.qvel[3:6], dtype=np.float64)
        rot_error = np.array(
            [
                -roll,
                -pitch,
                ((self.body_target.base_yaw - yaw + np.pi) % (2.0 * np.pi)) - np.pi,
            ]
        )
        torque = assist["rot_kp"] * rot_error - assist["rot_kd"] * omega
        torque = np.clip(torque, -assist["max_torque_nm"], assist["max_torque_nm"])
        self.data.xfrc_applied[pelvis, :3] += force
        self.data.xfrc_applied[pelvis, 3:6] += torque

    def _apply_carry_load(self) -> None:
        """Phase-1 载荷模型：抱持时把载荷重力以等效力旋加到躯干。

        载荷位置由双手中点决定（运动学抱持），其重量通过躯干等效力旋真实加载，
        使得关节力矩、基座辅助与平衡裕度都必须应对「约半体重」的负载。
        完整 WBC 版本应改为接触力/惯量一致的多刚体约束（后续路线）。
        """
        scale = self._load_scale()
        if scale <= 0.0 or self.object_mass <= 0.0:
            return
        data = self.data
        load_pos = np.array(data.xpos[self.object_body_id], dtype=np.float64)
        torso_id = self.model.body("torso_link").id
        wrench = np.array([0.0, 0.0, -scale * self.object_mass * 9.81], dtype=np.float64)
        lever = load_pos - np.array(data.xpos[torso_id], dtype=np.float64)
        data.xfrc_applied[torso_id, :3] += wrench
        data.xfrc_applied[torso_id, 3:6] += np.cross(lever, wrench)

    def _load_scale(self) -> float:
        """载荷传递系数 [0,1]：抓取后随「收拢贴身」过程线性递增加载（避免阶跃冲击）。"""
        if not self.grasp.grasped:
            return 0.0
        pull = (self.time - self._grasp_time) / max(self.carry_pull_duration_s, 1e-6)
        return float(quintic(min(1.0, max(0.0, pull))))

    def _load_com(self) -> np.ndarray:
        """机器人本体 + 抱持载荷的合成质心（未抱持时即本体质心）。

        注意：箱体挂在 worldbody 下（自由关节），`subtree_com[0]` 会把箱体质量算进来，
        因此本体质心取「骨盆子树」质心，再按质量加权叠加载荷。
        """
        com = self._robot_com()
        scale = self._load_scale()
        if scale <= 0.0 or self.object_mass <= 0.0:
            return com
        load_pos = np.array(self.data.xpos[self.object_body_id], dtype=np.float64)
        load_mass = scale * self.object_mass
        return (self.robot_mass * com + load_mass * load_pos) / (self.robot_mass + load_mass)

    def _robot_com(self) -> np.ndarray:
        """机器人本体（不含载荷箱体）质心：骨盆子树质心。"""
        return np.array(self.data.subtree_com[self.model.body("pelvis").id], dtype=np.float64)

    def _box_clearance(self) -> float:
        """箱体与机器人各几何体的最小间隙（m，负值表示穿模）。

        箱体碰撞被关闭（载荷由等效重力矩承担），因此穿模只能靠几何距离检查：
        本方法用 `mj_geomDistance` 逐对求最小距离，作为「不穿模」的硬指标。
        """
        mj = self.mujoco
        model, data = self.model, self.data
        best = float("inf")
        for geom in self._robot_geom_ids:
            distance = float(mj.mj_geomDistance(model, data, self.object_geom_id, geom, 5.0, None))
            if distance < best:
                best = distance
        return best


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """旋转矩阵误差 → 世界系旋转向量。"""
    rot = target @ current.T
    angle = np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(3)
    return axis / norm * angle


def _rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """wxyz 四元数 → roll/pitch/yaw（rad）。"""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm > 1e-9:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY → wxyz 四元数。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _quat_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 旋转矩阵。"""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

"""平衡系统压力测试场景：绕柱避障（pillar_avoid）与负重搬运（carry_box）。

两个场景都建立在既有平衡栈之上（教师 ONNX + 技能 + 安全层 + Balance Metrics），
只增加「环境侧扰动」：绕柱用闭环导航改写运动技能速度指令，搬运用等效重力旋加载载荷。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GRAVITY = 9.81

#: MjSpec 生命周期保活：spec 被 GC 后其编译出的 MjModel 内部指针悬空，
#: 实测表现为 viewer 关闭阶段 segfault（exit 139）。这里显式持有引用。
_SPEC_KEEPALIVE: list[Any] = []


def add_pillar(
    spec: Any,
    *,
    pos: tuple[float, float],
    radius: float = 0.25,
    height: float = 1.6,
    name: str = "pillar",
) -> None:
    """在世界系加入圆柱柱子（视觉 + 碰撞体，无质量静态障碍）。"""
    import mujoco

    body = spec.worldbody.add_body(name=name, pos=[float(pos[0]), float(pos[1]), height / 2.0])
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[float(radius), height / 2.0, 0.0],
        rgba=[0.32, 0.52, 0.90, 1.0],
        mass=0.0,  # 静态障碍：不参与系统质心（否则会把 subtree_com[0] 拉到柱子上）
        name=f"{name}_geom",
    )


def add_carry_box(
    spec: Any,
    *,
    half: tuple[float, float, float] = (0.12, 0.12, 0.15),
    mass: float = 8.0,
    pos: tuple[float, float, float] = (0.20, 0.0, 0.85),
) -> None:
    """加入自由箱体（质量真实，位姿由控制器每步运动学跟随躯干）。"""
    import mujoco

    body = spec.worldbody.add_body(name="carry_box", pos=list(pos))
    body.add_freejoint(name="carry_box_free")
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        rgba=[0.86, 0.30, 0.22, 1.0],
        mass=float(mass),
        name="carry_box_geom",
    )


def build_augmented_model(
    mujoco: Any,
    xml_path: str,
    *,
    pillars: list[tuple[float, float]] | None = None,
    carry_box_mass: float = 0.0,
) -> Any:
    """在教师 MJCF 基础上追加柱子/箱体（只追加 body，既有索引保持不变）。"""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for index, pos in enumerate(pillars or []):
        add_pillar(spec, pos=pos, name=f"pillar{index}")
    if carry_box_mass > 0.0:
        add_carry_box(spec, mass=carry_box_mass)
    model = spec.compile()
    _SPEC_KEEPALIVE.append(spec)
    if carry_box_mass > 0.0:
        # 载荷碰撞关闭：接触力与等效重力旋会互相冲突（实测把机器人弹飞，关节速度 >20 rad/s），
        # 与 `cb_pickup` 的 Phase-1 载荷模型保持一致（力真实、几何不参与接触求解）
        geom = model.geom("carry_box_geom").id
        model.geom_contype[geom] = 0
        model.geom_conaffinity[geom] = 0
    return model


def swap_runtime_model(env: Any, model: Any) -> None:
    """替换环境运行时模型（保持教师索引不变：仅追加 body/geom）。"""
    mujoco = env.rt.mujoco
    env.rt.model = model
    env.rt.data = mujoco.MjData(model)
    env.rt.base_body_mass = np.array(model.body_mass, dtype=np.float64)
    env.rt.base_geom_friction = np.array(model.geom_friction, dtype=np.float64)
    env.rt.model.opt.timestep = env.rt.core.sim_dt


@dataclass
class CarriedLoad:
    """负重搬运控制器：箱体运动学跟随躯干，并把重力以等效力旋加到躯干。"""

    mass: float
    offset_local: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.0, -0.10]))
    host_body: str = "torso_link"
    box_body: str = "carry_box"
    ramp_s: float = 1.0
    height: float = 0.0
    horizontal_arm: float = 0.0
    _t: float = 0.0
    _dt: float = 0.02

    def scale(self) -> float:
        """载荷传递系数：0→1 在 ramp_s 内线性递增（真实取物的重量转移过程）。"""
        return float(min(1.0, self._t / max(self.ramp_s, 1e-6)))

    def apply(self, mujoco: Any, model: Any, data: Any) -> None:
        """每控制步调用：先跟随躯干，再加载荷（在 env.step 之前调用）。"""
        torso_id = model.body(self.host_body).id
        torso_pos = np.array(data.xpos[torso_id], dtype=np.float64)
        torso_rot = np.array(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
        box_pos = torso_pos + torso_rot @ self.offset_local
        address = int(model.jnt_qposadr[model.joint(f"{self.box_body}_free").id])
        data.qpos[address : address + 3] = box_pos
        data.qpos[address + 3 : address + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        dof = int(model.jnt_dofadr[model.joint(f"{self.box_body}_free").id])
        data.qvel[dof : dof + 6] = 0.0
        self._t += self._dt
        force = np.array([0.0, 0.0, -self.scale() * self.mass * GRAVITY], dtype=np.float64)
        lever = box_pos - torso_pos
        # 先清零本体验力旋：MuJoCo 会保留 xfrc_applied 直到被改写，逐控制步累加会把载荷放大 N 倍
        data.xfrc_applied[torso_id, :] = 0.0
        data.xfrc_applied[torso_id, :3] += force
        data.xfrc_applied[torso_id, 3:6] += np.cross(lever, force)
        self.height = float(box_pos[2])
        # 水平力臂：载荷质心到支撑中心（双足中点）的水平距离，反映倾覆力矩大小
        feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        support = 0.5 * sum(np.array(data.xpos[fid], dtype=np.float64) for fid in feet)
        self.horizontal_arm = float(np.linalg.norm(box_pos[:2] - support[:2]))


@dataclass
class PillarNavigator:
    """闭环绕柱导航（横向让位 + 航向控制，支持多柱绕桩）：输出 (vx, yaw_rate)。

    与「追航点」不同，这里在世界系里沿前进方向取一个 look-ahead 点，
    只用横向让位量决定航向：接近柱子时把目标横向位置移到柱侧（左右交替），
    通过后自动回到中线。这样不会出现追着身后航点原地绕圈的问题。
    通过改写 `LocomotionSkill.vx/yaw_rate` 实现，复用既有跑步教师与安全层。
    """

    pillars: list[tuple[float, float]] = field(default_factory=lambda: [(5.0, 0.0)])
    radius: float = 0.25
    clearance: float = 0.30          # 让位量 = 柱半径 + 0.30（≥ 机身半宽 0.20 + 余量）
    sides: list[float] = field(default_factory=list)
    cruise_vx: float = 1.5
    lookahead_m: float = 1.0
    approach_m: float = 5.0
    yaw_gain: float = 2.0
    yaw_rate_limit: float = 0.9
    min_distance: float = float("inf")
    passed: bool = False
    pass_records: list[dict[str, float]] = field(default_factory=list)
    _prev_x: float = field(default=float("-inf"), init=False)

    def side_of(self, index: int) -> float:
        """第 index 根柱子的绕行侧（默认左右交替）。"""
        if index < len(self.sides):
            return float(self.sides[index])
        return 1.0 if index % 2 == 0 else -1.0

    def command(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        """返回 (vx, yaw_rate)。"""
        for index, (px, py) in enumerate(self.pillars):
            self.min_distance = min(self.min_distance, float(math.hypot(x - px, y - py)))
            # 通过判定：机身前缘越过柱心时记录横向让位，必须落在「预定侧」且满足最小净距
            if self._prev_x <= px < x and index >= len(self.pass_records):
                side = self.side_of(index)
                lateral = (y - py) * side
                # 判定阈值：柱半径 + 机身半宽（0.20 m），低于此值属于「贴柱擦过」不算绕开
                required = self.radius + 0.20
                self.pass_records.append(
                    {
                        "index": float(index),
                        "pillar_x": px,
                        "pillar_y": py,
                        "y_at_pass": float(y),
                        "lateral_m": float(lateral),
                        "required_m": float(required),
                        "ok": float(1.0 if lateral >= required else 0.0),
                    }
                )
        self._prev_x = float(x)
        self.passed = self.passed or x > float(self.pillars[-1][0]) + 0.3
        # 目标横向位置：尚未通过的柱子里，取最近一根的让位量；通过后回到中线
        lateral = self.radius + self.clearance
        target_y = 0.0
        for index, (px, py) in enumerate(self.pillars):
            if x < px - 0.2:  # 该柱尚未通过
                if x > px - self.approach_m:
                    target_y = py + self.side_of(index) * lateral
                break
        desired_yaw = math.atan2(target_y - y, self.lookahead_m)
        error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        yaw_rate = float(np.clip(self.yaw_gain * error, -self.yaw_rate_limit, self.yaw_rate_limit))
        vx = self.cruise_vx * max(0.4, 1.0 - 0.5 * abs(error))
        return vx, yaw_rate

    def override_skill(self, env: Any, x: float, y: float, yaw: float) -> tuple[float, float]:
        """把导航指令写入当前运动技能（若当前技能支持速度指令）。"""
        vx, yaw_rate = self.command(x, y, yaw)
        skill = getattr(env, "_active_skill", None)
        if skill is not None and hasattr(skill, "vx") and hasattr(skill, "yaw_rate"):
            skill.vx = float(vx)
            skill.yaw_rate = float(yaw_rate)
            skill._t = max(skill._t, 0.8)  # 起步渐入已完成，避免转弯指令被 ramp 削弱
        return vx, yaw_rate

"""关节/姿态/支撑域安全阈值与命令条件器（fail-closed）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cb_common.config import load_config
from cb_common.errors import SafetyError
from cb_common.joints import NUM_JOINTS, POLICY_JOINT_NAMES, mujoco_to_isaac
from cb_common.types import JointCommand, RobotState, SafetyDecision, SafetyEvent, SafetyLevel


@dataclass(frozen=True)
class JointLimits:
    """29 DOF 位置限位（策略序）+ 每关节速率/力矩分组限值。"""

    position_min: np.ndarray
    position_max: np.ndarray
    group_of_joint: Mapping[str, str]
    group_rated: Mapping[str, tuple[float, float, float]]  # group -> (vel, acc, jerk)
    group_torque_abort: Mapping[str, float]
    emergency_velocity: float

    def check(self, joint_pos: np.ndarray, joint_vel: np.ndarray | None, *, margin: float) -> list[SafetyEvent]:
        """检查位置（含 margin）与速度；返回事件列表。"""
        events: list[SafetyEvent] = []
        pos = np.asarray(joint_pos, dtype=np.float64)
        if pos.shape != (NUM_JOINTS,) or not np.isfinite(pos).all():
            return [SafetyEvent("joint_limit", SafetyLevel.EMERGENCY, "joint_pos_nan_or_shape", 0.0, value=float(pos.size))]
        below = pos < (self.position_min + margin)
        above = pos > (self.position_max - margin)
        for idx in np.flatnonzero(below | above):
            name = POLICY_JOINT_NAMES[int(idx)]
            bound = self.position_min[idx] if below[idx] else self.position_max[idx]
            events.append(
                SafetyEvent(
                    "joint_limit",
                    SafetyLevel.ABORT,
                    "joint_position_out_of_range",
                    0.0,
                    joint=name,
                    value=float(pos[idx]),
                    threshold=float(bound),
                    unit="rad",
                )
            )
        if joint_vel is not None:
            vel = np.asarray(joint_vel, dtype=np.float64)
            if not np.isfinite(vel).all():
                events.append(SafetyEvent("joint_limit", SafetyLevel.EMERGENCY, "joint_vel_nan", 0.0))
            else:
                for idx, name in enumerate(POLICY_JOINT_NAMES):
                    limit = self.group_rated[self.group_of_joint[name]][0]
                    if abs(vel[idx]) > limit:
                        events.append(
                            SafetyEvent(
                                "joint_limit",
                                SafetyLevel.ABORT,
                                "joint_velocity_exceeded",
                                0.0,
                                joint=name,
                                value=float(abs(vel[idx])),
                                threshold=float(limit),
                                unit="rad/s",
                            )
                        )
        return events


@dataclass(frozen=True)
class AttitudeLimits:
    """姿态三级阈值（deg, deg/s）。"""

    roll_warn_deg: float
    roll_abort_deg: float
    roll_emergency_deg: float
    pitch_warn_deg: float
    pitch_abort_deg: float
    pitch_emergency_deg: float
    ang_vel_warn_deg_s: float
    ang_vel_abort_deg_s: float
    ang_vel_emergency_deg_s: float


@dataclass(frozen=True)
class SupportLimits:
    """支撑域与单脚支撑阈值。"""

    com_radius_warn_m: float
    com_radius_abort_m: float
    contact_loss_warn_s: float
    contact_loss_abort_s: float
    single_foot_max_s: float


@dataclass(frozen=True)
class BaseLimits:
    """基座高度/漂移阈值。"""

    height_warn_m: float
    height_abort_m: float
    height_emergency_m: float
    drift_warn_m: float
    drift_abort_m: float
    vertical_velocity_abort_m_s: float


@dataclass(frozen=True)
class EStopConfig:
    """急停配置。"""

    action: str
    latching: bool
    requires_manual_reset: bool
    independent_channel: bool


@dataclass
class SafetyLimits:
    """安全阈值总装；缺失/非法字段一律拒绝构造。"""

    joints: JointLimits
    attitude: AttitudeLimits
    support: SupportLimits
    base: BaseLimits
    estop: EStopConfig
    watchdog: Mapping[str, float]
    position_margin_rad: float
    limits_version: str = "1.0"

    def validate(self) -> None:
        """配置自检；任何问题抛 SafetyError（fail-closed）。"""
        problems: list[str] = []
        if self.position_margin_rad < 0.0:
            problems.append("position_margin_rad < 0")
        for label, warn, abort, emergency in (
            ("roll", self.attitude.roll_warn_deg, self.attitude.roll_abort_deg, self.attitude.roll_emergency_deg),
            ("pitch", self.attitude.pitch_warn_deg, self.attitude.pitch_abort_deg, self.attitude.pitch_emergency_deg),
            (
                "ang_vel",
                self.attitude.ang_vel_warn_deg_s,
                self.attitude.ang_vel_abort_deg_s,
                self.attitude.ang_vel_emergency_deg_s,
            ),
        ):
            if not (0.0 < warn < abort < emergency):
                problems.append(f"{label} requires 0 < warn < abort < emergency, got {warn}/{abort}/{emergency}")
        if not (0.0 < self.support.com_radius_warn_m < self.support.com_radius_abort_m):
            problems.append("support.com_radius requires 0 < warn < abort")
        if not (self.support.single_foot_max_s > 0.0):
            problems.append("support.single_foot_max_s must be > 0")
        if not (0.0 < self.base.height_emergency_m < self.base.height_abort_m < self.base.height_warn_m):
            problems.append("base height requires 0 < emergency < abort < warn")
        if self.estop.action not in ("damp", "hold", "zero_torque"):
            problems.append(f"estop.action invalid: {self.estop.action!r}")
        for key in ("state_timeout_s", "command_timeout_s", "loop_timeout_s"):
            if float(self.watchdog.get(key, 0.0)) <= 0.0:
                problems.append(f"watchdog.{key} must be > 0")
        if problems:
            raise SafetyError("invalid safety limits", problems=problems)

    def joint_group(self, joint: str) -> str:
        """返回关节所属限值分组。"""
        try:
            return self.joints.group_of_joint[joint]
        except KeyError as exc:
            raise SafetyError("joint not registered in safety limits", joint=joint) from exc


def load_joint_limits_mjcf(scene_path: str | Path, *, policy_order: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """从 G1 MJCF 读取 29 关节位置限位（默认策略序）。

    只接受恰好 29 个 hinge joint 的模型；否则拒绝运行（禁止隐式 23/29 兼容）。
    """
    import mujoco  # 延迟导入，便于纯逻辑单测

    path = Path(scene_path)
    if not path.is_file():
        raise SafetyError("MJCF scene not found", path=str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    if len(names) != NUM_JOINTS:
        raise SafetyError("expected 29 actuated joints", found=len(names), path=str(path))
    lo = np.array([model.jnt_range[i + 1][0] for i in range(NUM_JOINTS)], dtype=np.float64)
    hi = np.array([model.jnt_range[i + 1][1] for i in range(NUM_JOINTS)], dtype=np.float64)
    if policy_order:
        lo, hi = mujoco_to_isaac(lo), mujoco_to_isaac(hi)
    return lo, hi


def load_safety_limits(
    config_path: str | Path,
    *,
    scene_path: str | Path | None = None,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> SafetyLimits:
    """加载 `configs/safety/limits.yaml` 并构造 SafetyLimits。"""
    cfg = load_config(config_path, required=["joints", "attitude", "support", "base", "watchdog", "estop"])
    if joint_limits is None:
        if scene_path is None:
            raise SafetyError("scene_path or joint_limits is required (fail-closed)")
        joint_limits = load_joint_limits_mjcf(scene_path)
    lo, hi = joint_limits
    joints_cfg = cfg.section("joints")
    group_rated: dict[str, tuple[float, float, float]] = {}
    group_torque: dict[str, float] = {}
    for group, values in joints_cfg.get("rated", {}).items():
        group_rated[group] = (
            float(values["velocity_rad_s"]),
            float(values["acceleration_rad_s2"]),
            float(values["jerk_rad_s3"]),
        )
    for group, values in joints_cfg.get("groups", {}).items():
        group_torque[group] = float(values["torque_abort_nm"])
    group_of_joint = {str(k): str(v) for k, v in joints_cfg.get("joint_group", {}).items()}
    missing = [name for name in POLICY_JOINT_NAMES if name not in group_of_joint]
    unknown = [name for name in group_of_joint if name not in POLICY_JOINT_NAMES]
    if missing or unknown:
        raise SafetyError("joint_group table mismatch", missing=missing, unknown=unknown)
    joint_limits_obj = JointLimits(
        position_min=np.asarray(lo, dtype=np.float64),
        position_max=np.asarray(hi, dtype=np.float64),
        group_of_joint=group_of_joint,
        group_rated=group_rated,
        group_torque_abort=group_torque,
        emergency_velocity=float(joints_cfg.get("emergency_velocity_rad_s", 25.0)),
    )
    attitude_cfg = cfg.section("attitude")
    support_cfg = cfg.section("support")
    base_cfg = cfg.section("base")
    estop_cfg = cfg.section("estop")
    watch_cfg = cfg.section("watchdog")
    limits = SafetyLimits(
        joints=joint_limits_obj,
        attitude=AttitudeLimits(
            roll_warn_deg=float(attitude_cfg.get("roll_warn_deg")),
            roll_abort_deg=float(attitude_cfg.get("roll_abort_deg")),
            roll_emergency_deg=float(attitude_cfg.get("roll_emergency_deg")),
            pitch_warn_deg=float(attitude_cfg.get("pitch_warn_deg")),
            pitch_abort_deg=float(attitude_cfg.get("pitch_abort_deg")),
            pitch_emergency_deg=float(attitude_cfg.get("pitch_emergency_deg")),
            ang_vel_warn_deg_s=float(attitude_cfg.get("angular_velocity_warn_deg_s")),
            ang_vel_abort_deg_s=float(attitude_cfg.get("angular_velocity_abort_deg_s")),
            ang_vel_emergency_deg_s=float(attitude_cfg.get("angular_velocity_emergency_deg_s")),
        ),
        support=SupportLimits(
            com_radius_warn_m=float(support_cfg.get("com_radius_warn_m")),
            com_radius_abort_m=float(support_cfg.get("com_radius_abort_m")),
            contact_loss_warn_s=float(support_cfg.get("contact_loss_warn_s")),
            contact_loss_abort_s=float(support_cfg.get("contact_loss_abort_s")),
            single_foot_max_s=float(support_cfg.get("single_foot_max_s")),
        ),
        base=BaseLimits(
            height_warn_m=float(base_cfg.get("height_warn_m")),
            height_abort_m=float(base_cfg.get("height_abort_m")),
            height_emergency_m=float(base_cfg.get("height_emergency_m")),
            drift_warn_m=float(base_cfg.get("drift_warn_m")),
            drift_abort_m=float(base_cfg.get("drift_abort_m")),
            vertical_velocity_abort_m_s=float(base_cfg.get("vertical_velocity_abort_m_s")),
        ),
        estop=EStopConfig(
            action=str(estop_cfg.get("action")),
            latching=bool(estop_cfg.get("latching", True)),
            requires_manual_reset=bool(estop_cfg.get("requires_manual_reset", True)),
            independent_channel=bool(estop_cfg.get("independent_channel", True)),
        ),
        watchdog={
            "state_timeout_s": float(watch_cfg.get("state_timeout_s")),
            "command_timeout_s": float(watch_cfg.get("command_timeout_s")),
            "loop_timeout_s": float(watch_cfg.get("loop_timeout_s")),
        },
        position_margin_rad=float(joints_cfg.get("position_margin_rad", 0.02)),
        limits_version=str(cfg.get("limits_version", "1.0")),
    )
    limits.validate()
    return limits


@dataclass
class CommandConditioner:
    """位置 → 速度 → 加速度 → jerk 条件器（可行制动校验）。"""

    limits: SafetyLimits
    dt: float
    _dq: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _ddq: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]
    _q_out: np.ndarray = field(default=None, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._dq = np.zeros(NUM_JOINTS)
        self._ddq = np.zeros(NUM_JOINTS)
        self._q_out = np.zeros(NUM_JOINTS)

    def reset(self, q: np.ndarray) -> None:
        """以当前关节位置重置条件器。"""
        self._q_out = np.asarray(q, dtype=np.float64).copy()
        self._dq = np.zeros(NUM_JOINTS)
        self._ddq = np.zeros(NUM_JOINTS)

    def _group_limits(self, key: str) -> np.ndarray:
        index = {"vel": 0, "acc": 1, "jerk": 2}[key]
        return np.array(
            [self.limits.joints.group_rated[self.limits.joints.group_of_joint[name]][index] for name in POLICY_JOINT_NAMES],
            dtype=np.float64,
        )

    def condition(self, q_target: np.ndarray, *, current_q: np.ndarray | None = None) -> np.ndarray:
        """条件化目标位置，返回可安全执行的关节目标（含制动可行性裁剪）。"""
        target = np.asarray(q_target, dtype=np.float64).copy()
        if target.shape != (NUM_JOINTS,):
            raise SafetyError("q_target shape invalid", shape=str(target.shape))
        if not np.isfinite(target).all():
            raise SafetyError("q_target contains NaN/Inf")
        if current_q is not None and self._q_out is None:
            self.reset(current_q)
        lo = self.limits.joints.position_min + self.limits.position_margin_rad
        hi = self.limits.joints.position_max - self.limits.position_margin_rad
        target = np.clip(target, lo, hi)

        v_max = self._group_limits("vel")
        a_max = self._group_limits("acc")
        j_max = self._group_limits("jerk")
        # 可行制动：若当前速度下无法在剩余距离内停住，则限制速度上界。
        remaining = target - self._q_out
        brake_v = np.sqrt(np.maximum(2.0 * a_max * np.abs(remaining), 0.0))
        v_cap = np.minimum(v_max, np.maximum(brake_v, 0.05))
        desired_dq = np.clip(remaining / self.dt, -v_cap, v_cap)
        dq = np.clip(desired_dq, self._dq - a_max * self.dt, self._dq + a_max * self.dt)
        dq = np.clip(dq, -v_max, v_max)
        ddq = (dq - self._dq) / self.dt
        ddq = np.clip(ddq, self._ddq - j_max * self.dt, self._ddq + j_max * self.dt)
        ddq = np.clip(ddq, -a_max, a_max)
        dq = self._dq + ddq * self.dt
        q_out = self._q_out + dq * self.dt
        q_out = np.clip(q_out, lo, hi)
        self._dq, self._ddq, self._q_out = dq, ddq, q_out
        return q_out.copy()


def check_safety(
    cmd: JointCommand,
    state: RobotState,
    limits: SafetyLimits,
    *,
    timestamp: float | None = None,
) -> SafetyDecision:
    """对单条命令做安全判定；返回事件与是否放行。"""
    now = float(state.timestamp if timestamp is None else timestamp)
    events: list[SafetyEvent] = []
    level = SafetyLevel.OK

    def raise_level(new: SafetyLevel) -> None:
        nonlocal level
        level = max(level, new)

    if not np.isfinite(cmd.q_target).all():
        events.append(SafetyEvent("nan_check", SafetyLevel.EMERGENCY, "command_nan", now))
        raise_level(SafetyLevel.EMERGENCY)
    events.extend(limits.joints.check(cmd.q_target, state.joint_vel, margin=limits.position_margin_rad))
    # 测量状态允许 1e-2 rad 的数值/软约束越界容差；命令目标仍按 margin 严格检查
    events.extend(limits.joints.check(state.joint_pos, None, margin=-0.01))
    for event in events:
        raise_level(event.level)

    roll, pitch, _ = state.rpy_deg()
    ang_vel_deg = np.degrees(np.linalg.norm(state.base_ang_vel))
    att = limits.attitude
    if abs(roll) >= att.roll_emergency_deg or abs(pitch) >= att.pitch_emergency_deg or ang_vel_deg >= att.ang_vel_emergency_deg_s:
        tilt_value = max(abs(roll), abs(pitch))
        trigger_rate = ang_vel_deg >= att.ang_vel_emergency_deg_s and tilt_value < att.roll_emergency_deg
        events.append(
            SafetyEvent(
                "attitude",
                SafetyLevel.EMERGENCY,
                "attitude_emergency",
                now,
                value=ang_vel_deg if trigger_rate else tilt_value,
                threshold=att.ang_vel_emergency_deg_s if trigger_rate else att.roll_emergency_deg,
                unit="deg/s" if trigger_rate else "deg",
            )
        )
        raise_level(SafetyLevel.EMERGENCY)
    elif abs(roll) >= att.roll_abort_deg or abs(pitch) >= att.pitch_abort_deg or ang_vel_deg >= att.ang_vel_abort_deg_s:
        tilt_value = max(abs(roll), abs(pitch))
        trigger_rate = ang_vel_deg >= att.ang_vel_abort_deg_s and tilt_value < att.roll_abort_deg
        events.append(
            SafetyEvent(
                "attitude",
                SafetyLevel.ABORT,
                "attitude_abort",
                now,
                value=ang_vel_deg if trigger_rate else tilt_value,
                threshold=att.ang_vel_abort_deg_s if trigger_rate else att.roll_abort_deg,
                unit="deg/s" if trigger_rate else "deg",
            )
        )
        raise_level(SafetyLevel.ABORT)
    elif abs(roll) >= att.roll_warn_deg or abs(pitch) >= att.pitch_warn_deg or ang_vel_deg >= att.ang_vel_warn_deg_s:
        tilt_value = max(abs(roll), abs(pitch))
        trigger_rate = ang_vel_deg >= att.ang_vel_warn_deg_s and tilt_value < att.roll_warn_deg
        events.append(
            SafetyEvent(
                "attitude",
                SafetyLevel.WARN,
                "attitude_warn",
                now,
                value=ang_vel_deg if trigger_rate else tilt_value,
                threshold=att.ang_vel_warn_deg_s if trigger_rate else att.roll_warn_deg,
                unit="deg/s" if trigger_rate else "deg",
            )
        )
        raise_level(SafetyLevel.WARN)

    height = float(state.base_pos[2])
    if height <= limits.base.height_emergency_m:
        events.append(SafetyEvent("base", SafetyLevel.EMERGENCY, "base_height_emergency", now, value=height, unit="m"))
        raise_level(SafetyLevel.EMERGENCY)
    elif height <= limits.base.height_abort_m:
        events.append(SafetyEvent("base", SafetyLevel.ABORT, "base_height_abort", now, value=height, unit="m"))
        raise_level(SafetyLevel.ABORT)
    elif height <= limits.base.height_warn_m:
        events.append(SafetyEvent("base", SafetyLevel.WARN, "base_height_warn", now, value=height, unit="m"))
        raise_level(SafetyLevel.WARN)
    if abs(float(state.base_lin_vel[2])) > limits.base.vertical_velocity_abort_m_s:
        events.append(SafetyEvent("base", SafetyLevel.ABORT, "vertical_velocity_abort", now, value=float(state.base_lin_vel[2]), unit="m/s"))
        raise_level(SafetyLevel.ABORT)

    if state.support_margin < -limits.support.com_radius_abort_m:
        events.append(SafetyEvent("support", SafetyLevel.ABORT, "com_outside_support", now, value=state.support_margin, unit="m"))
        raise_level(SafetyLevel.ABORT)
    elif state.support_margin < limits.support.com_radius_warn_m:
        events.append(SafetyEvent("support", SafetyLevel.WARN, "com_margin_warn", now, value=state.support_margin, unit="m"))
        raise_level(SafetyLevel.WARN)

    allowed = level < SafetyLevel.ABORT
    reason = events[-1].reason if events else "ok"
    return SafetyDecision(allowed=allowed, level=level, events=events, command=cmd, reason=reason)

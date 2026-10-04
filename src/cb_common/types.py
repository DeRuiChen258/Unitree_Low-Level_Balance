"""跨层共享的类型定义（第 11.5 节：跨层只走协议/消息）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np

from .joints import NUM_JOINTS, POLICY_JOINT_NAMES


class SkillId(IntEnum):
    """技能 id（与 configs/skills.yaml 绑定，代码中不出现裸数字语义）。"""

    STAND = 0
    WALK = 1
    RUN = 2
    JUMP = 3
    WAVE = 4
    TURN = 5
    RECOVER = 6
    SAFE_STOP = 7


SKILL_NAMES: tuple[str, ...] = tuple(s.name.lower() for s in SkillId)


class SafetyLevel(IntEnum):
    """安全等级（数值越大越严重）。"""

    OK = 0
    WARN = 1
    ABORT = 2
    EMERGENCY = 3


class BalanceMode(StrEnum):
    """平衡补偿模式：真机默认 pass-through。"""

    PASS_THROUGH = "pass_through"
    LOW_LEVEL_SIM_ONLY = "low_level_sim_only"
    LEARNED_RESIDUAL = "learned_residual"


class GaitPhase(StrEnum):
    """步态相位（由接触序列估计）。"""

    UNKNOWN = "unknown"
    LEFT_STANCE = "left_stance"
    RIGHT_STANCE = "right_stance"
    DOUBLE_SUPPORT = "double_support"
    FLIGHT = "flight"


class JumpPhase(StrEnum):
    """跳跃五段相位（第 5.3 节）。"""

    NONE = "none"
    APPROACH = "approach"
    TAKEOFF = "takeoff"
    FLIGHT = "flight"
    LANDING = "landing"
    RECOVER = "recover"


class RobotMode(StrEnum):
    """安全状态机状态（第 10.2 节）。"""

    IDLE = "IDLE"
    STAND_UP = "STAND_UP"
    STAND = "STAND"
    LOCOMOTION = "LOCOMOTION"
    SKILL_EXECUTION = "SKILL_EXECUTION"
    RECOVER = "RECOVER"
    FALLEN = "FALLEN"
    FALL_RECOVERY = "FALL_RECOVERY"
    SAFE_STOP = "SAFE_STOP"
    FAULT_RECOVERY = "FAULT_RECOVERY"
    ERROR = "ERROR"


@dataclass
class RobotState:
    """统一机器人状态（仿真 / 真机同接口）。单位：m, rad, s, m/s, rad/s, Nm。"""

    timestamp: float
    base_pos: np.ndarray
    base_quat: np.ndarray          # wxyz
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    joint_pos: np.ndarray          # 策略序
    joint_vel: np.ndarray          # 策略序
    joint_torque: np.ndarray | None = None
    contact: tuple[bool, bool] = (False, False)
    com: np.ndarray | None = None       # 世界系 CoM
    com_vel: np.ndarray | None = None
    cp: np.ndarray | None = None        # 捕获点 xy
    support_margin: float = 0.0         # 支撑域余量（m，负值表示越界）
    joint_names: Sequence[str] = POLICY_JOINT_NAMES
    frame_id: int = 0

    def __post_init__(self) -> None:
        for name in ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            setattr(self, name, arr)
        if self.joint_pos.shape != (NUM_JOINTS,) or self.joint_vel.shape != (NUM_JOINTS,):
            raise ValueError(f"joint_pos/joint_vel must be ({NUM_JOINTS},), got {self.joint_pos.shape}/{self.joint_vel.shape}")
        if self.base_quat.shape != (4,):
            raise ValueError(f"base_quat must be (4,) wxyz, got {self.base_quat.shape}")

    def rpy(self) -> tuple[float, float, float]:
        """返回横滚/俯仰/偏航（rad）。"""
        w, x, y, z = self.base_quat
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def rpy_deg(self) -> tuple[float, float, float]:
        """返回横滚/俯仰/偏航（deg）。"""
        return tuple(float(np.degrees(v)) for v in self.rpy())  # type: ignore[return-value]


@dataclass
class JointCommand:
    """关节命令（策略序绝对位置目标；真机经 SafetyWrapper 后才下发）。"""

    q_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    timestamp: float
    torque_ff: np.ndarray | None = None
    source: str = "policy"
    skill: str = "stand"

    def __post_init__(self) -> None:
        self.q_target = np.asarray(self.q_target, dtype=np.float64)
        self.kp = np.asarray(self.kp, dtype=np.float64)
        self.kd = np.asarray(self.kd, dtype=np.float64)
        if self.q_target.shape != (NUM_JOINTS,):
            raise ValueError(f"q_target must be ({NUM_JOINTS},), got {self.q_target.shape}")

    def copy_with(self, q_target: np.ndarray | None = None, **kwargs: Any) -> JointCommand:
        """返回替换字段后的副本（不修改原对象）。"""
        payload = {
            "q_target": self.q_target if q_target is None else q_target,
            "kp": self.kp,
            "kd": self.kd,
            "timestamp": self.timestamp,
            "torque_ff": self.torque_ff,
            "source": self.source,
            "skill": self.skill,
        }
        payload.update(kwargs)
        return JointCommand(**payload)


@dataclass
class SkillCommand:
    """调度器输出的技能指令（不直接写关节目标）。"""

    skill: str
    params: Mapping[str, Any] = field(default_factory=dict)
    blend_weight: float = 1.0
    reason_code: str = "rule"
    source: str = "scheduler"
    timestamp: float = 0.0
    transition_window_s: float = 0.0
    from_skill: str | None = None


@dataclass
class SkillContext:
    """技能进入时的上下文。"""

    command: SkillCommand
    default_pose: np.ndarray
    joint_names: Sequence[str] = POLICY_JOINT_NAMES
    dt: float = 0.02
    limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    """技能输出：关节目标增量 + 相位 + 接触期望。"""

    delta_q: np.ndarray                     # 策略序增量（rad）
    skill: str
    phase: float = 0.0
    jump_phase: JumpPhase = JumpPhase.NONE
    expected_contact: tuple[bool, bool] = (True, True)
    active_groups: tuple[str, ...] = ("lower_body", "torso", "arms")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta_q = np.asarray(self.delta_q, dtype=np.float64)
        if self.delta_q.shape != (NUM_JOINTS,):
            raise ValueError(f"delta_q must be ({NUM_JOINTS},), got {self.delta_q.shape}")


@dataclass
class ObsHistory:
    """固定长度观测历史（不足时重复填充并打标记）。"""

    length: int
    key_dim: int
    buffer: list[np.ndarray] = field(default_factory=list)
    padded: bool = False

    def append(self, key_obs: np.ndarray) -> None:
        """追加一帧关键观测组。"""
        arr = np.asarray(key_obs, dtype=np.float64).reshape(-1)
        if arr.size != self.key_dim:
            raise ValueError(f"key obs must be {self.key_dim}, got {arr.size}")
        self.buffer.append(arr.copy())
        if len(self.buffer) > self.length:
            self.buffer.pop(0)
            self.padded = False

    def reset(self, fill: np.ndarray | None = None) -> None:
        """清空历史；可指定填充帧。"""
        self.buffer.clear()
        self.padded = False
        if fill is not None:
            self.buffer.append(np.asarray(fill, dtype=np.float64).reshape(-1).copy())

    def stack(self) -> np.ndarray:
        """返回按时间顺序展开的历史（旧 → 新）。"""
        if not self.buffer:
            return np.zeros(self.length * self.key_dim, dtype=np.float64)
        frames = list(self.buffer)
        if len(frames) < self.length:
            self.padded = True
            frames = [frames[0]] * (self.length - len(frames)) + frames
        return np.concatenate(frames[-self.length :])


@dataclass
class TaskContext:
    """决策层任务上下文（第 5.6 节文本化状态的输入之一）。"""

    task: str = "stand"
    terrain: str = "flat"
    external_push: bool = False
    latency_ms: float = 0.0
    battery_ok: bool = True
    falls_last_60s: int = 0
    watchdog_trips_last_60s: int = 0
    candidates: Sequence[str] = field(default_factory=lambda: ["continue_current", "stand", "recover", "safe_stop"])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    """LayA 结构化决策输出（第 8.1 节，字段齐全）。"""

    action_id: str
    primitive: str
    confidence: float
    risk: float
    need_human_review: bool
    fallback_action: str
    probabilities: dict[str, float] = field(default_factory=dict)
    reason_code: str = "ok"
    heads: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """越界/NaN 立即拒绝，避免污染调度器。"""
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.risk <= 1.0:
            raise ValueError(f"confidence/risk out of range: {self.confidence}, {self.risk}")
        if not self.action_id:
            raise ValueError("action_id must not be empty")


@dataclass
class SafetyEvent:
    """安全事件（reason/value/threshold/unit 必填，可定位）。"""

    kind: str
    level: SafetyLevel
    reason: str
    timestamp: float
    joint: str | None = None
    value: float | None = None
    threshold: float | None = None
    unit: str = ""
    action: str = ""


@dataclass
class SafetyDecision:
    """安全层输出：是否放行 + 事件 +（可选）修正后的命令。"""

    allowed: bool
    level: SafetyLevel
    events: list[SafetyEvent] = field(default_factory=list)
    command: JointCommand | None = None
    reason: str = ""

    @property
    def emergency(self) -> bool:
        """是否达到急停等级。"""
        return self.level >= SafetyLevel.EMERGENCY

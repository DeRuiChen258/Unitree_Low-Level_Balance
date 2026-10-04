"""行为原语：Stand / Bend / Step / Lunge / Reach / Grasp / Lift / StandUp / Recovery。

每个原语输出 `BodyTarget`（基座/双脚/躯干/手目标），由 WholeBodyIK 转成关节目标；
轨迹使用 quintic/三角包络，保证连续、类人的加减速与支撑相/摆动相时序（第 10 节）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .footstep import FootstepPlan


def quintic(s: float) -> float:
    """最小 jerk 插值曲线（0→1）。"""
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def yaw_matrix(yaw: float) -> np.ndarray:
    """绕 z 轴 yaw 的旋转矩阵。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def embrace_points(
    grasp_point: np.ndarray,
    half_width: float,
    clearance_m: float = 0.055,
    half_depth: float = 0.14,
    side_drop_m: float = 0.06,
) -> dict[str, np.ndarray]:
    """双手抱取触点（世界系）：双手分别握在箱体**左右两侧面**（不是环抱顶角）。

    以抓取参考点（近侧顶边中点）为基准：
    - 横向：左右各外移「半宽 + clearance」，掌面贴住箱体侧面而不是插进箱内；
    - 纵向：从近侧面向后退回一体深（到箱体中线附近），使手掌压在侧面中部；
    - 高度：从顶边下降 `side_drop_m`，落在侧面上部（抱两侧而非托顶）。
    """
    base = np.asarray(grasp_point, dtype=np.float64)
    lateral = float(half_width) + float(clearance_m)
    back = float(half_depth) - 0.02
    return {
        "right": base + np.array([back, -lateral, -float(side_drop_m)]),
        "left": base + np.array([back, lateral, -float(side_drop_m)]),
    }


@dataclass
class BodyTarget:
    """全身目标（世界系），控制器据此解 IK。"""

    base_pos: np.ndarray
    base_yaw: float
    left_foot_pos: np.ndarray
    left_foot_yaw: float
    right_foot_pos: np.ndarray
    right_foot_yaw: float
    trunk_pitch: float = 0.0
    trunk_roll: float = 0.0
    hand_target: np.ndarray | None = None
    reach_side: str = "right"
    offhand_target: np.ndarray | None = None
    offhand_side: str = "left"
    grasp: bool = False
    phase: float = 0.0
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self, **kwargs: Any) -> BodyTarget:
        """返回替换字段后的副本。"""
        payload = {
            "base_pos": self.base_pos.copy(),
            "base_yaw": self.base_yaw,
            "left_foot_pos": self.left_foot_pos.copy(),
            "left_foot_yaw": self.left_foot_yaw,
            "right_foot_pos": self.right_foot_pos.copy(),
            "right_foot_yaw": self.right_foot_yaw,
            "trunk_pitch": self.trunk_pitch,
            "trunk_roll": self.trunk_roll,
            "hand_target": None if self.hand_target is None else self.hand_target.copy(),
            "reach_side": self.reach_side,
            "offhand_target": None if self.offhand_target is None else self.offhand_target.copy(),
            "offhand_side": self.offhand_side,
            "grasp": self.grasp,
            "phase": self.phase,
            "finished": self.finished,
            "metadata": dict(self.metadata),
        }
        payload.update(kwargs)
        return BodyTarget(**payload)


class Primitive:
    """原语接口：reset / update / is_finished / is_safe（第 10 节）。"""

    name = "primitive"
    duration_s: float = 1.0

    def __init__(self) -> None:
        self.start: BodyTarget | None = None
        self.goal: BodyTarget | None = None
        self.context: dict[str, Any] = {}
        self.t = 0.0
        self.finished = False

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """以当前目标为起点复位。"""
        self.start = target.copy()
        self.goal = target.copy()
        self.t = 0.0
        self.context = dict(context or {})
        self.finished = False

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """推进原语（子类实现）。"""
        raise NotImplementedError

    def is_finished(self) -> bool:
        """是否完成。"""
        return self.finished

    def is_safe(self, balance: Any) -> bool:
        """原语特定安全条件（默认安全）。"""
        return True

    def describe(self) -> dict[str, Any]:
        """日志摘要。"""
        return {"primitive": self.name, "t": self.t, "duration": self.duration_s, "finished": self.finished}


class StandPrimitive(Primitive):
    """站立/保持（默认安全态）。"""

    name = "STAND"
    duration_s = 0.8

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """保持当前脚位，回到直立躯干。"""
        self.t += dt
        w = quintic(self.t / self.duration_s)
        out = target.copy(
            trunk_pitch=float((1.0 - w) * target.trunk_pitch),
            trunk_roll=float((1.0 - w) * target.trunk_roll),
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
        )
        self.finished = out.finished
        return out


class BendPrimitive(Primitive):
    """弯腰：髋-膝-踝协同 + 躯干前倾 + 手臂反向配重（类人髋铰链）。"""

    name = "BEND"
    duration_s = 2.2

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """记录目标弯腰深度与目标位置。"""
        super().reset(target, context)
        self.target_pitch = float(self.context.get("trunk_pitch", 0.6))
        self.crouch_depth = float(self.context.get("crouch_depth", 0.14))
        self.hip_shift = float(self.context.get("hip_shift", 0.0))
        # 下蹲量以「站立基座高度」为绝对基准：恢复后重入不会累积下沉
        self.stand_base_z = float(self.context.get("stand_base_z", self.start.base_pos[2]))
        self.target_pos = self.context.get("object_position")
        self.speed_scale = float(self.context.get("speed_scale", 1.0))
        self.duration_s = max(0.6, 2.2 / max(self.speed_scale, 0.1))

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """逐步前倾；基座后移下沉，手臂向后配重。"""
        self.t += dt
        progress = min(1.0, self.t / self.duration_s)
        # 类人时序：先躯干前倾（0–55%），再屈膝下沉（25–100%），避免 CoM 先向后失稳
        pitch_w = quintic(min(progress / 0.55, 1.0))
        crouch_w = quintic(max(0.0, (progress - 0.25) / 0.75))
        back = -self.hip_shift * crouch_w
        base_z = self.stand_base_z - self.crouch_depth * crouch_w
        base = np.array([self.start.base_pos[0] + back, self.start.base_pos[1], base_z], dtype=np.float64)
        out = target.copy(
            base_pos=base,
            trunk_pitch=float(self.start.trunk_pitch + (self.target_pitch - self.start.trunk_pitch) * pitch_w),
            phase=progress,
            finished=self.t >= self.duration_s,
            metadata={
                **target.metadata,
                "bend_progress": progress,
                "pitch_progress": pitch_w,
                "crouch_progress": crouch_w,
            },
        )
        self.finished = out.finished
        return out


class StepPrimitive(Primitive):
    """单步：先重心转移，再抬脚-摆腿-落脚（含 5 cm 足端间隙）。"""

    name = "STEP"

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """读取 FootstepPlan。"""
        super().reset(target, context)
        plan = self.context.get("plan")
        if not isinstance(plan, FootstepPlan) or not plan.feasible:
            raise ValueError("StepPrimitive requires a feasible FootstepPlan")
        self.plan = plan
        self.duration_s = float(plan.step_duration)
        self.swing = plan.swing_foot
        self.support = plan.support_foot
        self.start_swing = target.left_foot_pos.copy() if self.swing == "left" else target.right_foot_pos.copy()
        self.landing = np.array([plan.landing_xy[0], plan.landing_xy[1], self.start_swing[2]])

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """按 支撑相(20%)-摆动相(60%)-落脚相(20%) 执行。"""
        self.t += dt
        progress = min(1.0, self.t / self.duration_s)
        # 重心转移到支撑脚（前 20% 与后 20%）
        transfer = quintic(min(progress / 0.2, 1.0)) * (1.0 - quintic(max(0.0, (progress - 0.8) / 0.2)))
        support_pos = target.right_foot_pos if self.support == "right" else target.left_foot_pos
        # 重心转移只做水平方向，保持当前下蹲高度（避免把 base 拉回站立高度）
        lateral_shift = np.array(
            [support_pos[0] - self.start.base_pos[0], support_pos[1] - self.start.base_pos[1], 0.0]
        )
        base_target = self.start.base_pos + lateral_shift * 0.25 * transfer
        # 摆动腿轨迹
        if progress <= 0.2:
            swing_pos = self.start_swing.copy()
        elif progress >= 0.8:
            swing_pos = self.landing.copy()
        else:
            s = quintic((progress - 0.2) / 0.6)
            swing_pos = self.start_swing + (self.landing - self.start_swing) * s
            swing_pos[2] = self.start_swing[2] + 0.05 * np.sin(np.pi * s)
        left = swing_pos if self.swing == "left" else target.left_foot_pos
        right = swing_pos if self.swing == "right" else target.right_foot_pos
        out = target.copy(
            base_pos=base_target,
            left_foot_pos=left.copy(),
            right_foot_pos=right.copy(),
            phase=progress,
            finished=progress >= 1.0,
            metadata={**target.metadata, "swing_foot": self.swing, "landing": self.landing.tolist()},
        )
        self.finished = out.finished
        return out


class LungePrimitive(StepPrimitive):
    """弓步：朝目标方向的较大前步，落脚后重心前移形成前后错开支撑。"""

    name = "LUNGE"

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """执行弓步（步幅更大、重心更靠前）。"""
        out = super().update(target, dt)
        # 弓步完成后，基座向目标方向前移，形成前后脚支撑
        forward = 0.05 * quintic(out.phase)
        direction = np.sign(self.landing[0] - self.start_swing[0]) or 1.0
        out = out.copy(base_pos=out.base_pos + np.array([direction * forward, 0.0, 0.0]))
        return out


class ReachPrimitive(Primitive):
    """抱取伸手：参考 G1_Waving 第一/二阶段（抬臂准备位 → 小幅摆动接近），双手抱箱体近侧上角。

    小臂（肘）弯曲幅度按需求收小：准备位肘角从 waving 的 −0.85 降到 −0.45，
    双臂不交叉抱臂，避免既遮挡视线又造成前臂/躯干自接触。
    """

    name = "REACH"
    duration_s = 1.8
    # G1_Waving configs/g1_wave.yaml 的 prepare_pose（右侧绝对角，左侧镜像）；肘角按需求收小
    PREPARE_POSE: dict[str, dict[str, float]] = {
        "right": {
            "shoulder_pitch": -0.50,
            "shoulder_roll": -0.10,
            "shoulder_yaw": 0.70,
            "elbow": -0.45,
            "wrist_roll": -1.20,
            "wrist_pitch": -0.20,
            "wrist_yaw": 0.0,
        },
        "left": {
            "shoulder_pitch": -0.50,
            "shoulder_roll": 0.10,
            "shoulder_yaw": -0.70,
            "elbow": -0.45,
            "wrist_roll": 1.20,
            "wrist_pitch": -0.20,
            "wrist_yaw": 0.0,
        },
    }

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """记录目标点、手臂侧与箱体尺寸（双手触点由几何算出）。"""
        super().reset(target, context)
        self.object_position = np.asarray(self.context.get("object_position", target.base_pos + np.array([0.35, 0.0, 0.2])), dtype=np.float64)
        self.reach_side = str(self.context.get("reach_side", "right"))
        self.offhand_side = "left" if self.reach_side == "right" else "right"
        self.two_handed = bool(self.context.get("two_handed", True))
        half_width = float(self.context.get("object_half_width", 0.14))
        clearance = float(self.context.get("embrace_clearance_m", 0.055))
        half_depth = float(self.context.get("object_half_depth", 0.14))
        side_drop = float(self.context.get("embrace_side_drop_m", 0.06))
        # 双手触点：箱体左右两侧面（相对抓取参考点偏移，见 embrace_points）
        points = embrace_points(self.object_position, half_width, clearance, half_depth, side_drop)
        self.hand_goal = points[self.reach_side]
        self.offhand_goal = points[self.offhand_side]
        # 第一阶段：抬臂准备（waving prepare）；第二阶段：小幅摆动接近（0.35 Hz）
        # 双手抱两侧时不做冻结准备位：低头抱取姿态下 waving 大抬臂会把前臂扫进箱体
        # （实测穿模 ≈7 cm），改由两段接近路径完成「先外移后前推」。
        default_prepare = 0.0 if self.two_handed else 0.9
        self.prepare_duration_s = float(self.context.get("prepare_duration_s", default_prepare))
        self.oscillation_hz = float(self.context.get("oscillation_hz", 0.35))
        self.oscillation_amp_m = float(self.context.get("oscillation_amp_m", 0.02))
        # 起点必须是「实测手位」，保证手从当前位置单调向前/向下接近物体（避免先向后甩）
        measured = self.context.get("hand_position")
        if measured is not None:
            self.start_hand = np.asarray(measured, dtype=np.float64).reshape(3).copy()
        elif target.hand_target is not None:
            self.start_hand = target.hand_target.copy()
        else:
            self.start_hand = self.object_position + np.array([-0.25, 0.0, 0.25])
        measured_off = self.context.get("offhand_position")
        if measured_off is not None:
            self.start_offhand = np.asarray(measured_off, dtype=np.float64).reshape(3).copy()
        elif target.offhand_target is not None:
            self.start_offhand = target.offhand_target.copy()
        else:
            # 无实测副手位时按 y 轴镜像估计（左右手关于躯干中线对称）
            self.start_offhand = self.start_hand + np.array([0.0, -2.0 * self.start_hand[1], 0.0])
        self.stop_on_warning = bool(self.context.get("stop_on_warning", False))

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """两段接近（先外移到侧面、再前推到触点）+ 二阶小摆动；躯干保持当前前倾。

        直线插值会让手掌从贴身位斜穿箱体（实测穿模 6–7 cm），因此拆成两段：
        A 段先在箱体近侧面之前对齐横向/高度，B 段再沿侧面前推贴住箱面。
        """
        self.t += dt
        progress = min(1.0, self.t / self.duration_s)
        split = 0.55
        w_a = quintic(min(1.0, progress / split))
        w_b = quintic(max(0.0, (progress - split) / (1.0 - split)))
        # 接近点：退到箱体近侧之前，并再向外让开一点（避免横向偏移目标时抄近路切进箱体）
        def approach_of(goal: np.ndarray) -> np.ndarray:
            outward = 0.03 * (1.0 if goal[1] >= 0.0 else -1.0)
            return goal + np.array([-0.18, outward, 0.02])

        approach = approach_of(self.hand_goal)
        approach_off = approach_of(self.offhand_goal)
        hand = self.start_hand + (approach - self.start_hand) * w_a
        hand = hand + (self.hand_goal - approach) * w_b
        decay = max(0.0, 1.0 - max(w_a, w_b))
        angle = 2.0 * np.pi * self.oscillation_hz * self.t
        wobble = np.array([0.0, self.oscillation_amp_m * decay * np.sin(angle), 0.5 * self.oscillation_amp_m * decay * np.cos(angle)])
        hand = hand + wobble
        offhand = None
        if self.two_handed:
            offhand = self.start_offhand + (approach_off - self.start_offhand) * w_a
            offhand = offhand + (self.offhand_goal - approach_off) * w_b
        if offhand is not None:
            offhand = offhand + wobble
        out = target.copy(
            hand_target=hand,
            reach_side=self.reach_side,
            offhand_target=offhand,
            offhand_side=self.offhand_side,
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
            metadata={
                **target.metadata,
                "hand_error": float(np.linalg.norm(hand - self.hand_goal)),
                "offhand_error": float(np.linalg.norm(offhand - self.offhand_goal)) if offhand is not None else float("nan"),
            },
        )
        self.finished = out.finished
        return out

    def prepare_weight(self) -> float:
        """waving 第一阶段进度 [0,1]：>0 时手臂先走准备位，不追手部 IK 目标。"""
        if self.prepare_duration_s <= 0:
            return 1.0
        return quintic(min(self.t / self.prepare_duration_s, 1.0))

    def is_safe(self, balance: Any) -> bool:
        """平衡裕度不足时不可继续伸手。"""
        return bool(getattr(balance, "stability_margin", 0.0) >= 0.0)


class GraspPrimitive(Primitive):
    """抱取：双手保持在箱体近侧上角并触发 mock grasp（载荷由控制器施加）。"""

    name = "GRASP"
    duration_s = 0.5

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """记录双手触点的保持目标。"""
        super().reset(target, context)
        self.object_position = np.asarray(self.context.get("object_position", target.base_pos + np.array([0.3, 0.0, 0.1])), dtype=np.float64)
        self.reach_side = str(self.context.get("reach_side", "right"))
        self.offhand_side = "left" if self.reach_side == "right" else "right"
        half_width = float(self.context.get("object_half_width", 0.14))
        clearance = float(self.context.get("embrace_clearance_m", 0.055))
        half_depth = float(self.context.get("object_half_depth", 0.14))
        side_drop = float(self.context.get("embrace_side_drop_m", 0.06))
        points = embrace_points(self.object_position, half_width, clearance, half_depth, side_drop)
        self.hand_goal = (
            np.asarray(target.hand_target, dtype=np.float64).copy() if target.hand_target is not None else points[self.reach_side]
        )
        self.offhand_goal = (
            np.asarray(target.offhand_target, dtype=np.float64).copy()
            if target.offhand_target is not None
            else points[self.offhand_side]
        )

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """保持双手在箱体上并标志 grasp=True。"""
        self.t += dt
        out = target.copy(
            hand_target=self.hand_goal,
            reach_side=self.reach_side,
            offhand_target=self.offhand_goal,
            offhand_side=self.offhand_side,
            grasp=True,
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
        )
        self.finished = out.finished
        return out


class LiftPrimitive(Primitive):
    """抬升：双手（抱持）同步上抬 + 躯干轻微后仰回到直立。"""

    name = "LIFT"
    duration_s = 1.0

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """记录抬升高度与双手起点。"""
        super().reset(target, context)
        self.lift_height = float(self.context.get("lift_height", 0.18))
        self.object_position = np.asarray(self.context.get("object_position", target.base_pos + np.array([0.3, 0.0, 0.2])), dtype=np.float64)
        self.start_hand = target.hand_target.copy() if target.hand_target is not None else self.object_position
        self.start_offhand = target.offhand_target.copy() if target.offhand_target is not None else None
        self.reach_side = target.reach_side
        self.offhand_side = target.offhand_side

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """抬升双手并逐步减小躯干前倾。"""
        self.t += dt
        w = quintic(self.t / self.duration_s)
        rise = np.array([0.0, 0.0, self.lift_height * w])
        hand = self.start_hand + rise
        out = target.copy(
            hand_target=hand,
            offhand_target=None if self.start_offhand is None else self.start_offhand + rise,
            reach_side=self.reach_side,
            offhand_side=self.offhand_side,
            trunk_pitch=float(target.trunk_pitch * (1.0 - 0.4 * w)),
            grasp=True,
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
        )
        self.finished = out.finished
        return out


class StandUpPrimitive(Primitive):
    """起身：解除弯腰、回到平行站姿（含弓步后脚回位）。"""

    name = "STAND_UP"
    duration_s = 1.8

    def reset(self, target: BodyTarget, context: Mapping[str, Any] | None = None) -> None:
        """记录起始目标与站姿参数。"""
        super().reset(target, context)
        self.stance_y = float(self.context.get("stance_width", 0.14))
        self.stand_base_z = float(self.context.get("stand_base_z", target.base_pos[2]))
        self.start_left = target.left_foot_pos.copy()
        self.start_right = target.right_foot_pos.copy()
        self.stance_x = float(self.context.get("stance_x", 0.0))

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """躯干回正 + 双脚回到平行站姿。"""
        self.t += dt
        w = quintic(self.t / self.duration_s)
        base = target.base_pos.copy()
        base[0] += (self.stance_x - base[0]) * w
        # 骨盆升回站立高度（弯腰时下沉了 crouch_depth，起身必须补回来）
        base[2] += (self.stand_base_z - base[2]) * w
        half = max(0.10, self.stance_y / 2.0)
        left = self.start_left + (np.array([self.stance_x, half, 0.0]) - self.start_left) * w
        right = self.start_right + (np.array([self.stance_x, -half, 0.0]) - self.start_right) * w
        # 抱持中起身：双手目标随基座一起平移，保持「箱体贴身」构型不变
        delta = base - target.base_pos
        hand = None if target.hand_target is None or not target.grasp else target.hand_target + delta
        offhand = None if target.offhand_target is None or not target.grasp else target.offhand_target + delta
        out = target.copy(
            base_pos=base,
            trunk_pitch=float(target.trunk_pitch * (1.0 - w)),
            trunk_roll=float(target.trunk_roll * (1.0 - w)),
            left_foot_pos=left,
            right_foot_pos=right,
            hand_target=hand,
            offhand_target=offhand,
            grasp=bool(target.grasp),
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
        )
        self.finished = out.finished
        return out


class RecoveryPrimitive(Primitive):
    """恢复：停止上肢、减小弯腰、脚步回退到安全支撑（危险时退回站姿）。"""

    name = "RECOVER"
    duration_s = 1.2

    def update(self, target: BodyTarget, dt: float) -> BodyTarget:
        """躯干快速回到安全前倾角并把手臂收回。"""
        self.t += dt
        w = quintic(self.t / self.duration_s)
        safe_pitch = 0.25 * np.sign(target.trunk_pitch)
        # 抱持载荷时恢复动作必须把腿伸直（否则箱体会压进屈膝空间并破坏力矩闭环）
        base = target.base_pos.copy()
        if target.grasp:
            stand_z = float(self.context.get("stand_base_z", base[2]))
            base[2] = float(base[2] + (stand_z - 0.05 - base[2]) * w)
        out = target.copy(
            base_pos=base,
            trunk_pitch=float(target.trunk_pitch * (1.0 - w) + safe_pitch * w),
            hand_target=target.hand_target if target.grasp else None,
            offhand_target=target.offhand_target if target.grasp else None,
            grasp=bool(target.grasp),
            phase=min(1.0, self.t / self.duration_s),
            finished=self.t >= self.duration_s,
            metadata={**target.metadata, "recovery": True},
        )
        self.finished = out.finished
        return out

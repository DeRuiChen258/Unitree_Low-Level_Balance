"""Motion Manager：行为原语选择与状态机，强制稳定性门禁（第 11、12 节）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from .balance_monitor import BalanceState
from .decision import DecisionModel, PickupAction, PickupDecision
from .footstep import FootstepPlanner


class PickupPhase(StrEnum):
    """Motion Manager 状态（第 11 节）。"""

    IDLE = "IDLE"
    BENDING = "BENDING"
    BALANCE_WARNING = "BALANCE_WARNING"
    FOOT_ADJUSTMENT = "FOOT_ADJUSTMENT"
    LUNGE = "LUNGE"
    REACHING = "REACHING"
    GRASPING = "GRASPING"
    LIFTING = "LIFTING"
    STANDING_UP = "STANDING_UP"
    RECOVERY = "RECOVERY"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class ManagerCommand:
    """Motion Manager 输出：原语 + 参数 + 阶段 + 决策（供控制器执行）。"""

    primitive: str
    params: dict[str, Any] = field(default_factory=dict)
    phase: PickupPhase = PickupPhase.IDLE
    reason: str = ""
    decision: PickupDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        """日志友好输出。"""
        return {
            "primitive": self.primitive,
            "phase": self.phase.value,
            "reason": self.reason,
            "params": {k: (v.value if isinstance(v, PickupAction) else v) for k, v in self.params.items()},
            "decision": None if self.decision is None else self.decision.to_dict(),
        }


@dataclass
class MotionManager:
    """行为选择与门禁：不允许稳定裕度不足时 BEND → REACH。"""

    planner: FootstepPlanner = field(default_factory=FootstepPlanner)
    decision_model: DecisionModel | None = None
    reach_margin_m: float = 0.015
    step_margin_m: float = 0.0
    critical_margin_m: float = -0.05
    #: 负重（已抱持）时的失稳阈值：Phase-1 存在虚拟 gantry（外部支撑反力），
    #: 此时「CoM 投影是否在支撑多边形内」不再是正确的失稳判据，
    #: 因此放宽触发阈值，真实裕度仍逐帧记录并在评测中如实报告。
    carry_critical_margin_m: float = -0.25
    target_bend_pitch: float = 0.65
    crouch_depth: float = 0.17
    hip_shift: float = 0.05
    reach_extension_m: float = 0.12
    max_steps: int = 3
    phase: PickupPhase = PickupPhase.IDLE
    step_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    last_decision: PickupDecision | None = None
    _bend_done: bool = False
    _step_in_progress: bool = False
    _step_request_streak: int = 0
    _proactive_lunge_done: bool = False

    def reset(self) -> None:
        """复位状态机。"""
        self.phase = PickupPhase.IDLE
        self.step_count = 0
        self.history.clear()
        self.last_decision = None
        self._bend_done = False
        self._step_in_progress = False
        self._step_request_streak = 0
        self._proactive_lunge_done = False

    def _log(self, reason: str, decision: PickupDecision | None = None) -> None:
        self.history.append(
            {
                "phase": self.phase.value,
                "reason": reason,
                "step_count": self.step_count,
                "decision": None if decision is None else decision.to_dict(),
            }
        )

    def _transition(self, phase: PickupPhase, reason: str, decision: PickupDecision | None = None) -> None:
        if phase != self.phase:
            self.phase = phase
            self._log(reason, decision)

    def update(
        self,
        *,
        balance: BalanceState,
        goal: Mapping[str, Any],
        context: Mapping[str, Any],
        step_finished: bool = False,
    ) -> ManagerCommand:
        """每个决策周期调用：返回要执行的原语（含脚步计划）。"""
        candidates: Sequence[PickupAction] = tuple(PickupAction)
        decision_model = self.decision_model
        if decision_model is None:
            from .decision import RuleBasedDecision

            decision_model = RuleBasedDecision()
        decision = decision_model.predict(balance=balance, goal=goal, phase=self.phase.value, candidates=candidates)
        self.last_decision = decision
        if step_finished:
            self._step_in_progress = False
        margin = decision.stability_margin
        object_position = np.asarray(goal.get("object_position", balance.com_projection + np.array([0.35, 0.0])), dtype=np.float64)
        active = str(context.get("active_primitive", ""))
        carrying = bool(context.get("grasped", False))
        critical_margin = self.carry_critical_margin_m if carrying else self.critical_margin_m
        step_airborne_grace = (
            active in ("STEP", "LUNGE") and not step_finished and margin > critical_margin - 0.05
        )
        # Tier 0：不稳定 → 恢复/停止
        # 负重阶段用 carry 阈值统一判定（含预测裕度），避免监控器的分级阈值把
        # 「外部支撑（gantry）承担倾覆力矩」的负重相误判为失稳
        effective_margin = min(margin, balance.predicted_margin) if carrying else margin
        unstable = (not carrying and balance.recovery_level >= 4) or effective_margin < critical_margin
        if unstable and not step_airborne_grace:
            self._transition(PickupPhase.RECOVERY, "critical_margin", decision)
            return ManagerCommand("RECOVER", {}, self.phase, "critical_margin", decision)
        if (
            decision.action in (PickupAction.STOP,)
            or (balance.recovery_level >= 3 and self.phase in (PickupPhase.BENDING, PickupPhase.REACHING))
        ) and self.phase not in (PickupPhase.RECOVERY,):
            self._transition(PickupPhase.RECOVERY, "decision_stop", decision)
            return ManagerCommand("RECOVER", {}, self.phase, "decision_stop", decision)
        # 状态机主流程
        if self.phase in (PickupPhase.IDLE,):
            self._transition(PickupPhase.BENDING, "task_start", decision)
            return ManagerCommand(
                "BEND",
                {
                    "trunk_pitch": self.target_bend_pitch,
                    "crouch_depth": self.crouch_depth,
                    "hip_shift": self.hip_shift,
                    "object_position": object_position,
                },
                self.phase,
                "task_start",
                decision,
            )
        if self.phase in (PickupPhase.BENDING, PickupPhase.BALANCE_WARNING):
            active = str(context.get("active_primitive", ""))
            # 主动弓步：弯腰达到 30% 且目标超出当前臂展时提前迈步（不等 margin 变负）
            if bool(context.get("lunge_needed", False)) and not self._proactive_lunge_done and self.step_count < self.max_steps:
                plan = self.planner.plan(
                    balance=balance,
                    left_foot_pos=np.asarray(context["left_foot_pos"]),
                    left_foot_yaw=float(context.get("left_foot_yaw", 0.0)),
                    right_foot_pos=np.asarray(context["right_foot_pos"]),
                    right_foot_yaw=float(context.get("right_foot_yaw", 0.0)),
                    target_position=object_position,
                    use_capture_point=bool(context.get("use_capture_point", True)),
                    use_prediction=bool(context.get("use_prediction", True)),
                    target_bias_m=0.18,
                    obstacle_xy=context.get("obstacle_xy"),
                    obstacle_radius=float(context.get("obstacle_radius", 0.0)),
                )
                if plan.feasible:
                    self._transition(PickupPhase.LUNGE, "proactive_lunge", decision)
                    self.step_count += 1
                    self._step_in_progress = True
                    self._proactive_lunge_done = True
                    self._step_request_streak = 0
                    return ManagerCommand("LUNGE", {"plan": plan}, self.phase, "proactive_lunge", decision)
            if (
                active in ("BEND", "REACH")
                and not step_finished
                and decision.action
                not in (
                    PickupAction.STEP_LEFT,
                    PickupAction.STEP_RIGHT,
                    PickupAction.LUNGE_LEFT,
                    PickupAction.LUNGE_RIGHT,
                    PickupAction.RECOVER,
                    PickupAction.STOP,
                )
            ):
                # 正在执行的连续原语不允许被普通决策打断（只有安全/脚步可抢占）
                return ManagerCommand(
                    active,
                    {
                        "trunk_pitch": self.target_bend_pitch,
                        "crouch_depth": self.crouch_depth,
                        "hip_shift": self.hip_shift,
                        "object_position": object_position,
                        "speed_scale": 1.0,
                    },
                    self.phase,
                    "continue_active_primitive",
                    decision,
                )
            if balance.needs_step or decision.action in (
                PickupAction.STEP_LEFT,
                PickupAction.STEP_RIGHT,
                PickupAction.LUNGE_LEFT,
                PickupAction.LUNGE_RIGHT,
            ):
                # 去抖：连续 3 个决策周期（≈0.15 s）请求脚步才真正规划，避免起弯瞬态误触发
                self._step_request_streak += 1
                if self._step_request_streak < 3:
                    return ManagerCommand(
                        "BEND",
                        {
                            "trunk_pitch": self.target_bend_pitch,
                            "crouch_depth": self.crouch_depth,
                            "hip_shift": self.hip_shift,
                            "object_position": object_position,
                            "speed_scale": 0.7,
                        },
                        self.phase,
                        "step_request_debounce",
                        decision,
                    )
                if self.step_count >= self.max_steps:
                    self._transition(PickupPhase.RECOVERY, "max_steps_reached", decision)
                    return ManagerCommand("RECOVER", {}, self.phase, "max_steps_reached", decision)
                plan = self.planner.plan(
                    balance=balance,
                    left_foot_pos=np.asarray(context["left_foot_pos"]),
                    left_foot_yaw=float(context.get("left_foot_yaw", 0.0)),
                    right_foot_pos=np.asarray(context["right_foot_pos"]),
                    right_foot_yaw=float(context.get("right_foot_yaw", 0.0)),
                    target_position=object_position,
                    use_capture_point=bool(context.get("use_capture_point", True)),
                    use_prediction=bool(context.get("use_prediction", True)),
                    obstacle_xy=context.get("obstacle_xy"),
                    obstacle_radius=float(context.get("obstacle_radius", 0.0)),
                )
                if not plan.feasible:
                    self._transition(PickupPhase.RECOVERY, f"step_infeasible:{plan.reason}", decision)
                    return ManagerCommand("RECOVER", {}, self.phase, plan.reason, decision)
                lunge = decision.action in (PickupAction.LUNGE_LEFT, PickupAction.LUNGE_RIGHT) or abs(plan.dx) > 0.16
                self._transition(PickupPhase.LUNGE if lunge else PickupPhase.FOOT_ADJUSTMENT, "autonomous_step", decision)
                self.step_count += 1
                self._step_in_progress = True
                self._step_request_streak = 0
                return ManagerCommand(
                    "LUNGE" if lunge else "STEP",
                    {"plan": plan},
                    self.phase,
                    "autonomous_step",
                    decision,
                )
            if self._step_in_progress:
                return ManagerCommand(
                    "STEP",
                    {"plan": self.planner.last_plan} if self.planner.last_plan is not None else {},
                    self.phase,
                    "step_in_progress",
                    decision,
                )
            # 弯腰达到目标后判断是否允许伸手（强制门禁）
            if balance.stability_margin >= self.reach_margin_m:
                self._bend_done = True
                hand_error = float(context.get("hand_object_distance", 1.0))
                if hand_error <= float(context.get("reach_threshold_m", 0.12)):
                    self._transition(PickupPhase.GRASPING, "object_in_hand_range", decision)
                    return ManagerCommand("GRASP", {"object_position": object_position, "reach_side": context.get("reach_side", "right")}, self.phase, "object_in_hand_range", decision)
                self._transition(PickupPhase.REACHING, "stable_reach", decision)
                return ManagerCommand("REACH", {"object_position": object_position, "reach_side": context.get("reach_side", "right")}, self.phase, "stable_reach", decision)
            speed = 0.5 if decision.action == PickupAction.BEND_SLOW else 1.0
            self._step_request_streak = 0
            return ManagerCommand(
                "BEND",
                {
                    "trunk_pitch": self.target_bend_pitch,
                    "crouch_depth": self.crouch_depth,
                    "hip_shift": self.hip_shift,
                    "object_position": object_position,
                    "speed_scale": speed,
                },
                self.phase,
                "bend_continue",
                decision,
            )
        if self.phase in (PickupPhase.FOOT_ADJUSTMENT, PickupPhase.LUNGE):
            if step_finished:
                self._transition(PickupPhase.BENDING, "step_complete_reassess", decision)
                return ManagerCommand(
                    "BEND",
                    {
                        "trunk_pitch": self.target_bend_pitch,
                        "crouch_depth": self.crouch_depth,
                        "hip_shift": self.hip_shift,
                        "object_position": object_position,
                    },
                    self.phase,
                    "step_complete_reassess",
                    decision,
                )
            return ManagerCommand(
                "STEP",
                {"plan": self.planner.last_plan} if self.planner.last_plan is not None else {},
                self.phase,
                "step_in_progress",
                decision,
            )
        if self.phase == PickupPhase.REACHING:
            if str(context.get("active_primitive", "")) == "REACH" and not step_finished:
                return ManagerCommand(
                    "REACH",
                    {"object_position": object_position, "reach_side": context.get("reach_side", "right")},
                    self.phase,
                    "reach_in_progress",
                    decision,
                )
            if balance.stability_margin < 0.0:
                self._transition(PickupPhase.RECOVERY, "reach_unstable", decision)
                return ManagerCommand("RECOVER", {}, self.phase, "reach_unstable", decision)
            if bool(context.get("grasp_available", False)):
                self._transition(PickupPhase.GRASPING, "reach_complete", decision)
                return ManagerCommand("GRASP", {"object_position": object_position, "reach_side": context.get("reach_side", "right")}, self.phase, "reach_complete", decision)
            return ManagerCommand("REACH", {"object_position": object_position, "reach_side": context.get("reach_side", "right")}, self.phase, "reach", decision)
        if self.phase == PickupPhase.GRASPING:
            if bool(context.get("grasped", False)):
                self._transition(PickupPhase.LIFTING, "grasp_success", decision)
                return ManagerCommand("LIFT", {"object_position": object_position, "lift_height": float(context.get("lift_height", 0.18))}, self.phase, "grasp_success", decision)
            return ManagerCommand("GRASP", {"object_position": object_position, "reach_side": context.get("reach_side", "right")}, self.phase, "grasping", decision)
        if self.phase == PickupPhase.LIFTING:
            if bool(context.get("lift_complete", False)):
                self._transition(PickupPhase.STANDING_UP, "lift_complete", decision)
                return ManagerCommand("STAND_UP", {"stance_width": float(context.get("stance_width", 0.14))}, self.phase, "lift_complete", decision)
            return ManagerCommand("LIFT", {"object_position": object_position, "lift_height": float(context.get("lift_height", 0.18))}, self.phase, "lifting", decision)
        if self.phase == PickupPhase.STANDING_UP:
            if bool(context.get("stand_complete", False)):
                self._transition(PickupPhase.SUCCESS, "stand_complete", decision)
                return ManagerCommand("STAND", {}, self.phase, "success", decision)
            return ManagerCommand("STAND_UP", {}, self.phase, "standing_up", decision)
        if self.phase == PickupPhase.RECOVERY:
            if bool(context.get("recovered", False)):
                self._transition(PickupPhase.BENDING, "recovered_resume", decision)
                return ManagerCommand(
                    "BEND",
                    {
                        "trunk_pitch": self.target_bend_pitch,
                        "crouch_depth": self.crouch_depth,
                        "hip_shift": self.hip_shift,
                        "object_position": object_position,
                    },
                    self.phase,
                    "recovered_resume",
                    decision,
                )
            return ManagerCommand("RECOVER", {}, self.phase, "recovering", decision)
        return ManagerCommand("STAND", {}, self.phase, "terminal_hold", decision)

    def export_log(self) -> list[dict[str, Any]]:
        """导出状态机日志。"""
        return list(self.history)

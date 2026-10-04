"""PickupBalanceEnv：Gymnasium 风格行为级环境（第 14 节，action = primitive ID）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .controller import PickupConfig, PickupController, PickupStepInfo
from .decision import PickupAction
from .motion_manager import ManagerCommand, PickupPhase
from .reward import PickupRewardConfig, compute_pickup_reward
from .scenarios import PickupScenario, sample_scenario


@dataclass
class PickupEnvConfig:
    """环境配置。"""

    scene_path: str
    primitive_step_s: float = 0.25
    max_episode_s: float = 35.0
    randomize: bool = False
    reward: PickupRewardConfig = field(default_factory=PickupRewardConfig)
    controller: PickupConfig | None = None
    observation_keys: Sequence[str] = (
        "joint_pos",
        "joint_vel",
        "base_quat",
        "base_lin_vel",
        "base_ang_vel",
        "com",
        "com_vel",
        "feet",
        "contacts",
        "target",
        "trunk",
        "margins",
        "phase",
    )


class PickupBalanceEnv:
    """行为级环境：高层选择 primitive，低层由 PickupController 执行。"""

    metadata = {"render_modes": ["video", "viewer", "none"]}

    def __init__(self, config: PickupEnvConfig, *, scenario: PickupScenario | None = None, seed: int = 0) -> None:
        self.config = config
        self.controller = PickupController(config.controller or PickupConfig(scene_path=config.scene_path))
        self.scenario = scenario
        self.seed_value = seed
        self._rng = np.random.default_rng(seed)
        self._previous_info: dict[str, Any] | None = None
        self._action: PickupAction = PickupAction.STAND
        self._override_command: ManagerCommand | None = None
        self._step_info: PickupStepInfo | None = None
        self._episode_reward = 0.0

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, scenario: PickupScenario | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境；返回 (obs, info)。"""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.seed_value = seed
        chosen = scenario or self.scenario or sample_scenario("front", self._rng)
        self.scenario = chosen
        self.controller.reset(chosen, seed=self.seed_value, randomize=self.config.randomize)
        self._previous_info = None
        self._episode_reward = 0.0
        self._action = PickupAction.BEND
        self._override_command = None
        info = self._info_dict(self.controller.step())
        self._step_info = self.controller._last_info
        return self._observation(), info

    def step(self, action: int | str | Mapping[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一个 primitive（内部执行多个控制步）。"""
        primitive = self._parse_action(action)
        self._action = primitive
        # 覆盖 manager：由 env 选择 primitive，controller 执行
        self.controller.manager.update = lambda **kwargs: self._command_for(primitive, kwargs)
        steps = max(1, int(round(self.config.primitive_step_s / self.controller.config.control_dt)))
        total_reward = 0.0
        info: dict[str, Any] = {}
        for _ in range(steps):
            step_info = self.controller.step()
            info = self._info_dict(step_info)
            reward, components = compute_pickup_reward(
                config=self.config.reward,
                info=info,
                previous=self._previous_info,
                action_primitive=primitive.value,
                dt=self.controller.config.control_dt,
            )
            total_reward += reward
            info["reward_components"] = components
            self._previous_info = info
            self._step_info = step_info
            if info["success"] or info["fall"]:
                break
        self._episode_reward += total_reward
        info["episode_reward"] = self._episode_reward
        terminated = bool(info.get("success", False) or info.get("fall", False))
        truncated = self.controller.time >= self.config.max_episode_s
        return self._observation(), float(total_reward), terminated, truncated, info

    def _command_for(self, primitive: PickupAction, kwargs: Mapping[str, Any]) -> ManagerCommand:
        """把 env 的 primitive ID 翻译成 ManagerCommand（保持状态机门禁）。"""
        phase_map = {
            PickupAction.STAND: PickupPhase.BENDING,
            PickupAction.BEND: PickupPhase.BENDING,
            PickupAction.BEND_SLOW: PickupPhase.BALANCE_WARNING,
            PickupAction.STEP_LEFT: PickupPhase.FOOT_ADJUSTMENT,
            PickupAction.STEP_RIGHT: PickupPhase.FOOT_ADJUSTMENT,
            PickupAction.LUNGE_LEFT: PickupPhase.LUNGE,
            PickupAction.LUNGE_RIGHT: PickupPhase.LUNGE,
            PickupAction.REACH: PickupPhase.REACHING,
            PickupAction.GRASP: PickupPhase.GRASPING,
            PickupAction.LIFT: PickupPhase.LIFTING,
            PickupAction.STAND_UP: PickupPhase.STANDING_UP,
            PickupAction.RECOVER: PickupPhase.RECOVERY,
            PickupAction.STOP: PickupPhase.RECOVERY,
        }
        context = dict(kwargs.get("context", {}))
        balance = kwargs.get("balance")
        goal = dict(kwargs.get("goal", {}))
        object_position = np.asarray(goal.get("object_position", np.zeros(3)), dtype=np.float64)
        primitive_name = {
            PickupAction.STAND: "STAND",
            PickupAction.BEND: "BEND",
            PickupAction.BEND_SLOW: "BEND",
            PickupAction.STEP_LEFT: "STEP",
            PickupAction.STEP_RIGHT: "STEP",
            PickupAction.LUNGE_LEFT: "LUNGE",
            PickupAction.LUNGE_RIGHT: "LUNGE",
            PickupAction.REACH: "REACH",
            PickupAction.GRASP: "GRASP",
            PickupAction.LIFT: "LIFT",
            PickupAction.STAND_UP: "STAND_UP",
            PickupAction.RECOVER: "RECOVER",
            PickupAction.STOP: "RECOVER",
        }[primitive]
        params: dict[str, Any] = {"object_position": object_position}
        if primitive in (PickupAction.BEND, PickupAction.BEND_SLOW):
            params.update(
                {
                    "trunk_pitch": self.controller.config.trunk_pitch_target,
                    "crouch_depth": 0.17,
                    "hip_shift": 0.05,
                    "speed_scale": 0.5 if primitive == PickupAction.BEND_SLOW else 1.0,
                }
            )
        if primitive in (PickupAction.STEP_LEFT, PickupAction.STEP_RIGHT, PickupAction.LUNGE_LEFT, PickupAction.LUNGE_RIGHT):
            plan = self.controller.planner.plan(
                balance=balance,
                left_foot_pos=np.asarray(context["left_foot_pos"]),
                left_foot_yaw=float(context.get("left_foot_yaw", 0.0)),
                right_foot_pos=np.asarray(context["right_foot_pos"]),
                right_foot_yaw=float(context.get("right_foot_yaw", 0.0)),
                target_position=object_position,
                target_bias_m=0.18 if "LUNGE" in primitive.value else 0.0,
            )
            params["plan"] = plan
        return ManagerCommand(primitive_name, params, phase_map[primitive], f"env_action:{primitive.value}", None)

    def _parse_action(self, action: int | str | Mapping[str, Any]) -> PickupAction:
        """解析 primitive ID/名称/字典动作。"""
        if isinstance(action, Mapping):
            return PickupAction(str(action["primitive"]).upper())
        if isinstance(action, (int, np.integer)):
            return tuple(PickupAction)[int(action) % len(PickupAction)]
        return PickupAction(str(action).upper())

    def _observation(self) -> np.ndarray:
        """构造行为级观测向量。"""
        ctrl = self.controller
        data = ctrl.data
        info = ctrl._last_info
        if info is None:
            info = ctrl.step()
            ctrl._last_info = info
        hand = np.asarray(info.hand_position, dtype=np.float64)
        object_grasp = np.asarray(info.object_position, dtype=np.float64)
        q = np.array(data.qpos[ctrl.qpos_addr], dtype=np.float64)
        dq = np.array(data.qvel[ctrl.qvel_addr], dtype=np.float64)
        base_quat = np.array(data.qpos[3:7], dtype=np.float64)
        base_lin = np.array(data.qvel[0:3], dtype=np.float64)
        base_ang = np.array(data.qvel[3:6], dtype=np.float64)
        com = ctrl.monitor._previous_com if ctrl.monitor._previous_com is not None else ctrl._load_com()
        com_vel = ctrl.monitor._previous_com_velocity if ctrl.monitor._previous_com_velocity is not None else np.zeros(3)
        left_pos, left_rot = ctrl._foot_pose("left")
        right_pos, right_rot = ctrl._foot_pose("right")
        feet = np.concatenate(
            [
                left_pos,
                left_rot.reshape(-1)[:3],
                right_pos,
                right_rot.reshape(-1)[:3],
            ]
        )
        contacts = np.array([1.0 if c else 0.0 for c in ctrl._contacts()])
        target = np.concatenate([object_grasp, object_grasp - hand])
        trunk = np.array([info.trunk_pitch, info.balance.trunk_roll])
        margins = np.array([info.stability_margin, info.balance.predicted_margin])
        phase = np.zeros(len(PickupPhase))
        from contextlib import suppress

        with suppress(ValueError):
            phase[list(PickupPhase).index(ctrl.manager.phase)] = 1.0
        return np.concatenate(
            [q, dq, base_quat, base_lin, base_ang, np.asarray(com), np.asarray(com_vel), feet, contacts, target, trunk, margins, phase]
        ).astype(np.float32)

    @property
    def observation_dim(self) -> int:
        """观测维度。"""
        return int(self._observation().size)

    @property
    def action_space_size(self) -> int:
        """离散 primitive 数量。"""
        return len(PickupAction)

    def _info_dict(self, step_info: PickupStepInfo) -> dict[str, Any]:
        """把 controller 输出转成 env info（含奖励所需字段）。"""
        grasp_point = np.asarray(step_info.object_position, dtype=np.float64)
        hand = np.asarray(step_info.hand_position, dtype=np.float64)
        return {
            "time": step_info.time,
            "phase": step_info.phase,
            "primitive": step_info.primitive,
            "success": step_info.success,
            "fall": step_info.fall,
            "grasped": step_info.grasped,
            "stability_margin": step_info.stability_margin,
            "predicted_margin": step_info.balance.predicted_margin,
            "trunk_pitch": step_info.trunk_pitch,
            "hand_distance": float(np.linalg.norm(hand - grasp_point)),
            "object_height": float(step_info.object_position[2] + 0.31),
            "initial_object_height": float(self.controller._initial_grasp_z),
            "self_contacts": int(self._count_self_contacts()),
            "energy_delta": float(step_info.energy - (self._previous_info or {}).get("energy", 0.0)),
            "energy": float(step_info.energy),
            "steps": int(step_info.steps),
            "decision_latency_ms": float(step_info.decision_latency_ms),
            "com": step_info.com.tolist(),
            "zmp": step_info.zmp.tolist(),
            "support_polygon": step_info.support_polygon.tolist(),
            "left_foot": step_info.left_foot_pos.tolist(),
            "right_foot": step_info.right_foot_pos.tolist(),
            "hand_position": step_info.hand_position.tolist(),
            "object_position": step_info.object_position.tolist(),
        }

    def _count_self_contacts(self) -> int:
        """机器人体段自接触数（碰撞惩罚用）。"""
        model, data = self.controller.model, self.controller.data
        count = 0
        for i in range(data.ncon):
            b1 = int(model.geom_bodyid[data.contact[i].geom1])
            b2 = int(model.geom_bodyid[data.contact[i].geom2])
            n1 = self.controller.mujoco.mj_id2name(model, self.controller.mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            n2 = self.controller.mujoco.mj_id2name(model, self.controller.mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            if "pickup_object" in (n1, n2):
                continue  # 手-物体接触是抓取本身，不计碰撞
            if b1 > 0 and b2 > 0 and not (
                "ankle" in n1 and "ankle" in n2
            ):
                count += 1
        return count

    def close(self) -> None:
        """关闭控制器。"""
        self.controller.close()

"""G1 平衡训练环境：教师策略 + 技能叠加 + 学习残差（MuJoCo，第 6 节）。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cb_common.errors import PolicyError
from cb_common.joints import JOINT_GROUPS, NUM_JOINTS, POLICY_INDEX, mujoco_to_isaac
from cb_common.types import JumpPhase, RobotState, SkillCommand, SkillContext
from cb_features.balance_feats import balance_features
from cb_features.command import encode_skill
from cb_features.obs_spec import ObsSpec, default_obs_spec
from cb_features.phase_clock import PhaseClock
from cb_features.proprio import build_proprio
from cb_policy.teacher import TeacherCore
from cb_skills.jump import JumpSkill
from cb_skills.locomotion import LocomotionSkill
from cb_skills.recover import RecoverSkill
from cb_skills.stand import StandSkill
from cb_skills.turn import TurnSkill
from cb_skills.wave import WaveSkill

from .dr_profiles import DomainRandomizer
from .terminations import check_termination

KEY_GROUPS = ("ang_vel", "gravity", "lin_vel", "command", "joint_pos", "joint_vel", "last_action")


@dataclass
class ScenarioScript:
    """场景脚本：技能序列与参数。"""

    name: str
    segments: list[dict[str, Any]]

    @classmethod
    def from_config(cls, name: str, spec: Mapping[str, Any]) -> ScenarioScript:
        """从 train 配置构造。"""
        return cls(name=name, segments=[dict(item) for item in spec.get("segments", [])])

    def segment_at(self, t: float) -> tuple[dict[str, Any], float]:
        """返回 (当前段, 段内时间)。"""
        cursor = 0.0
        for item in self.segments:
            duration = float(item.get("duration_s", 1.0))
            if t < cursor + duration:
                return item, t - cursor
            cursor += duration
        return self.segments[-1], t - cursor + float(self.segments[-1].get("duration_s", 1.0))

    @property
    def duration_s(self) -> float:
        """脚本总时长。"""
        return float(sum(float(item.get("duration_s", 1.0)) for item in self.segments))


@dataclass
class EnvConfig:
    """训练环境配置。"""

    scene_path: str
    control_dt: float = 0.02
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 12.0
    history_length: int = 4
    residual_scale: float = 0.25
    action_clip: float = 1.0
    skill_embedding_dim: int = 16
    reward: Mapping[str, Any] = field(default_factory=dict)
    termination: Mapping[str, Any] = field(default_factory=dict)
    dr: Mapping[str, Any] = field(default_factory=dict)
    obs_spec: ObsSpec = field(default_factory=default_obs_spec)
    policy_mode: str = "residual"      # residual（教师+残差）| absolute（学生直接输出增量）


def _support_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """点到凸多边形最小有符号距离（内部为正）。"""
    if polygon.shape[0] < 3:
        return 0.0
    inside = True
    best = float("inf")
    for i in range(polygon.shape[0]):
        a, b = polygon[i], polygon[(i + 1) % polygon.shape[0]]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        dist = float(np.dot(point - a, normal))
        if dist < 0.0:
            inside = False
        best = min(best, abs(dist))
    return best if inside else -best


class G1BalanceEnv:
    """单环境（CPU MuJoCo 步进 + 教师 CPU ONNX + 学习残差）。"""

    def __init__(
        self,
        teacher_core: TeacherCore,
        config: EnvConfig,
        scenario: ScenarioScript,
        *,
        seed: int = 0,
        skill_specs: Mapping[str, Any] | None = None,
    ) -> None:
        self.core = teacher_core
        self.config = config
        self.scenario = scenario
        self.rt = teacher_core.new_runtime()
        self.obs_spec = config.obs_spec
        self.key_dim = 3 + 3 + 3 + 5 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS
        self.hist = deque(maxlen=config.history_length)
        self.phase = PhaseClock()
        self.rng = np.random.default_rng(seed)
        self.dr = DomainRandomizer(config.dr, seed=seed)
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_action = np.zeros(NUM_JOINTS)
        self.prev_action = np.zeros(NUM_JOINTS)
        self.last_target = np.array(self.core.default_qpos_policy)
        self.action_delay_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(config.dr.get("actuator_delay_steps", [0, 0])[1]) + 1))
        self.torque = np.zeros(NUM_JOINTS)
        self.metrics: dict[str, float] = {}
        limits = _load_limits(config.scene_path)
        self.limits_lo, self.limits_hi = limits
        self._skill_specs = dict(skill_specs or {})
        self._active_skill: Any = None
        self._active_name: str = ""
        self._overlay_skill: Any = None
        self._overlay_name: str = ""
        self._current_segment: dict[str, Any] = {}
        self._segment_time = 0.0
        self._last_command3 = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, randomize: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境；返回 (obs, info)。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.dr = DomainRandomizer(self.config.dr, seed=seed)
        self.rt.reset()
        info: dict[str, Any] = {}
        if randomize and self.config.dr:
            info = self.dr.sample(episode_steps=int(self.config.episode_length_s / self.config.control_dt), dt=self.config.control_dt)
            self.dr.apply_model(
                self.rt.model,
                base_mass=self.rt.base_body_mass,
                base_friction=self.rt.base_geom_friction,
            )
        self.step_count = 0
        self.episode_reward = 0.0
        self.hist.clear()
        self.action_delay_buffer.clear()
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        self.last_target = np.array(self.core.default_qpos_policy)
        self.phase = PhaseClock()
        self.torque[:] = 0.0
        self.metrics = {
            "roll_peak": 0.0,
            "pitch_peak": 0.0,
            "support_margin_min": 1.0,
            "height_min": 1.0,
            "fall": 0.0,
            "jump_apex": 0.0,
            "foot_clearance_max": 0.0,
            "wave_error": 0.0,
            "turn_error": 0.0,
            "distance": 0.0,
            "speed_error": 0.0,
            "speed_error_sum": 0.0,
            "speed_error_count": 0.0,
            "speed_ratio_sum": 0.0,
            "speed_ratio_count": 0.0,
            "speed_retention": 0.0,
            "support_violation_frames": 0.0,
            "support_violation_max_s": 0.0,
            "support_contact_frames": 0.0,
            "support_violation_ratio": 0.0,
            "turn_yaw_start": float("nan"),
            "yaw_unwrapped": 0.0,
        }
        self._last_push_step = -10_000
        self._last_yaw_meas: float | None = None
        self.tilt_series: list[tuple[float, float, float]] = []
        self._support_violation_run = 0.0
        self._activate_skill(self.scenario.segments[0]["skill"], params=dict(self.scenario.segments[0]))
        self._last_command3 = self._command3_from_segment(self.scenario.segments[0])
        self._refresh_teacher(np.zeros(3))
        obs = self._observation()
        for _ in range(self.config.history_length):
            self.hist.append(self._key_obs())
        return obs, info

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步控制（含技能叠加与残差）。"""
        raw_action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -self.config.action_clip, self.config.action_clip)
        self.action_delay_buffer.append(raw_action.copy())
        delayed = self.action_delay_buffer[0] if len(self.action_delay_buffer) > 1 else raw_action
        t = self.step_count * self.config.control_dt
        segment, segment_time = self.scenario.segment_at(t)
        if segment.get("skill") != self._active_name:
            self._activate_skill(str(segment["skill"]), params=dict(segment))
        self._segment_time = segment_time
        state = self._robot_state()
        skill_out = self._active_skill.update(state, self.config.control_dt)
        if self._overlay_skill is not None:
            from cb_skills.composer import compose_parallel

            overlay_out = self._overlay_skill.update(state, self.config.control_dt)
            skill_out = compose_parallel(skill_out, [overlay_out], weights=[1.0])
        command3 = self._command_from_skill(skill_out, segment)
        self._last_command3 = command3
        teacher_target = self._refresh_teacher(command3)
        if self.config.policy_mode == "absolute":
            target = np.array(self.core.default_qpos_policy) + delayed
        else:
            target = teacher_target + skill_out.delta_q + self.config.residual_scale * delayed
        target = np.clip(target, self.limits_lo + 0.02, self.limits_hi - 0.02)
        self.last_target = target
        self.prev_action = self.last_action.copy()
        if self.config.policy_mode == "absolute":
            # 绝对模式：观测中的 last_action 与部署一致（实际下发的目标增量）
            self.last_action = (target - np.array(self.core.default_qpos_policy)).copy()
        else:
            self.last_action = delayed.copy()
        push = self.dr.push_at(self.step_count)
        if push is not None:
            self._last_push_step = self.step_count
        for _ in range(self.config.decimation):
            if push is not None:
                self.rt.data.xfrc_applied[1, 0:3] = push
            else:
                self.rt.data.xfrc_applied[1, 0:3] = 0.0
            self.rt.apply_pd(target, kp_scale=self.dr.kp_scale, kd_scale=self.dr.kd_scale)
            self.rt.mujoco.mj_step(self.rt.model, self.rt.data)
        self.step_count += 1
        self.torque = mujoco_to_isaac(
            np.asarray(self.rt.data.actuator_force[:NUM_JOINTS], dtype=np.float64)
        )
        state = self._robot_state()
        self._update_metrics(state, skill_out, command3)
        reward, components = self._reward(state, skill_out, command3, target)
        terminated, reason = check_termination(
            base_height=float(state.base_pos[2]),
            base_quat=state.base_quat,
            joint_pos=state.joint_pos,
            joint_vel=state.joint_vel,
            config=self.config.termination,
        )
        truncated = self.step_count >= int(self.config.episode_length_s / self.config.control_dt)
        if terminated:
            self.metrics["fall"] = 1.0
        self.episode_reward += reward
        obs = self._observation()
        self.hist.append(self._key_obs())
        info = {
            "skill": self._active_name,
            "skill_phase": skill_out.phase,
            "jump_phase": skill_out.jump_phase.value,
            "reward_components": components,
            "termination_reason": reason,
            "episode_reward": self.episode_reward,
            "metrics": dict(self.metrics),
            "command": command3.tolist(),
            "target": target.tolist(),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------------------------------------------------- internals
    def _activate_skill(self, name: str, *, params: dict[str, Any]) -> None:
        """切换当前技能并调用 enter。"""
        if self._active_skill is not None:
            self._active_skill.exit()
        spec = self._skill_specs.get(name)
        if spec is None:
            raise PolicyError("skill spec missing", skill=name)
        if name in ("stand",):
            skill: Any = StandSkill(spec)
        elif name in ("walk", "run"):
            skill = LocomotionSkill(spec, skill_name=name)
        elif name == "jump":
            skill = JumpSkill(spec)
        elif name == "wave":
            skill = WaveSkill(spec)
        elif name == "turn":
            skill = TurnSkill(spec)
        elif name == "recover":
            skill = RecoverSkill(spec)
        else:
            raise PolicyError("unsupported skill in environment", skill=name)
        command = SkillCommand(skill=name, params=params, reason_code="script", source="scenario")
        ctx = SkillContext(command=command, default_pose=np.array(self.core.default_qpos_policy), dt=self.config.control_dt)
        skill.enter(ctx)
        self._active_skill = skill
        self._active_name = name
        self._current_segment = params
        if self._overlay_skill is not None:
            self._overlay_skill.exit()
        self._overlay_skill = None
        self._overlay_name = ""
        wave_params = params.get("wave")
        if isinstance(wave_params, Mapping):
            wave_spec = self._skill_specs.get("wave")
            if wave_spec is not None:
                overlay = WaveSkill(wave_spec)
                merge = {
                    "skill": "wave",
                    "side": wave_params.get("side", "right"),
                    "cycles": wave_params.get("cycles", 2.0),
                    "frequency_hz": wave_params.get("frequency_hz", 1.2),
                }
                overlay.enter(
                    SkillContext(
                        command=SkillCommand(skill="wave", params=merge, source="overlay"),
                        default_pose=np.array(self.core.default_qpos_policy),
                        dt=self.config.control_dt,
                    )
                )
                self._overlay_skill = overlay
                self._overlay_name = "wave"

    def _command_from_skill(self, skill_out: Any, segment: Mapping[str, Any]) -> np.ndarray:
        """优先使用技能输出的指令（转弯 yaw_rate 等），否则回退到场景脚本。"""
        command = skill_out.metadata.get("command") if skill_out.metadata else None
        if isinstance(command, Mapping) and "vx" in command:
            return np.array(
                [float(command.get("vx", 0.0)), float(command.get("vy", 0.0)), float(command.get("yaw_rate", 0.0))],
                dtype=np.float64,
            )
        return self._command3_from_segment(segment)

    def _command3_from_segment(self, segment: Mapping[str, Any]) -> np.ndarray:
        """提取教师速度指令。"""
        vx = float(segment.get("vx", 0.0))
        vy = float(segment.get("vy", 0.0))
        yaw_rate = float(segment.get("yaw_rate", 0.0))
        if segment.get("skill") == "jump" and "vx" not in segment:
            vx = float(self._current_segment.get("vx", 1.0))
        return np.array([vx, vy, yaw_rate], dtype=np.float64)

    def _refresh_teacher(self, command3: np.ndarray) -> np.ndarray:
        """运行教师 ONNX 并返回策略序目标。"""
        _action, target = self.rt.act(command3)
        return target

    def _robot_state(self) -> RobotState:
        """从 MjData 组装统一状态。"""
        data = self.rt.data
        com = np.array(data.subtree_com[0], dtype=np.float64)
        foot_positions = []
        foot_corners: list[np.ndarray] = []
        contacts = [False, False]
        for body_id in range(self.rt.model.nbody):
            name = self.rt.mujoco.mj_id2name(self.rt.model, self.rt.mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.endswith("ankle_roll_link"):
                foot_positions.append(np.array(data.xpos[body_id], dtype=np.float64))
                rot = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                for forward, lateral in ((0.09, 0.04), (0.09, -0.04), (-0.05, 0.04), (-0.05, -0.04)):
                    offset = rot @ np.array([forward, lateral, 0.0])
                    foot_corners.append(np.array(data.xpos[body_id], dtype=np.float64) + offset)
        for i in range(len(foot_positions)):
            geom_z = float(foot_positions[i][2])
            contacts[i] = geom_z < 0.10
        polygon = np.array([p[[0, 1]] for p in foot_corners]) if foot_corners else np.zeros((0, 2))
        margin = _support_margin(com[:2], _convex_hull(polygon)) if polygon.shape[0] >= 3 else 0.0
        joint_pos = self.rt.joint_pos_policy
        joint_vel = self.rt.joint_vel_policy
        return RobotState(
            timestamp=float(data.time),
            base_pos=np.array(data.qpos[0:3], dtype=np.float64),
            base_quat=np.array(data.qpos[3:7], dtype=np.float64),
            base_lin_vel=np.array(data.qvel[0:3], dtype=np.float64),
            base_ang_vel=np.array(data.qvel[3:6], dtype=np.float64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torque=self.torque.copy(),
            contact=(bool(contacts[0]), bool(contacts[1])),
            com=com,
            com_vel=np.zeros(3),
            cp=np.array([com[0], com[1]]),
            support_margin=float(margin),
            frame_id=self.step_count,
        )

    def _obs_frames(self, state: RobotState, command5: np.ndarray, skill: str, jump_phase: JumpPhase) -> dict[str, np.ndarray]:
        """构造单帧观测组。"""
        proprio = build_proprio(
            state,
            last_action=self.last_action,
            default_pose=np.array(self.core.default_qpos_policy),
        )
        contact = np.array([1.0 if c else 0.0 for c in state.contact])
        phase = self.phase.update(self.config.control_dt, state.contact)
        skill_embed = encode_skill(skill, jump_phase, self.config.skill_embedding_dim)
        balance = balance_features(state)
        return {
            "ang_vel": proprio["ang_vel"],
            "gravity": proprio["gravity"],
            "lin_vel": proprio["lin_vel"],
            "command": command5,
            "joint_pos": proprio["joint_pos"],
            "joint_vel": proprio["joint_vel"],
            "last_action": proprio["last_action"],
            "contact": contact,
            "phase_clock": phase,
            "skill_embedding": skill_embed,
            "balance_feats": balance,
        }

    def _observation(self) -> np.ndarray:
        """组装最终观测（历史组按 spec 堆叠）。"""
        state = self._robot_state()
        command3 = self._last_command3
        jump_flag = 1.0 if self._active_name == "jump" else 0.0
        wave_skill = self._overlay_skill if self._overlay_name == "wave" else (self._active_skill if self._active_name == "wave" else None)
        wave_side = 0.0 if wave_skill is None else (1.0 if getattr(wave_skill, "side", "right") == "right" else -1.0)
        command5 = np.array([command3[0], command3[1], command3[2], jump_flag, wave_side])
        frames = self._obs_frames(state, command5, self._active_name, self._active_skill.jump_phase if hasattr(self._active_skill, "jump_phase") else JumpPhase.NONE)
        history = self._history_arrays(frames)
        obs = self.obs_spec.assemble(frames, history)
        noise = self.dr.obs_noise_std
        if noise > 0.0:
            obs = obs + self.rng.normal(0.0, noise, size=obs.shape)
        return obs

    def _key_obs(self) -> np.ndarray:
        """构造历史关键组（与 obs_spec.history_groups 顺序一致）。"""
        state = self._robot_state()
        command3 = self._last_command3
        jump_flag = 1.0 if self._active_name == "jump" else 0.0
        wave_skill = self._overlay_skill if self._overlay_name == "wave" else (self._active_skill if self._active_name == "wave" else None)
        wave_side = 0.0 if wave_skill is None else (1.0 if getattr(wave_skill, "side", "right") == "right" else -1.0)
        command5 = np.array([command3[0], command3[1], command3[2], jump_flag, wave_side])
        frames = self._obs_frames(state, command5, self._active_name, getattr(self._active_skill, "jump_phase", JumpPhase.NONE))
        return np.concatenate(
            [
                frames["ang_vel"],
                frames["gravity"],
                frames["lin_vel"],
                frames["command"],
                frames["joint_pos"],
                frames["joint_vel"],
                frames["last_action"],
            ]
        )

    def _history_arrays(self, frames: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """把 deque 历史转为 spec 需要的 (history, dim) 数组。"""
        if len(self.hist) == 0:
            key = np.concatenate(
                [
                    frames["ang_vel"],
                    frames["gravity"],
                    frames["lin_vel"],
                    frames["command"],
                    frames["joint_pos"],
                    frames["joint_vel"],
                    frames["last_action"],
                ]
            )
            self.hist.append(key)
        buffer = list(self.hist)
        while len(buffer) < self.config.history_length:
            buffer.insert(0, buffer[0])
        arr = np.stack(buffer[-self.config.history_length :])
        return {
            "ang_vel": arr[:, 0:3],
            "gravity": arr[:, 3:6],
            "lin_vel": arr[:, 6:9],
            "command": arr[:, 9:14],
            "joint_pos": arr[:, 14 : 14 + NUM_JOINTS],
            "joint_vel": arr[:, 14 + NUM_JOINTS : 14 + 2 * NUM_JOINTS],
            "last_action": arr[:, 14 + 2 * NUM_JOINTS : 14 + 3 * NUM_JOINTS],
        }

    def _reward(self, state: RobotState, skill_out: Any, command3: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
        """计算奖励。"""
        from .rewards_adapter import compute_reward_adapter

        return compute_reward_adapter(
            config=self.config.reward,
            state=state,
            command=command3,
            action=self.last_action,
            last_action=self.prev_action,
            torque=self.torque,
            expected_contacts=skill_out.expected_contact,
            target=target,
            teacher_target=self.rt.last_target_policy,
        )

    def _update_metrics(self, state: RobotState, skill_out: Any, command3: np.ndarray) -> None:
        """累计场景指标。"""
        roll, pitch, yaw = state.rpy_deg()
        self.tilt_series.append((float(self.step_count) * self.config.control_dt, float(roll), float(pitch)))
        self.metrics["roll_peak"] = max(self.metrics["roll_peak"], abs(roll))
        self.metrics["pitch_peak"] = max(self.metrics["pitch_peak"], abs(pitch))
        if any(state.contact):
            self.metrics["support_margin_min"] = min(self.metrics["support_margin_min"], float(state.support_margin))
            if state.support_margin < -0.02:
                self.metrics["support_violation_frames"] += 1.0
                self._support_violation_run += self.config.control_dt
                self.metrics["support_violation_max_s"] = max(
                    self.metrics["support_violation_max_s"], self._support_violation_run
                )
            else:
                self._support_violation_run = 0.0
            self.metrics["support_contact_frames"] += 1.0
            self.metrics["support_violation_ratio"] = self.metrics["support_violation_frames"] / max(
                1.0, self.metrics["support_contact_frames"]
            )
        self.metrics["height_min"] = min(self.metrics["height_min"], float(state.base_pos[2]))
        if self._active_name == "jump":
            self.metrics["jump_apex"] = max(self.metrics["jump_apex"], float(state.base_pos[2]))
            self.metrics["foot_clearance_max"] = max(self.metrics["foot_clearance_max"], self._foot_clearance())
        if self._active_name == "wave" or self._overlay_name == "wave":
            arm_indices = [POLICY_INDEX[name] for name in JOINT_GROUPS["arms"]]
            reference = np.array(self.core.default_qpos_policy) + skill_out.delta_q
            self.metrics["wave_error"] = float(np.mean(np.abs(state.joint_pos[arm_indices] - reference[arm_indices])))
        if self._active_name == "turn":
            if self._last_yaw_meas is None:
                self.metrics["yaw_unwrapped"] = 0.0
            else:
                delta = np.deg2rad(((yaw - self._last_yaw_meas + 180.0) % 360.0) - 180.0)
                self.metrics["yaw_unwrapped"] += float(delta)
            self._last_yaw_meas = float(yaw)
            target_deg = float(self._current_segment.get("target_yaw_deg", 0.0))
            start = self.metrics["turn_yaw_start"]
            if not np.isfinite(start):
                self.metrics["turn_yaw_start"] = float(self.metrics["yaw_unwrapped"])
                start = float(self.metrics["yaw_unwrapped"])
            self.metrics["turn_error"] = float(self.metrics["yaw_unwrapped"] - start - np.deg2rad(target_deg))
        else:
            self._last_yaw_meas = float(yaw)
        if self.step_count - self._last_push_step > 10 and self.step_count * self.config.control_dt > 2.0:
            self.metrics["speed_error_sum"] += float(abs(state.base_lin_vel[0] - command3[0]))
            self.metrics["speed_error_count"] += 1.0
            self.metrics["speed_error"] = self.metrics["speed_error_sum"] / max(1.0, self.metrics["speed_error_count"])
            if abs(command3[0]) > 0.2:
                self.metrics["speed_ratio_sum"] += float(state.base_lin_vel[0] / command3[0])
                self.metrics["speed_ratio_count"] += 1.0
                self.metrics["speed_retention"] = self.metrics["speed_ratio_sum"] / max(1.0, self.metrics["speed_ratio_count"])

    def _foot_clearance(self) -> float:
        """足端最低点高度（腾空指标）。"""
        heights = []
        for body_id in range(self.rt.model.nbody):
            name = self.rt.mujoco.mj_id2name(self.rt.model, self.rt.mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.endswith("ankle_roll_link"):
                heights.append(float(self.rt.data.xpos[body_id][2]))
        return min(heights) if heights else 0.0


def _load_limits(scene_path: str) -> tuple[np.ndarray, np.ndarray]:
    """从 MJCF 读取策略序关节限位（缓存到模块级）。"""
    from cb_safety import load_joint_limits_mjcf

    lo_mj, hi_mj = load_joint_limits_mjcf(scene_path, policy_order=False)
    return mujoco_to_isaac(lo_mj), mujoco_to_isaac(hi_mj)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """二维凸包（Andrew monotone chain）。"""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).round(9))))
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o: Any, a: Any, b: Any) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])

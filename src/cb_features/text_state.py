"""机器人数值状态 → 稳定 JSON 文本（LayA 输入，第 5.6 节）。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cb_common.types import RobotState, SafetyLevel, SkillCommand, TaskContext

TEXT_STATE_TEMPLATE_VERSION = "1.0"


def build_text_state(
    state: RobotState,
    command: SkillCommand | None = None,
    context: TaskContext | None = None,
    *,
    robot: str = "unitree_g1_29dof",
    contact_hold_s: float = 0.0,
    stability: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """构造紧凑、字段顺序稳定、可解析的 JSON 字符串（数值固定精度）。"""
    ctx = context or TaskContext()
    cmd = command or SkillCommand(skill="stand")
    roll, pitch, _ = state.rpy_deg()
    margin = float(state.support_margin)
    stability = stability or _stability_from_margin(margin, roll, pitch)
    contacts = [name for name, active in zip(("left", "right"), state.contact, strict=False) if active]
    payload: dict[str, Any] = {
        "state_version": TEXT_STATE_TEMPLATE_VERSION,
        "robot": robot,
        "task": ctx.task,
        "balance": {
            "roll_deg": round(float(roll), 2),
            "pitch_deg": round(float(pitch), 2),
            "com_margin_m": round(margin, 3),
            "cp_margin_m": round(_cp_margin(state), 3),
            "contact": contacts,
            "contact_hold_s": round(float(contact_hold_s), 2),
            "stability": stability,
        },
        "motion": {
            "vx": round(float(state.base_lin_vel[0]), 3),
            "vy": round(float(state.base_lin_vel[1]), 3),
            "yaw_rate": round(float(state.base_ang_vel[2]), 3),
            "skill": cmd.skill,
            "phase": round(float(cmd.params.get("phase", 0.0)), 3),
            "time_in_skill_s": round(float(ctx.extra.get("time_in_skill_s", 0.0)), 2),
        },
        "context": {
            "terrain": ctx.terrain,
            "external_push": bool(ctx.external_push),
            "latency_ms": round(float(ctx.latency_ms), 1),
            "battery_ok": bool(ctx.battery_ok),
        },
        "history": {
            "falls_last_60s": int(ctx.falls_last_60s),
            "watchdog_trips_last_60s": int(ctx.watchdog_trips_last_60s),
        },
        "candidates": list(ctx.candidates),
    }
    if extra:
        payload["extra"] = dict(extra)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_text_state(text: str) -> dict[str, Any]:
    """解析文本化状态（用于回放一致性核对）。"""
    data = json.loads(text)
    if data.get("state_version") != TEXT_STATE_TEMPLATE_VERSION:
        raise ValueError(f"unsupported state template version: {data.get('state_version')!r}")
    return data


def candidates_from_skills(skills: Sequence[str], *, include_hold: bool = True) -> list[str]:
    """由技能名构造候选集合（调度器注入，模型不得自由生成）。"""
    out = ["continue_current"] if include_hold else []
    out.extend(skills)
    return out


def risk_score(
    state: RobotState,
    *,
    weights: Mapping[str, float] | None = None,
    model_uncertainty: float = 0.0,
) -> float:
    """可解释风险分 [0,1]（Tier 0 规则与决策层共用）。"""
    w = {"roll": 0.30, "pitch": 0.25, "com_margin": 0.25, "contact_loss": 0.10, "model_uncertainty": 0.10}
    if weights:
        w.update(weights)
    roll, pitch, _ = state.rpy_deg()
    roll_term = min(abs(roll) / 15.0, 1.0)
    pitch_term = min(abs(pitch) / 15.0, 1.0)
    margin = float(state.support_margin)
    margin_term = min(max((0.05 - margin) / 0.10, 0.0), 1.0)
    contact_term = 0.0 if all(state.contact) else (0.5 if any(state.contact) else 1.0)
    value = (
        w["roll"] * roll_term
        + w["pitch"] * pitch_term
        + w["com_margin"] * margin_term
        + w["contact_loss"] * contact_term
        + w["model_uncertainty"] * float(np.clip(model_uncertainty, 0.0, 1.0))
    )
    return float(min(max(value, 0.0), 1.0))


def safety_level_from_state(state: RobotState, *, abort_roll_deg: float = 8.0, abort_pitch_deg: float = 8.0) -> SafetyLevel:
    """Tier 0 规则：由状态直接给出安全等级（fail-closed）。"""
    roll, pitch, _ = state.rpy_deg()
    if not np.isfinite(state.joint_pos).all() or abs(roll) >= 15.0 or abs(pitch) >= 15.0:
        return SafetyLevel.EMERGENCY
    if abs(roll) >= abort_roll_deg or abs(pitch) >= abort_pitch_deg or state.support_margin < -0.10:
        return SafetyLevel.ABORT
    if abs(roll) >= 5.0 or abs(pitch) >= 5.0 or state.support_margin < 0.02:
        return SafetyLevel.WARN
    return SafetyLevel.OK


def _stability_from_margin(margin: float, roll: float, pitch: float) -> str:
    if margin >= 0.04 and abs(roll) < 3.0 and abs(pitch) < 3.0:
        return "stable"
    if margin >= -0.02 and abs(roll) < 8.0 and abs(pitch) < 8.0:
        return "marginal"
    return "unstable"


def _cp_margin(state: RobotState) -> float:
    if state.cp is None:
        return 0.0
    cp = np.asarray(state.cp, dtype=np.float64)
    com = np.asarray(state.com[:2] if state.com is not None else np.zeros(2), dtype=np.float64)
    extra = max(0.0, float(np.linalg.norm(cp - com)) - 0.02)
    return float(state.support_margin - extra)

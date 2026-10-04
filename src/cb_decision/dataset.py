"""决策数据集构建：仿真 rollout + 规则专家标签（第 8.5 节）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config
from cb_common.errors import DataError
from cb_common.logging import JsonlLogger, new_run_id
from cb_common.types import SafetyLevel, SkillCommand, TaskContext
from cb_features.text_state import build_text_state, safety_level_from_state
from cb_train.train_ppo import _build_env_factory

PRIMITIVE_LABELS = (
    "continue_current",
    "stand",
    "walk",
    "run",
    "run_jump",
    "wave_while_run",
    "turn_left",
    "recover",
    "safe_stop",
)


@dataclass
class DecisionSample:
    """单条决策样本（状态 + 四头标签 + 特性）。"""

    sample_id: str
    rollout_id: str
    skill: str
    state_text: str
    features: dict[str, float]
    labels: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def primitive_label(skill: str, *, vx: float, yaw_rate: float) -> str:
    """由场景技能与运动状态生成 primitive 标签。"""
    if skill == "jump":
        return "run_jump"
    if skill == "wave":
        return "wave_while_run" if abs(vx) > 0.5 else "continue_current"
    if skill == "turn":
        return "turn_left" if yaw_rate >= 0.0 else "continue_current"
    if skill == "recover":
        return "recover"
    if skill == "run":
        return "run"
    if skill == "walk":
        return "walk"
    if skill == "stand":
        return "stand"
    return "continue_current"


def build_decision_dataset(
    train_cfg: Config,
    system_cfg: Config,
    reward_cfg: Config,
    *,
    output_dir: str | Path,
    seeds: int = 4,
    decision_every: int = 4,
    scenarios: Sequence[str] | None = None,
    logger: JsonlLogger | None = None,
) -> dict[str, Any]:
    """运行带规则标签的仿真 rollout，产出 train/val/test JSONL。"""
    factory, env_cfg, scenario_map, teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed=7)
    names = list(scenarios or scenario_map.keys())
    samples: list[DecisionSample] = []
    for scenario_name in names:
        for seed in range(seeds):
            env = factory(seed, scenario_name)
            obs, _ = env.reset(seed=seed, randomize=True)
            rollout_id = f"{scenario_name}-{seed}"
            step = 0
            while True:
                obs, _reward, terminated, truncated, info = env.step(np.zeros(29))
                step += 1
                if step % decision_every == 0:
                    state = env._robot_state()
                    command3 = env._command3_from_segment(env._current_segment)
                    label = primitive_label(env._active_name, vx=command3[0], yaw_rate=command3[2])
                    level = safety_level_from_state(state)
                    veto = level >= SafetyLevel.ABORT
                    margin = float(state.support_margin)
                    recovery = "safe_stop" if level >= SafetyLevel.EMERGENCY else (
                        "stand_recover" if level >= SafetyLevel.ABORT else ("small_step" if level >= SafetyLevel.WARN else "none")
                    )
                    roll, pitch, _ = state.rpy_deg()
                    context = TaskContext(
                        task=scenario_name,
                        external_push=bool(env.dr.push_at(env.step_count) is not None),
                        latency_ms=float(env.dr.action_delay_steps) * env_cfg.control_dt * 1000.0,
                        candidates=list(PRIMITIVE_LABELS),
                    )
                    text = build_text_state(state, SkillCommand(skill=env._active_name), context)
                    sample = DecisionSample(
                        sample_id=f"{rollout_id}-{step}",
                        rollout_id=rollout_id,
                        skill=env._active_name,
                        state_text=text,
                        features={
                            "roll_deg": float(roll),
                            "pitch_deg": float(pitch),
                            "support_margin_m": margin,
                            "vx": float(state.base_lin_vel[0]),
                            "yaw_rate": float(state.base_ang_vel[2]),
                            "height_m": float(state.base_pos[2]),
                            "contact_left": float(state.contact[0]),
                            "contact_right": float(state.contact[1]),
                            "emergency": float(level >= SafetyLevel.EMERGENCY),
                        },
                        labels={
                            "motion_primitive_head": label,
                            "safety_veto_head": int(veto),
                            "recovery_head": recovery,
                            "escalation_head": int(
                                bool(info["metrics"].get("fall", 0.0))
                                or level >= SafetyLevel.ABORT
                                or recovery == "safe_stop"
                            ),
                        },
                        meta={"scenario": scenario_name, "seed": seed, "step": step, "safety_level": level.name},
                    )
                    samples.append(sample)
                if terminated or truncated:
                    break
    if not samples:
        raise DataError("decision dataset is empty")
    # 按 rollout 切分，防泄漏
    rollouts = sorted({sample.rollout_id for sample in samples})
    split_map: dict[str, str] = {}
    for rollout in rollouts:
        digest = hashlib.sha256(f"decision:{rollout}".encode()).hexdigest()
        ratio = int(digest[:8], 16) / 0xFFFFFFFF
        split_map[rollout] = "train" if ratio < 0.7 else ("val" if ratio < 0.85 else "test")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                if split_map[sample.rollout_id] != split:
                    continue
                handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                counts[split] += 1
    summary = {
        "dataset_version": new_run_id("decision-data"),
        "samples": len(samples),
        "rollouts": len(rollouts),
        "split_counts": counts,
        "label_distribution": _label_distribution(samples),
        "scenarios": names,
        "seeds": seeds,
        "decision_every": decision_every,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if logger:
        logger.log("data_pipeline", stage="decision_dataset", **summary)
    return summary


def load_decision_dataset(directory: str | Path, split: str) -> list[DecisionSample]:
    """读取某个 split 的决策样本。"""
    path = Path(directory) / f"{split}.jsonl"
    if not path.is_file():
        raise DataError("decision split not found", path=str(path))
    out: list[DecisionSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(DecisionSample(**json.loads(line)))
    if not out:
        raise DataError("decision split is empty", path=str(path))
    return out


def _label_distribution(samples: Sequence[DecisionSample]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sample in samples:
        for head, label in sample.labels.items():
            out.setdefault(head, {})
            key = str(label)
            out[head][key] = out[head].get(key, 0) + 1
    return out

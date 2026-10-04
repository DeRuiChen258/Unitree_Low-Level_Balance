"""批量 rollout 评测：三场景 × 四扰动 × N seeds（含最差 seed）。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config
from cb_common.errors import PolicyError
from cb_common.logging import JsonlLogger, new_run_id
from cb_train.train_ppo import _build_env_factory

from .metrics import recovery_time, summarize_episodes

PolicyFn = Callable[[np.ndarray], np.ndarray]


def zero_policy(_obs: np.ndarray) -> np.ndarray:
    """基线：零残差（教师策略单独运行）。"""
    return np.zeros(29)


@dataclass
class EvalConfig:
    """评测配置。"""

    scenarios: tuple[str, ...] = ("run_jump", "run_wave", "turn_wave")
    profiles: tuple[str, ...] = ("none", "push", "noise", "delay")
    seeds: int = 10
    thresholds: Mapping[str, float] = field(default_factory=dict)
    output_dir: str = "runs/reports"
    policy_mode: str = "residual"
    profile_overrides: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: {
            "none": {"push_prob": 0.0, "obs_noise_std": 0.0, "actuator_delay_steps": [0, 0],
                     "mass_scale": [1.0, 1.0], "friction_scale": [1.0, 1.0]},
            # 每个扰动 profile 隔离单一因素：质量/摩擦回到 1.0
            "push": {
                "push_prob": 0.006,
                "push_force_n": [15.0, 30.0],
                "push_duration_s": [0.05, 0.12],
                "mass_scale": [1.0, 1.0],
                "friction_scale": [1.0, 1.0],
            },
            "noise": {
                "obs_noise_std": 0.02,
                "push_prob": 0.0,
                "mass_scale": [1.0, 1.0],
                "friction_scale": [1.0, 1.0],
            },
            "delay": {
                "actuator_delay_steps": [2, 2],
                "push_prob": 0.0,
                "mass_scale": [1.0, 1.0],
                "friction_scale": [1.0, 1.0],
            },
        }
    )


def _policy_fn(policy: Any) -> PolicyFn:
    """把 OnnxPolicy/TorchPolicy/callable 统一成函数。"""
    if policy is None:
        return zero_policy
    if callable(policy):
        return policy
    if hasattr(policy, "act"):
        return lambda obs: np.asarray(policy.act(obs), dtype=np.float64)
    raise PolicyError("unsupported policy object", type=type(policy).__name__)


def run_scenario(
    *,
    train_cfg: Config,
    system_cfg: Config,
    reward_cfg: Config,
    scenario_name: str,
    profile: str,
    seeds: int,
    policy: Any = None,
    profile_override: Mapping[str, Any] | None = None,
    threshold_cfg: Mapping[str, float] | None = None,
    policy_mode: str = "residual",
) -> list[dict[str, Any]]:
    """运行一个场景 × 一个扰动 profile × N seeds。"""
    factory, env_cfg, scenarios, teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed=1)
    dr = dict(train_cfg.get("dr", {}))
    dr.update(dict(profile_override or {}))
    env_cfg.dr = dr
    env_cfg.policy_mode = policy_mode
    act = _policy_fn(policy)
    episodes: list[dict[str, Any]] = []
    for seed in range(seeds):
        env = factory(seed, scenario_name)
        obs, _ = env.reset(seed=seed, randomize=profile != "none")
        step = 0
        while True:
            action = act(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                break
        metrics = info["metrics"]
        push_steps = [start for start, _end, _force in env.dr.push_schedule]
        if push_steps:
            # 只评估「扰动后仍有 >=2.5 s 观测窗口」的 episode；其余为删失数据，记为 NaN
            valid_onsets = [step for step in push_steps if (len(env.tilt_series) - step) * env_cfg.control_dt >= 2.5]
            onset = min(valid_onsets) if valid_onsets else None
            times = [item[0] for item in env.tilt_series]
            tilt_deg = [max(abs(np.degrees(item[1])), abs(np.degrees(item[2]))) for item in env.tilt_series]
            if onset is None:
                rec = float("nan")
            else:
                pre = tilt_deg[: max(1, min(onset, len(tilt_deg)))]
                baseline = float(np.median(pre)) if pre else 0.0
                threshold = max(6.0, baseline + 4.0)
                rec = recovery_time(
                    times,
                    tilt_deg,
                    onset_index=min(onset, len(times) - 1),
                    threshold_deg=threshold,
                    smooth_window_s=0.3,
                )
        else:
            rec = float("nan")
        wave_success = float(metrics["wave_error"] < 5.0) if "wave" in scenario_name else float("nan")
        jump_success = (
            float(metrics["foot_clearance_max"] > 0.05 and metrics["fall"] == 0.0)
            if scenario_name == "run_jump"
            else float("nan")
        )
        turn_ok = float(abs(metrics["turn_error"]) < np.radians(12.0)) if "turn" in scenario_name else float("nan")
        episodes.append(
            {
                "seed": seed,
                "profile": profile,
                "scenario": scenario_name,
                "episode_reward": float(info["episode_reward"]),
                "roll_peak_deg": float(metrics["roll_peak"]),
                "pitch_peak_deg": float(metrics["pitch_peak"]),
                "fall": float(metrics["fall"]),
                "support_margin_min": float(metrics["support_margin_min"]),
                "speed_error_mean": float(metrics["speed_error"]),
                "speed_retention": float(metrics.get("speed_retention", 0.0)),
                "support_violation_max_s": float(metrics.get("support_violation_max_s", 0.0)),
                "support_violation_ratio": float(metrics.get("support_violation_ratio", 0.0)),
                "height_min": float(metrics["height_min"]),
                "jump_apex": float(metrics["jump_apex"]),
                "foot_clearance_max": float(metrics["foot_clearance_max"]),
                "jump_success": jump_success,
                "wave_success": wave_success,
                "turn_error_deg": float(np.degrees(metrics["turn_error"])),
                "turn_success": turn_ok,
                "recovery_time_s": float(rec) if np.isfinite(rec) else float("nan"),
            }
        )
    return episodes


def evaluate_policy(
    *,
    train_cfg: Config,
    system_cfg: Config,
    reward_cfg: Config,
    config: EvalConfig,
    policy: Any = None,
    label: str = "policy",
    logger: JsonlLogger | None = None,
) -> dict[str, Any]:
    """完整评测：三场景 × 四扰动 × seeds，输出 JSON + Markdown。"""
    run_id = new_run_id("eval")
    out_dir = Path(config.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    for scenario in config.scenarios:
        results[scenario] = {}
        for profile in config.profiles:
            override = dict(config.profile_overrides.get(profile, {}))
            episodes = run_scenario(
                train_cfg=train_cfg,
                system_cfg=system_cfg,
                reward_cfg=reward_cfg,
                scenario_name=scenario,
                profile=profile,
                seeds=config.seeds,
                policy=policy,
                profile_override=override,
                policy_mode=config.policy_mode,
            )
            summary = summarize_episodes(episodes, config.thresholds)
            results[scenario][profile] = {"summary": summary, "episodes": episodes}
            all_episodes.extend(episodes)
            if logger:
                logger.log("eval_scenario", scenario=scenario, profile=profile, **summary)
    overall = summarize_episodes(all_episodes, config.thresholds)
    report = {
        "run_id": run_id,
        "label": label,
        "scenarios": list(config.scenarios),
        "profiles": list(config.profiles),
        "seeds": config.seeds,
        "overall": overall,
        "results": results,
        "pass": bool(overall["pass"]) and all(
            results[s][p]["summary"]["pass"] for s in config.scenarios for p in config.profiles
        ),
    }
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(out_dir / "eval.md", report)
    return report


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    """写可 diff 的 Markdown 报告。"""
    lines = [
        f"# Policy Evaluation `{report['run_id']}`",
        "",
        f"- label: `{report['label']}`",
        f"- scenarios: {', '.join(report['scenarios'])}",
        f"- profiles: {', '.join(report['profiles'])}",
        f"- seeds: {report['seeds']}",
        f"- overall pass: `{report['pass']}`",
        "",
        "| scenario | profile | pass | fall_rate | roll_worst | pitch_worst | speed_err | jump | wave | turn_err |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario, profiles in report["results"].items():
        for profile, payload in profiles.items():
            s = payload["summary"]
            lines.append(
                f"| {scenario} | {profile} | {s['pass']} | {s['fall_rate']:.2f} | {s['roll_peak_worst_deg']:.2f} | "
                f"{s['pitch_peak_worst_deg']:.2f} | {s['speed_error_worst_mps']:.2f} | "
                f"{s.get('jump_success_rate')} | {s.get('wave_success_rate')} | {s.get('turn_error_worst_deg')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

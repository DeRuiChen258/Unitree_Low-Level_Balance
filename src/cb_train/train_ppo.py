"""PPO 训练入口：MuJoCo 环境 + 教师策略 + 学习残差（第 6.2/6.6 节）。"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.config import Config, load_config, resolve_path
from cb_common.logging import JsonlLogger, new_run_id
from cb_common.seeding import set_global_seed
from cb_features.normalizer import RunningMeanStd
from cb_features.obs_spec import default_obs_spec
from cb_policy.teacher import TeacherCore
from cb_skills import load_skill_specs

from .envs.g1_balance_env import EnvConfig, G1BalanceEnv, ScenarioScript
from .networks import ActorCritic


@dataclass
class TrainResult:
    """训练结果摘要。"""

    run_id: str
    run_dir: str
    checkpoint: str
    updates: int
    env_steps: int
    final_reward: float
    baseline_reward: float
    trained_reward: float
    summary: dict[str, Any]


def _build_env_factory(train_cfg: Config, system_cfg: Config, reward_cfg: Config, seed: int):
    """构造环境工厂（共享一个 TeacherCore，CPU 推理）。"""
    run_teacher = TeacherCore(
        deploy_dir=str(train_cfg.get("teacher.run.deploy_dir")),
        config_path=str(train_cfg.get("teacher.run.config")),
        onnx_name=Path(str(train_cfg.get("teacher.run.onnx"))).name,
    )
    walk_teacher = TeacherCore(
        deploy_dir=str(train_cfg.get("teacher.walk.deploy_dir")),
        config_path=str(train_cfg.get("teacher.walk.config")),
        onnx_name=Path(str(train_cfg.get("teacher.walk.onnx"))).name,
    )
    specs = load_skill_specs("configs/skills.yaml")
    obs_spec = default_obs_spec(
        history_length=int(train_cfg.get("obs.history_length", 4) or system_cfg.get("history_length", 4)),
        skill_embedding_dim=16,
    )
    scenarios = {
        name: ScenarioScript.from_config(name, dict(spec))
        for name, spec in dict(train_cfg.get("scenarios", {})).items()
    }
    env_cfg = EnvConfig(
        scene_path=str(system_cfg.get("limits.mjcf_scene")),
        control_dt=float(train_cfg.get("control_dt", 0.02)),
        sim_dt=float(train_cfg.get("sim_dt", 0.005)),
        decimation=int(round(float(train_cfg.get("control_dt", 0.02)) / float(train_cfg.get("sim_dt", 0.005)))),
        episode_length_s=float(train_cfg.get("episode_length_s", 12.0)),
        history_length=obs_spec.history_length,
        residual_scale=float(train_cfg.get("network.residual_scale", 0.25)),
        reward=reward_cfg.to_dict(),
        termination=dict(reward_cfg.get("termination", {})),
        dr=dict(train_cfg.get("dr", {})),
        obs_spec=obs_spec,
    )

    def factory(index: int, scenario_name: str) -> G1BalanceEnv:
        # 低速场景使用 AMP 走路教师，高速场景使用跑步教师（复用现有 ONNX 资产）
        teacher = walk_teacher if scenario_name == "run_wave" else run_teacher
        return G1BalanceEnv(
            teacher,
            env_cfg,
            scenarios[scenario_name],
            seed=seed + index,
            skill_specs=specs,
        )

    return factory, env_cfg, scenarios, run_teacher


def train(
    config_path: str | Path,
    *,
    system_config_path: str | Path = "configs/system.yaml",
    reward_config_path: str | Path = "configs/reward.yaml",
    seed: int | None = None,
    updates: int | None = None,
    num_envs: int | None = None,
    run_dir: str | Path | None = None,
    device: str | None = None,
) -> TrainResult:
    """执行 PPO 训练并落盘 checkpoint/报告。"""
    train_cfg = load_config(config_path)
    system_cfg = load_config(system_config_path)
    reward_cfg = load_config(reward_config_path)
    seed = int(train_cfg.get("seed", 1) if seed is None else seed)
    updates = int(train_cfg.get("total_updates", 100) if updates is None else updates)
    num_envs = int(train_cfg.get("num_envs", 8) if num_envs is None else num_envs)
    rollout_steps = int(train_cfg.get("rollout_steps", 128))
    device = str(device or train_cfg.get("device", "cuda"))
    seed_record = set_global_seed(seed)
    run_id = new_run_id("ppo")
    base_dir = Path(run_dir) if run_dir else resolve_path(system_cfg, "paths.runs_dir") / "exp" / run_id
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    factory, env_cfg, scenarios, teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed)
    scenario_names = list(scenarios)
    envs = [factory(i, scenario_names[i % len(scenario_names)]) for i in range(num_envs)]
    obs_dim = env_cfg.obs_spec.total_dim
    import torch

    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    torch.set_num_threads(int(system_cfg.get("runtime.cpu_threads", 8)))
    network_cfg = dict(train_cfg.get("network", {}))
    policy = ActorCritic(
        obs_dim,
        29,
        actor_hidden=tuple(network_cfg.get("actor_hidden", [256, 256])),
        critic_hidden=tuple(network_cfg.get("critic_hidden", [256, 256])),
        init_noise_std=float(network_cfg.get("init_noise_std", 0.35)),
        log_std_min=float(network_cfg.get("log_std_min", -3.0)),
        log_std_max=float(network_cfg.get("log_std_max", 0.5)),
    ).to(torch_device)
    ppo = dict(train_cfg.get("ppo", {}))
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(ppo.get("lr", 3e-4)))
    normalizer = RunningMeanStd((obs_dim,))
    logger = JsonlLogger(
        base_dir / "training.jsonl",
        event_types=["training_epoch"],
        module="cb_train.train_ppo",
    )
    metrics_path = base_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["update", "reward_mean", "episode_reward", "fall_rate", "value_loss", "policy_loss", "entropy", "wall_s"])

    obs_batch = []
    for env in envs:
        obs, _ = env.reset(seed=seed + len(obs_batch), randomize=True)
        obs_batch.append(obs)
    episode_stats: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = 0
    update_metrics: dict[str, float] = {}
    for update in range(updates):
        transitions = {"obs": [], "actions": [], "logp": [], "values": [], "rewards": [], "dones": [], "raw": []}
        reward_scale = float(ppo.get("reward_scale", 0.1))
        rewards_window: list[float] = []
        falls = 0
        episodes = 0
        for _step in range(rollout_steps):
            obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=torch_device)
            norm_obs = normalizer.normalize(obs_tensor.cpu().numpy())
            with torch.no_grad():
                dist, mean, value = policy.distribution(torch.tensor(norm_obs, dtype=torch.float32, device=torch_device))
                raw_action = dist.sample()
                action = torch.tanh(raw_action)
                logp = ActorCritic.log_prob_of(dist, raw_action, action)
            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(action[i].cpu().numpy())
                transitions["obs"].append(norm_obs[i])
                transitions["raw"].append(raw_action[i].cpu().numpy())
                transitions["actions"].append(action[i].cpu().numpy())
                transitions["logp"].append(float(logp[i].cpu()))
                transitions["values"].append(float(value[i].cpu()))
                transitions["rewards"].append(float(reward) * reward_scale)
                transitions["dones"].append(bool(terminated or truncated))
                rewards_window.append(float(reward))
                env_steps += 1
                if terminated or truncated:
                    episodes += 1
                    falls += int(info["metrics"].get("fall", 0.0))
                    episode_stats.append(
                        {
                            "reward": info["episode_reward"],
                            "fall": info["metrics"].get("fall", 0.0),
                            "skill": info["skill"],
                            "metrics": info["metrics"],
                        }
                    )
                    next_obs, _ = env.reset(seed=seed + env_steps, randomize=True)
                obs_batch[i] = next_obs
        obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=torch_device)
        next_norm = normalizer.normalize(obs_tensor.cpu().numpy())
        with torch.no_grad():
            _, _, next_value = policy.distribution(torch.tensor(next_norm, dtype=torch.float32, device=torch_device))
            next_values = next_value.cpu().numpy()
        # 按环境分段计算 GAE
        obs_arr = np.stack(transitions["obs"])
        act_arr = np.stack(transitions["actions"])
        logp_arr = np.array(transitions["logp"], dtype=np.float64)
        val_arr = np.array(transitions["values"], dtype=np.float64)
        rew_arr = np.array(transitions["rewards"], dtype=np.float64)
        done_arr = np.array(transitions["dones"], dtype=bool)
        adv = np.zeros_like(rew_arr)
        gamma, lam = float(ppo.get("gamma", 0.99)), float(ppo.get("lam", 0.95))
        rew_env = rew_arr.reshape(rollout_steps, num_envs)
        val_env = val_arr.reshape(rollout_steps, num_envs)
        done_env = done_arr.reshape(rollout_steps, num_envs)
        adv_env = np.zeros_like(rew_env)
        last_adv = np.zeros(num_envs)
        for t in reversed(range(rollout_steps)):
            next_val = next_values if t == rollout_steps - 1 else val_env[t + 1]
            nonterminal = 1.0 - done_env[t].astype(np.float64)
            delta = rew_env[t] + gamma * next_val * nonterminal - val_env[t]
            last_adv = delta + gamma * lam * nonterminal * last_adv
            adv_env[t] = last_adv
        adv = adv_env.reshape(-1)
        returns = adv + val_arr
        if bool(ppo.get("normalize_advantage", True)):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=torch_device)
        act_t = torch.tensor(act_arr, dtype=torch.float32, device=torch_device)
        logp_old = torch.tensor(logp_arr, dtype=torch.float32, device=torch_device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=torch_device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=torch_device)
        minibatches = int(ppo.get("minibatches", 4))
        batch_size = max(1, len(obs_arr) // max(1, minibatches))
        last_metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0}
        for _epoch in range(int(ppo.get("epochs", 4))):
            indices = torch.randperm(len(obs_arr), device=torch_device)
            for start in range(0, len(obs_arr), batch_size):
                idx = indices[start : start + batch_size]
                dist, _mean, value = policy.distribution(obs_t[idx])
                raw_actions = torch.atanh(torch.clamp(act_t[idx], -0.999, 0.999))
                logp = ActorCritic.log_prob_of(dist, raw_actions, act_t[idx])
                ratio = torch.exp(logp - logp_old[idx])
                clip = float(ppo.get("clip", 0.2))
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.functional.mse_loss(value, ret_t[idx])
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + float(ppo.get("value_coef", 0.5)) * value_loss
                    - float(ppo.get("entropy_coef", 0.002)) * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(ppo.get("max_grad_norm", 1.0)))
                optimizer.step()
                last_metrics = {
                    "value_loss": float(value_loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                }
        update_metrics = last_metrics
        wall = time.time() - start_time
        recent_episodes = episode_stats[-20:]
        episode_reward = float(np.mean([e["reward"] for e in recent_episodes])) if recent_episodes else float("nan")
        fall_rate = float(np.mean([e["fall"] for e in recent_episodes])) if recent_episodes else 0.0
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [update, float(np.mean(rewards_window)), episode_reward, fall_rate, last_metrics["value_loss"], last_metrics["policy_loss"], last_metrics["entropy"], round(wall, 2)]
            )
        logger.log(
            "training_epoch",
            update=update,
            env_steps=env_steps,
            reward_mean=float(np.mean(rewards_window)),
            episode_reward=episode_reward,
            fall_rate=fall_rate,
            **last_metrics,
        )
    checkpoint_path = base_dir / "checkpoints" / "latest.pt"
    metadata = {
        "feature_version": str(system_cfg.get("feature_version", "1.0")),
        "obs_spec_hash": env_cfg.obs_spec.hash(),
        "obs_dim": obs_dim,
        "action_dim": 29,
        "action_kind": "residual_delta",
        "output_scale": float(env_cfg.residual_scale),
        "dataset_version": "n/a",
        "model_version": "1.0.0",
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": 29,
            "hidden_sizes": list(network_cfg.get("actor_hidden", [256, 256])),
            "metadata": metadata,
            "normalizer": normalizer.state_dict(),
            "config": train_cfg.to_dict(),
            "seed": seed,
        },
        checkpoint_path,
    )
    summary = {
        "run_id": run_id,
        "updates": updates,
        "env_steps": env_steps,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "device": str(torch_device),
        "seed_record": seed_record,
        "reward_mean_last_update": float(np.mean(rewards_window)),
        "episode_reward_mean": float(np.mean([e["reward"] for e in episode_stats])) if episode_stats else float("nan"),
        "fall_rate": float(np.mean([e["fall"] for e in episode_stats])) if episode_stats else float("nan"),
        "skill_distribution": _skill_distribution(episode_stats),
        "skill_fall_rate": _skill_fall_rate(episode_stats),
        "final_metrics": update_metrics,
        "wall_s": round(time.time() - start_time, 2),
        "config_hash": train_cfg.hash(),
    }
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (base_dir / "config.json").write_text(json.dumps(train_cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.close()
    _ = asdict  # 保留 dataclass 引用，避免误删
    return TrainResult(
        run_id=run_id,
        run_dir=str(base_dir),
        checkpoint=str(checkpoint_path),
        updates=updates,
        env_steps=env_steps,
        final_reward=float(np.mean(rewards_window)),
        baseline_reward=float("nan"),
        trained_reward=float("nan"),
        summary=summary,
    )


def _skill_distribution(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in episodes:
        skill = str(item.get("skill", "unknown"))
        counts[skill] = counts.get(skill, 0) + 1
    return counts


def _skill_fall_rate(episodes: list[dict[str, Any]]) -> dict[str, float]:
    """按技能统计摔倒率（最差 seed 分析的基础）。"""
    counts: dict[str, list[float]] = {}
    for item in episodes:
        counts.setdefault(str(item.get("skill", "unknown")), []).append(float(item.get("fall", 0.0)))
    return {skill: float(np.mean(values)) for skill, values in counts.items()}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = train(args.config, seed=args.seed, updates=args.updates, num_envs=args.num_envs, run_dir=args.run_dir)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))

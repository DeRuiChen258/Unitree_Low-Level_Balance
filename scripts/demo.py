#!/usr/bin/env python3
"""端到端 demo：状态 → 特征 → 决策 → 技能 → 安全层 → MuJoCo 执行 + 可视化。

用法：
    python scripts/demo.py --scenario run_wave --render viewer
    python scripts/demo.py --scenario run_jump --render video --steps 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_common.joints import KD_POLICY, KP_POLICY
from cb_common.logging import JsonlLogger, new_run_id
from cb_common.types import JointCommand, SkillCommand, TaskContext
from cb_decision.policy import DecisionPolicy
from cb_features.text_state import build_text_state
from cb_policy.balance_policy import OnnxPolicy
from cb_safety import EStopChannel, SafetyWrapper, Watchdog, WatchdogLimits, load_safety_limits
from cb_train.train_ppo import _build_env_factory


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--train-config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--scenario", default="run_wave", choices=["run_jump", "run_wave", "turn_wave"])
    parser.add_argument("--policy", default=None, help="ONNX 学生/残差策略；缺省使用教师基线")
    parser.add_argument("--policy-mode", choices=["residual", "absolute"], default="residual")
    parser.add_argument("--render", choices=["none", "viewer", "video"], default="video")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--use-laya", action="store_true", help="启用 LayA 决策（默认规则后端，避免加载大模型）")
    args = parser.parse_args()
    system_cfg = load_config(args.config)
    train_cfg = load_config(args.train_config)
    reward_cfg = load_config(args.reward_config)
    run_id = new_run_id("demo")
    out_dir = Path(args.out_dir) if args.out_dir else Path(str(system_cfg.get("paths.runs_dir"))) / "demo" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    factory, env_cfg, scenarios, _teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed=3)
    env_cfg.policy_mode = args.policy_mode
    env_cfg.episode_length_s = max(4.0, args.steps * env_cfg.control_dt + 1.0)
    env = factory(args.seed, args.scenario)
    policy = OnnxPolicy(args.policy) if args.policy else None
    limits = load_safety_limits("configs/safety/limits.yaml", scene_path=str(system_cfg.get("limits.mjcf_scene")))
    wrapper = SafetyWrapper(limits, env_cfg.control_dt, estop=EStopChannel(), watchdog=Watchdog(WatchdogLimits.from_mapping(limits.watchdog)))
    logger = JsonlLogger(out_dir / "demo.jsonl", event_types=["skill_switch", "decision", "safety_event", "demo_frame"], module="scripts.demo")
    decider = DecisionPolicy() if args.use_laya else None
    viewer = None
    renderer = None
    writer = None
    if args.render == "viewer":
        import mujoco
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.rt.model, env.rt.data)
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
    elif args.render == "video":
        import cv2
        import mujoco

        renderer = mujoco.Renderer(env.rt.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.rt.model, camera)
        camera.distance = 2.5
        camera.azimuth = 120
        camera.elevation = -20
        writer = cv2.VideoWriter(str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (640, 480))

    obs, _ = env.reset(seed=args.seed, randomize=False)
    step = 0
    blocked = False
    start = time.time()
    decision_interval = max(1, int(round(0.2 / env_cfg.control_dt)))
    while step < args.steps and not blocked:
        action = policy.act(obs) if policy is not None else np.zeros(29)
        obs, reward, terminated, truncated, info = env.step(action)
        state = env._robot_state()
        target = np.asarray(info["target"], dtype=np.float64)
        command = JointCommand(
            q_target=target,
            kp=np.array(KP_POLICY),
            kd=np.array(KD_POLICY),
            timestamp=state.timestamp,
            source="demo",
            skill=str(info["skill"]),
        )
        decision = wrapper.filter(command, state, now=state.timestamp)
        if not decision.allowed:
            logger.log("safety_event", level=decision.level.name, reason=decision.reason, skill=info["skill"])
            blocked = True
            break
        if decider is not None and step % decision_interval == 0:
            ctx = TaskContext(task=args.scenario, candidates=["continue_current", "stand", "recover", "safe_stop"])
            out = decider.decide(state, ctx, command=SkillCommand(skill=info["skill"]))
            logger.log("decision", action_id=out.action_id, confidence=out.confidence, risk=out.risk, reason_code=out.reason_code)
        logger.log(
            "demo_frame",
            step=step,
            skill=info["skill"],
            reward=float(reward),
            target=np.round(target, 4),
            state_hash=hash(build_text_state(state, SkillCommand(skill=info["skill"]), TaskContext(task=args.scenario))) & 0xFFFFFFFF,
            roll=float(state.rpy_deg()[0]),
            pitch=float(state.rpy_deg()[1]),
        )
        if renderer is not None:
            camera.lookat[:] = env.rt.data.qpos[:3]
            renderer.update_scene(env.rt.data, camera=camera)
            frame = renderer.render().copy()
            writer.write(frame[:, :, ::-1])
        if viewer is not None:
            viewer.cam.lookat[:] = env.rt.data.qpos[:3]
            viewer.sync()
            time.sleep(max(0.0, env_cfg.control_dt - 0.004))
        step += 1
        if terminated or truncated:
            break
    if writer is not None:
        writer.release()
    if renderer is not None:
        renderer.close()
    if viewer is not None:
        viewer.close()
    logger.close()
    summary = {
        "run_id": run_id,
        "scenario": args.scenario,
        "steps": step,
        "render": args.render,
        "policy": args.policy or "teacher_baseline",
        "policy_mode": args.policy_mode,
        "blocked_by_safety": blocked,
        "final_skill": info["skill"],
        "metrics": info["metrics"],
        "wall_s": round(time.time() - start, 2),
        "video": str(out_dir / "demo.mp4") if writer is not None else None,
        "log": str(out_dir / "demo.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

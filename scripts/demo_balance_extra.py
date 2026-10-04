#!/usr/bin/env python3
"""平衡系统压力测试演示：绕柱避障（pillar_avoid）与负重搬运（carry_box）。

用法：
    python scripts/demo_balance_extra.py --scenario pillar_avoid --render viewer
    python scripts/demo_balance_extra.py --scenario carry_box --carry-mass 8 --render viewer
    python scripts/demo_balance_extra.py --scenario carry_box --carry-mass 16.67 --render video

链路与 scripts/demo.py 一致：教师 ONNX + 技能 + 安全层 + 平衡指标；
额外注入环境侧扰动（柱子 / 载荷），用于压测平衡系统。
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
from cb_common.types import JointCommand
from cb_safety import EStopChannel, SafetyWrapper, Watchdog, WatchdogLimits, load_safety_limits
from cb_train.envs.balance_extra import CarriedLoad, PillarNavigator, build_augmented_model, swap_runtime_model
from cb_train.train_ppo import _build_env_factory


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--train-config", default="configs/train_g1_balance.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument(
        "--scenario",
        default="pillar_slalom",
        choices=["walk", "walk_run", "run_stop", "pillar_avoid", "pillar_slalom", "carry_box"],
    )
    parser.add_argument("--carry-mass", type=float, default=8.0, help="载荷质量 kg（机器人本体 33.34 kg）")
    parser.add_argument("--carry-forward", type=float, default=0.20, help="载荷质心相对躯干的前向偏移 m（贴身程度）")
    parser.add_argument("--carry-drop", type=float, default=0.10, help="载荷质心相对躯干的高度偏移 m")
    parser.add_argument("--pillar-x", type=float, default=5.0, help="单柱场景的柱子 x")
    parser.add_argument(
        "--pillars",
        default="4.5,0.45;8.0,-0.45;11.5,0.45",
        help="绕桩柱位列表 'x,y;x,y;...'（pillar_slalom 用；空串表示默认单柱）",
    )
    parser.add_argument("--render", choices=["none", "viewer", "video"], default="video")
    parser.add_argument("--record", action="store_true", help="viewer 模式同时录制 MP4（用于留档）")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    system_cfg = load_config(args.config)
    train_cfg = load_config(args.train_config)
    reward_cfg = load_config(args.reward_config)
    run_id = new_run_id("balance_extra")
    out_dir = Path(args.out_dir) if args.out_dir else Path(str(system_cfg.get("paths.runs_dir"))) / "demo" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    factory, env_cfg, _scenarios, _teacher = _build_env_factory(train_cfg, system_cfg, reward_cfg, seed=args.seed)
    env_cfg.episode_length_s = max(6.0, args.steps * env_cfg.control_dt + 1.0)
    env = factory(args.seed, args.scenario)

    # 环境侧扰动注入：追加柱子 / 载荷箱体（只追加 body，既有索引不变）
    import mujoco

    pillar_positions: list[tuple[float, float]] = []
    if args.scenario == "pillar_avoid":
        pillar_positions = [(args.pillar_x, 0.0)]
    elif args.scenario == "pillar_slalom":
        for item in args.pillars.split(";"):
            if item.strip():
                px, py = item.split(",")
                pillar_positions.append((float(px), float(py)))
    model = build_augmented_model(
        mujoco,
        str(env.rt.core.xml_path),
        pillars=pillar_positions,
        carry_box_mass=args.carry_mass if args.scenario == "carry_box" else 0.0,
    )
    swap_runtime_model(env, model)
    navigator = PillarNavigator(
        pillars=pillar_positions or [(args.pillar_x, 0.0)],
        cruise_vx=float(_scenarios[args.scenario].segments[0].get("vx", 1.5)),
    )
    load = CarriedLoad(
        mass=args.carry_mass,
        offset_local=np.array([float(args.carry_forward), 0.0, -float(args.carry_drop)]),
        _dt=float(env_cfg.control_dt),
    )

    limits = load_safety_limits("configs/safety/limits.yaml", scene_path=str(system_cfg.get("limits.mjcf_scene")))
    wrapper = SafetyWrapper(
        limits, env_cfg.control_dt, estop=EStopChannel(), watchdog=Watchdog(WatchdogLimits.from_mapping(limits.watchdog))
    )
    logger = JsonlLogger(
        out_dir / "demo.jsonl",
        event_types=["safety_event", "demo_frame", "waypoint"],
        module="scripts.demo_balance_extra",
    )

    viewer = None
    renderer = None
    writer = None
    if args.render == "viewer":
        viewer = mujoco.viewer.launch_passive(env.rt.model, env.rt.data)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -18
    if args.render == "video" or args.record:
        import cv2

        renderer = mujoco.Renderer(env.rt.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.rt.model, camera)
        camera.distance = 4.0
        camera.azimuth = 120
        camera.elevation = -18
        writer = cv2.VideoWriter(
            str(out_dir / "demo.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            round(1.0 / float(env_cfg.control_dt)),
            (640, 480),
        )

    env.reset(seed=args.seed, randomize=False)
    step = 0
    blocked = False
    start = time.time()
    circling = args.scenario in ("pillar_avoid", "pillar_slalom")
    stop_watch: dict[str, float] = {}
    while step < args.steps and not blocked:
        state = env._robot_state()
        if circling:
            _, _, yaw = state.rpy()
            navigator.override_skill(env, float(state.base_pos[0]), float(state.base_pos[1]), float(yaw))
        elif args.scenario == "carry_box" and args.carry_mass > 0.0:
            load.apply(mujoco, env.rt.model, env.rt.data)
        obs, reward, terminated, truncated, info = env.step(np.zeros(29))
        state = env._robot_state()
        if args.scenario == "run_stop" and str(info["skill"]) == "stand":
            # 记录「跑步→急停」切换点与停止后的位置，用于统计急停距离
            stop_watch.setdefault("stop_start", float(state.base_pos[0]))
            stop_watch["stop_end"] = float(state.base_pos[0])
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
            logger.log(
                "safety_event",
                level=decision.level.name,
                reason=decision.reason,
                skill=info["skill"],
                events=[(item.level.name, item.reason, round(float(item.value), 4)) for item in decision.events],
            )
            blocked = True
            break
        logger.log(
            "demo_frame",
            step=step,
            skill=info["skill"],
            roll=float(state.rpy_deg()[0]),
            pitch=float(state.rpy_deg()[1]),
            support_margin=float(state.support_margin),
            base_x=float(state.base_pos[0]),
            base_y=float(state.base_pos[1]),
        )
        if renderer is not None:
            camera.lookat[:] = env.rt.data.qpos[:3]
            renderer.update_scene(env.rt.data, camera=camera)
            writer.write(renderer.render()[:, :, ::-1])
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
        # 该环境下显式 close() 会有概率在 GL/EGL 上下文回收时 segfault（exit 139，
        # 本文件与 scripts/demo.py 均可复现，且发生在结果落盘之后）。
        # 这里只释放引用，交给进程退出回收窗口，避免把整条演示脚本打断。
        viewer = None
    logger.close()

    metrics = dict(info["metrics"])
    extra = {
        "load_mass_kg": args.carry_mass if args.scenario == "carry_box" else 0.0,
        "load_mass_ratio": (args.carry_mass / 33.34) if args.scenario == "carry_box" else 0.0,
        "load_height_final_m": load.height if args.scenario == "carry_box" else 0.0,
        "load_horizontal_arm_m": load.horizontal_arm if args.scenario == "carry_box" else 0.0,
        "pillar_count": len(pillar_positions),
        "pillar_min_distance_m": navigator.min_distance if circling else 0.0,
        "pillar_passed": bool(navigator.passed) if circling else False,
        # 判定必须覆盖每一根柱子：缺记录（没走到）也算未通过
        "pillar_passed_all": (
            circling
            and len(navigator.pass_records) == len(pillar_positions)
            and all(item["ok"] > 0.5 for item in navigator.pass_records)
        ),
        "pillar_records": navigator.pass_records if circling else [],
        "stop_distance_m": (stop_watch.get("stop_end", 0.0) - stop_watch.get("stop_start", 0.0)) if stop_watch else 0.0,
    }
    summary = {
        "run_id": run_id,
        "scenario": args.scenario,
        "steps": step,
        "render": args.render,
        "policy": "teacher_baseline",
        "blocked_by_safety": blocked,
        "final_skill": info["skill"],
        "metrics": {**metrics, **extra},
        "wall_s": round(time.time() - start, 2),
        "video": str(out_dir / "demo.mp4") if writer is not None else None,
        "log": str(out_dir / "demo.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

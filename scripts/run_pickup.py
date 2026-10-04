#!/usr/bin/env python3
"""一条命令运行完整「弯腰 → 自主弓步 → 伸手 → 抓取 → 起身 → 恢复」仿真 Demo。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_pickup.ablation import apply_ablation
from cb_pickup.baselines import make_decision_model
from cb_pickup.controller import PickupConfig, PickupController
from cb_pickup.experiment import ExperimentLogger
from cb_pickup.scenarios import sample_scenario


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="front")
    parser.add_argument("--baseline", default="C", choices=["A", "B", "C", "D"])
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", choices=["none", "video", "viewer"], default="none")
    parser.add_argument("--record", action="store_true", help="viewer 模式同时录制 MP4（用于留档）")
    parser.add_argument("--max-time", type=float, default=35.0)
    parser.add_argument("--out", default="experiments")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--pickup-config", default="configs/g1_pickup.yaml")
    args = parser.parse_args()
    system_cfg = load_config(args.system_config)
    pickup_cfg = load_config(args.pickup_config)
    scene = str(pickup_cfg.get("scene_path", system_cfg.get("paths.menagerie_scene")))
    decision_model, _overrides = make_decision_model(args.baseline)
    controller = PickupController(
        PickupConfig.from_mapping(pickup_cfg, scene_path=scene, max_episode_s=args.max_time),
        decision_model=decision_model,
    )
    if args.ablation:
        apply_ablation(controller, args.ablation)
    rng = np.random.default_rng(args.seed)
    scenario = sample_scenario(args.scenario, rng)
    logger = ExperimentLogger(
        root=args.out,
        tag=f"pickup_{args.scenario}_baseline{args.baseline}",
        config={"scenario": scenario.__dict__, "baseline": args.baseline, "ablation": args.ablation, "render": args.render},
        seed=args.seed,
        scenario=scenario.name,
        versions={
            "feature_version": system_cfg.get("feature_version"),
            "model_version": system_cfg.get("model_version"),
            "questions_version": system_cfg.get("questions_version"),
        },
    )
    # 渲染装配
    renderer = None
    writer = None
    viewer = None
    if args.render == "video":
        import cv2
        import mujoco

        renderer = mujoco.Renderer(controller.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(controller.model, camera)
        camera.distance = 2.2
        camera.azimuth = 130
        camera.elevation = -18
        writer = cv2.VideoWriter(str(logger.run_dir / "video" / "pickup.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (640, 480))
    elif args.render == "viewer":
        import mujoco
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(controller.model, controller.data)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -18
        if args.record:
            import cv2

            renderer = mujoco.Renderer(controller.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(controller.model, camera)
            camera.distance = 2.2
            camera.azimuth = 130
            camera.elevation = -18
            writer = cv2.VideoWriter(
                str(logger.run_dir / "video" / "pickup.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                50.0,
                (640, 480),
            )
    controller.reset(scenario, seed=args.seed, randomize=False)
    start = time.time()
    last_phase = None
    while controller.time < args.max_time:
        info = controller.step()
        decision = controller.manager.last_decision
        logger.log_step(
            {
                "time": info.time,
                "phase": info.phase,
                "primitive": info.primitive,
                "stability_margin": info.stability_margin,
                "predicted_margin": info.balance.predicted_margin,
                "trunk_pitch": info.trunk_pitch,
                "trunk_roll": info.balance.trunk_roll,
                "hand_distance": float(np.linalg.norm(info.hand_position - (info.object_position + np.array([0, 0, 0.31])))),
                "object_height": float(info.object_position[2] + 0.31),
                "energy": info.energy,
                "steps": info.steps,
                "decision_latency_ms": info.decision_latency_ms,
                "com": info.com,
                "zmp": info.zmp,
                "left_foot": info.left_foot_pos,
                "right_foot": info.right_foot_pos,
                "hand_position": info.hand_position,
                "object_position": info.object_position,
                "support_polygon": info.support_polygon,
                "box_clearance": controller._last_box_clearance,
            },
            decision=None if decision is None else decision.to_dict(),
            event={"phase": info.phase} if info.phase != last_phase else None,
        )
        last_phase = info.phase
        if renderer is not None:
            camera.lookat[:] = controller.data.qpos[:3]
            renderer.update_scene(controller.data, camera=camera)
            writer.write(renderer.render()[:, :, ::-1])
        if viewer is not None:
            viewer.cam.lookat[:] = controller.data.qpos[:3]
            viewer.sync()
            time.sleep(max(0.0, controller.config.control_dt - 0.004))
        if info.success or info.fall:
            break
    if writer is not None:
        writer.release()
    if renderer is not None:
        renderer.close()
    if viewer is not None:
        viewer.close()
    metrics = {
        "success": bool(info.success),
        "fall": bool(info.fall),
        "time_s": info.time,
        "steps": info.steps,
        "min_stability_margin": float(min(logger.trajectories["stability_margin"])),
        "final_margin": info.stability_margin,
        "energy": info.energy,
        "grasped": info.grasped,
        "decision_count": len(logger.decisions),
        "wall_s": round(time.time() - start, 2),
    }
    run_dir = logger.finalize(metrics)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2, ensure_ascii=False))
    return 0 if info.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

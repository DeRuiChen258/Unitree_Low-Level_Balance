"""S3 retarget 单测：IK 收敛、关节限位、误差记录。"""

from __future__ import annotations

import numpy as np

from cb_common import load_config
from cb_data.canonicalize import canonicalize
from cb_data.retarget import G1IKSolver, IKConfig, retarget_clip
from tests.helpers import make_motion


def _scene() -> str:
    return str(load_config("configs/system.yaml").get("limits.mjcf_scene"))


def test_ik_accepts_reachable_targets() -> None:
    """默认站立目标 IK 误差应低于 6 cm。"""
    cfg = IKConfig(scene_path=_scene(), ik_iters=30, ik_tolerance_m=0.06)
    solver = G1IKSolver(cfg)
    # 用 G1 默认姿态生成目标
    q = np.zeros(29)
    targets = []
    for _ in range(4):
        solver.data.qpos[:] = 0.0
        solver.data.qpos[2] = 0.79
        solver.data.qpos[3] = 1.0
        solver.data.qpos[solver.qpos_addr] = q
        solver.mujoco.mj_forward(solver.model, solver.data)
        targets.append(np.array([solver.data.xpos[b] for b in solver.body_ids]))
    result = solver.solve(np.stack(targets[0]), base_pos=np.array([0.0, 0.0, 0.79]), base_quat=np.array([1.0, 0, 0, 0]), q_init=q)
    assert result[1] < 0.06


def test_retarget_clip_shapes() -> None:
    """重定向输出形状与误差记录。"""
    cfg = IKConfig(scene_path=_scene(), ik_iters=8)
    solver = G1IKSolver(cfg)
    clip = make_motion(frames=4)
    clip.joint_world_pos[:, :, 1] += 0.2
    canonical = canonicalize(clip, {"target_fps": 30.0})
    result = retarget_clip(canonical, cfg, solver=solver)
    assert result.qpos.shape == (4, 29)
    assert result.retarget_err.shape == (4,)
    assert result.joint_limit_hits.shape == (4,)
    assert np.isfinite(result.qpos).all()

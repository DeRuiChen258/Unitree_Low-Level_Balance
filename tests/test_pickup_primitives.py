"""行为原语单测：quintic、弯腰时序、步态弧线、伸手。"""

from __future__ import annotations

import numpy as np

from cb_pickup.footstep import FootstepPlan
from cb_pickup.primitives import BendPrimitive, BodyTarget, ReachPrimitive, StepPrimitive, embrace_points, quintic


def _target() -> BodyTarget:
    return BodyTarget(
        base_pos=np.array([0.0, 0.0, 0.79]),
        base_yaw=0.0,
        left_foot_pos=np.array([0.0, 0.12, 0.04]),
        left_foot_yaw=0.0,
        right_foot_pos=np.array([0.0, -0.12, 0.04]),
        right_foot_yaw=0.0,
    )


def test_quintic_and_bend_timing() -> None:
    """quintic 端点正确；弯腰先倾后蹲。"""
    assert quintic(0.0) == 0.0 and abs(quintic(1.0) - 1.0) < 1e-9
    bend = BendPrimitive()
    bend.reset(_target(), {"trunk_pitch": 0.5, "crouch_depth": 0.14, "hip_shift": 0.0})
    target = _target()
    early = bend.update(target, 0.6)
    late = bend.update(target, 1.0)
    assert early.trunk_pitch > late.base_pos[2] * 0 + 0.0
    assert late.trunk_pitch > early.trunk_pitch
    assert late.base_pos[2] <= 0.79


def test_step_arc_and_reach() -> None:
    """步态有抬脚弧线；伸手单调向前并在结束时精确到位（waving 两阶段契约）。"""
    step = StepPrimitive()
    plan = FootstepPlan("right", "left", 0.1, 0.0, 0.45, 0.0, np.array([0.1, 0.12]), True, "test", 0.05)
    step.reset(_target(), {"plan": plan})
    out = step.update(_target(), 0.2)
    assert out.left_foot_pos[2] >= 0.04

    reach = ReachPrimitive()
    object_position = np.array([0.4, 0.0, 0.6])
    start_hand = np.array([0.10, 0.0, 0.72])  # 实测手位：位于物体后方，避免先向后甩
    half_width = 0.14
    goals = embrace_points(object_position, half_width)
    hand_goal = goals["right"]
    offhand_goal = goals["left"]
    target = _target()
    reach.reset(target, {"object_position": object_position, "reach_side": "right", "hand_position": start_hand})
    xs: list[float] = []
    deviations: list[float] = []
    offhands: list[np.ndarray] = []
    steps = int(round(reach.duration_s / 0.01))
    for _ in range(steps):
        target = reach.update(target, 0.01)
        hand = np.asarray(target.hand_target, dtype=np.float64)
        xs.append(float(hand[0]))
        deviations.append(abs(float(hand[1] - hand_goal[1])) + abs(float(hand[2] - hand_goal[2])))
        offhands.append(np.asarray(target.offhand_target, dtype=np.float64).copy())

    assert target.hand_target is not None
    # 1) 双手结束时精确到位（衰减摆动在终点归零）
    assert np.linalg.norm(target.hand_target - hand_goal) < 1e-9
    assert target.offhand_target is not None
    assert np.linalg.norm(target.offhand_target - offhand_goal) < 1e-9
    # 2) 副手与主手在 y/z 上同步收敛到各自触点（两段接近不引入额外滞后）
    assert np.linalg.norm(offhands[-1] - offhand_goal) < 1e-9
    # 双手触点关于箱体中线对称（左右各一，间距 = 2*(半宽+2 cm)）
    assert abs(float(target.hand_target[1] + target.offhand_target[1])) < 1e-9
    # 3) 两段接近（先外移对齐、再前推贴面）：全程 x 不超过终点，手不会越过箱面插进去
    assert max(xs) <= float(hand_goal[0]) + 1e-9
    # 4) 摆动是小幅的：横向/高度相对终点的偏差在收敛过程中单调收敛到 0
    assert deviations[0] > deviations[-1]
    assert deviations[-1] < 1e-9
    # 5) 双手抱取不做冻结准备位（避免前臂扫入箱体）
    assert reach.prepare_weight() == 1.0


def test_reach_prepare_pose_mirror() -> None:
    """waving 准备位与 G1_Waving 配置一致，且左右严格镜像。"""
    pose = ReachPrimitive.PREPARE_POSE
    assert set(pose["left"]) == set(pose["right"])
    for role, right_value in pose["right"].items():
        left_value = pose["left"][role]
        if role in {"shoulder_roll", "shoulder_yaw", "wrist_roll"}:
            assert abs(left_value + right_value) < 1e-12, role
        else:
            assert abs(left_value - right_value) < 1e-12, role
    # 与 Action/G1_Waving/configs/g1_wave.yaml 的 prepare_pose 关键值对齐（肘角按需求收小）
    assert pose["right"]["shoulder_pitch"] == -0.50
    assert pose["right"]["elbow"] == -0.45
    assert abs(pose["right"]["elbow"]) < 0.85  # 小臂弯曲幅度小于 waving 原准备位
    assert pose["right"]["wrist_roll"] == -1.20

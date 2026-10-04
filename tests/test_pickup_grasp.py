"""Mock grasp 单测：接近触发、锁存、显式释放。"""

from __future__ import annotations

import numpy as np

from cb_pickup.grasp import GraspMock


def test_grasp_latch_and_release() -> None:
    """双手接近且请求抓取时锁存；release 后释放。"""
    grasp = GraspMock(grasp_distance_m=0.08)
    hand = np.array([0.4, 0.0, 0.6])
    obj = np.array([0.42, 0.0, 0.6])
    hands = (hand, hand + np.array([0.0, 0.32, 0.0]))
    objects = (obj, obj + np.array([0.0, 0.32, 0.0]))
    assert not grasp.update(hand_positions=hands, object_positions=objects, request_grasp=False, time_s=0.0)
    assert grasp.update(hand_positions=hands, object_positions=objects, request_grasp=True, time_s=0.1)
    # 抓取后即使距离变大也保持锁存
    assert grasp.update(
        hand_positions=(hand + 0.5, hand + 0.8),
        object_positions=objects,
        request_grasp=False,
        time_s=0.2,
    )
    grasp.release(time_s=1.0)
    assert not grasp.grasped
    assert len(grasp.events) >= 2


def test_grasp_requires_all_hands() -> None:
    """双手抱取：任一只手未到位都不允许抓取。"""
    grasp = GraspMock(grasp_distance_m=0.08)
    hand = np.array([0.4, 0.0, 0.6])
    objects = (np.array([0.42, 0.0, 0.6]), np.array([0.42, 0.32, 0.6]))
    assert not grasp.update(
        hand_positions=(hand, hand + np.array([0.0, 0.32, 0.0]) + np.array([0.4, 0.0, 0.0])),
        object_positions=objects,
        request_grasp=True,
        time_s=0.0,
    )
    assert not grasp.grasped
    assert grasp.update(
        hand_positions=(hand, hand + np.array([0.0, 0.32, 0.0])),
        object_positions=objects,
        request_grasp=True,
        time_s=0.1,
    )
    assert grasp.grasped

"""Mock grasp：当手接近物体时用 mocap/weld 式附着模拟抓起（第 8 节允许的研究占位）。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraspMock:
    """研究用抓取：接近 + 距离阈值 + 运动学附着（不依赖真实灵巧手模型）。"""

    grasp_distance_m: float = 0.07
    release_distance_m: float = 0.12
    grasped: bool = False
    attach_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.02]))
    events: list[dict[str, float]] = field(default_factory=list)

    def reset(self) -> None:
        """复位抓取状态。"""
        self.grasped = False
        self.events.clear()

    def update(
        self,
        *,
        hand_positions: Sequence[np.ndarray],
        object_positions: Sequence[np.ndarray],
        request_grasp: bool,
        time_s: float,
    ) -> bool:
        """返回是否处于抓取状态；所有手都在接近阈值内且请求抓取则附着（双手抱取）。"""
        pairs = list(zip(hand_positions, object_positions, strict=True))
        if not pairs:
            raise ValueError("grasp requires at least one hand/object pair")
        distances = [float(np.linalg.norm(np.asarray(hand) - np.asarray(obj))) for hand, obj in pairs]
        distance = max(distances)
        if not self.grasped and request_grasp and distance <= self.grasp_distance_m:
            self.grasped = True
            self.events.append({"time": time_s, "event": "grasp", "distance": distance, "hands": float(len(pairs))})
        # 抓取后锁存（研究用 mocap 附着）；释放由上层显式调用 reset()/release()
        return self.grasped

    def release(self, *, time_s: float = 0.0) -> None:
        """显式释放（由放置/任务逻辑调用）。"""
        if self.grasped:
            self.grasped = False
            self.events.append({"time": time_s, "event": "release", "distance": 0.0})

    def attached_object_pose(self, hand_position: np.ndarray, hand_quat: np.ndarray) -> np.ndarray:
        """返回附着时物体应处的位姿（qpos 7 维）。"""
        quat = np.asarray(hand_quat, dtype=np.float64)
        return np.concatenate([np.asarray(hand_position, dtype=np.float64) + self.attach_offset, quat])

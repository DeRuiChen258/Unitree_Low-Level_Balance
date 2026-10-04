"""拾取过程几何门禁：箱体与机器人全程不得穿模（最小间隙 > 0）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config  # noqa: E402
from cb_pickup.controller import PickupConfig, PickupController  # noqa: E402
from cb_pickup.scenarios import SCENARIOS  # noqa: E402


def _run_front_episode() -> tuple[PickupController, float]:
    """跑完 front 场景，返回控制器与全程最小箱体间隙。"""
    config = PickupConfig.from_mapping(load_config("configs/g1_pickup.yaml"))
    controller = PickupController(config)
    controller.reset(SCENARIOS["front"], seed=0, randomize=False)
    clearance = float("inf")
    for _ in range(1200):
        info = controller.step()
        clearance = min(clearance, controller._last_box_clearance)
        if info.phase in ("SUCCESS", "FAILURE"):
            break
    return controller, clearance


@pytest.mark.slow
def test_pickup_no_box_penetration() -> None:
    """双手抱取全程箱体与机器人保持正间隙，且任务成功、不摔倒。"""
    controller, clearance = _run_front_episode()
    assert clearance > 0.0, f"箱体穿模：最小间隙 {clearance:.4f} m"
    assert not controller.fall
    assert controller.success


@pytest.mark.slow
def test_carry_load_is_half_body_mass() -> None:
    """载荷质量约为机器人本体质量的一半，且手部目标在箱体表面之外。"""
    controller, _clearance = _run_front_episode()
    ratio = controller.object_mass / controller.robot_mass
    assert 0.45 <= ratio <= 0.55, f"载荷质量比 {ratio:.2f} 不在半体重范围"
    clearance = controller.config.embrace_clearance_m
    assert clearance >= 0.05, "腕部目标外移量过小，手掌会插入箱体"

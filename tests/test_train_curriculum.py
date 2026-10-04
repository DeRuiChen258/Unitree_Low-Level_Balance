"""课程单测：阶段推进、回退。"""

from __future__ import annotations

from cb_train.curriculum import Curriculum


def test_curriculum_advances_and_regresses() -> None:
    """通过指标推进；摔倒率过高回退。"""
    config = {
        "curriculum": {
            "stages": [
                {"name": "stand", "min_episodes": 2, "metrics": {"fall_rate": 0.05, "vx_err": 0.3}, "allowed_skills": ["stand"]},
                {"name": "walk", "min_episodes": 2, "metrics": {"fall_rate": 0.1}, "allowed_skills": ["stand", "walk"]},
            ],
            "regression_fall_rate": 0.2,
        }
    }
    curriculum = Curriculum.from_config(config)
    assert curriculum.stage.name == "stand"
    curriculum.update({"fall_rate": 0.0, "vx_err": 0.0})
    assert curriculum.update({"fall_rate": 0.0, "vx_err": 0.0})
    assert curriculum.stage.name == "walk"
    curriculum.update({"fall_rate": 0.5})
    curriculum.update({"fall_rate": 0.5})
    assert curriculum.index == 0

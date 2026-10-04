"""S1 ingest 单测：272 维布局解析、文本解析、缺失文件报错。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cb_common.errors import DataError
from cb_data.ingest import load_272d_clip, parse_272d, parse_text_file, skill_from_text


def fake_272d(frames: int = 6) -> np.ndarray:
    """构造合法 272 维向量（站立）。"""
    array = np.zeros((frames, 272))
    array[:, 2:8] = np.array([1.0, 0, 0, 0, 1.0, 0])
    array[:, 8:74] = array[:, 8:74].reshape(frames, 22, 3).reshape(frames, 66)
    positions = array[:, 8:74].reshape(frames, 22, 3)
    positions[:, :, 1] = 0.9
    positions[:, 10, 1] = 0.05
    positions[:, 11, 1] = 0.05
    array[:, 8:74] = positions.reshape(frames, 66)
    rotations = np.zeros((frames, 22, 6))
    rotations[:, :, 0] = 1.0
    rotations[:, :, 4] = 1.0
    array[:, 140:272] = rotations.reshape(frames, 132)
    return array


def test_parse_272d_shapes() -> None:
    """解析后的形状与接触判定。"""
    parsed = parse_272d(fake_272d())
    assert parsed["root_pos"].shape == (6, 3)
    assert parsed["joint_world_pos"].shape == (6, 22, 3)
    assert parsed["contacts"].shape == (6, 2)
    assert parsed["contacts"].all()


def test_parse_272d_rejects_wrong_dim() -> None:
    """维度不符必须拒绝。"""
    with pytest.raises(DataError):
        parse_272d(np.zeros((4, 263)))


def test_load_clip_from_fake_dataset(tmp_path: Path) -> None:
    """从最小数据集目录加载片段与文本。"""
    root = tmp_path
    (root / "motion_data").mkdir()
    (root / "texts").mkdir()
    (root / "split").mkdir()
    np.save(root / "motion_data" / "000001.npy", fake_272d())
    (root / "texts" / "000001.txt").write_text("a person is walking forward#a/DET person/NOUN walk/VERB#0.0#1.0\n", encoding="utf-8")
    (root / "split" / "train.txt").write_text("000001\n", encoding="utf-8")
    clip = load_272d_clip(root, "000001")
    assert clip.frames == 6
    assert clip.text == "a person is walking forward"
    assert clip.meta["upright"] in (True, False)


def test_text_and_skill_rules(tmp_path: Path) -> None:
    """文本解析与关键词分桶。"""
    path = tmp_path / "x.txt"
    path.write_text("a man waves his right hand#a/DET man/NOUN#0.0#1.0\n", encoding="utf-8")
    assert parse_text_file(path).startswith("a man waves")
    rules = {"wave": ["wave"], "run": ["run"]}
    assert skill_from_text("a man waves his right hand", rules) == "wave"
    assert skill_from_text("nothing matches", rules) == "stand"
    with pytest.raises(DataError):
        parse_text_file(tmp_path / "missing.txt")

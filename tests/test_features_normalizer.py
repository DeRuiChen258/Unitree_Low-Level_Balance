"""归一化单测：running stats、裁剪命中率、序列化。"""

from __future__ import annotations

import numpy as np

from cb_features.normalizer import RunningMeanStd


def test_running_stats_converge() -> None:
    """统计量收敛到批次均值/方差。"""
    normalizer = RunningMeanStd((4,))
    batch = np.random.default_rng(0).normal(2.0, 0.5, size=(1000, 4))
    normalizer.update(batch)
    assert np.allclose(normalizer.mean, batch.mean(axis=0), atol=0.05)
    normalized = normalizer.normalize(batch, clip=3.0)
    assert normalized.shape == batch.shape
    assert normalizer.clip_ratio() >= 0.0


def test_save_load(tmp_path) -> None:
    """npz 保存/恢复保持统计量。"""
    normalizer = RunningMeanStd((3,))
    normalizer.update(np.ones((10, 3)))
    path = tmp_path / "stats.npz"
    normalizer.save(path)
    restored = RunningMeanStd.load(path)
    assert np.allclose(restored.mean, normalizer.mean)

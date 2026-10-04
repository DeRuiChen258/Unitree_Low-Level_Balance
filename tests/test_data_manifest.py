"""S8 manifest 单测：hash、版本、文件校验。"""

from __future__ import annotations

from pathlib import Path

from cb_data.manifest import build_manifest
from cb_data.split import split_segments
from tests.helpers import make_segment


def test_manifest_hash_and_verify(tmp_path: Path) -> None:
    """manifest 记录文件 hash，缺失/篡改可检测。"""
    segments = [make_segment(frames=20)]
    splits = split_segments(segments, ratios={"train": 0.8, "val": 0.1, "test": 0.1})
    artifact = tmp_path / "segments" / "a.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"hello")
    manifest = build_manifest(
        dataset_version="v-test",
        root=tmp_path,
        segments=segments,
        transitions=[],
        splits=splits,
        profile="smoke",
        aug_profile="{}",
        feature_version="1.0",
        obs_spec_hash="abc",
    )
    manifest.save(tmp_path / "manifest.json")
    assert not manifest.verify_files(tmp_path)
    artifact.write_bytes(b"tampered")
    assert manifest.verify_files(tmp_path)

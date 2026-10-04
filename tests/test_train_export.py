"""ONNX 导出单测：PyTorch↔ONNX 误差 ≤ 1e-5 与 metadata。"""

from __future__ import annotations

import pytest
import torch

from cb_features.obs_spec import default_obs_spec
from cb_train.export_onnx import export_checkpoint
from cb_train.networks import StudentPolicy


@pytest.mark.slow
def test_export_student_consistency(tmp_path) -> None:
    """导出学生网络并校验一致性。"""
    spec = default_obs_spec(4, 16)
    student = StudentPolicy(spec.total_dim, 29, (32, 32))
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "obs_dim": spec.total_dim,
            "action_dim": 29,
            "hidden_sizes": [32, 32],
            "metadata": {"obs_spec_hash": spec.hash(), "feature_version": "1.0", "action_kind": "absolute_delta"},
        },
        checkpoint,
    )
    result = export_checkpoint(checkpoint, tmp_path / "student.onnx")
    assert result.max_abs_diff <= 1e-5
    assert result.metadata["obs_spec_hash"] == spec.hash()

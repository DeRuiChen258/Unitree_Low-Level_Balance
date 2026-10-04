"""服务 API 单测：health/version/单条/批量/异常。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from cb_serving.api import LocalCalibratedBackend, create_app

STATE = {
    "state_version": "1.0",
    "robot": "unitree_g1_29dof",
    "balance": {"roll_deg": 1.0, "pitch_deg": 0.5, "com_margin_m": 0.05},
    "motion": {"vx": 1.0, "skill": "run"},
    "candidates": ["continue_current", "stand", "safe_stop"],
}


def test_api_endpoints() -> None:
    """四个端点语义正确，非法请求 4xx。"""
    client = TestClient(create_app("configs/serving.yaml", backend=LocalCalibratedBackend()))
    health = client.get("/v1/health")
    assert health.status_code == 200 and health.json()["model_loaded"] is True
    version = client.get("/v1/version")
    assert version.status_code == 200
    assert version.json()["questions_version"] == "1.0"
    single = client.post("/v1/decision", json={"state": STATE})
    assert single.status_code == 200
    body = single.json()
    assert "answers" in body and "meta" in body
    assert body["meta"]["questions_hash"]
    batch = client.post("/v1/decision:batch", json={"items": [{"state": STATE}, {"state": STATE}]})
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 2
    assert client.post("/v1/decision", json={}).status_code == 422


def test_api_fail_closed_on_dangerous_state() -> None:
    """危险状态返回 veto=true。"""
    client = TestClient(create_app("configs/serving.yaml", backend=LocalCalibratedBackend()))
    dangerous = {"state_version": "1.0", "balance": {"roll_deg": 20.0, "pitch_deg": 0.0, "com_margin_m": -0.2}}
    response = client.post("/v1/decision", json={"state": dangerous})
    assert response.status_code == 200
    assert response.json()["answers"]["safety_veto_head"]["noul"] == 1.0

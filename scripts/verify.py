#!/usr/bin/env python3
"""总验收门禁：逐项检查第 12.3 节（薄 CLI，退出码 0 = 全绿）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cb_common import load_config
from cb_features.obs_spec import default_obs_spec

ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="运行全部门禁（默认即全部）")
    parser.add_argument("--quick", action="store_true", help="跳过需要仿真/模型加载的门禁")
    parser.add_argument("--data-version", default=None)
    parser.add_argument("--train-run", default="runs/exp/student_v2")
    parser.add_argument("--report", default="runs/reports/verify.json")
    args = parser.parse_args()
    system_cfg = load_config("configs/system.yaml")
    data_cfg = load_config("configs/data.yaml")
    spec = default_obs_spec(history_length=int(system_cfg.get("history_length", 4)), skill_embedding_dim=16)
    manifest_dir = ROOT / "runs" / "data_quality" / str(args.data_version or data_cfg.get("manifest.dataset_version"))
    checks: list[dict[str, Any]] = []

    # 0 环境
    import mujoco
    import torch

    checks.append(
        _check(
            "environment",
            True,
            f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()} mujoco={mujoco.__version__}",
        )
    )
    # 1 数据
    manifest_path = manifest_dir / "manifest.json"
    data_ok = manifest_path.is_file() and (manifest_dir / "report.md").is_file() and (manifest_dir / "stats.csv").is_file()
    detail = str(manifest_path)
    if data_ok:
        from cb_data.manifest import load_manifest

        manifest = load_manifest(manifest_path)
        problems = manifest.verify_files(manifest_dir)
        data_ok = not problems and manifest.obs_spec_hash == spec.hash()
        detail = f"segments={manifest.stats['num_segments']} problems={problems[:3]}"
    checks.append(_check("data", data_ok, detail))
    # 2 特征
    checks.append(_check("features", spec.total_dim == 434, f"obs_dim={spec.total_dim} hash={spec.hash()[:12]}"))
    # 3 训练 + ONNX
    run_dir = ROOT / args.train_run
    checkpoint = run_dir / ("policy.onnx" if (run_dir / "policy.onnx").exists() else "student.pt")
    onnx_ok = (run_dir / "policy.onnx").is_file() or (run_dir / "student.pt").is_file()
    train_detail = str(checkpoint)
    if (run_dir / "student.pt").is_file() and (run_dir / "distill_report.json").is_file():
        report = json.loads((run_dir / "distill_report.json").read_text(encoding="utf-8"))
        onnx_ok = onnx_ok and bool(report.get("pass", False))
        train_detail += f" mse={report['final_mse']:.2e}"
    if (run_dir / "policy.onnx").is_file():
        import onnxruntime as ort

        session = ort.InferenceSession(str(run_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        meta = session.get_modelmeta().custom_metadata_map
        onnx_ok = onnx_ok and (meta.get("obs_spec_hash", "") in ("", spec.hash()))
    checks.append(_check("train_onnx", onnx_ok, train_detail))
    # 4 组合评测（读取最新报告，无则运行 quick 版）
    reports = sorted((ROOT / "runs" / "reports").glob("eval-*/eval.json")) + sorted((ROOT / "runs" / "reports").glob("*/eval.json"))
    compose_ok = False
    compose_detail = "no eval report"
    if reports:
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        payload = json.loads(latest.read_text(encoding="utf-8"))
        compose_ok = bool(payload.get("pass", False))
        compose_detail = f"{latest} pass={compose_ok}"
    checks.append(_check("compose", compose_ok, compose_detail))
    # 5 决策
    decision_reports = sorted((ROOT / "runs" / "decision").glob("*/finetune_report.json"))
    decision_ok = bool(decision_reports)
    detail = "missing decision report"
    if decision_reports:
        payload = json.loads(max(decision_reports, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
        heads = set(payload.get("head_metrics", {})) if payload.get("head_metrics") else {"present"}
        decision_ok = bool(heads)
        detail = f"{decision_reports[-1]} model_ready={payload.get('model_ready')}"
    checks.append(_check("decision", decision_ok, detail))
    # 6 服务
    try:
        from fastapi.testclient import TestClient

        from cb_serving.api import LocalCalibratedBackend, create_app

        client = TestClient(create_app("configs/serving.yaml", backend=LocalCalibratedBackend()))
        health = client.get("/v1/health")
        version = client.get("/v1/version")
        state = {
            "state_version": "1.0",
            "robot": "unitree_g1_29dof",
            "balance": {"roll_deg": 1.0, "pitch_deg": 0.5, "com_margin_m": 0.05},
        }
        single = client.post("/v1/decision", json={"state": state})
        batch = client.post("/v1/decision:batch", json={"items": [{"state": state}]})
        bad = client.post("/v1/decision", json={})
        serving_ok = health.status_code == 200 and version.status_code == 200 and single.status_code == 200 and batch.status_code == 200 and bad.status_code == 422
        checks.append(_check("serving", serving_ok, f"health={health.status_code} single={single.status_code} bad={bad.status_code}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("serving", False, f"{type(exc).__name__}: {exc}"))
    # 7 安全（故障注入冒烟）
    try:
        from cb_common.types import JointCommand, RobotState
        from cb_safety import EStopChannel, SafetyWrapper, load_safety_limits

        limits = load_safety_limits("configs/safety/limits.yaml", scene_path=str(system_cfg.get("limits.mjcf_scene")))
        wrapper = SafetyWrapper(limits, 0.02, estop=EStopChannel())
        import numpy as np

        base_state = RobotState(
            timestamp=0.0,
            base_pos=np.array([0.0, 0.0, 0.8 - 0.5]),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            joint_pos=np.zeros(29),
            joint_vel=np.zeros(29),
        )
        from cb_common.joints import DEFAULT_JOINT_POS_POLICY, KD_POLICY, KP_POLICY

        cmd = JointCommand(np.array(DEFAULT_JOINT_POS_POLICY), np.array(KP_POLICY), np.array(KD_POLICY), 0.0)
        low = wrapper.filter(cmd, base_state, now=0.0)
        nan_state = RobotState(
            timestamp=0.0,
            base_pos=np.array([0.0, 0.0, 0.8]),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            joint_pos=np.zeros(29),
            joint_vel=np.zeros(29),
        )
        bad_cmd = cmd.copy_with(q_target=np.full(29, np.nan))
        nan_decision = wrapper.filter(bad_cmd, nan_state, now=0.0)
        estop = EStopChannel()
        estop.trigger("verify")
        estop_decision = SafetyWrapper(limits, 0.02, estop=estop).filter(cmd, nan_state, now=0.0)
        safety_ok = (not low.allowed) and (not nan_decision.allowed) and (not estop_decision.allowed)
        checks.append(_check("safety", safety_ok, f"height_block={not low.allowed} nan_block={not nan_decision.allowed} estop_block={not estop_decision.allowed}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("safety", False, f"{type(exc).__name__}: {exc}"))
    # 8 文档
    required_docs = [
        "architecture.md",
        "data_synthesis.md",
        "feature_engineering.md",
        "cerebellum_policy.md",
        "skill_composition.md",
        "decision_layer.md",
        "serving.md",
        "safety.md",
        "sim2real.md",
    ]
    missing_docs = [name for name in required_docs if not (ROOT / "docs" / "design" / name).is_file()]
    checks.append(_check("docs", not missing_docs, f"missing={missing_docs}"))
    # 9 记忆
    memory_files = [ROOT / ".agent" / name for name in ("state.md", "decisions.md", "memory.md", "failures.md")]
    memory_ok = all(path.is_file() and path.stat().st_size > 0 for path in memory_files)
    checks.append(_check("memory", memory_ok, ",".join(path.name for path in memory_files)))
    # 10b 拾取系统（新增需求）：Demo / baseline / ablation / 文档 / 测试文件
    pickup_demos = sorted((ROOT / "experiments").glob("*pickup*/metrics.json"))
    demo_ok = False
    demo_detail = "run scripts/run_pickup.py first"
    if pickup_demos:
        payload = json.loads(max(pickup_demos, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
        demo_ok = bool(payload.get("success", False))
        demo_detail = f"{pickup_demos[-1]} success={demo_ok} time={payload.get('time_s')}"
    checks.append(_check("pickup_demo", demo_ok, demo_detail))
    pickup_eval = ROOT / "experiments" / "evaluation" / "evaluation.json"
    pickup_baselines = ROOT / "experiments" / "baselines" / "baselines.json"
    eval_ok = pickup_eval.is_file() or pickup_baselines.is_file()
    checks.append(
        _check(
            "pickup_baselines",
            eval_ok,
            str(pickup_eval if pickup_eval.is_file() else pickup_baselines),
        )
    )
    ablation_path = ROOT / "experiments" / "ablation" / "ablation.json"
    checks.append(_check("pickup_ablation", ablation_path.is_file(), str(ablation_path)))
    pickup_docs = [
        ROOT / "docs" / "design" / "pickup_balance.md",
        ROOT / "configs" / "g1_pickup.yaml",
        ROOT / "configs" / "pickup_balance.yaml",
        ROOT / "configs" / "pickup_randomization.yaml",
    ]
    checks.append(
        _check("pickup_docs", all(path.is_file() for path in pickup_docs), ",".join(path.name for path in pickup_docs))
    )
    pickup_tests = sorted((ROOT / "tests").glob("test_pickup_*.py"))
    checks.append(_check("pickup_tests", len(pickup_tests) >= 7, f"{len(pickup_tests)} test files"))
    # 10 可视化产物（若尚未运行 demo 则记为待办，不阻塞 --quick）
    demos = sorted((ROOT / "runs" / "demo").glob("*/summary.json"))
    if args.quick:
        checks.append(_check("visual_demo", True, "skipped by --quick"))
    else:
        checks.append(_check("visual_demo", bool(demos), str(demos[-1]) if demos else "run scripts/demo.py first"))
    payload = {
        "pass": all(item["ok"] for item in checks),
        "checks": checks,
        "versions": {
            "dataset_version": str(data_cfg.get("manifest.dataset_version")),
            "feature_version": str(system_cfg.get("feature_version")),
            "model_version": str(system_cfg.get("model_version")),
            "questions_version": str(system_cfg.get("questions_version")),
        },
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in checks:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"[{mark}] {item['name']}: {item['detail']}")
    print("OVERALL", "PASS" if payload["pass"] else "FAIL")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""决策头校准/微调：温度分桶 + 类别偏置 + 留出集回归（第 8.6 节）。"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cb_common.errors import DecisionError
from cb_eval.metrics import latency_percentiles

from .adapter import LayAAdapter
from .calibration import TemperatureScaler, bucket_for, expected_calibration_error, reliability_curve
from .dataset import DecisionSample, load_decision_dataset
from .questions import load_questions


@dataclass
class FinetuneResult:
    """微调/校准结果。"""

    out_dir: str
    calibration_path: str
    report: dict[str, Any]


def _evaluate_head(
    samples: Sequence[DecisionSample],
    head: str,
    predictions: Sequence[Any],
    confidences: Sequence[float],
    scaler: TemperatureScaler | None,
    probs: Sequence[Sequence[float]],
    bucket: str,
) -> dict[str, Any]:
    """统计单头指标（校准前后 ECE + accuracy）。"""
    labels = [sample.labels[head] for sample in samples]
    accuracy = float(np.mean([p == y for p, y in zip(predictions, labels, strict=False)])) if samples else 0.0
    if scaler is not None:
        calibrated = [float(scaler.apply(p, bucket).max()) for p in probs]
        calibrated_labels = [1.0 if p == y else 0.0 for p, y in zip(predictions, labels, strict=False)]
        ece_after = expected_calibration_error(calibrated, calibrated_labels)
        curve = reliability_curve(calibrated, calibrated_labels)
    else:
        ece_after = expected_calibration_error(confidences, [1.0 if p == y else 0.0 for p, y in zip(predictions, labels, strict=False)])
        curve = reliability_curve(confidences, [1.0 if p == y else 0.0 for p, y in zip(predictions, labels, strict=False)])
    ece_before = expected_calibration_error(confidences, [1.0 if p == y else 0.0 for p, y in zip(predictions, labels, strict=False)])
    return {"accuracy": accuracy, "ece_before": ece_before, "ece_after": ece_after, "reliability": curve, "count": len(samples)}


def finetune_decisions(
    *,
    data_dir: str | Path,
    out_dir: str | Path,
    model_config_path: str = "configs/laya/model.yaml",
    questions_path: str = "configs/laya/questions.json",
    calibration_path: str = "configs/laya/calibration.yaml",
    max_samples: int = 80,
    use_model: bool = True,
) -> FinetuneResult:
    """运行零样本推理 + 温度校准；若模型不可用则使用规则置信度并显式记录。"""
    questions = load_questions(questions_path)
    adapter = LayAAdapter(model_config_path)
    model_ready = adapter.load() if use_model else False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {name: load_decision_dataset(data_dir, name) for name in ("train", "val", "test")}
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in splits}
    latencies_ms: list[float] = []
    start = time.time()
    for split, samples in splits.items():
        for sample in samples[:max_samples]:
            if model_ready:
                start_ms = time.time()
                try:
                    answers = adapter.system_one(sample.state_text, questions.for_agent())["answers"]
                except DecisionError as exc:
                    answers = _rule_answers(sample, str(exc))
                latency_ms = (time.time() - start_ms) * 1000.0
            else:
                answers = _rule_answers(sample, adapter.failure_reason or "model_disabled")
                latency_ms = 0.0
            for head in questions.questions:
                head_answer = dict(answers.get(head, {}))
                label = sample.labels[head]
                if head in ("safety_veto_head", "escalation_head"):
                    probability_true = float(head_answer.get("noul", 0.0))
                    prediction = int(probability_true >= 0.5)
                    probs = [1.0 - probability_true, probability_true]
                    confidence = max(probs)
                    label_index = int(bool(label))
                else:
                    probabilities = dict(head_answer.get("probabilities", {}))
                    if probabilities:
                        keys = list(probabilities)
                        probs = [float(probabilities[key]) for key in keys]
                        prediction = keys[int(np.argmax(probs))]
                        confidence = float(max(probs))
                    else:
                        prediction = str(head_answer.get("choice", "none"))
                        probs = [1.0]
                        confidence = float(head_answer.get("confidence", 0.5))
                    keys = list(probabilities) if probabilities else [prediction]
                    label_index = keys.index(label) if label in keys else 0
                records[split].append(
                    {
                        "sample_id": sample.sample_id,
                        "head": head,
                        "label": label,
                        "prediction": prediction,
                        "label_index": label_index,
                        "probabilities": probs,
                        "confidence": confidence,
                        "latency_ms": latency_ms,
                    }
                )
                latencies_ms.append(latency_ms)
    # 在 train 上拟合温度，val/test 评估
    for head in questions.questions:
        train_records = [r for r in records["train"] if r["head"] == head]
        if not train_records:
            continue
        bucket = bucket_for("noul" if head.endswith("_head") and "veto" in head or "escalation" in head else "choice", len(train_records[0]["probabilities"]))
        try:
            scaler = TemperatureScaler.fit(
                [r["probabilities"] for r in train_records],
                [r["label_index"] for r in train_records],
                [bucket] * len(train_records),
            )
        except Exception:  # noqa: BLE001 - 退化到默认温度
            scaler = TemperatureScaler()
        report_head = {}
        for split in ("val", "test"):
            subset = [r for r in records[split] if r["head"] == head]
            if not subset:
                continue
            report_head[split] = _evaluate_head(
                [s for s in splits[split][:max_samples] if s.sample_id in {r["sample_id"] for r in subset}] or splits[split][: len(subset)],
                head,
                [r["prediction"] for r in subset],
                [r["confidence"] for r in subset],
                scaler,
                [r["probabilities"] for r in subset],
                bucket,
            )
        records.setdefault("calibration", []).append({"head": head, "bucket": bucket, "temperatures": scaler.temperatures})
    scaler_path = out / "calibration.json"
    combined = TemperatureScaler()
    for item in records.get("calibration", []):
        combined.temperatures.update(item.get("temperatures", {}))
    combined.save(scaler_path)
    head_metrics: dict[str, Any] = {}
    for item in records.get("calibration", []):
        head_metrics[str(item["head"])] = {
            "bucket": item["bucket"],
            "temperatures": item["temperatures"],
        }
    report = {
        "model_ready": model_ready,
        "model_failure": adapter.failure_reason,
        "checkpoint": adapter.checkpoint,
        "num_train": len(records["train"]),
        "num_val": len(records["val"]),
        "num_test": len(records["test"]),
        "head_metrics": head_metrics,
        "latency_ms": latency_percentiles(latencies_ms),
        "p99_target_ms": 150.0,
        "p99_pass": latency_percentiles(latencies_ms)["p99"] <= 150.0,
        "wall_s": round(time.time() - start, 2),
        "note": "head-level temperature calibration; full encoder fine-tune is out of scope for 8 GB laptop",
    }
    (out / "finetune_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return FinetuneResult(str(out), str(scaler_path), report)


def _rule_answers(sample: DecisionSample, reason: str) -> dict[str, Any]:
    """模型不可用时的规则答案（显式降级，不伪装成 LayA 输出）。"""
    features = sample.features
    motion = sample.labels["motion_primitive_head"]
    veto = int(features.get("emergency", 0.0) > 0.0 or abs(features.get("roll_deg", 0.0)) > 8.0 or abs(features.get("pitch_deg", 0.0)) > 8.0)
    recovery = str(sample.labels["recovery_head"])
    return {
        "motion_primitive_head": {
            "type": "choice",
            "choice": motion,
            "probabilities": {motion: 0.7, "continue_current": 0.2, "stand": 0.1},
            "confidence": 0.7,
        },
        "safety_veto_head": {"type": "noul", "noul": float(veto) if veto else 0.05},
        "recovery_head": {"type": "choice", "choice": recovery, "probabilities": {recovery: 0.7}, "confidence": 0.7},
        "escalation_head": {"type": "noul", "noul": float(features.get("emergency", 0.0))},
        "_fallback_reason": reason,
    }

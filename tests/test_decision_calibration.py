"""校准单测：温度拟合降低 NLL、输出归一化、ECE。"""

from __future__ import annotations

import numpy as np

from cb_decision.calibration import TemperatureScaler, bucket_for, expected_calibration_error


def test_temperature_fit_reduces_nll() -> None:
    """过度自信的概率经拟合后 NLL 下降。"""
    rng = np.random.default_rng(0)
    predicted = rng.integers(0, 2, size=400)
    labels = np.where(rng.random(400) < 0.2, 1 - predicted, predicted)
    probabilities = []
    for prediction in predicted:
        probabilities.append(np.array([0.02, 0.98]) if prediction == 1 else np.array([0.98, 0.02]))
    probabilities = np.array(probabilities)
    scaler = TemperatureScaler.fit(probabilities, labels, ["noul:2"] * len(labels))
    calibrated = np.stack([scaler.apply(p, "noul:2") for p in probabilities])
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert scaler.temperatures["noul:2"] > 1.0


def test_bucket_and_ece() -> None:
    """温度桶规则与 ECE 边界。"""
    assert bucket_for("choice", 2) == "choice:2"
    assert bucket_for("choice", 4) == "choice:3-5"
    assert bucket_for("noul", 2) == "noul:2"
    ece = expected_calibration_error([0.9, 0.9], [1, 0])
    assert ece > 0.0

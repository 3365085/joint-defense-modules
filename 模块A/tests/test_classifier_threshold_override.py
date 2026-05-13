"""Classifier threshold override — operators can tighten/loosen the A4
classifier without retraining the artifact."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from defense.module_a.fusion.classifier_fusion import TorchLogisticFusion


def _tiny_logistic_artifact(tmp_path: Path, threshold: float = 0.5) -> Path:
    """Write a deterministic 3-feature logistic-regression artifact."""
    artifact = {
        "kind": "torch_logistic_regression",
        "feature_names": ["x1", "x2", "x3"],
        "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "weights": [1.0, 1.0, 1.0],
        "bias": 0.0,
        "threshold": threshold,
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(artifact))
    return path


def test_default_threshold_matches_artifact(cuda_device: str, tmp_path: Path) -> None:
    artifact = _tiny_logistic_artifact(tmp_path, threshold=0.5)
    clf = TorchLogisticFusion(artifact, cuda_device)
    assert clf.threshold == 0.5
    assert clf.threshold_artifact == 0.5
    # x1+x2+x3 = 0 → sigmoid = 0.5 → just above threshold.
    out = clf.compute({"x1": 0.0, "x2": 0.0, "x3": 0.0})
    assert out["classifier_p_adv"] == pytest.approx(0.5, abs=1e-5)
    assert out["classifier_threshold"] == 0.5
    assert out["classifier_threshold_overridden"] is False


def test_override_tightens_threshold(cuda_device: str, tmp_path: Path) -> None:
    artifact = _tiny_logistic_artifact(tmp_path, threshold=0.5)
    clf = TorchLogisticFusion(artifact, cuda_device, threshold_override=0.95)
    assert clf.threshold == 0.95
    assert clf.threshold_artifact == 0.5
    out = clf.compute({"x1": 0.0, "x2": 0.0, "x3": 0.0})
    # Same sigmoid 0.5 now below 0.95 → not triggered.
    assert out["classifier_triggered"] is False
    assert out["classifier_threshold_overridden"] is True


def test_override_loosens_threshold(cuda_device: str, tmp_path: Path) -> None:
    artifact = _tiny_logistic_artifact(tmp_path, threshold=0.9)
    clf = TorchLogisticFusion(artifact, cuda_device, threshold_override=0.1)
    out = clf.compute({"x1": 0.0, "x2": 0.0, "x3": 0.0})
    # Sigmoid 0.5 above override 0.1 → triggered.
    assert out["classifier_triggered"] is True
    assert out["classifier_threshold_overridden"] is True

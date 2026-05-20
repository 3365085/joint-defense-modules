from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..calibration import FeatureCalibration


class TorchLogisticFusion:
    """Tiny torch fusion classifier exported by tools/train_module_a_classifier.py."""

    def __init__(
        self,
        artifact_path: str | Path,
        device: str,
        calibration_model: str | None = None,
        threshold_override: float | None = None,
    ):
        self.artifact_path = Path(artifact_path)
        with self.artifact_path.open("r", encoding="utf-8") as fp:
            artifact = json.load(fp)
        self.kind = str(artifact.get("kind", "torch_logistic_regression"))
        self.feature_names = [str(name) for name in artifact["feature_names"]]
        normalization = artifact.get("normalization") or artifact
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=device)
        self.std = torch.tensor(normalization["std"], dtype=torch.float32, device=device)
        self.device = self.mean.device
        self.calibration: FeatureCalibration | None = None
        if self.kind.startswith("universal_calibrated_"):
            self.calibration = FeatureCalibration(
                feature_names=self.feature_names,
                artifact=artifact,
                calibration_model=calibration_model,
            )

        if "weights1" in artifact:
            self.weights1 = torch.tensor(artifact["weights1"], dtype=torch.float32, device=device)
            self.bias1 = torch.tensor(artifact["bias1"], dtype=torch.float32, device=device)
            self.weights2 = torch.tensor(artifact["weights2"], dtype=torch.float32, device=device)
            self.bias2 = torch.tensor(float(artifact["bias2"]), dtype=torch.float32, device=device)
            self.is_mlp = True
        else:
            self.weights = torch.tensor(artifact["weights"], dtype=torch.float32, device=device)
            self.bias = torch.tensor(float(artifact["bias"]), dtype=torch.float32, device=device)
            self.is_mlp = False
        # Threshold resolution:
        #   1. ``threshold_override`` (set by ModuleADetector config) wins.
        #   2. Artifact-embedded ``threshold`` is the default.
        # The override is the cheap, no-retrain knob for operators who want
        # to trade detection rate against false-alarm rate without touching
        # the classifier artifact. Only consumed by ``compute`` — the
        # original artifact value is preserved on ``self.threshold_artifact``
        # for diagnostics.
        self.threshold_artifact = float(artifact["threshold"])
        self.threshold = (
            float(threshold_override)
            if threshold_override is not None
            else self.threshold_artifact
        )
        self.transform_mode = (
            self.calibration.transform_mode if self.calibration is not None else "raw"
        )
        self.calibration_model = (
            self.calibration.model_name if self.calibration is not None else calibration_model
        )

    def compute(self, features: dict[str, float]) -> dict[str, Any]:
        if self.calibration is None:
            feature_values = [float(features.get(name, 0.0)) for name in self.feature_names]
        else:
            feature_values = self.calibration.transform(features)
        values = torch.tensor(
            feature_values,
            dtype=torch.float32,
            device=self.device,
        )
        if values.numel() != self.mean.numel():
            raise ValueError(
                f"Classifier feature length mismatch: got {values.numel()}, "
                f"expected {self.mean.numel()} from {self.artifact_path}"
            )
        normalized = (values - self.mean) / self.std
        if self.is_mlp:
            hidden = torch.relu(torch.mv(self.weights1, normalized) + self.bias1)
            logit = torch.dot(hidden, self.weights2) + self.bias2
        else:
            logit = torch.dot(normalized, self.weights) + self.bias
        p_adv = float(torch.sigmoid(logit).item())
        return {
            "classifier_p_adv": p_adv,
            "classifier_threshold": self.threshold,
            "classifier_threshold_artifact": self.threshold_artifact,
            "classifier_threshold_overridden": self.threshold != self.threshold_artifact,
            "classifier_triggered": p_adv >= self.threshold,
            "classifier_artifact": str(self.artifact_path),
            "classifier_kind": self.kind,
            "classifier_transform_mode": self.transform_mode,
            "classifier_calibration_model": self.calibration_model or "",
        }

from __future__ import annotations

import bisect
from typing import Any


class FeatureCalibration:
    """Per-model feature distribution calibration before universal A4 fusion."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        artifact: dict[str, Any],
        calibration_model: str | None,
    ) -> None:
        self.feature_names = feature_names
        self.transform_mode = str(artifact.get("transform_mode", "raw"))
        calibration = artifact.get("calibration") or {}
        models = calibration.get("models") or {}
        if self.transform_mode == "raw":
            self.model_name = calibration_model or str(
                artifact.get("default_calibration_model", "raw")
            )
            self.model_stats: dict[str, Any] = {}
            return

        if not models:
            raise ValueError("Universal calibrated classifier requires artifact.calibration.models")
        if calibration_model is None:
            if len(models) == 1:
                calibration_model = next(iter(models))
            else:
                available = ", ".join(sorted(models))
                raise ValueError(
                    "classifier_calibration_model is required for universal classifier; "
                    f"available models: {available}"
                )
        if calibration_model not in models:
            available = ", ".join(sorted(models))
            raise ValueError(
                f"Unknown classifier_calibration_model={calibration_model}; "
                f"available models: {available}"
            )
        self.model_name = str(calibration_model)
        self.model_stats = models[self.model_name]

    def transform(self, features: dict[str, float]) -> list[float]:
        if self.transform_mode == "raw":
            return [float(features.get(name, 0.0)) for name in self.feature_names]
        if self.transform_mode == "calibrated_percentile":
            return [
                self._percentile(self._stat(name), float(features.get(name, 0.0)))
                for name in self.feature_names
            ]
        if self.transform_mode == "calibrated_hybrid":
            output: list[float] = []
            for name in self.feature_names:
                value = float(features.get(name, 0.0))
                stat = self._stat(name)
                percentile = self._percentile(stat, value)
                scale = max(1e-6, float(stat.get("scale", 1.0)))
                q50 = float(stat.get("q50", 0.0))
                q95 = float(stat.get("q95", 0.0))
                q99 = float(stat.get("q99", q95 + 1.0))
                robust = max(-2.0, min(4.0, (value - q50) / scale)) / 4.0
                tail = max(0.0, min(4.0, (value - q95) / max(1e-6, q99 - q95))) / 4.0
                output.extend([percentile, robust, tail])
            return output
        raise ValueError(f"Unsupported calibration transform_mode: {self.transform_mode}")

    def _stat(self, name: str) -> dict[str, Any]:
        return dict(self.model_stats.get(name) or {})

    @staticmethod
    def _percentile(stat: dict[str, Any], value: float) -> float:
        sorted_values = stat.get("sorted") or []
        if sorted_values:
            return bisect.bisect_right(sorted_values, value) / len(sorted_values)

        quantiles = stat.get("quantiles") or []
        if quantiles:
            return bisect.bisect_right(quantiles, value) / len(quantiles)

        q01 = float(stat.get("q01", 0.0))
        q99 = float(stat.get("q99", 1.0))
        if q99 <= q01:
            return 0.5 if value >= q01 else 0.0
        return max(0.0, min(1.0, (value - q01) / (q99 - q01)))

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends.detector_backend import create_detector_backend
from model_security_gate.adapters.base import Detection

from .device_policy import ModelSecurityDevicePolicy, resolve_model_security_device


class ModuleADetectorAdapter:
    """Expose the Module A detector backend through Module B's adapter protocol."""

    def __init__(
        self,
        backend: Any,
        *,
        device_policy: ModelSecurityDevicePolicy | None = None,
    ) -> None:
        self.backend = backend
        self.device_policy = device_policy
        self.device_status = device_policy.to_dict() if device_policy else None
        self.names = {int(k): str(v) for k, v in getattr(backend, "names", {}).items()}

    def predict_image(
        self,
        image: str | Path | np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        del iou, imgsz
        array = self._read_image(image)
        old_conf = getattr(self.backend, "confidence", None)
        if conf is not None and old_conf is not None:
            self.backend.confidence = float(conf)
        try:
            result = self.backend.predict(array)
        finally:
            if conf is not None and old_conf is not None:
                self.backend.confidence = old_conf
        self.names = {int(k): str(v) for k, v in getattr(result, "names", self.names).items()}
        out: list[Detection] = []
        for box, cls_id, score in zip(result.boxes, result.classes, result.confidences):
            x1, y1, x2, y2 = [float(v) for v in box]
            cls_int = int(cls_id)
            out.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    conf=float(score),
                    cls_id=cls_int,
                    cls_name=self.names.get(cls_int, str(cls_int)),
                )
            )
        return out

    def predict_batch(
        self,
        images: list[str | Path | np.ndarray] | tuple[str | Path | np.ndarray, ...],
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[list[Detection]]:
        return [self.predict_image(image, conf=conf, iou=iou, imgsz=imgsz) for image in images]

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        array = cv2.imread(str(image))
        if array is None:
            raise FileNotFoundError(f"Cannot read image for B-module external scan: {image}")
        return array


def apply_model_security_device_policy(
    config: dict[str, Any],
    *,
    device_policy: ModelSecurityDevicePolicy | None = None,
) -> tuple[dict[str, Any], ModelSecurityDevicePolicy]:
    policy = device_policy or resolve_model_security_device(config)
    runtime_config = deepcopy(config)
    inference = runtime_config.setdefault("inference", {})
    if not isinstance(inference, dict):
        raise TypeError("inference_config_must_be_object")
    inference["device"] = policy.effective_device
    if not policy.uses_cuda:
        inference["half"] = False
    return runtime_config, policy


def create_module_a_detector_adapter(
    config: dict[str, Any],
    root: str | Path,
    *,
    device_policy: ModelSecurityDevicePolicy | None = None,
) -> ModuleADetectorAdapter:
    runtime_config, policy = apply_model_security_device_policy(
        config,
        device_policy=device_policy,
    )
    inference = runtime_config.get("inference", {})
    backend_name = str(inference.get("backend", "")).strip().lower()
    if backend_name in {"tensorrt", "engine", "trt"} and not policy.uses_cuda:
        raise RuntimeError("tensorrt_requires_cuda:model_security_cpu_fallback_unavailable")
    backend = create_detector_backend(runtime_config, Path(root))
    return ModuleADetectorAdapter(backend, device_policy=policy)

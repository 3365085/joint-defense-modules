from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends.detector_backend import create_detector_backend
from model_security_gate.adapters.base import Detection


class ModuleADetectorAdapter:
    """Expose the Module A detector backend through Module B's adapter protocol."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
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


def create_module_a_detector_adapter(config: dict[str, Any], root: str | Path) -> ModuleADetectorAdapter:
    backend = create_detector_backend(config, Path(root))
    return ModuleADetectorAdapter(backend)

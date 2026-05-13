from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class DetectionFrameResult:
    """Normalized detector output consumed by Module A."""

    image: np.ndarray
    boxes: list[list[int]]
    classes: list[int]
    confidences: list[float]
    names: dict[int, str]
    backend: str
    artifact_path: str
    inference_ms: float
    raw_result: Any | None = None

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cls_id in self.classes:
            name = self.names.get(int(cls_id), f"class_{int(cls_id)}")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def plot(self) -> np.ndarray:
        if self.raw_result is not None and hasattr(self.raw_result, "plot"):
            return self.raw_result.plot()

        rendered = self.image.copy()
        for box, cls_id, conf in zip(self.boxes, self.classes, self.confidences):
            x1, y1, x2, y2 = [int(v) for v in box]
            label = self.names.get(int(cls_id), f"class_{int(cls_id)}")
            text = f"{label} {float(conf):.2f}"
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (40, 220, 80), 2)
            cv2.putText(
                rendered,
                text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 220, 80),
                1,
                cv2.LINE_AA,
            )
        return rendered


class UltralyticsDetectorBackend:
    """Single interface for TensorRT, ONNX, and PyTorch Ultralytics artifacts."""

    def __init__(
        self,
        artifact_path: str | Path,
        backend: str,
        device: str = "cuda:0",
        half: bool = True,
        confidence: float = 0.25,
        image_size: int = 640,
    ):
        from ultralytics import YOLO

        self.artifact_path = Path(artifact_path)
        self.backend = str(backend)
        self.device = str(device)
        self.half = bool(half)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.model = YOLO(str(self.artifact_path))
        self.names = self._normalize_names(getattr(self.model, "names", {}))

        if self.backend == "pytorch":
            self.model.to(self.device)

    def predict(self, image: np.ndarray) -> DetectionFrameResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self.confidence,
            "imgsz": self.image_size,
        }
        if self.backend == "pytorch":
            kwargs["device"] = self.device
            kwargs["half"] = self.half and self.device.startswith("cuda")
        elif self.backend == "onnx":
            kwargs["device"] = self.device

        result = self.model(image, **kwargs)[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not self.names:
            self.names = self._normalize_names(getattr(result, "names", {}))

        boxes: list[list[int]] = []
        classes: list[int] = []
        confidences: list[float] = []
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is not None:
            for box in result_boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                boxes.append([x1, y1, x2, y2])
                classes.append(int(box.cls[0]))
                confidences.append(float(box.conf[0]))

        return DetectionFrameResult(
            image=image,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            names=self.names,
            backend=self.backend,
            artifact_path=str(self.artifact_path),
            inference_ms=inference_ms,
            raw_result=result,
        )

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}


class YoloV5DetectorBackend:
    """Detector backend for original YOLOv5 PyTorch, ONNX, and TensorRT exports."""

    def __init__(
        self,
        artifact_path: str | Path,
        backend: str,
        device: str = "cuda:0",
        half: bool = True,
        confidence: float = 0.25,
        image_size: int = 640,
    ):
        import torch

        self.artifact_path = Path(artifact_path)
        self.backend = str(backend)
        self.device = str(device)
        self.half = bool(half)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.names = {0: "helmet", 1: "head", 2: "person"}
        self.torch = torch
        self.model: Any | None = None
        self.session: Any | None = None
        self.input_name = "images"
        self.output_name = "output0"
        self.engine: Any | None = None
        self.context: Any | None = None
        self.input_tensor_name: str | None = None
        self.output_tensor_name: str | None = None
        self.output_shape: tuple[int, ...] | None = None
        self.output_dtype: Any | None = None
        self.onnx_input_dtype: Any = np.float32

        if self.backend == "pytorch":
            import sys

            project_root = Path(__file__).resolve().parents[3]
            yolov5_root = project_root / "external" / "yolov5_official"
            if not yolov5_root.exists():
                raise FileNotFoundError(
                    "YOLOv5 PyTorch backend requires external/yolov5_official. "
                    "Use TensorRT/ONNX artifacts for the packaged runtime."
                )
            if str(yolov5_root) not in sys.path:
                sys.path.insert(0, str(yolov5_root))
            from models.experimental import attempt_load  # type: ignore

            model = attempt_load(
                str(self.artifact_path), device=torch.device(self.device), inplace=True, fuse=True
            )
            model = model.to(self.device).eval()
            if self.half and self.device.startswith("cuda"):
                model = model.half()
            self.model = model
        elif self.backend == "onnx":
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.startswith("cuda")
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(str(self.artifact_path), providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.output_name = self.session.get_outputs()[0].name
            self.onnx_input_dtype = (
                np.float16 if "float16" in str(input_meta.type).lower() else np.float32
            )
        elif self.backend == "tensorrt":
            if not self.device.startswith("cuda"):
                raise ValueError("YOLOv5 TensorRT backend requires a CUDA device")
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(self.artifact_path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"Cannot deserialize TensorRT engine: {self.artifact_path}")
            self.context = self.engine.create_execution_context()
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_tensor_name = name
                else:
                    self.output_tensor_name = name
            if self.input_tensor_name is None or self.output_tensor_name is None:
                raise RuntimeError(f"Cannot discover TensorRT IO tensors: {self.artifact_path}")
            shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.output_tensor_name))
            self.output_shape = shape
            self.output_dtype = torch.float16 if self.half else torch.float32
        else:
            raise ValueError(f"Unsupported YOLOv5 backend: {self.backend}")

    def predict(self, image: np.ndarray) -> DetectionFrameResult:
        started = time.perf_counter()
        batch = self._preprocess(image)
        if self.backend == "pytorch":
            with self.torch.no_grad():
                pred = self.model(batch)  # type: ignore[misc]
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
        elif self.backend == "onnx":
            array = batch.detach().cpu().numpy().astype(self.onnx_input_dtype, copy=False)
            output = self.session.run([self.output_name], {self.input_name: array})[0]  # type: ignore[union-attr]
            pred = self.torch.from_numpy(output).to(
                self.device if self.device.startswith("cuda") else "cpu"
            )
        else:
            pred = self._predict_tensorrt(batch)

        det = self._non_max_suppression(
            pred.float(),
            conf_thres=self.confidence,
            iou_thres=0.7,
            max_det=100,
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        boxes: list[list[int]] = []
        classes: list[int] = []
        confidences: list[float] = []
        if det is not None:
            for row in det:
                x1, y1, x2, y2 = [int(v) for v in row[:4]]
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(row[4]))
                classes.append(int(row[5]))

        return DetectionFrameResult(
            image=image,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            names=self.names,
            backend=self.backend,
            artifact_path=str(self.artifact_path),
            inference_ms=inference_ms,
            raw_result=None,
        )

    def _preprocess(self, image: np.ndarray) -> Any:
        resized = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self.device)
        tensor = tensor.half() if self.half and self.device.startswith("cuda") else tensor.float()
        return tensor.unsqueeze(0).div(255.0)

    def _predict_tensorrt(self, batch: Any) -> Any:
        if (
            self.context is None
            or self.input_tensor_name is None
            or self.output_tensor_name is None
        ):
            raise RuntimeError("TensorRT context is not initialized")
        if self.output_shape is None:
            raise RuntimeError("TensorRT output shape is not initialized")
        output = self.torch.empty(self.output_shape, device=batch.device, dtype=self.output_dtype)
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_tensor_name, tuple(batch.shape))
        self.context.set_tensor_address(self.input_tensor_name, int(batch.data_ptr()))
        self.context.set_tensor_address(self.output_tensor_name, int(output.data_ptr()))
        stream = self.torch.cuda.current_stream(device=batch.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed: {self.artifact_path}")
        stream.synchronize()
        return output

    def _non_max_suppression(
        self,
        prediction: Any,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ) -> list[Any]:
        torch = self.torch
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(0)
        if prediction.ndim != 3:
            raise ValueError(f"Unexpected YOLOv5 output shape: {tuple(prediction.shape)}")
        if prediction.shape[1] <= 16 and prediction.shape[2] > prediction.shape[1]:
            prediction = prediction.transpose(1, 2).contiguous()

        output: list[Any] = []
        for pred in prediction:
            if pred.numel() == 0:
                output.append(pred.new_zeros((0, 6)))
                continue
            if pred.shape[1] == 6:
                det = pred[pred[:, 4] > conf_thres]
            else:
                if pred.shape[1] < 6:
                    output.append(pred.new_zeros((0, 6)))
                    continue
                pred = pred[pred[:, 4] > conf_thres]
                if pred.numel() == 0:
                    output.append(pred.new_zeros((0, 6)))
                    continue
                pred[:, 5:] *= pred[:, 4:5]
                boxes = self._xywh_to_xyxy(pred[:, :4])
                conf, cls = pred[:, 5:].max(dim=1)
                keep = conf > conf_thres
                if not bool(keep.any()):
                    output.append(pred.new_zeros((0, 6)))
                    continue
                det = torch.cat((boxes[keep], conf[keep, None], cls[keep, None].float()), dim=1)

            det = det[torch.isfinite(det).all(dim=1)]
            if det.numel() == 0:
                output.append(pred.new_zeros((0, 6)))
                continue
            det = det[det[:, 4].argsort(descending=True)]
            class_offsets = det[:, 5:6] * 4096.0
            keep_idx = self._nms(det[:, :4] + class_offsets, det[:, 4], iou_thres)
            output.append(det[keep_idx[:max_det]])
        return output

    def _xywh_to_xyxy(self, xywh: Any) -> Any:
        xyxy = xywh.clone()
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy

    def _nms(self, boxes: Any, scores: Any, iou_thres: float) -> Any:
        try:
            from torchvision.ops import nms

            return nms(boxes, scores, iou_thres)
        except Exception:
            return self._nms_torch(boxes, scores, iou_thres)

    def _nms_torch(self, boxes: Any, scores: Any, iou_thres: float) -> Any:
        torch = self.torch
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep: list[Any] = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[i] + areas[rest] - inter
            iou = inter / union.clamp(min=1e-6)
            order = rest[iou <= iou_thres]
        return (
            torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)
        )


def resolve_detector_artifact(
    project_root: Path,
    candidates: list[str],
    backend: str,
) -> Path:
    checked: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = project_root / path
        checked.append(path)
        if path.exists():
            return path

    checked_text = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Missing {backend} detector artifact. Checked:\n  {checked_text}\n"
        "Run: pixi run build-inference"
    )


def create_detector_backend(
    config: dict[str, Any], project_root: Path
) -> UltralyticsDetectorBackend:
    inference = config.get("inference", {})
    backend = str(inference.get("backend", "tensorrt")).lower()
    if backend not in {"tensorrt", "onnx", "pytorch"}:
        raise ValueError(f"Unsupported detector backend: {backend}")

    artifact_key = "engine" if backend == "tensorrt" else backend
    artifacts = inference.get("artifacts", {})
    candidates = artifacts.get(artifact_key, [])
    if isinstance(candidates, str):
        candidates = [candidates]
    if not candidates:
        raise ValueError(f"No artifact candidates configured for backend: {backend}")

    artifact_path = resolve_detector_artifact(project_root, candidates, backend)
    family = str(inference.get("model_family", inference.get("family", "ultralytics"))).lower()
    if family == "yolov5":
        return YoloV5DetectorBackend(
            artifact_path=artifact_path,
            backend=backend,
            device=str(inference.get("device", config.get("module_a", {}).get("device", "cuda:0"))),
            half=bool(inference.get("half", True)),
            confidence=float(inference.get("confidence", 0.25)),
            image_size=int(
                inference.get("image_size", config.get("module_a", {}).get("frame_size", 640))
            ),
        )
    return UltralyticsDetectorBackend(
        artifact_path=artifact_path,
        backend=backend,
        device=str(inference.get("device", config.get("module_a", {}).get("device", "cuda:0"))),
        half=bool(inference.get("half", True)),
        confidence=float(inference.get("confidence", 0.25)),
        image_size=int(
            inference.get("image_size", config.get("module_a", {}).get("frame_size", 640))
        ),
    )

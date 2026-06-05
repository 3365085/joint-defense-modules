from __future__ import annotations

import gc
import time
from dataclasses import dataclass
import os
from pathlib import Path
import sys
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
        candidate_confidence: float | None = None,
        image_size: int = 640,
        class_names: Any | None = None,
    ):
        from ultralytics import YOLO

        self.artifact_path = Path(artifact_path)
        self.backend = str(backend)
        self.device = str(device)
        self.half = bool(half)
        self.confidence = float(confidence)
        self.candidate_confidence = (
            float(candidate_confidence) if candidate_confidence is not None else None
        )
        self.image_size = int(image_size)
        self.model = YOLO(str(self.artifact_path))
        configured_names = normalize_class_names(class_names)
        self.names = configured_names or self._normalize_names(getattr(self.model, "names", {}))

        if self.backend == "pytorch":
            self.model.to(self.device)

    def predict(self, image: np.ndarray) -> DetectionFrameResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self._prediction_confidence(),
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

    def close(self) -> None:
        self.model = None
        _release_torch_cuda_cache()

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def _prediction_confidence(self) -> float:
        if self.candidate_confidence is None:
            return self.confidence
        if not _is_head_helmet_only_names(self.names):
            return self.confidence
        return min(self.confidence, self.candidate_confidence)


class YoloV5DetectorBackend:
    """Detector backend for original YOLOv5 PyTorch, ONNX, and TensorRT exports."""

    def __init__(
        self,
        artifact_path: str | Path,
        backend: str,
        device: str = "cuda:0",
        half: bool = True,
        confidence: float = 0.25,
        candidate_confidence: float | None = None,
        image_size: int = 640,
        class_names: Any | None = None,
    ):
        import torch

        self.artifact_path = Path(artifact_path)
        self.backend = str(backend)
        self.device = str(device)
        self.half = bool(half)
        self.confidence = float(confidence)
        self.candidate_confidence = (
            float(candidate_confidence) if candidate_confidence is not None else None
        )
        self.image_size = int(image_size)
        self.names = normalize_class_names(class_names) or {0: "helmet", 1: "head", 2: "person"}
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
            project_root = Path(__file__).resolve().parents[3]
            package_root = Path(__file__).resolve().parents[2]
            yolov5_root = package_root / "model_bases" / "yolov5_official"
            if not yolov5_root.exists():
                raise FileNotFoundError(
                    "YOLOv5 PyTorch backend requires src/defense/model_bases/yolov5_official. "
                    "Use TensorRT/ONNX artifacts for the packaged runtime."
                )
            self._configure_yolov5_base_runtime(yolov5_root)
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

    def close(self) -> None:
        self.context = None
        self.engine = None
        self.session = None
        self.model = None
        _release_torch_cuda_cache()

    def warmup_postprocess(self) -> None:
        """Force lazy NMS imports/kernels before the first real detection frame."""
        device = self.device if self.device.startswith("cuda") else "cpu"
        dummy = self.torch.tensor(
            [
                [
                    [10.0, 10.0, 40.0, 40.0, 0.90, 0.0],
                    [12.0, 12.0, 42.0, 42.0, 0.80, 0.0],
                    [80.0, 80.0, 110.0, 115.0, 0.70, 1.0],
                    [120.0, 120.0, 160.0, 170.0, 0.60, 2.0],
                    [0.0, 0.0, 1.0, 1.0, 0.01, 0.0],
                    [1.0, 1.0, 2.0, 2.0, 0.01, 1.0],
                ]
            ],
            device=device,
            dtype=self.torch.float32,
        )
        self._non_max_suppression(
            dummy,
            conf_thres=min(self._prediction_confidence(), 0.25),
            iou_thres=0.7,
            max_det=10,
        )
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.torch.device(self.device))

    @staticmethod
    def _configure_yolov5_base_runtime(yolov5_root: Path) -> None:
        """Constrain bundled YOLOv5 code to local artifact loading only."""
        project_root = Path(__file__).resolve().parents[4]
        runtime_dir = project_root.parent / "logs" / "yolov5_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLOv5_AUTOINSTALL", "0")
        os.environ.setdefault("YOLOV5_AUTOINSTALL", "0")
        os.environ.setdefault("YOLOv5_OFFLINE", "1")
        os.environ.setdefault("YOLOV5_OFFLINE", "1")
        os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))

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
            conf_thres=self._prediction_confidence(),
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

    def _prediction_confidence(self) -> float:
        if self.candidate_confidence is None:
            return self.confidence
        if not _is_head_helmet_only_names(self.names):
            return self.confidence
        return min(self.confidence, self.candidate_confidence)

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


def _artifact_search_roots(project_root: Path) -> list[Path]:
    roots = [project_root]
    for env_name in ("MODULE_A_WORKSPACE_ROOT", "SECURITY_PROJECT_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())
    try:
        from defense.runtime.config import workspace_asset_roots

        roots.extend(workspace_asset_roots())
    except Exception:
        pass

    parent = project_root.parent
    roots.extend([parent, project_root / "模型和素材", parent / "模型和素材"])
    roots.extend([project_root / "素材", parent / "素材", parent / "训练素材"])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_detector_artifact(
    project_root: Path,
    candidates: list[str],
    backend: str,
) -> Path:
    checked: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute():
            checked.append(path)
            if path.exists():
                return path
            continue

        for root in _artifact_search_roots(project_root):
            resolved = root / path
            checked.append(resolved)
            if resolved.exists():
                return resolved

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
    confidence = float(inference.get("confidence", 0.25))
    ppe_config = config.get("ppe_tracking", {}) if isinstance(config.get("ppe_tracking"), dict) else {}
    candidate_confidence_value = inference.get(
        "candidate_confidence",
        ppe_config.get("temporal_candidate_min_confidence"),
    )
    candidate_confidence = (
        float(candidate_confidence_value) if candidate_confidence_value is not None else None
    )
    class_names = configured_class_names(config)
    if family == "yolov5":
        return YoloV5DetectorBackend(
            artifact_path=artifact_path,
            backend=backend,
            device=str(inference.get("device", config.get("module_a", {}).get("device", "cuda:0"))),
            half=bool(inference.get("half", True)),
            confidence=confidence,
            candidate_confidence=candidate_confidence,
            image_size=int(
                inference.get("image_size", config.get("module_a", {}).get("frame_size", 640))
            ),
            class_names=class_names,
        )
    return UltralyticsDetectorBackend(
        artifact_path=artifact_path,
        backend=backend,
        device=str(inference.get("device", config.get("module_a", {}).get("device", "cuda:0"))),
        half=bool(inference.get("half", True)),
        confidence=confidence,
        candidate_confidence=candidate_confidence,
        image_size=int(
            inference.get("image_size", config.get("module_a", {}).get("frame_size", 640))
        ),
        class_names=class_names,
    )


def normalize_class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def configured_class_names(config: dict[str, Any] | None) -> dict[int, str]:
    inference = (
        config.get("inference", {})
        if isinstance(config, dict) and isinstance(config.get("inference"), dict)
        else {}
    )
    for key in ("names", "class_names", "labels"):
        names = normalize_class_names(inference.get(key))
        if names:
            return names
    return {}


def _is_head_helmet_only_names(names: dict[int, str]) -> bool:
    labels = {str(name or "").strip().lower().replace("-", "_") for name in names.values()}
    has_head = any("head" in label for label in labels)
    has_helmet = any(
        "helmet" in label or "hard_hat" in label or "hardhat" in label for label in labels
    )
    has_person = any(
        "person" in label or "worker" in label or "human" in label or "pedestrian" in label for label in labels
    )
    return has_head and has_helmet and not has_person


def _release_torch_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

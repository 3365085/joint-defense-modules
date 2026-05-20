from __future__ import annotations

import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends.detector_backend import DetectionFrameResult, create_detector_backend
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline

from .artifacts import missing_artifact_message
from .config import load_runtime_config, normalize_custom_model_options, project_root

EMPTY_BACKEND_PROFILES = frozenset({"empty_smoke"})


def configure_runtime_threads() -> None:
    """Keep OpenCV/BLAS/PyTorch from creating CPU thread storms."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class EmptyDetectorBackend:
    """A no-model backend for UI smoke tests and CI compile checks.

    It is never selected by default. Set ``runtime.allow_empty_backend: true``
    when you want to exercise Web/recording logic without weights.
    """

    backend = "empty"
    artifact_path = "empty://no-model"
    names = {0: "helmet", 1: "head", 2: "person"}

    def predict(self, image: np.ndarray) -> DetectionFrameResult:
        return DetectionFrameResult(
            image=image,
            boxes=[],
            classes=[],
            confidences=[],
            names=self.names,
            backend=self.backend,
            artifact_path=self.artifact_path,
            inference_ms=0.0,
            raw_result=None,
        )


def allow_empty_backend_for_profile(profile: str) -> bool:
    return str(profile or "default") in EMPTY_BACKEND_PROFILES


@dataclass(slots=True)
class PipelineBundle:
    pipeline: VideoDefensePipeline
    backend: str
    artifact_path: str
    config: dict[str, Any]
    warmup_error: str = ""


class PipelineCache:
    """Thread-safe pipeline cache keyed by profile + runtime switches."""

    def __init__(self, *, config_path: str | Path | None = None, root: Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.root = root or project_root()
        self._lock = threading.Lock()
        self._key: tuple[Any, ...] | None = None
        self._bundle: PipelineBundle | None = None

    def clear(self) -> None:
        with self._lock:
            bundle = self._bundle
            self._bundle = None
            self._key = None
        if bundle is not None:
            close_pipeline = getattr(bundle.pipeline, "close", None)
            if callable(close_pipeline):
                close_pipeline()

    def get(
        self,
        *,
        profile: str,
        feature_options: dict[str, Any] | None = None,
        custom_model: dict[str, Any] | None = None,
    ) -> PipelineBundle:
        configure_runtime_threads()
        normalized_custom = normalize_custom_model_options(custom_model)
        key = (
            str(profile or "default"),
            bool((feature_options or {}).get("static_image_enabled", True)),
            bool(normalized_custom.get("enabled", False)),
            str(normalized_custom.get("path", "")),
            str(normalized_custom.get("backend", "auto")),
            str(normalized_custom.get("model_family", "yolov5")),
            str(self.config_path or ""),
        )
        with self._lock:
            if self._bundle is not None and self._key == key:
                self._bundle.pipeline.reset()
                return self._bundle

            config = load_runtime_config(
                config_path=self.config_path,
                profile=str(profile or "default"),
                feature_options=feature_options,
                custom_model=normalized_custom,
            )
            runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
            allow_empty_backend = bool(runtime_config.get("allow_empty_backend", False))
            if allow_empty_backend and not allow_empty_backend_for_profile(str(profile or "default")):
                raise RuntimeError(
                    f"runtime.allow_empty_backend is only allowed for profiles: {', '.join(sorted(EMPTY_BACKEND_PROFILES))}"
                )
            if allow_empty_backend:
                backend = EmptyDetectorBackend()
            else:
                try:
                    backend = create_detector_backend(config, self.root)
                except Exception as exc:
                    raise RuntimeError(f"{exc}\n{missing_artifact_message(config, self.root)}") from exc
            pipeline = VideoDefensePipeline(backend, config=config)
            warmup_frames = int(getattr(pipeline, "warmup_frames", 0) or 0)
            warmup_error = ""
            try:
                pipeline.warmup(warmup_frames)
                pipeline.reset()
            except Exception as exc:
                # Warmup is an optimization, not a correctness requirement. The
                # actual inference error will still surface during processing.
                warmup_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                pipeline.reset()
            bundle = PipelineBundle(
                pipeline=pipeline,
                backend=str(getattr(backend, "backend", "unknown")),
                artifact_path=str(getattr(backend, "artifact_path", "")),
                config=config,
                warmup_error=warmup_error,
            )
            self._bundle = bundle
            self._key = key
            return bundle

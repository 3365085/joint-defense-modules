from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from defense.module_a.backends.detector_backend import (
    DetectionFrameResult,
    configured_class_names,
    create_detector_backend,
)
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

    def __init__(self, names: dict[int, str] | None = None) -> None:
        self.names = names or {0: "helmet", 1: "head", 2: "person"}

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
    model_family: str
    artifact_path: str
    config: dict[str, Any]
    warmup_error: str = ""
    cache_hit: bool = False
    cache_get_ms: float = 0.0
    config_load_ms: float = 0.0
    backend_create_ms: float = 0.0
    pipeline_construct_ms: float = 0.0
    warmup_ms: float = 0.0
    warmup_frames: int = 0
    pipeline_reset_ms: float = 0.0


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
        self._close_bundle(bundle)

    @staticmethod
    def _close_bundle(bundle: PipelineBundle | None) -> None:
        if bundle is None:
            return
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
        get_started = time.perf_counter()
        configure_runtime_threads()
        normalized_custom = normalize_custom_model_options(custom_model)
        key = (
            str(profile or "default"),
            bool((feature_options or {}).get("static_image_enabled", True)),
            str((feature_options or {}).get("a3b_sensitivity") or "balanced"),
            bool(normalized_custom.get("enabled", False)),
            str(normalized_custom.get("path", "")),
            str(normalized_custom.get("backend", "auto")),
            str(normalized_custom.get("model_family", "yolov5")),
            json.dumps(normalized_custom.get("class_names"), sort_keys=True, ensure_ascii=True),
            str(self.config_path or ""),
        )
        with self._lock:
            if self._bundle is not None and self._key == key:
                reset_started = time.perf_counter()
                self._bundle.pipeline.reset()
                self._bundle.cache_hit = True
                self._bundle.cache_get_ms = (time.perf_counter() - get_started) * 1000.0
                self._bundle.config_load_ms = 0.0
                self._bundle.backend_create_ms = 0.0
                self._bundle.pipeline_construct_ms = 0.0
                self._bundle.warmup_ms = 0.0
                self._bundle.warmup_frames = 0
                self._bundle.pipeline_reset_ms = (time.perf_counter() - reset_started) * 1000.0
                return self._bundle

            old_bundle = self._bundle
            self._bundle = None
            self._key = None
            self._close_bundle(old_bundle)

            config_started = time.perf_counter()
            config = load_runtime_config(
                config_path=self.config_path,
                profile=str(profile or "default"),
                feature_options=feature_options,
                custom_model=normalized_custom,
            )
            config_load_ms = (time.perf_counter() - config_started) * 1000.0
            runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
            allow_empty_backend = bool(runtime_config.get("allow_empty_backend", False))
            if allow_empty_backend and not allow_empty_backend_for_profile(str(profile or "default")):
                raise RuntimeError(
                    f"runtime.allow_empty_backend is only allowed for profiles: {', '.join(sorted(EMPTY_BACKEND_PROFILES))}"
                )
            backend_started = time.perf_counter()
            if allow_empty_backend:
                backend = EmptyDetectorBackend(configured_class_names(config))
            else:
                try:
                    backend = create_detector_backend(config, self.root)
                except Exception as exc:
                    raise RuntimeError(f"{exc}\n{missing_artifact_message(config, self.root)}") from exc
            backend_create_ms = (time.perf_counter() - backend_started) * 1000.0
            construct_started = time.perf_counter()
            pipeline = VideoDefensePipeline(backend, config=config)
            pipeline_construct_ms = (time.perf_counter() - construct_started) * 1000.0
            warmup_frames = int(getattr(pipeline, "warmup_frames", 0) or 0)
            warmup_error = ""
            warmup_started = time.perf_counter()
            warmup_ms = 0.0
            reset_ms = 0.0
            try:
                pipeline.warmup(warmup_frames)
            except Exception as exc:
                # Warmup is an optimization, not a correctness requirement. The
                # actual inference error will still surface during processing.
                warmup_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
                reset_started = time.perf_counter()
                pipeline.reset()
                reset_ms = (time.perf_counter() - reset_started) * 1000.0
            bundle = PipelineBundle(
                pipeline=pipeline,
                backend=str(getattr(backend, "backend", "unknown")),
                model_family=str(config.get("inference", {}).get("model_family", "unknown")),
                artifact_path=str(getattr(backend, "artifact_path", "")),
                config=config,
                warmup_error=warmup_error,
                cache_hit=False,
                cache_get_ms=(time.perf_counter() - get_started) * 1000.0,
                config_load_ms=config_load_ms,
                backend_create_ms=backend_create_ms,
                pipeline_construct_ms=pipeline_construct_ms,
                warmup_ms=warmup_ms,
                warmup_frames=warmup_frames,
                pipeline_reset_ms=reset_ms,
            )
            self._bundle = bundle
            self._key = key
            return bundle

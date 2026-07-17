from __future__ import annotations

import copy
import time
from typing import Any, Callable

import cv2
import numpy as np

from defense.module_a import ModuleADetector, ModuleAInput
from defense.module_a.rebuilt import RebuiltModuleADetector
from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.backends import UltralyticsDetectorBackend
from defense.module_a.result_contract import adapt_a3b_result
from defense.module_a.roi_provider import DetectionROIProvider

from .stream_source import FrameEnvelope, StreamSource


class VideoDefensePipeline:
    """GPU-first Module A pipeline driven by a detector backend.

    The pipeline exposes two entry points:

    * :meth:`process_frame` — legacy signature used by the offline MP4 path
      (``tools/run_experiment.py`` and the standalone Monitor_App file input).
      Callers hand in a raw ``ndarray`` and may optionally provide source
      timestamp/FPS metadata. Missing timing is generated from a stable
      configured cadence instead of machine-speed wall clock.
    * :meth:`process_envelope` — streaming entry point used when a
      :class:`StreamSource` feeds the pipeline. It consumes a
      :class:`FrameEnvelope`, propagates the real frame timestamp to
      ``ModuleAInput.timestamp`` (which ``ModuleADetector`` in turn feeds into
      ``AlertState`` as ``frame_ts``), reacts to
      ``flags["stream_geometry_changed"]`` by resetting both the pipeline and
      the detector, and fills ``info["latency_breakdown"]`` with the real
      source→decode→process timings plus the end-to-end latency relative to
      ``envelope.source_ts``.

    The optional ``stream_source`` constructor argument lets callers attach a
    ``StreamSource`` instance for observability (geometry callbacks, stats),
    but it is **not** consulted by :meth:`process_frame`. Frames are always
    provided by the caller.
    """

    _DEFAULT_OFFLINE_SOURCE_FPS = 30.0
    # At most two 60 FPS source frames (about 33 ms).  A 40 ms wall keeps the
    # optimization useful for high-frame-rate files without relaxing 30 FPS
    # detector freshness beyond one source frame.
    _DEFAULT_REUSE_MAX_SOURCE_TIME_GAP_S = 0.04
    # High-FPS inputs, plus lower-FPS inputs already showing latest-only drops,
    # use a processed-frame budget for Module A.  Without this budget every
    # surviving frame looks overdue in source time, so the expensive analysis
    # runs on every processed frame and creates a self-reinforcing
    # low-throughput loop.  The stale cap still forces a fresh analysis when
    # the processed stream itself becomes sparse.
    _MODULE_A_CADENCE_MAX_STALE_INTERVALS = 3.0
    _A3B_SUPPRESSION_HOLD_FRAMES = 180
    _A3B_SUPPRESSION_HOLD_S = 6.0
    _A3B_SUPPRESSION_STALE_BRIDGE_S = 0.5

    def __init__(
        self,
        detector_backend: UltralyticsDetectorBackend,
        config: dict[str, Any] | None = None,
        stream_source: StreamSource | None = None,
    ):
        self.detector_backend = detector_backend
        self.class_names = detector_backend.names
        inference_config = (config or {}).get("inference", {})
        module_config = (config or {}).get("module_a", config or {})
        # Module A detection kernel selection (2026-06-30):
        #   ``rebuilt`` = ported rebuilt_demo kernel (A1-A4 + branch-B blinding
        #     + scene-adaptive baseline + joint decision; XGBoost A4).
        #   ``legacy``  = original in-tree detector.
        # Both honor the same ModuleAInput/ModuleAResult contract and run behind
        # the unchanged frame-skip / detection-reuse shell in this pipeline.
        self.detector_impl = str(module_config.get("detector_impl", "rebuilt")).lower()
        if self.detector_impl == "legacy":
            self.detector = ModuleADetector(config=config)
        else:
            self.detector = RebuiltModuleADetector(config=config)
        configured_warmup = int(inference_config.get("warmup_frames", 3))
        light_flow_warmup = int(module_config.get("light_flow_interval", 3)) + 1
        self.warmup_frames = max(
            configured_warmup,
            light_flow_warmup
            if module_config.get("light_flow_enabled", True)
            else configured_warmup,
        )
        self.roi_provider = DetectionROIProvider(
            self.class_names,
            min_confidence=module_config.get("roi_min_confidence", 0.25),
            margin=module_config.get("roi_margin", 8),
            class_aliases=module_config.get("roi_class_aliases", {}),
            target_labels=module_config.get(
                "roi_target_labels",
                module_config.get("static_image_target_labels", ["person", "helmet", "head"]),
            ),
            stabilize_overlaps=module_config.get("roi_stabilize_overlaps_enabled", True),
            same_label_iou=module_config.get("roi_stabilize_same_label_iou", 0.55),
            head_helmet_iou=module_config.get("roi_stabilize_head_helmet_iou", 0.20),
            head_helmet_center_distance=module_config.get(
                "roi_stabilize_head_helmet_center_distance", 0.05
            ),
        )
        self.stream_source = stream_source
        self.frame_idx = 0
        self._last_small_gray: np.ndarray | None = None
        self._last_detections: Any | None = None
        self._last_rois: list[Any] | None = None
        self._last_detector_frame_idx: int = -1
        self._last_detector_source_frame_idx: int | None = None
        self._last_detector_source_time_s: float | None = None
        self._current_reuse_source_frame_idx: int | None = None
        self._current_reuse_source_time_s: float | None = None
        self._temporal_reuse_threshold = float(
            module_config.get("temporal_detector_reuse_threshold", 0.010)
        )
        self._temporal_reuse_max_gap = max(
            1, int(module_config.get("temporal_detector_reuse_max_gap", 2))
        )
        self._temporal_reuse_max_source_time_gap_s = max(
            0.0,
            float(
                module_config.get(
                    "temporal_detector_reuse_max_source_time_gap_s",
                    self._DEFAULT_REUSE_MAX_SOURCE_TIME_GAP_S,
                )
            ),
        )
        self._temporal_reuse_ppe_change_threshold = max(
            0.0,
            float(
                module_config.get(
                    "temporal_detector_reuse_ppe_change_threshold",
                    self._temporal_reuse_threshold * 0.5,
                )
            ),
        )
        self._last_reuse_decision: dict[str, Any] = {
            "hit": False,
            "reason": "not_evaluated",
        }
        self._detector_reuse_attempt_count = 0
        self._detector_reuse_hit_count = 0
        self._detector_backend_predict_count = 0
        self._detector_reuse_miss_reasons: dict[str, int] = {}
        self._module_a_analysis_max_hz = max(
            0.0,
            float(module_config.get("analysis_max_hz", 25.0)),
        )
        self._last_module_a_result: Any | None = None
        self._last_module_a_source_frame_idx: int | None = None
        self._last_module_a_source_time_s: float | None = None
        self._last_module_a_seen_source_frame_idx: int | None = None
        self._last_module_a_seen_source_time_s: float | None = None
        self._module_a_source_fps: float | None = None
        self._module_a_cadence_attempt_count = 0
        self._module_a_cadence_hit_count = 0
        self._module_a_processed_cadence_phase = 0.0
        self._last_module_a_cadence: dict[str, Any] = {
            "hit": False,
            "reason": "not_evaluated",
        }
        self._offline_default_source_fps = max(
            0.1,
            float(
                inference_config.get(
                    "offline_source_fps",
                    module_config.get(
                        "offline_source_fps",
                        self._DEFAULT_OFFLINE_SOURCE_FPS,
                    ),
                )
            ),
        )
        self._offline_explicit_timestamp_offset_s: float | None = None
        self._last_runtime_source_frame_idx: int | None = None
        self._last_runtime_source_time_s: float | None = None
        self._last_runtime_frame_shape: tuple[int, ...] | None = None
        self._a3b_suppression_hold_s = max(
            0.0,
            float(
                module_config.get(
                    "a3b_suppression_hold_s",
                    self._A3B_SUPPRESSION_HOLD_S,
                )
            ),
        )
        self._a3b_suppression_stale_bridge_s = min(
            self._a3b_suppression_hold_s,
            max(
                0.0,
                float(
                    module_config.get(
                        "a3b_suppression_stale_bridge_s",
                        self._A3B_SUPPRESSION_STALE_BRIDGE_S,
                    )
                ),
            ),
        )

        # Per-target reuse state (2026-06-11 架构修复)
        # Each target gets tracked individually so small-region changes
        # don't get drowned out by the global change_score.
        self._temporal_reuse_target_state: dict[int, dict[str, Any]] = {}
        self._temporal_reuse_consecutive = 0

        # A3b 假视频区域检测框抑制：真实释放由 source timestamp 秒级租约驱动。
        # ``_a3b_suppress_remaining`` 仅保留为旧字段的帧数近似镜像。
        self._clear_a3b_suppression_lease()

    def reset(self) -> None:
        self.detector.reset()
        self.frame_idx = 0
        self._last_small_gray = None
        self._last_detections = None
        self._last_rois = None
        self._last_detector_frame_idx = -1
        self._last_detector_source_frame_idx = None
        self._last_detector_source_time_s = None
        self._current_reuse_source_frame_idx = None
        self._current_reuse_source_time_s = None
        self._last_reuse_decision = {
            "hit": False,
            "reason": "reset",
        }
        self._detector_reuse_attempt_count = 0
        self._detector_reuse_hit_count = 0
        self._detector_backend_predict_count = 0
        self._detector_reuse_miss_reasons = {}
        self._last_module_a_result = None
        self._last_module_a_source_frame_idx = None
        self._last_module_a_source_time_s = None
        self._last_module_a_seen_source_frame_idx = None
        self._last_module_a_seen_source_time_s = None
        self._module_a_source_fps = None
        self._module_a_cadence_attempt_count = 0
        self._module_a_cadence_hit_count = 0
        self._module_a_processed_cadence_phase = 0.0
        self._last_module_a_cadence = {
            "hit": False,
            "reason": "reset",
        }
        self._offline_explicit_timestamp_offset_s = None
        self._last_runtime_source_frame_idx = None
        self._last_runtime_source_time_s = None
        self._last_runtime_frame_shape = None
        self._temporal_reuse_target_state.clear()
        self._temporal_reuse_consecutive = 0
        self._clear_a3b_suppression_lease()

    def close(self) -> None:
        close_detector = getattr(self.detector, "close", None)
        try:
            if callable(close_detector):
                close_detector()
        finally:
            close_backend = getattr(self.detector_backend, "close", None)
            try:
                if callable(close_backend):
                    close_backend()
            finally:
                self._last_small_gray = None
                self._last_detections = None
                self._last_rois = None
                self._last_detector_frame_idx = -1
                self._last_detector_source_frame_idx = None
                self._last_detector_source_time_s = None
                self._current_reuse_source_frame_idx = None
                self._current_reuse_source_time_s = None
                self._last_reuse_decision = {
                    "hit": False,
                    "reason": "closed",
                }
                self._detector_reuse_attempt_count = 0
                self._detector_reuse_hit_count = 0
                self._detector_backend_predict_count = 0
                self._detector_reuse_miss_reasons = {}
                self._last_module_a_result = None
                self._last_module_a_source_frame_idx = None
                self._last_module_a_source_time_s = None
                self._last_module_a_seen_source_frame_idx = None
                self._last_module_a_seen_source_time_s = None
                self._module_a_source_fps = None
                self._module_a_cadence_attempt_count = 0
                self._module_a_cadence_hit_count = 0
                self._module_a_processed_cadence_phase = 0.0
                self._last_module_a_cadence = {
                    "hit": False,
                    "reason": "closed",
                }
                self._offline_explicit_timestamp_offset_s = None
                self._last_runtime_source_frame_idx = None
                self._last_runtime_source_time_s = None
                self._last_runtime_frame_shape = None
                self._temporal_reuse_target_state.clear()
                self._temporal_reuse_consecutive = 0
                self._clear_a3b_suppression_lease()

    def warmup(self, frames: int = 3) -> None:
        if frames <= 0:
            return
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(frames):
            self.process_frame(warmup_frame)
        warmup_postprocess = getattr(self.detector_backend, "warmup_postprocess", None)
        if callable(warmup_postprocess):
            warmup_postprocess()
        self.reset()

    # ------------------------------------------------------------------ core

    def _maybe_reuse_detections(
        self,
        frame_640: np.ndarray,
        *,
        current_source_frame_idx: int | None = None,
        current_source_time_s: float | None = None,
    ) -> tuple[Any | None, list[Any] | None, float, float]:
        """Reuse the last detector output when the current frame barely changed.

        Uses per-target ROI-level change detection to avoid missing small-region
        changes, plus a hard cap on consecutive reuse frames to prevent
        unbounded box-position drift (2026-06-11 架构修复). Reuse is allowed
        only when processed-frame, source-frame, and source-time gaps are all
        known, forward-moving, and inside their limits. Missing source context
        therefore fails closed to a real detector invocation.

        Returns
        -------
        tuple
            (detections_or_none, rois_or_none, detector_inference_ms, change_score)
        """
        def record_decision(
            *,
            hit: bool,
            reason: str,
            processed_gap: int | None = None,
            source_frame_gap: int | None = None,
            source_time_gap_s: float | None = None,
            ppe_sensitive: bool = False,
            change_score: float | None = None,
        ) -> None:
            if not hasattr(self, "_detector_reuse_attempt_count"):
                self._detector_reuse_attempt_count = 0
                self._detector_reuse_hit_count = 0
                self._detector_reuse_miss_reasons = {}
            self._detector_reuse_attempt_count += 1
            if hit:
                self._detector_reuse_hit_count += 1
            else:
                misses = self._detector_reuse_miss_reasons
                misses[reason] = int(misses.get(reason, 0)) + 1
            self._last_reuse_decision = {
                "hit": bool(hit),
                "reason": str(reason),
                "ppe_sensitive": bool(ppe_sensitive),
                "change_score": (
                    float(change_score) if change_score is not None else None
                ),
                "processed_gap": processed_gap,
                "source_frame_gap": source_frame_gap,
                "source_time_gap_s": source_time_gap_s,
                "max_processed_gap": int(
                    getattr(self, "_temporal_reuse_max_gap", 2)
                ),
                "max_source_frame_gap": int(
                    getattr(self, "_temporal_reuse_max_gap", 2)
                ),
                "max_source_time_gap_s": float(
                    getattr(
                        self,
                        "_temporal_reuse_max_source_time_gap_s",
                        self._DEFAULT_REUSE_MAX_SOURCE_TIME_GAP_S,
                    )
                ),
            }

        if (
            self._last_small_gray is None
            or self._last_detections is None
            or self._last_rois is None
        ):
            record_decision(
                hit=False,
                reason="no_cached_detection",
                change_score=1.0,
            )
            return None, None, 0.0, 1.0
        gray = cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(small, self._last_small_gray)
        change_score = float(diff.mean() / 255.0)
        processed_gap = self.frame_idx - self._last_detector_frame_idx

        last_source_frame_idx = getattr(
            self,
            "_last_detector_source_frame_idx",
            None,
        )
        last_source_time_s = getattr(
            self,
            "_last_detector_source_time_s",
            None,
        )
        source_frame_gap = (
            int(current_source_frame_idx) - int(last_source_frame_idx)
            if current_source_frame_idx is not None
            and last_source_frame_idx is not None
            else None
        )
        source_time_gap_s = (
            float(current_source_time_s) - float(last_source_time_s)
            if current_source_time_s is not None
            and last_source_time_s is not None
            else None
        )
        ppe_sensitive = False
        cached_classes = getattr(self._last_detections, "classes", None) or []
        cached_names = getattr(self._last_detections, "names", None) or {}
        roi_provider = getattr(self, "roi_provider", None)
        for class_id in cached_classes:
            raw_label = str(
                cached_names.get(int(class_id), f"class_{int(class_id)}")
            )
            normalize_label = getattr(roi_provider, "normalize_label", None)
            normalized = (
                str(normalize_label(raw_label))
                if callable(normalize_label)
                else raw_label.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if normalized in {"person", "head", "helmet"}:
                ppe_sensitive = True
                break

        # Lazy-init for _temporal_reuse_consecutive (tests may construct
        # pipeline without calling __init__, e.g. test_video_defense_pipeline_reuse)
        if not hasattr(self, '_temporal_reuse_consecutive'):
            self._temporal_reuse_consecutive = 0

        def force_detection(reason: str) -> tuple[None, None, float, float]:
            self._last_small_gray = small
            self._temporal_reuse_consecutive = 0
            record_decision(
                hit=False,
                reason=reason,
                processed_gap=processed_gap,
                source_frame_gap=source_frame_gap,
                source_time_gap_s=source_time_gap_s,
                ppe_sensitive=ppe_sensitive,
                change_score=change_score,
            )
            return None, None, 0.0, change_score

        if source_frame_gap is None or source_time_gap_s is None:
            return force_detection("source_context_missing")
        if processed_gap <= 0:
            return force_detection("processed_gap_not_forward")
        if processed_gap > self._temporal_reuse_max_gap:
            return force_detection("processed_gap_exceeded")
        if source_frame_gap <= 0:
            return force_detection("source_frame_gap_not_forward")
        if source_frame_gap > self._temporal_reuse_max_gap:
            return force_detection("source_frame_gap_exceeded")
        if not np.isfinite(source_time_gap_s) or source_time_gap_s <= 0.0:
            return force_detection("source_time_gap_not_forward")
        max_source_time_gap_s = float(
            getattr(
                self,
                "_temporal_reuse_max_source_time_gap_s",
                self._DEFAULT_REUSE_MAX_SOURCE_TIME_GAP_S,
            )
        )
        if source_time_gap_s > max_source_time_gap_s:
            return force_detection("source_time_gap_exceeded")

        ppe_change_threshold = float(
            getattr(
                self,
                "_temporal_reuse_ppe_change_threshold",
                self._temporal_reuse_threshold * 0.5,
            )
        )
        if ppe_sensitive and change_score > ppe_change_threshold:
            return force_detection(
                "ppe_change_exceeds_tighter_reuse_threshold"
            )

        # Hard cap on consecutive reuse frames (prevents unbounded drift)
        from defense.module_a.backends.detector_backend import temporal_reuse_max_consecutive as _reuse_max
        _max_consecutive = _reuse_max(
            getattr(
                self,
                '_temporal_reuse_max_consecutive',
                None,
            )
        )
        if self._temporal_reuse_consecutive >= _max_consecutive:
            return force_detection("consecutive_reuse_limit")

        # Per-target ROI-level change check: for each existing target bbox,
        # compute the local change score. If any target region changed
        # significantly, force re-detection.
        _boxes = getattr(self._last_detections, 'boxes', None)
        if _boxes:
            for _box in _boxes:
                if len(_box) < 4:
                    continue
                x1, y1, x2, y2 = [max(0, int(v)) for v in _box[:4]]
                x2 = min(x2, gray.shape[1] - 1)
                y2 = min(y2, gray.shape[0] - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                _roi_prev = self._last_small_gray[
                    max(0, y1 * 160 // gray.shape[0]):min(160, y2 * 160 // gray.shape[0]),
                    max(0, x1 * 160 // gray.shape[1]):min(160, x2 * 160 // gray.shape[1])
                ]
                _roi_cur = small[
                    max(0, y1 * 160 // gray.shape[0]):min(160, y2 * 160 // gray.shape[0]),
                    max(0, x1 * 160 // gray.shape[1]):min(160, x2 * 160 // gray.shape[1])
                ]
                if _roi_prev.size == 0 or _roi_cur.size == 0 or _roi_prev.shape != _roi_cur.shape:
                    continue
                _roi_change = float(cv2.absdiff(_roi_cur, _roi_prev).mean() / 255.0)
                if _roi_change > self._temporal_reuse_threshold * 1.5:
                    return force_detection("target_roi_change_exceeded")

        if change_score <= self._temporal_reuse_threshold:
            self._temporal_reuse_consecutive += 1
            record_decision(
                hit=True,
                reason="reused",
                processed_gap=processed_gap,
                source_frame_gap=source_frame_gap,
                source_time_gap_s=source_time_gap_s,
                ppe_sensitive=ppe_sensitive,
                change_score=change_score,
            )
            return self._last_detections, self._last_rois, 0.0, change_score
        return force_detection("global_change_exceeded")

    def _release_a3b_suppression_lease(self) -> None:
        """Release the active bbox lease without rewinding its logical clock."""
        self._a3b_suppress_remaining = 0
        self._a3b_suppress_bbox = None
        self._a3b_suppress_result_seq = None
        self._a3b_suppress_lease_expires_at_s: float | None = None

    def _clear_a3b_suppression_lease(self) -> None:
        """Clear suppression state across reset, close, and source discontinuity."""
        self._release_a3b_suppression_lease()
        self._a3b_suppress_clock_s: float | None = None
        self._a3b_suppress_clock_basis: str | None = None
        self._a3b_suppress_last_source_time_s: float | None = None

    def _resolve_a3b_suppression_time(
        self,
        source_timestamp_s: float | None,
    ) -> tuple[float, str, str | None]:
        """Resolve a monotonic lease clock from source time or fixed cadence."""
        previous_clock = getattr(self, "_a3b_suppress_clock_s", None)
        previous_basis = getattr(self, "_a3b_suppress_clock_basis", None)
        last_source_time = getattr(
            self,
            "_a3b_suppress_last_source_time_s",
            None,
        )
        source_time: float | None = None
        if source_timestamp_s is not None:
            candidate = float(source_timestamp_s)
            if np.isfinite(candidate) and candidate >= 0.0:
                source_time = candidate

        clock_reset_reason: str | None = None
        if source_time is not None:
            if (
                last_source_time is not None
                and source_time < float(last_source_time)
            ):
                self._clear_a3b_suppression_lease()
                previous_clock = None
                previous_basis = None
                clock_reset_reason = "source_timestamp_rewind"
            elif (
                last_source_time is None
                and previous_basis == "cadence_fallback"
                and previous_clock is not None
            ):
                expiry = getattr(
                    self,
                    "_a3b_suppress_lease_expires_at_s",
                    None,
                )
                if expiry is not None:
                    remaining = max(0.0, float(expiry) - float(previous_clock))
                    self._a3b_suppress_lease_expires_at_s = (
                        source_time + remaining
                    )

            self._a3b_suppress_clock_s = source_time
            self._a3b_suppress_clock_basis = "source_timestamp"
            self._a3b_suppress_last_source_time_s = source_time
            return source_time, "source_timestamp", clock_reset_reason

        fallback_fps = float(
            getattr(
                self,
                "_offline_default_source_fps",
                self._DEFAULT_OFFLINE_SOURCE_FPS,
            )
        )
        if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
            fallback_fps = self._DEFAULT_OFFLINE_SOURCE_FPS
        fallback_step_s = 1.0 / fallback_fps
        fallback_time = (
            float(previous_clock) + fallback_step_s
            if previous_clock is not None
            else fallback_step_s
        )
        self._a3b_suppress_clock_s = fallback_time
        self._a3b_suppress_clock_basis = "cadence_fallback"
        return fallback_time, "cadence_fallback", None

    def _apply_a3b_suppression(
        self,
        frame_640: np.ndarray,
        detections: DetectionFrameResult,
        rois: list[Any],
        info: dict[str, Any],
        *,
        source_timestamp_s: float | None = None,
    ) -> tuple[DetectionFrameResult, list[Any]]:
        """Suppress YOLO boxes inside A3b-triggered fake-video region.

        A fresh, policy-allowed, confirmed bbox grants a source-time lease.
        Stale or policy-blocked results cannot renew it and clamp any existing
        lease to a short bridge. Missing source time advances on a deterministic
        configured cadence; the legacy frame counter remains diagnostic only.
        """
        hold_s = max(
            0.0,
            float(
                getattr(
                    self,
                    "_a3b_suppression_hold_s",
                    self._A3B_SUPPRESSION_HOLD_S,
                )
            ),
        )
        stale_bridge_s = min(
            hold_s,
            max(
                0.0,
                float(
                    getattr(
                        self,
                        "_a3b_suppression_stale_bridge_s",
                        self._A3B_SUPPRESSION_STALE_BRIDGE_S,
                    )
                ),
            ),
        )
        now_s, clock_basis, clock_reset_reason = (
            self._resolve_a3b_suppression_time(source_timestamp_s)
        )
        if clock_reset_reason is not None:
            info["a3b_suppression_clock_reset_reason"] = clock_reset_reason

        # Read both legacy and rebuilt A3b through the shared result contract.
        static_media = adapt_a3b_result(info)
        if "media_confirmed" in static_media or "confirmed" in static_media:
            a3b_triggered = bool(
                static_media.get(
                    "media_confirmed",
                    static_media.get("confirmed", False),
                )
            )
        else:
            a3b_triggered = bool(
                static_media.get(
                    "static_image_triggered",
                    static_media.get("triggered", False),
                )
            )
        p_media_bbox = static_media.get("p_media_bbox")
        a3b_result_seq = static_media.get("a3b_result_seq")
        is_rebuilt_result = (
            static_media.get("result_contract_source") == "rebuilt"
        )
        policy = static_media.get("policy")
        policy = dict(policy) if isinstance(policy, dict) else {}
        suppression = static_media.get("suppression")
        suppression = (
            dict(suppression) if isinstance(suppression, dict) else {}
        )
        candidate_allowed_value = static_media.get(
            "media_candidate_allowed",
            policy.get(
                "media_candidate_allowed",
                suppression.get("media_candidate_allowed"),
            ),
        )
        policy_suppressed = bool(
            policy.get("suppressed", False)
            or suppression.get("suppressed", False)
        )
        freshness_value = static_media.get("a3b_result_fresh")
        result_fresh = bool(
            not is_rebuilt_result or freshness_value is True
        )
        refresh_blocked_reason: str | None = None
        if policy_suppressed:
            refresh_blocked_reason = "policy_suppressed"
        elif candidate_allowed_value is False:
            refresh_blocked_reason = "media_candidate_not_allowed"
        elif is_rebuilt_result and freshness_value is False:
            refresh_blocked_reason = "stale_result"
        elif is_rebuilt_result and freshness_value is None:
            refresh_blocked_reason = "freshness_missing"
        elif not a3b_triggered:
            refresh_blocked_reason = "not_confirmed"

        # --- 1. fresh + allowed + confirmed bbox → refresh source-time lease ---
        confirmed_bbox: tuple[int, int, int, int] | None = None
        if (
            a3b_triggered
            and isinstance(p_media_bbox, (list, tuple))
            and len(p_media_bbox) == 4
        ):
            x1, y1, x2, y2 = [max(0, int(v)) for v in p_media_bbox[:4]]
            x2 = min(x2, frame_640.shape[1] - 1)
            y2 = min(y2, frame_640.shape[0] - 1)
            if x2 > x1 and y2 > y1:
                confirmed_bbox = (x1, y1, x2, y2)
        if a3b_triggered and confirmed_bbox is None and refresh_blocked_reason is None:
            refresh_blocked_reason = "invalid_bbox"

        refresh_eligible = bool(
            confirmed_bbox is not None
            and result_fresh
            and candidate_allowed_value is not False
            and not policy_suppressed
            and refresh_blocked_reason is None
        )
        if refresh_eligible:
            self._a3b_suppress_bbox = confirmed_bbox
            self._a3b_suppress_lease_expires_at_s = now_s + hold_s
            self._a3b_suppress_result_seq = a3b_result_seq
            info["a3b_suppression_refreshed"] = True
        elif refresh_blocked_reason in {
            "policy_suppressed",
            "media_candidate_not_allowed",
            "stale_result",
            "freshness_missing",
        }:
            expiry = getattr(
                self,
                "_a3b_suppress_lease_expires_at_s",
                None,
            )
            if (
                expiry is not None
                and self._a3b_suppress_bbox is not None
                and float(expiry) > now_s
            ):
                clamped_expiry = min(
                    float(expiry),
                    now_s + stale_bridge_s,
                )
                if clamped_expiry < float(expiry):
                    self._a3b_suppress_lease_expires_at_s = clamped_expiry
                    info["a3b_suppression_lease_clamped"] = True

        expiry = getattr(
            self,
            "_a3b_suppress_lease_expires_at_s",
            None,
        )
        remaining_s = (
            max(0.0, float(expiry) - now_s)
            if expiry is not None and self._a3b_suppress_bbox is not None
            else 0.0
        )
        fallback_fps = float(
            getattr(
                self,
                "_offline_default_source_fps",
                self._DEFAULT_OFFLINE_SOURCE_FPS,
            )
        )
        if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
            fallback_fps = self._DEFAULT_OFFLINE_SOURCE_FPS
        hold_frames = max(0, int(np.ceil(hold_s * fallback_fps)))
        self._a3b_suppress_remaining = max(
            0,
            int(np.ceil(remaining_s * fallback_fps)),
        )

        suppression_relevant = bool(
            a3b_triggered
            or p_media_bbox is not None
            or self._a3b_suppress_bbox is not None
            or expiry is not None
        )
        if suppression_relevant:
            info["a3b_suppression_hold_frames"] = hold_frames
            info["a3b_suppression_hold_s"] = hold_s
            info["a3b_suppression_stale_bridge_s"] = stale_bridge_s
            info["a3b_suppression_remaining_s"] = remaining_s
            info["a3b_suppression_refresh_eligible"] = refresh_eligible
            info["a3b_suppression_clock_basis"] = clock_basis
            if refresh_blocked_reason is not None:
                info["a3b_suppression_refresh_blocked_reason"] = (
                    refresh_blocked_reason
                )

        # --- 2. Active lease filters detections; expiry is source-time based. ---
        if remaining_s > 0.0:
            info["a3b_suppression_active"] = True
            info["a3b_suppression_remaining"] = self._a3b_suppress_remaining

            # 过滤 bbox 内的框
            bbox = self._a3b_suppress_bbox
            if bbox:
                sx1, sy1, sx2, sy2 = bbox
                kept_boxes: list[list[int]] = []
                kept_classes: list[int] = []
                kept_confs: list[float] = []
                for box, cls_id, conf in zip(detections.boxes, detections.classes, detections.confidences):
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                        continue
                    kept_boxes.append(box)
                    kept_classes.append(cls_id)
                    kept_confs.append(conf)
                detections.boxes = kept_boxes
                detections.classes = kept_classes
                detections.confidences = kept_confs
                # 同步过滤 rois
                if rois:
                    kept_rois = []
                    for roi in rois:
                        if hasattr(roi, "bbox"):
                            rcx = (roi.bbox[0] + roi.bbox[2]) / 2.0
                            rcy = (roi.bbox[1] + roi.bbox[3]) / 2.0
                            if sx1 <= rcx <= sx2 and sy1 <= rcy <= sy2:
                                continue
                        kept_rois.append(roi)
                    rois = kept_rois
                info["a3b_suppression_filtered"] = True
        elif expiry is not None or self._a3b_suppress_bbox is not None:
            self._release_a3b_suppression_lease()
            info["a3b_suppression_remaining"] = 0
            info["a3b_suppression_remaining_s"] = 0.0
            info["a3b_suppression_released"] = True

        return detections, rois

    def _run_detection_with_source_context(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
        current_source_frame_idx: int | None,
        current_source_time_s: float | None,
        detector_input: Any | None = None,
    ) -> tuple[np.ndarray, Any, dict[str, Any], float, float]:
        """Run detection while exposing source lineage to the reuse gate."""
        previous_frame_idx = getattr(
            self,
            "_current_reuse_source_frame_idx",
            None,
        )
        previous_time_s = getattr(
            self,
            "_current_reuse_source_time_s",
            None,
        )
        self._current_reuse_source_frame_idx = current_source_frame_idx
        self._current_reuse_source_time_s = current_source_time_s
        try:
            if detector_input is None:
                return self._run_detection(
                    frame,
                    timestamp=timestamp,
                )
            return self._run_detection(
                frame,
                timestamp=timestamp,
                detector_input=detector_input,
            )
        finally:
            self._current_reuse_source_frame_idx = previous_frame_idx
            self._current_reuse_source_time_s = previous_time_s

    def _observe_module_a_source_cadence(
        self,
        *,
        current_source_frame_idx: int | None,
        current_source_time_s: float | None,
    ) -> float | None:
        """Estimate real source FPS even when heavy Module A work is sampled."""
        observed = getattr(self, "_module_a_source_fps", None)
        previous_frame_idx = getattr(
            self,
            "_last_module_a_seen_source_frame_idx",
            None,
        )
        previous_time_s = getattr(
            self,
            "_last_module_a_seen_source_time_s",
            None,
        )
        if current_source_frame_idx is not None and current_source_time_s is not None:
            current_frame_idx = int(current_source_frame_idx)
            current_time_s = float(current_source_time_s)
            if (
                previous_frame_idx is not None
                and previous_time_s is not None
                and current_frame_idx > int(previous_frame_idx)
                and np.isfinite(current_time_s)
                and current_time_s > float(previous_time_s)
            ):
                instant = (
                    float(current_frame_idx - int(previous_frame_idx))
                    / float(current_time_s - float(previous_time_s))
                )
                instant = min(240.0, max(1.0, instant))
                observed = (
                    instant
                    if observed is None or not np.isfinite(float(observed))
                    else 0.80 * float(observed) + 0.20 * instant
                )
                self._module_a_source_fps = float(observed)
            self._last_module_a_seen_source_frame_idx = current_frame_idx
            self._last_module_a_seen_source_time_s = current_time_s

        detector = getattr(self, "detector", None)
        if detector is not None and observed is not None:
            detector.source_fps = float(observed)
        return float(observed) if observed is not None else None

    def _module_a_cadence_decision(
        self,
        *,
        current_source_frame_idx: int | None,
        current_source_time_s: float | None,
    ) -> dict[str, Any]:
        """Return a fail-closed cadence decision for heavy Module A reuse.

        Sources at or below the 30 FPS calibration rate keep analysing every
        processed frame while source-frame lineage is contiguous.  Faster
        sources, and lower-rate sources whose latest-only consumer has already
        skipped frames, use a fractional processed-frame budget derived from
        ``analysis_max_hz / source_fps``.  This is deliberately independent
        from the gap between surviving packets: a large source gap must not
        force heavy Module A work on every processed frame and thereby make the
        gap grow even larger.

        A source-time stale cap remains fail-closed.  Whenever the cached
        analysis is older than a small multiple of the configured interval, a
        fresh analysis is forced.  Actual analysis still receives the strict
        source predecessor supplied by the runtime packet.
        """
        if not hasattr(self, "_module_a_cadence_attempt_count"):
            self._module_a_cadence_attempt_count = 0
            self._module_a_cadence_hit_count = 0
        self._module_a_cadence_attempt_count += 1
        source_fps = self._observe_module_a_source_cadence(
            current_source_frame_idx=current_source_frame_idx,
            current_source_time_s=current_source_time_s,
        )
        max_hz = float(getattr(self, "_module_a_analysis_max_hz", 25.0))
        interval_s = 1.0 / max_hz if max_hz > 0.0 else 0.0
        last_frame_idx = getattr(self, "_last_module_a_source_frame_idx", None)
        last_time_s = getattr(self, "_last_module_a_source_time_s", None)
        source_frame_gap = (
            int(current_source_frame_idx) - int(last_frame_idx)
            if current_source_frame_idx is not None and last_frame_idx is not None
            else None
        )
        source_time_gap_s = (
            float(current_source_time_s) - float(last_time_s)
            if current_source_time_s is not None and last_time_s is not None
            else None
        )
        processed_phase_before = float(
            getattr(self, "_module_a_processed_cadence_phase", 0.0)
        )
        processed_budget_share = None
        processed_phase_after = processed_phase_before
        max_staleness_s = (
            interval_s * float(self._MODULE_A_CADENCE_MAX_STALE_INTERVALS)
            if interval_s > 0.0
            else 0.0
        )
        reason = "cadence_due"
        hit = False
        if max_hz <= 0.0:
            reason = "cadence_disabled"
            self._module_a_processed_cadence_phase = 0.0
        elif getattr(self, "_last_module_a_result", None) is None:
            reason = "no_cached_module_a_result"
            self._module_a_processed_cadence_phase = 0.0
        elif source_frame_gap is None or source_time_gap_s is None:
            reason = "source_context_missing"
            self._module_a_processed_cadence_phase = 0.0
        elif source_frame_gap <= 0:
            reason = "source_frame_gap_not_forward"
            self._module_a_processed_cadence_phase = 0.0
        elif not np.isfinite(source_time_gap_s) or source_time_gap_s <= 0.0:
            reason = "source_time_gap_not_forward"
            self._module_a_processed_cadence_phase = 0.0
        elif source_fps is None or not np.isfinite(float(source_fps)):
            reason = "source_rate_unavailable"
            self._module_a_processed_cadence_phase = 0.0
        elif float(source_fps) <= 30.5 and source_frame_gap <= 1:
            # Preserve the calibrated every-frame path while the consumer is
            # actually keeping up.  If latest-only has already skipped source
            # frames, fall through to the processed budget below so a 30 FPS
            # source cannot enter the same every-survivor feedback loop.
            reason = "source_at_or_below_calibration_rate"
            self._module_a_processed_cadence_phase = 0.0
        else:
            processed_budget_share = min(
                1.0,
                max(0.0, max_hz / max(float(source_fps), 1e-6)),
            )
            processed_phase_after = processed_phase_before + processed_budget_share
            stale_due = bool(
                max_staleness_s > 0.0
                and source_time_gap_s + 1e-6 >= max_staleness_s
            )
            if stale_due:
                reason = "source_staleness_due"
                processed_phase_after = 0.0
            elif processed_phase_after + 1e-9 < 1.0:
                hit = True
                reason = "reused_processed_budget"
            else:
                reason = "processed_budget_due"
                processed_phase_after = max(0.0, processed_phase_after - 1.0)
            self._module_a_processed_cadence_phase = float(
                min(1.0, processed_phase_after)
            )

        decision = {
            "hit": bool(hit),
            "reason": reason,
            "analysis_max_hz": max_hz,
            "analysis_interval_s": interval_s,
            "source_frame_gap": source_frame_gap,
            "source_time_gap_s": source_time_gap_s,
            "source_fps": source_fps,
            "processed_budget_share": processed_budget_share,
            "processed_phase_before": processed_phase_before,
            "processed_phase_after": float(
                getattr(self, "_module_a_processed_cadence_phase", 0.0)
            ),
            "max_staleness_s": max_staleness_s,
            "last_analysis_source_frame_idx": last_frame_idx,
            "last_analysis_source_time_s": last_time_s,
        }
        self._last_module_a_cadence = dict(decision)
        return decision

    @staticmethod
    def _zero_reused_module_a_timing(module_result: Any) -> None:
        if hasattr(module_result, "timing_ms"):
            module_result.timing_ms = 0.0
        details = getattr(module_result, "details", None)
        if not isinstance(details, dict):
            return
        timing = details.get("timing")
        if isinstance(timing, dict):
            details["timing"] = {
                key: 0.0 if isinstance(value, (int, float)) else value
                for key, value in timing.items()
            }

    def _run_detection(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
        detector_input: Any | None = None,
    ) -> tuple[np.ndarray, Any, dict[str, Any], float, float]:
        """Shared Module A inference used by both offline and streaming paths.

        Returns
        -------
        tuple
            ``(frame_640, detections, info, total_timing_ms, module_a_timing_ms)``.
            The ``info`` dict already carries the triple-channel contract, the
            detection details, and a stub ``latency_breakdown`` block with only
            the detector / Module A / total timings filled in. Streaming
            callers overwrite the stream-specific fields afterwards.
        """
        started = time.perf_counter()
        source_frame_shape = tuple(int(v) for v in frame.shape[:2])
        frame_resize_ms = 0.0
        if frame.shape[0] == 640 and frame.shape[1] == 640:
            frame_640 = frame
        else:
            resize_started = time.perf_counter()
            frame_640 = cv2.resize(frame, (640, 640))
            frame_resize_ms = (time.perf_counter() - resize_started) * 1000.0
        current_source_frame_idx = getattr(
            self,
            "_current_reuse_source_frame_idx",
            None,
        )
        current_source_time_s = getattr(
            self,
            "_current_reuse_source_time_s",
            None,
        )
        reused_detections, reused_rois, reused_detector_ms, change_score = self._maybe_reuse_detections(
            frame_640,
            current_source_frame_idx=current_source_frame_idx,
            current_source_time_s=current_source_time_s,
        )
        detector_reuse_hit = reused_detections is not None and reused_rois is not None
        if reused_detections is not None and reused_rois is not None:
            detections = DetectionFrameResult(
                image=frame_640,
                boxes=[list(box) for box in reused_detections.boxes],
                classes=list(reused_detections.classes),
                confidences=list(reused_detections.confidences),
                names=reused_detections.names,
                backend=reused_detections.backend,
                artifact_path=reused_detections.artifact_path,
                inference_ms=float(reused_detector_ms),
                raw_result=None,
                preprocess_ms=float(
                    getattr(reused_detections, "preprocess_ms", 0.0) or 0.0
                ),
                input_device=str(
                    getattr(reused_detections, "input_device", "host")
                    or "host"
                ),
                input_format=str(
                    getattr(reused_detections, "input_format", "bgr24")
                    or "bgr24"
                ),
            )
            rois = reused_rois
            detector_inference_ms = reused_detector_ms
        else:
            predict_cuda = getattr(self.detector_backend, "predict_cuda", None)
            if detector_input is not None and callable(predict_cuda):
                detections = predict_cuda(detector_input, image=frame_640)
            else:
                detections = self.detector_backend.predict(frame_640)
            self._detector_backend_predict_count = (
                int(getattr(self, "_detector_backend_predict_count", 0)) + 1
            )
            rois = self.roi_provider.from_detections(
                detections.boxes, detections.classes, detections.confidences
            )
            gray_small = cv2.resize(cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY), (160, 160), interpolation=cv2.INTER_AREA)
            self._last_small_gray = gray_small
            self._last_detections = detections
            self._last_rois = rois
            self._last_detector_frame_idx = self.frame_idx
            self._last_detector_source_frame_idx = (
                int(current_source_frame_idx)
                if current_source_frame_idx is not None
                else None
            )
            self._last_detector_source_time_s = (
                float(current_source_time_s)
                if current_source_time_s is not None
                else None
            )
            detector_inference_ms = float(detections.inference_ms)
        module_a_source_frame_idx = (
            int(current_source_frame_idx)
            if current_source_frame_idx is not None
            else int(self.frame_idx)
        )
        precomputed_module_a_cadence = getattr(
            self,
            "_current_module_a_cadence_decision",
            None,
        )
        if isinstance(precomputed_module_a_cadence, dict):
            module_a_cadence = dict(precomputed_module_a_cadence)
        else:
            module_a_cadence = {
                "hit": False,
                "reason": "strict_predecessor_unavailable",
                "analysis_max_hz": float(
                    getattr(self, "_module_a_analysis_max_hz", 25.0)
                ),
                "analysis_interval_s": 0.0,
                "source_frame_gap": None,
                "source_time_gap_s": None,
                "source_fps": self._observe_module_a_source_cadence(
                    current_source_frame_idx=current_source_frame_idx,
                    current_source_time_s=current_source_time_s,
                ),
                "last_analysis_source_frame_idx": getattr(
                    self,
                    "_last_module_a_source_frame_idx",
                    None,
                ),
                "last_analysis_source_time_s": getattr(
                    self,
                    "_last_module_a_source_time_s",
                    None,
                ),
            }
            self._last_module_a_cadence = dict(module_a_cadence)
        module_result = None
        if module_a_cadence["hit"]:
            try:
                module_result = copy.deepcopy(self._last_module_a_result)
            except Exception:
                module_a_cadence["hit"] = False
                module_a_cadence["reason"] = "cached_result_copy_failed"
                self._last_module_a_cadence = dict(module_a_cadence)
            else:
                self._module_a_cadence_hit_count += 1
                if hasattr(module_result, "frame_idx"):
                    module_result.frame_idx = module_a_source_frame_idx
                self._zero_reused_module_a_timing(module_result)
        if module_result is None:
            module_result = self.detector.process(
                ModuleAInput(
                    frame=frame_640,
                    # A3b scheduling/result lineage must follow the real source
                    # clock.  ``self.frame_idx`` is only the processed-frame
                    # sequence and diverges under latest-only drops.
                    frame_idx=module_a_source_frame_idx,
                    timestamp=float(timestamp),
                    rois=rois,
                )
            )
            self._last_module_a_result = copy.deepcopy(module_result)
            self._last_module_a_source_frame_idx = (
                int(current_source_frame_idx)
                if current_source_frame_idx is not None
                else None
            )
            self._last_module_a_source_time_s = (
                float(current_source_time_s)
                if current_source_time_s is not None
                else None
            )
        info = module_result.to_info_dict()
        info.setdefault("details", {})["module_a_cadence"] = dict(
            module_a_cadence
        )
        info.setdefault("details", {})["runtime_frame_lineage"] = {
            "processed_frame_idx": int(self.frame_idx),
            "source_frame_idx": (
                int(current_source_frame_idx)
                if current_source_frame_idx is not None
                else None
            ),
            "module_a_input_frame_idx": module_a_source_frame_idx,
        }
        total_timing_ms = (time.perf_counter() - started) * 1000.0
        module_a_timing_ms = float(info.get("timing_ms", 0.0))
        info["timing_ms"] = total_timing_ms
        info["module_a_timing_ms"] = module_a_timing_ms
        raw_classes = [self.class_names.get(c, f"class_{c}") for c in detections.classes[:20]]
        info["details"]["detections"] = {
            "roi_count": len(rois),
            "boxes": detections.boxes[:20],
            "classes": raw_classes,
            "normalized_classes": [self.roi_provider.normalize_label(v) for v in raw_classes],
            "target_labels": sorted(self.roi_provider.target_labels),
            "class_ids": detections.classes[:20],
            "confidences": [float(v) for v in detections.confidences[:20]],
            "backend": detections.backend,
            "artifact_path": detections.artifact_path,
            "inference_ms": float(detector_inference_ms),
        }

        # --- Branch contract (p_adv / p_safety) ---
        # A3b is exposed through details.module_a_features.static_media and the monitor status.
        p_adv_value = getattr(module_result, "p_adv", None)
        p_adv_display_value = (
            info.get("details", {})
            .get("module_a", {})
            .get("p_adv_display")
        )
        if p_adv_value is None:
            info["p_adv"] = None
            info["p_adv_missing_reason"] = "module_a_p_adv_unavailable"
        else:
            info["p_adv"] = float(p_adv_value)
            info["p_adv_display"] = float(
                p_adv_display_value if p_adv_display_value is not None else p_adv_value
            )

        # p_safety 业务侧尚未接入 Module A pipeline，此处固定落 null + 原因。
        info["p_safety"] = None
        info["p_safety_missing_reason"] = "p_safety 业务侧未接入"

        info["reason_codes"] = module_result.reason_codes
        info["detector_backend"] = detections.backend
        info["detector_inference_ms"] = float(detector_inference_ms)
        info["detector_change_score"] = float(change_score)

        # --- Latency breakdown 预留结构（Requirements 5.1-5.6）---
        # 离线路径下 source_to_decode_ms / decode_to_process_ms / e2e_ms 没有真实
        # 时间戳可填，保持 ``None``；streaming 路径由 process_envelope 覆写。
        #
        # ``module_a_breakdown`` 来自 ``ModuleADetector.process``：6 个字段分别
        # 对应 A1 / A2 / A3 / A3b / A4 / Source_Authenticity 的打点耗时（tasks.md
        # §3.1）。若 ModuleADetector 出现老版本（未注入 breakdown）或 details 被
        # 外部截断导致字段缺失，这里保持 ``{}`` 以兼容现有聚合脚本。
        details = info.setdefault("details", {})
        stage_timing_raw = details.get("timing")
        stage_timing: dict[str, float] = {}
        if isinstance(stage_timing_raw, dict):
            for key, value in stage_timing_raw.items():
                try:
                    stage_timing[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        breakdown_raw = details.get("module_a_breakdown")
        module_a_breakdown: dict[str, float] = {}
        if isinstance(breakdown_raw, dict):
            for key, value in breakdown_raw.items():
                try:
                    module_a_breakdown[str(key)] = float(value)
                except (TypeError, ValueError):
                    # Silently skip non-numeric entries rather than raise, so a
                    # single malformed field cannot break an entire run.
                    continue
        if not module_a_breakdown:
            module_a_breakdown = {
                key: value
                for key, value in stage_timing.items()
                if key
                not in {
                    "detector_ms",
                    "module_a_ms",
                    "pipeline_ms",
                    "total",
                    "total_ms",
                }
            }

        info["latency_breakdown"] = {
            "source_to_decode_ms": None,
            "decode_to_process_ms": None,
            "detector_ms": float(detections.inference_ms),
            "detector_preprocess_ms": float(
                getattr(detections, "preprocess_ms", 0.0) or 0.0
            ),
            "detector_input_device": str(
                getattr(detections, "input_device", "host") or "host"
            ),
            "detector_input_format": str(
                getattr(detections, "input_format", "bgr24") or "bgr24"
            ),
            "module_a_total_ms": float(module_a_timing_ms),
            "module_a_reuse_hit": bool(module_a_cadence["hit"]),
            "module_a_cadence": dict(module_a_cadence),
            "module_a_cadence_counters": {
                "attempt_count": int(
                    getattr(self, "_module_a_cadence_attempt_count", 0)
                ),
                "hit_count": int(
                    getattr(self, "_module_a_cadence_hit_count", 0)
                ),
                "hit_rate": (
                    float(getattr(self, "_module_a_cadence_hit_count", 0))
                    / float(getattr(self, "_module_a_cadence_attempt_count", 0))
                    if int(getattr(self, "_module_a_cadence_attempt_count", 0)) > 0
                    else 0.0
                ),
            },
            "frame_resize_ms": float(frame_resize_ms),
            "detector_reuse_hit": bool(detector_reuse_hit),
            "detector_change_score": float(change_score),
            "detector_reuse": dict(
                getattr(
                    self,
                    "_last_reuse_decision",
                    {"hit": False, "reason": "unavailable"},
                )
            ),
            "detector_reuse_counters": {
                "attempt_count": int(
                    getattr(self, "_detector_reuse_attempt_count", 0)
                ),
                "hit_count": int(
                    getattr(self, "_detector_reuse_hit_count", 0)
                ),
                "miss_count": max(
                    0,
                    int(getattr(self, "_detector_reuse_attempt_count", 0))
                    - int(getattr(self, "_detector_reuse_hit_count", 0)),
                ),
                "hit_rate": (
                    float(getattr(self, "_detector_reuse_hit_count", 0))
                    / float(getattr(self, "_detector_reuse_attempt_count", 0))
                    if int(getattr(self, "_detector_reuse_attempt_count", 0)) > 0
                    else 0.0
                ),
                "backend_predict_count": int(
                    getattr(self, "_detector_backend_predict_count", 0)
                ),
                "miss_reasons": dict(
                    getattr(self, "_detector_reuse_miss_reasons", {})
                ),
            },
            "source_frame_shape": list(source_frame_shape),
            "detector_frame_shape": [int(frame_640.shape[0]), int(frame_640.shape[1])],
            "module_a_breakdown": module_a_breakdown,
            "e2e_ms": float(total_timing_ms),
        }

        stage_timing.update(
            {
                "pipeline_ms": float(total_timing_ms),
                "detector_ms": float(detections.inference_ms),
                "module_a_ms": float(module_a_timing_ms),
            }
        )
        details["timing"] = stage_timing

        # A3b 假视频区域检测框抑制 (2026-06-11)
        detections, rois = self._apply_a3b_suppression(
            frame_640,
            detections,
            rois,
            info,
            source_timestamp_s=current_source_time_s,
        )
        # Rebuild detection info dict after suppression
        if info.get("a3b_suppression_active"):
            raw_classes = [self.class_names.get(c, f"class_{c}") for c in detections.classes[:20]]
            info["details"]["detections"].update({
                "roi_count": len(rois),
                "boxes": detections.boxes[:20],
                "classes": raw_classes,
                "normalized_classes": [self.roi_provider.normalize_label(v) for v in raw_classes],
                "class_ids": detections.classes[:20],
                "confidences": [float(v) for v in detections.confidences[:20]],
            })

        self.frame_idx += 1
        return frame_640, detections, info, total_timing_ms, module_a_timing_ms

    def _can_reuse_internal_temporal_predecessor(
        self,
        frame: Any,
        previous_frame: Any | None,
        *,
        current_source_frame_idx: int | None,
        previous_source_frame_idx: int | None,
        previous_source_time_s: float | None,
    ) -> bool:
        """Keep detector temporal state when it already represents the predecessor."""
        if (
            previous_frame is None
            or current_source_frame_idx is None
            or previous_source_frame_idx is None
        ):
            return False
        last_frame_idx = getattr(self, "_last_runtime_source_frame_idx", None)
        if (
            last_frame_idx is None
            or int(previous_source_frame_idx) != int(last_frame_idx)
            or int(current_source_frame_idx) != int(previous_source_frame_idx) + 1
        ):
            return False
        # Runtime lineage advances on every processed source frame, while the
        # heavy Module A cadence may intentionally hold a decision.  In that
        # case detector.prev_gray still represents an older analyzed frame and
        # must be replaced with the strict source predecessor before the next
        # analysis; otherwise 60 FPS cadence sampling silently turns A2/A3 into
        # multi-frame differences and reintroduces target-motion false alarms.
        last_module_a_frame_idx = getattr(
            self,
            "_last_module_a_source_frame_idx",
            None,
        )
        if (
            last_module_a_frame_idx is not None
            and int(previous_source_frame_idx) != int(last_module_a_frame_idx)
        ):
            return False
        previous_shape = tuple(int(value) for value in previous_frame.shape[:2])
        current_shape = tuple(int(value) for value in frame.shape[:2])
        last_shape = getattr(self, "_last_runtime_frame_shape", None)
        if previous_shape != current_shape or last_shape != previous_shape:
            return False
        last_time_s = getattr(self, "_last_runtime_source_time_s", None)
        if previous_source_time_s is None or last_time_s is None:
            return False
        if abs(float(previous_source_time_s) - float(last_time_s)) > 1e-4:
            return False
        detector = getattr(self, "detector", None)
        return bool(
            detector is not None
            and getattr(detector, "prev_gray", None) is not None
            and getattr(detector, "prev_lbp", None) is not None
        )

    def _inject_temporal_previous_frame(self, previous_frame: Any | None) -> bool:
        """Prime A2/A3 with the source frame immediately before the current frame."""
        if previous_frame is None:
            return False
        try:
            if previous_frame.shape[0] == 640 and previous_frame.shape[1] == 640:
                previous_640 = previous_frame
            else:
                previous_640 = cv2.resize(previous_frame, (640, 640))
            previous_gray = (
                previous_640.astype(np.uint8)
                if previous_640.ndim == 2
                else cv2.cvtColor(previous_640, cv2.COLOR_BGR2GRAY)
            )
            compute_lbp = getattr(self.detector, "_compute_lbp", None)
            if not callable(compute_lbp) or not hasattr(self.detector, "prev_gray"):
                return False
            self.detector.prev_gray = previous_gray
            self.detector.prev_lbp = compute_lbp(previous_gray)
            if hasattr(self.detector, "prev_brightness"):
                self.detector.prev_brightness = float(np.mean(previous_gray))
            return True
        except Exception:
            return False

    def _clear_temporal_predecessor_state(self) -> None:
        """Fail closed instead of comparing against a stale predecessor."""
        detector = getattr(self, "detector", None)
        if detector is None:
            return
        for attribute_name in (
            "prev_gray",
            "prev_lbp",
            "_last_computed_lbp",
            "prev_brightness",
        ):
            if hasattr(detector, attribute_name):
                setattr(detector, attribute_name, None)
        flownet = getattr(detector, "_flownet", None)
        if isinstance(flownet, dict):
            flownet.pop("prev_ref", None)
            flownet.pop("prev_small", None)
        overexposure = getattr(detector, "overexposure", None)
        if overexposure is not None and hasattr(overexposure, "_prev_gray"):
            overexposure._prev_gray = None

    def _resolve_offline_timing(
        self,
        *,
        timestamp: float | None,
        source_fps: float | None,
    ) -> tuple[float, float, bool]:
        """Resolve deterministic offline time without consulting wall clock."""
        effective_fps = (
            float(source_fps)
            if source_fps is not None
            else float(
                getattr(
                    self,
                    "_offline_default_source_fps",
                    self._DEFAULT_OFFLINE_SOURCE_FPS,
                )
            )
        )
        if not np.isfinite(effective_fps) or effective_fps <= 0.0:
            raise ValueError("source_fps must be a finite positive number")

        if timestamp is None:
            return (
                (int(getattr(self, "frame_idx", 0)) + 1) / effective_fps,
                effective_fps,
                True,
            )

        raw_timestamp = float(timestamp)
        if not np.isfinite(raw_timestamp) or raw_timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative number")
        offset = getattr(
            self,
            "_offline_explicit_timestamp_offset_s",
            None,
        )
        if offset is None:
            # Rebuilt currently treats timestamp=0 as a wall-clock sentinel.
            # Preserve source cadence by applying one constant positive offset
            # to a zero-based explicit timeline.
            offset = 1.0 / effective_fps if raw_timestamp == 0.0 else 0.0
            self._offline_explicit_timestamp_offset_s = offset
        return raw_timestamp + float(offset), effective_fps, False

    def process_runtime_frame(
        self,
        frame: Any,
        *,
        timestamp: float,
        previous_frame: Any | None = None,
        current_source_frame_idx: int | None = None,
        previous_source_frame_idx: int | None = None,
        previous_source_time_s: float | None = None,
        detector_input: Any | None = None,
        previous_frame_provider: Callable[[], Any] | None = None,
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        """Runtime entry point with source time and strict temporal predecessor."""
        module_a_cadence = self._module_a_cadence_decision(
            current_source_frame_idx=current_source_frame_idx,
            current_source_time_s=float(timestamp),
        )
        previous_cadence_decision = getattr(
            self,
            "_current_module_a_cadence_decision",
            None,
        )
        self._current_module_a_cadence_decision = module_a_cadence
        try:
            temporal_previous_reused = False
            temporal_previous_injected = False
            temporal_previous_reset = False
            temporal_previous_failure_reason = "none"
            if not module_a_cadence["hit"]:
                if previous_frame is None and callable(previous_frame_provider):
                    try:
                        previous_frame = previous_frame_provider()
                    except Exception:
                        previous_frame = None
                        temporal_previous_failure_reason = (
                            "strict_predecessor_materialization_failed"
                        )
                temporal_previous_reused = (
                    self._can_reuse_internal_temporal_predecessor(
                        frame,
                        previous_frame,
                        current_source_frame_idx=current_source_frame_idx,
                        previous_source_frame_idx=previous_source_frame_idx,
                        previous_source_time_s=previous_source_time_s,
                    )
                )
                if not temporal_previous_reused:
                    if previous_frame is None:
                        self._clear_temporal_predecessor_state()
                        temporal_previous_reset = True
                        if temporal_previous_failure_reason == "none":
                            temporal_previous_failure_reason = (
                                "strict_predecessor_missing"
                            )
                    else:
                        temporal_previous_injected = (
                            self._inject_temporal_previous_frame(previous_frame)
                        )
                        if not temporal_previous_injected:
                            self._clear_temporal_predecessor_state()
                            temporal_previous_reset = True
                            temporal_previous_failure_reason = (
                                "strict_predecessor_injection_failed"
                            )
            temporal_previous_applied = bool(
                temporal_previous_reused or temporal_previous_injected
            )
            frame_640, detections, info, _, _ = (
                self._run_detection_with_source_context(
                    frame,
                    timestamp=float(timestamp),
                    current_source_frame_idx=current_source_frame_idx,
                    current_source_time_s=float(timestamp),
                    detector_input=detector_input,
                )
            )
        finally:
            self._current_module_a_cadence_decision = (
                previous_cadence_decision
            )
        self._last_runtime_source_frame_idx = (
            int(current_source_frame_idx)
            if current_source_frame_idx is not None
            else None
        )
        self._last_runtime_source_time_s = float(timestamp)
        self._last_runtime_frame_shape = tuple(
            int(value) for value in frame.shape[:2]
        )
        gap_frames = None
        if current_source_frame_idx is not None and previous_source_frame_idx is not None:
            gap_frames = int(current_source_frame_idx) - int(previous_source_frame_idx)
        info["temporal_input"] = {
            "previous_frame_applied": bool(temporal_previous_applied),
            "previous_frame_injected": bool(temporal_previous_injected),
            "previous_frame_reused_internal_state": bool(
                temporal_previous_reused
            ),
            "previous_frame_temporal_state_reset": bool(
                temporal_previous_reset
            ),
            "previous_frame_failure_reason": temporal_previous_failure_reason,
            "current_source_frame_idx": (
                int(current_source_frame_idx) if current_source_frame_idx is not None else None
            ),
            "previous_source_frame_idx": (
                int(previous_source_frame_idx) if previous_source_frame_idx is not None else None
            ),
            "source_gap_frames": gap_frames,
            "strict_source_predecessor": bool(temporal_previous_applied and gap_frames == 1),
            "current_source_time_s": float(timestamp),
            "previous_source_time_s": (
                float(previous_source_time_s) if previous_source_time_s is not None else None
            ),
        }
        return frame_640, detections, info

    # ------------------------------------------------------------------ offline

    def process_frame(
        self,
        frame: Any,
        *,
        timestamp: float | None = None,
        source_fps: float | None = None,
        source_frame_idx: int | None = None,
    ):
        """Offline MP4 / legacy path with deterministic source cadence.

        Existing ``process_frame(frame)`` callers remain valid. When the caller
        does not provide timing, frames are placed on a stable 30 FPS timeline
        (or ``offline_source_fps`` from config) instead of using machine-speed
        wall clock. Explicit zero-based timestamps are shifted by one constant
        frame interval because rebuilt reserves ``0`` as its wall-clock
        sentinel; this preserves all inter-frame deltas.

        Detection reuse remains conservative: callers must also provide the
        real ``source_frame_idx`` before cached detector output may be reused.
        """
        effective_timestamp, effective_fps, generated_timestamp = (
            self._resolve_offline_timing(
                timestamp=timestamp,
                source_fps=source_fps,
            )
        )
        frame_640, detections, info, _, _ = self._run_detection_with_source_context(
            frame,
            timestamp=effective_timestamp,
            current_source_frame_idx=source_frame_idx,
            current_source_time_s=effective_timestamp,
        )
        info["offline_timing"] = {
            "source_timestamp_s": float(effective_timestamp),
            "source_fps": float(effective_fps),
            "timestamp_generated": bool(generated_timestamp),
            "source_frame_idx": (
                int(source_frame_idx)
                if source_frame_idx is not None
                else None
            ),
        }
        return frame_640, detections, info

    # ------------------------------------------------------------------ streaming

    def process_envelope(
        self,
        envelope: FrameEnvelope,
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        """Streaming path. Consumes a :class:`FrameEnvelope` from StreamSource.

        Behaviour vs :meth:`process_frame`:

        * ``envelope.flags["stream_geometry_changed"]`` — reset the pipeline
          (and therefore ``ModuleADetector``) before running detection, so the
          new frame starts a fresh 3/5 window / fresh track state on the new
          geometry (Requirement 2.3 / design §2).
        * ``envelope.source_ts`` is forwarded to ``ModuleAInput.timestamp``
          so ``AlertState`` uses the real frame timestamp for its soft window
          constraint (Requirement 2.6).
        * ``info["latency_breakdown"]`` is filled with real stream timings:

            - ``source_to_decode_ms`` = decode_ts - source_ts
            - ``decode_to_process_ms`` = process_ts - decode_ts
            - ``e2e_ms``              = pipeline_end - source_ts
              (i.e. "帧到达拉流线程 → pipeline 处理完毕" per design.md)

        * ``info["source_ts"]`` / ``info["decode_ts"]`` / ``info["process_ts"]``
          / ``info["stream_flags"]`` are exposed at the top level so event
          files and latency aggregators can consume them without digging into
          the envelope object.
        """
        if envelope.flags.get("stream_geometry_changed"):
            # Reset the full pipeline (frame_idx) and ModuleADetector state
            # (alert queue, tracks, light-flow history, source-authenticity
            # window, static-media hold). After reset the current envelope
            # becomes frame 0 on the new geometry.
            self.reset()

        frame_640, detections, info, _, _ = self._run_detection_with_source_context(
            envelope.frame,
            timestamp=float(envelope.source_ts),
            current_source_frame_idx=int(envelope.frame_idx),
            current_source_time_s=float(envelope.source_ts),
        )
        pipeline_end_ts = time.monotonic()

        # --- Real stream-timing fields ---
        source_ts = float(envelope.source_ts)
        decode_ts = float(envelope.decode_ts)
        process_ts = float(envelope.process_ts)
        source_to_decode_ms = max(0.0, (decode_ts - source_ts) * 1000.0)
        decode_to_process_ms = max(0.0, (process_ts - decode_ts) * 1000.0)
        e2e_ms = max(0.0, (pipeline_end_ts - source_ts) * 1000.0)

        latency = info["latency_breakdown"]
        latency["source_to_decode_ms"] = source_to_decode_ms
        latency["decode_to_process_ms"] = decode_to_process_ms
        latency["e2e_ms"] = e2e_ms

        # --- Expose the raw stream timestamps + flags at info top level ---
        info["source_ts"] = source_ts
        info["decode_ts"] = decode_ts
        info["process_ts"] = process_ts
        # Copy flags so downstream consumers (run_experiment frame_events,
        # Monitor_App event files) can retain the snapshot even if the
        # envelope object is garbage-collected or mutated.
        info["stream_flags"] = dict(envelope.flags)

        return frame_640, detections, info

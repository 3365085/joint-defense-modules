from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from defense.module_a import ModuleADetector, ModuleAInput
from defense.module_a.rebuilt import RebuiltModuleADetector
from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.backends import UltralyticsDetectorBackend
from defense.module_a.roi_provider import DetectionROIProvider

from .stream_source import FrameEnvelope, StreamSource


class VideoDefensePipeline:
    """GPU-first Module A pipeline driven by a detector backend.

    The pipeline exposes two entry points:

    * :meth:`process_frame` — legacy signature used by the offline MP4 path
      (``tools/run_experiment.py`` and the standalone Monitor_App file input).
      Callers hand in a raw ``ndarray`` and the pipeline behaves exactly like
      it did before the stream-aware refactor: ``AlertState`` runs with
      ``frame_ts=None`` and ``info["latency_breakdown"]`` carries only the
      timings the offline path actually has.
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
        self._temporal_reuse_threshold = float(
            module_config.get("temporal_detector_reuse_threshold", 0.010)
        )
        self._temporal_reuse_max_gap = max(
            1, int(module_config.get("temporal_detector_reuse_max_gap", 2))
        )

        # Per-target reuse state (2026-06-11 架构修复)
        # Each target gets tracked individually so small-region changes
        # don't get drowned out by the global change_score.
        self._temporal_reuse_target_state: dict[int, dict[str, Any]] = {}
        self._temporal_reuse_consecutive = 0

        # A3b 假视频区域检测框抑制 (2026-06-11)
        # 报警后假视频区域内的 person/helmet/head 框不再显示,
        # 场景变化（视频切走/人站起）后自动恢复。
        self._a3b_suppress_remaining: int = 0       # 剩余抑制帧数, 0=无抑制
        self._a3b_suppress_bbox: tuple[int, ...] | None = None  # (x1,y1,x2,y2)

    def reset(self) -> None:
        self.detector.reset()
        self.frame_idx = 0
        self._last_small_gray = None
        self._last_detections = None
        self._last_rois = None
        self._last_detector_frame_idx = -1
        self._temporal_reuse_target_state.clear()
        self._temporal_reuse_consecutive = 0
        self._a3b_suppress_remaining = 0
        self._a3b_suppress_bbox = None

    def close(self) -> None:
        close_backend = getattr(self.detector_backend, "close", None)
        if callable(close_backend):
            close_backend()
        self._last_small_gray = None
        self._last_detections = None
        self._last_rois = None
        self._temporal_reuse_target_state.clear()
        self._temporal_reuse_consecutive = 0

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

    def _maybe_reuse_detections(self, frame_640: np.ndarray) -> tuple[Any | None, list[Any] | None, float, float]:
        """Reuse the last detector output when the current frame barely changed.

        Uses per-target ROI-level change detection to avoid missing small-region
        changes, plus a hard cap on consecutive reuse frames to prevent
        unbounded box-position drift (2026-06-11 架构修复).

        Returns
        -------
        tuple
            (detections_or_none, rois_or_none, detector_inference_ms, change_score)
        """
        if (
            self._last_small_gray is None
            or self._last_detections is None
            or self._last_rois is None
        ):
            return None, None, 0.0, 1.0
        gray = cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(small, self._last_small_gray)
        change_score = float(diff.mean() / 255.0)
        gap = self.frame_idx - self._last_detector_frame_idx

        # Lazy-init for _temporal_reuse_consecutive (tests may construct
        # pipeline without calling __init__, e.g. test_video_defense_pipeline_reuse)
        if not hasattr(self, '_temporal_reuse_consecutive'):
            self._temporal_reuse_consecutive = 0

        # Hard cap on consecutive reuse frames (prevents unbounded drift)
        from defense.module_a.backends.detector_backend import temporal_reuse_max_consecutive as _reuse_max
        _max_consecutive = getattr(self, '_temporal_reuse_max_consecutive', 3)
        if self._temporal_reuse_consecutive >= _max_consecutive:
            self._last_small_gray = small
            self._temporal_reuse_consecutive = 0
            return None, None, 0.0, change_score

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
                    self._last_small_gray = small
                    self._temporal_reuse_consecutive = 0
                    return None, None, 0.0, change_score

        if change_score <= self._temporal_reuse_threshold and gap <= self._temporal_reuse_max_gap:
            self._temporal_reuse_consecutive += 1
            return self._last_detections, self._last_rois, 0.0, change_score
        self._last_small_gray = small
        self._temporal_reuse_consecutive = 0
        return None, None, 0.0, change_score

    def _apply_a3b_suppression(
        self,
        frame_640: np.ndarray,
        detections: DetectionFrameResult,
        rois: list[Any],
        info: dict[str, Any],
    ) -> tuple[DetectionFrameResult, list[Any]]:
        """Suppress YOLO boxes inside A3b-triggered fake-video region.

        Once A3b confirms a static-media attack, person/helmet/head boxes
        inside the attack bbox are hidden until the hold expires, at which
        point A3b re-evaluates — if the fake video is still there it will
        re-trigger suppression automatically.

        The release is purely timer-based (no pixel-change check) because
        the fake video content itself moves, making pixel diff unreliable.
        """
        # Read A3b state from info
        static_media = (
            info.get("details", {})
            .get("module_a_features", {})
            .get("static_media", {})
        )
        a3b_triggered = bool(static_media.get("static_image_triggered", False))
        p_media_bbox = static_media.get("p_media_bbox")

        # --- 1. 新触发 → 启动抑制 ---
        if a3b_triggered and self._a3b_suppress_remaining <= 0:
            if isinstance(p_media_bbox, (list, tuple)) and len(p_media_bbox) == 4:
                x1, y1, x2, y2 = [max(0, int(v)) for v in p_media_bbox[:4]]
                x2 = min(x2, frame_640.shape[1] - 1)
                y2 = min(y2, frame_640.shape[0] - 1)
                if x2 > x1 and y2 > y1:
                    self._a3b_suppress_bbox = (x1, y1, x2, y2)
                    # ~3s at 60fps, ~1.5s at 30fps
                    self._a3b_suppress_remaining = 180

        # --- 2. 抑制持续期：纯计时，不检测像素变化 ---
        if self._a3b_suppress_remaining > 0:
            self._a3b_suppress_remaining -= 1
            info["a3b_suppression_active"] = True
            info["a3b_suppression_remaining"] = self._a3b_suppress_remaining

            if self._a3b_suppress_remaining <= 0:
                # 到期释放，下一帧 A3b 会判断是否需要重新抑制
                self._a3b_suppress_bbox = None
                info["a3b_suppression_released"] = True
                return detections, rois

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

        return detections, rois

    def _run_detection(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
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
        reused_detections, reused_rois, reused_detector_ms, change_score = self._maybe_reuse_detections(frame_640)
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
            )
            rois = reused_rois
            detector_inference_ms = reused_detector_ms
        else:
            detections = self.detector_backend.predict(frame_640)
            rois = self.roi_provider.from_detections(
                detections.boxes, detections.classes, detections.confidences
            )
            gray_small = cv2.resize(cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY), (160, 160), interpolation=cv2.INTER_AREA)
            self._last_small_gray = gray_small
            self._last_detections = detections
            self._last_rois = rois
            self._last_detector_frame_idx = self.frame_idx
            detector_inference_ms = float(detections.inference_ms)
        module_result = self.detector.process(
            ModuleAInput(
                frame=frame_640,
                frame_idx=self.frame_idx,
                timestamp=float(timestamp),
                rois=rois,
            )
        )
        info = module_result.to_info_dict()
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
        breakdown_raw = info.get("details", {}).get("module_a_breakdown")
        module_a_breakdown: dict[str, float] = {}
        if isinstance(breakdown_raw, dict):
            for key, value in breakdown_raw.items():
                try:
                    module_a_breakdown[str(key)] = float(value)
                except (TypeError, ValueError):
                    # Silently skip non-numeric entries rather than raise, so a
                    # single malformed field cannot break an entire run.
                    continue

        info["latency_breakdown"] = {
            "source_to_decode_ms": None,
            "decode_to_process_ms": None,
            "detector_ms": float(detections.inference_ms),
            "module_a_total_ms": float(module_a_timing_ms),
            "frame_resize_ms": float(frame_resize_ms),
            "detector_reuse_hit": bool(detector_reuse_hit),
            "detector_change_score": float(change_score),
            "source_frame_shape": list(source_frame_shape),
            "detector_frame_shape": [int(frame_640.shape[0]), int(frame_640.shape[1])],
            "module_a_breakdown": module_a_breakdown,
            "e2e_ms": float(total_timing_ms),
        }

        info["details"]["timing"] = {
            "pipeline_ms": float(total_timing_ms),
            "detector_ms": float(detections.inference_ms),
            "module_a_ms": float(module_a_timing_ms),
        }

        # A3b 假视频区域检测框抑制 (2026-06-11)
        detections, rois = self._apply_a3b_suppression(frame_640, detections, rois, info)
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
            return True
        except Exception:
            return False

    def process_runtime_frame(
        self,
        frame: Any,
        *,
        timestamp: float,
        previous_frame: Any | None = None,
        current_source_frame_idx: int | None = None,
        previous_source_frame_idx: int | None = None,
        previous_source_time_s: float | None = None,
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        """Runtime entry point with source time and strict temporal predecessor."""
        temporal_previous_applied = self._inject_temporal_previous_frame(previous_frame)
        frame_640, detections, info, _, _ = self._run_detection(
            frame,
            timestamp=float(timestamp),
        )
        gap_frames = None
        if current_source_frame_idx is not None and previous_source_frame_idx is not None:
            gap_frames = int(current_source_frame_idx) - int(previous_source_frame_idx)
        info["temporal_input"] = {
            "previous_frame_applied": bool(temporal_previous_applied),
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

    def process_frame(self, frame):
        """Offline MP4 / legacy path. Signature and behaviour are preserved.

        ``ModuleAInput.timestamp`` is left at ``0.0`` so ``AlertState`` routes
        through the legacy (``frame_ts=None``) branch, keeping every offline
        regression bit-for-bit identical to the pre-stream refactor.
        """
        frame_640, detections, info, _, _ = self._run_detection(frame, timestamp=0.0)
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

        pipeline_started_ts = time.monotonic()
        frame_640, detections, info, _, _ = self._run_detection(
            envelope.frame,
            timestamp=float(envelope.source_ts),
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

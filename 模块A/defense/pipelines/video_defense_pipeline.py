from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from defense.module_a import ModuleADetector, ModuleAInput
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
        self.detector = ModuleADetector(config=config)
        inference_config = (config or {}).get("inference", {})
        module_config = (config or {}).get("module_a", config or {})
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
        )
        self.stream_source = stream_source
        self.frame_idx = 0

    def reset(self) -> None:
        self.detector.reset()
        self.frame_idx = 0

    def warmup(self, frames: int = 3) -> None:
        if frames <= 0:
            return
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(frames):
            self.process_frame(warmup_frame)
        self.reset()

    # ------------------------------------------------------------------ core

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
        frame_640 = cv2.resize(frame, (640, 640))
        detections = self.detector_backend.predict(frame_640)
        rois = self.roi_provider.from_detections(
            detections.boxes, detections.classes, detections.confidences
        )
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
            "inference_ms": float(detections.inference_ms),
        }

        # --- Triple-channel contract (p_adv / p_safety / p_synth) ---
        # 三路告警分路契约：每路独立写入，缺失时写 null + <branch>_missing_reason,
        # 不得用其他分路的值互填。参见 requirements.md 1.1 / 1.6。
        p_adv_value = getattr(module_result, "p_adv", None)
        if p_adv_value is None:
            info["p_adv"] = None
            info["p_adv_missing_reason"] = "module_a_p_adv_unavailable"
        else:
            info["p_adv"] = float(p_adv_value)

        # p_safety 业务侧尚未接入 Module A pipeline，此处固定落 null + 原因。
        info["p_safety"] = None
        info["p_safety_missing_reason"] = "p_safety 业务侧未接入"

        info["reason_codes"] = module_result.reason_codes
        source_auth = (
            info.get("details", {}).get("module_a_features", {}).get("source_authenticity", {})
        )
        source_auth_enabled = bool(source_auth.get("enabled", False))
        source_auth_available = bool(source_auth.get("available", False))
        info["source_authenticity_enabled"] = source_auth_enabled
        info["source_authenticity_warning"] = bool(source_auth.get("warning", False))
        info["source_authenticity_confirmed"] = bool(source_auth.get("confirmed", False))
        info["source_authenticity_available"] = source_auth_available
        info["source_authenticity_reason"] = str(source_auth.get("reason", ""))

        # p_synth：未启用或数据尚未就绪时一律标记缺失，不回落成 0.0 也不使用其它分路的值。
        if not source_auth_enabled:
            info["p_synth"] = None
            info["p_synth_missing_reason"] = "source_authenticity_disabled"
        elif not source_auth_available:
            info["p_synth"] = None
            info["p_synth_missing_reason"] = "source_authenticity_window_not_ready"
        elif "p_synth" not in source_auth:
            info["p_synth"] = None
            info["p_synth_missing_reason"] = "source_authenticity_p_synth_missing"
        else:
            info["p_synth"] = float(source_auth.get("p_synth"))

        info["detector_backend"] = detections.backend
        info["detector_inference_ms"] = float(detections.inference_ms)

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
            "module_a_breakdown": module_a_breakdown,
            "e2e_ms": float(total_timing_ms),
        }

        info["details"]["timing"] = {
            "pipeline_ms": float(total_timing_ms),
            "detector_ms": float(detections.inference_ms),
            "module_a_ms": float(module_a_timing_ms),
        }
        self.frame_idx += 1
        return frame_640, detections, info, total_timing_ms, module_a_timing_ms

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

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from defense.module_a.postprocess import PPEDisplayTracker, merge_roi_detections
from .a3b_soft_trigger import A3BSoftTriggerState
from .pipeline_factory import PipelineBundle
from .ppe_business import evaluate_ppe_business
from .ppe_state import SafetyHelmetState


REASON_TEXT = {
    "overexposure": "强光/过曝异常",
    "temporal_texture_change": "时序纹理突变",
    "local_temporal_texture_change": "局部纹理突变",
    "motion_artifact": "运动/光流伪影",
    "light_optical_flow_artifact": "局部光流异常",
    "local_blur_degradation": "局部模糊退化",
    "track_consistency_drop": "目标轨迹一致性下降",
    "static_image_spoof": "静态媒介/翻拍疑似",
    "static_media_spoof": "静态媒介攻击确认",
    "p_adv": "融合分数越过阈值",
    "natural_exposure_suppressed": "自然曝光变化已抑制",
}


def prepare_frame_640(frame: np.ndarray, max_input: int = 1280) -> np.ndarray:
    h_src, w_src = frame.shape[:2]
    if h_src > max_input or w_src > max_input:
        scale = max_input / max(h_src, w_src)
        frame = cv2.resize(
            frame,
            (max(1, int(w_src * scale)), max(1, int(h_src * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if frame.shape[:2] == (640, 640):
        return frame
    # Use the same final interpolation as the direct pipeline. A3b relies on
    # fine screen/edge texture; INTER_AREA at this last square resize was
    # washing out some positive screen-spoof cues in BrowserFrameSource mode.
    return cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def info_reason(info: dict[str, Any]) -> str:
    reason_codes = info.get("reason_codes") or []
    if not reason_codes:
        reason_codes = info.get("details", {}).get("reason_codes") or []
    translated = [REASON_TEXT.get(str(code), str(code)) for code in reason_codes]
    return "，".join(translated[:5])


def _static_media_details(info: dict[str, Any]) -> dict[str, Any]:
    return (
        info.get("details", {})
        .get("module_a_features", {})
        .get("static_media", {})
        if isinstance(info.get("details"), dict)
        else {}
    )


@dataclass(slots=True)
class ProcessedFrame:
    frame_idx: int
    frame_640: np.ndarray
    rendered_frame: np.ndarray
    info: dict[str, Any]
    ppe: dict[str, Any]
    ppe_tracks: list[dict[str, Any]]
    status: dict[str, Any]


class FrameProcessor:
    """Algorithm/runtime bridge.

    The Web server never calls model code directly. It only asks MonitorEngine
    for status/frames; MonitorEngine delegates per-frame work here.
    """

    def __init__(self, bundle: PipelineBundle, *, jpeg_quality: int = 82) -> None:
        self.bundle = bundle
        self.pipeline = bundle.pipeline
        config = bundle.config if isinstance(bundle.config, dict) else {}
        ppe_config = config.get("ppe_tracking", {}) if isinstance(config.get("ppe_tracking"), dict) else {}
        hold_frames = int(ppe_config.get("max_missed_frames", 10 if ppe_config else 8))
        self.ppe_state = SafetyHelmetState()
        self.ppe_tracker = PPEDisplayTracker(
            history=9,
            hold_frames=hold_frames,
            small_hold_frames=int(ppe_config.get("max_missed_frames", hold_frames if ppe_config else 18)),
            switch_count=4,
            small_area_ratio=0.020,
            small_confidence=0.62,
            redetect_interval=3,
            iou_match_threshold=float(ppe_config.get("iou_match_threshold", 0.30 if ppe_config else 0.12)),
            max_missed_ms=float(ppe_config.get("max_missed_ms", 700.0)),
            hold_last_box=bool(ppe_config.get("hold_last_box", True)),
            smooth_alpha=float(ppe_config.get("smooth_alpha", 0.82 if ppe_config else 0.82)),
            show_held_boxes=bool(ppe_config.get("show_held_boxes", True)),
        )
        self.ppe_tracking_enabled = bool(ppe_config.get("enabled", True))
        a3b_config = config.get("a3b", {}) if isinstance(config.get("a3b"), dict) else {}
        self.a3b_soft = A3BSoftTriggerState(a3b_config)
        self.processing_history: deque[float] = deque(maxlen=30)
        self.jpeg_quality = int(jpeg_quality)

    def reset(self) -> None:
        self.pipeline.reset()
        self.ppe_state.reset()
        self.ppe_tracker.reset()
        self.a3b_soft.reset()
        self.processing_history.clear()

    def process(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool,
        video_time_s: float,
        source_fps: float,
        dropped_frames: int,
        display_options: dict[str, Any],
        feature_options: dict[str, Any],
        custom_model: dict[str, Any],
        target_frame_budget_ms: float,
    ) -> ProcessedFrame:
        started = time.perf_counter()
        frame_640 = prepare_frame_640(frame)
        _, detections, info = self.pipeline.process_frame(frame_640)
        redetect_ms = 0.0
        redetect_count = 0
        avg_processing_ms = (
            (sum(self.processing_history) / len(self.processing_history)) * 1000.0
            if self.processing_history
            else 0.0
        )
        budget_ok = (not realtime) or not self.processing_history or avg_processing_ms <= target_frame_budget_ms * 0.85
        if display_options.get("show_boxes", True) and budget_ok:
            rois = self.ppe_tracker.recommend_redetect_rois(
                detections,
                frame_640.shape[:2],
                frame_idx,
                enabled=True,
                max_rois=1,
            )
            if rois:
                roi_results = []
                redetect_started = time.perf_counter()
                for roi in rois:
                    x1, y1, x2, y2 = [int(v) for v in roi]
                    crop = frame_640[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    roi_results.append((roi, self.pipeline.detector_backend.predict(crop)))
                if roi_results:
                    detections = merge_roi_detections(detections, roi_results, frame_640.shape[:2])
                    redetect_count = len(roi_results)
                redetect_ms = (time.perf_counter() - redetect_started) * 1000.0

        ppe_result = evaluate_ppe_business(
            detections,
            frame_shape=frame_640.shape[:2],
            ppe_state=self.ppe_state,
            ppe_tracker=self.ppe_tracker,
            tracking_enabled=self.ppe_tracking_enabled,
        )
        ppe = ppe_result.ppe
        ppe_tracks = ppe_result.tracks
        process_total_s = time.perf_counter() - started
        self.processing_history.append(process_total_s)
        fps = 1.0 / (sum(self.processing_history) / len(self.processing_history)) if self.processing_history else 0.0
        status = self._build_status(
            source_type=source_type,
            source=source,
            profile=profile,
            realtime=realtime,
            frame_idx=frame_idx,
            video_time_s=video_time_s,
            source_fps=source_fps,
            fps=fps,
            dropped_frames=dropped_frames,
            info=info,
            ppe=ppe,
            ppe_tracks=ppe_tracks,
            display_options=display_options,
            feature_options=feature_options,
            custom_model=custom_model,
            redetect_budget_ok=budget_ok,
            redetect_count=redetect_count,
            redetect_ms=redetect_ms,
            processing_ms=process_total_s * 1000.0,
            raw_boxes_count=len(getattr(detections, "boxes", []) or []),
        )
        return ProcessedFrame(
            frame_idx=frame_idx,
            frame_640=frame_640,
            rendered_frame=frame_640,
            info=info,
            ppe=ppe,
            ppe_tracks=ppe_tracks,
            status=status,
        )

    def _build_status(
        self,
        *,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool,
        frame_idx: int,
        video_time_s: float,
        source_fps: float,
        fps: float,
        dropped_frames: int,
        info: dict[str, Any],
        ppe: dict[str, Any],
        ppe_tracks: list[dict[str, Any]],
        display_options: dict[str, Any],
        feature_options: dict[str, Any],
        custom_model: dict[str, Any],
        redetect_budget_ok: bool,
        redetect_count: int,
        redetect_ms: float,
        processing_ms: float,
        raw_boxes_count: int,
    ) -> dict[str, Any]:
        static_media = dict(_static_media_details(info))
        static_media["source_path"] = source
        latency = info.get("latency_breakdown", {}) if isinstance(info.get("latency_breakdown"), dict) else {}
        p_adv = info.get("p_adv")
        a3b_soft = self.a3b_soft.update(static_media)
        a3b_triggered = bool(a3b_soft["triggered"])
        a3b_observed_score = _float(a3b_soft.get("observed_score"))
        a3b_smoothed_score = _float(
            static_media.get("live_score_display", static_media.get("score", a3b_observed_score))
        )
        a3b_confirmed_score = _float(a3b_soft.get("confirmed_score"))
        a3b_confidence = _float(a3b_soft.get("confidence", a3b_confirmed_score))
        a3b_display_score = _float(a3b_soft.get("display_score"), a3b_confidence)
        a3b_card_score = a3b_confidence
        a3b_event_score = a3b_confidence if a3b_confidence > 0 else a3b_observed_score
        a3b_state = str(a3b_soft.get("state") or ("confirmed" if a3b_soft.get("triggered") else "normal"))
        ppe_boxes_count = _int(ppe.get("person_count")) + _int(ppe.get("helmet_count")) + _int(ppe.get("head_count"))
        tracked_boxes_count = len([track for track in ppe_tracks if str(track.get("source", "detected")) in {"tracked", "held"}])
        render_boxes_count = len(ppe_tracks)
        bundle_config = self.bundle.config if isinstance(self.bundle.config, dict) else {}
        runtime_config = bundle_config.get("runtime", {}) if isinstance(bundle_config.get("runtime"), dict) else {}
        resolved_custom_model = runtime_config.get("custom_model", custom_model)
        if not isinstance(resolved_custom_model, dict):
            resolved_custom_model = custom_model
        status = {
            "running": True,
            "source_type": source_type,
            "source": source,
            "profile": profile,
            "realtime": bool(realtime),
            "backend": self.bundle.backend,
            "model_family": self.bundle.model_family,
            "artifact": self.bundle.artifact_path,
            "frame_idx": int(frame_idx),
            "video_time_s": float(video_time_s),
            "fps": float(fps),
            "source_fps": float(source_fps),
            "dropped_frames": int(dropped_frames),
            "timing_ms": _float(info.get("timing_ms"), processing_ms),
            "processing_ms": float(processing_ms),
            "detector_inference_ms": _float(info.get("detector_inference_ms")),
            "module_a_timing_ms": _float(info.get("module_a_timing_ms")),
            "latency_breakdown": latency,
            "detector_reuse_hit": bool(latency.get("detector_reuse_hit", False)),
            "detector_change_score": _float(latency.get("detector_change_score")),
            "source_frame_shape": latency.get("source_frame_shape", []),
            "detector_frame_shape": latency.get("detector_frame_shape", []),
            "p_adv": None if p_adv is None else _float(p_adv),
            "p_adv_display": _float(info.get("p_adv_display", p_adv or 0.0)),
            "p_adv_missing_reason": str(info.get("p_adv_missing_reason", "")),
            "alert_confirmed": bool(info.get("alert_confirmed", False)),
            "attack_detected": bool(info.get("attack_detected", False)),
            "attack_state_active": bool(info.get("attack_state_active", False)),
            "reason": info_reason(info),
            "reason_codes": list(info.get("reason_codes") or []),
            "a3b_score": float(a3b_card_score),
            "a3b_confidence": float(a3b_confidence),
            "a3b_observed_score": float(a3b_observed_score),
            "a3b_smoothed_score": float(a3b_smoothed_score),
            "a3b_confirmed_score": float(a3b_confirmed_score),
            "a3b_display_score": float(a3b_display_score),
            "a3b_event_score": float(a3b_event_score),
            "a3b_state": a3b_state,
            "a3b_triggered": bool(a3b_triggered),
            "a3b_p_media": _float(static_media.get("p_media")),
            "a3b_bbox": static_media.get("p_media_bbox"),
            "a3b_triggered_source": str(a3b_soft.get("triggered_source") or "none"),
            "a3b_reason": str(a3b_soft.get("reason") or ""),
            "a3b_debug": dict(a3b_soft.get("debug") or {}),
            "ppe_warning": bool(ppe.get("warning", False)),
            "ppe_candidate": bool(ppe.get("candidate", False)),
            "ppe_confirmed": bool(ppe.get("confirmed", False)),
            "ppe_person_count": _int(ppe.get("person_count")),
            "ppe_raw_person_count": _int(ppe.get("raw_person_count", ppe.get("person_count"))),
            "ppe_inferred_person_count": _int(ppe.get("inferred_person_count", ppe.get("person_count"))),
            "ppe_person_context_count": _int(ppe.get("person_context_count", ppe.get("person_count"))),
            "ppe_helmet_count": _int(ppe.get("helmet_count")),
            "ppe_head_count": _int(ppe.get("head_count")),
            "ppe_missing_helmet_count": _int(ppe.get("missing_helmet_count")),
            "ppe_has_person_class": bool(ppe.get("has_person_class", False)),
            "ppe_evidence_mode": str(ppe.get("evidence_mode", "")),
            "ppe_uncertain": bool(ppe.get("uncertain", False)),
            "ppe_reason": str(ppe.get("reason", "")),
            "ppe_window_positive": _int(ppe.get("window_positive")),
            "ppe_window": _int(ppe.get("window")),
            "ppe_track_count": len(ppe_tracks),
            "ppe_tracks": [dict(track) for track in ppe_tracks],
            "ppe_class_counts": ppe.get("class_counts", {}),
            "raw_boxes_count": int(raw_boxes_count),
            "ppe_boxes_count": int(ppe_boxes_count),
            "tracked_boxes_count": int(tracked_boxes_count),
            "render_boxes_count": int(render_boxes_count),
            "ppe_roi_redetect_budget_ok": bool(redetect_budget_ok),
            "ppe_roi_redetect_count": int(redetect_count),
            "ppe_roi_redetect_ms": float(redetect_ms),
            "feature_options": dict(feature_options),
            "custom_model": dict(resolved_custom_model),
            "display_options": dict(display_options),
            "preview_mode": "async_latest_frame" if realtime else "sync_detection_frame",
            "error": "",
        }
        status["branch_cards"] = build_branch_cards(status)
        return status


def _score_display(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _bar_ratio(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def build_branch_cards(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the right-panel branch cards consumed by the Web UI."""
    p_adv = status.get("p_adv")
    p_adv_missing = p_adv is None
    p_adv_confirmed = bool(status.get("alert_confirmed"))
    p_adv_active = bool(status.get("attack_state_active") or status.get("attack_detected"))
    if p_adv_missing:
        adv_class = "card-missing"
        adv_state = "待检测"
        adv_detail = status.get("p_adv_missing_reason") or "尚未产生物理扰动检测结果。"
    elif p_adv_confirmed:
        adv_class = "card-confirmed"
        adv_state = "确认告警"
        adv_detail = "连续帧满足模块A告警条件。"
    elif p_adv_active:
        adv_class = "card-warning"
        adv_state = "疑似扰动"
        adv_detail = "当前帧存在物理扰动迹象，等待连续帧确认。"
    else:
        adv_class = "card-idle"
        adv_state = "OK"
        adv_detail = "未触发物理扰动检测。"

    feature_options = status.get("feature_options") if isinstance(status.get("feature_options"), dict) else {}
    a3b_enabled = feature_options.get("static_image_enabled", True) is not False
    a3b_observed = _float(status.get("a3b_observed_score"))
    a3b_confirmed = _float(status.get("a3b_confirmed_score"))
    a3b_confidence = _float(status.get("a3b_confidence"), a3b_confirmed)
    a3b_triggered = bool(status.get("a3b_triggered"))
    recent_source_events = status.get("recent_source_auth_events")
    if not isinstance(recent_source_events, list):
        recent_source_events = []
    recent_a3b_peak = 0.0
    recent_a3b_reason = ""
    for event in recent_source_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("channel") or "a3b") not in {"a3b", "source_auth"}:
            continue
        score = _float(
            event.get("peak_a3b_score")
            or event.get("a3b_event_score")
            or event.get("peak_score")
            or event.get("max_score")
        )
        if score > recent_a3b_peak:
            recent_a3b_peak = score
            recent_a3b_reason = str(event.get("reason") or event.get("close_reason") or "")
    a3b_has_record = recent_a3b_peak > 0.0
    a3b_display = max(
        _float(status.get("a3b_display_score")),
        _float(status.get("a3b_event_score")),
        a3b_observed if a3b_triggered else 0.0,
        a3b_confirmed * 0.8,
        recent_a3b_peak,
    )
    a3b_source = str(status.get("a3b_triggered_source") or "none")
    a3b_machine_state = str(status.get("a3b_state") or "normal")
    a3b_debug = status.get("a3b_debug") if isinstance(status.get("a3b_debug"), dict) else {}
    failed_gates = a3b_debug.get("failed_gates") if isinstance(a3b_debug.get("failed_gates"), list) else []
    state_labels = {"normal": "OK", "observing": "观察中", "suspect": "疑似", "confirmed": "确认", "disabled": "未启用"}
    if not a3b_enabled:
        a3b_class = "card-missing"
        a3b_state = "未启用"
        a3b_detail = "A3b 翻拍/假目标检测未启用。"
    elif a3b_machine_state == "confirmed" or (
        a3b_triggered and a3b_source not in {"single_strong", "observed_strong", "observed_window"}
    ):
        a3b_class = "card-warning"
        a3b_state = "确认"
        a3b_detail = f"展示分数 {a3b_display:.3f}，确认置信度 {a3b_confidence:.3f}，观察分数 {a3b_observed:.3f}，来源 {a3b_source}。"
    elif a3b_machine_state == "suspect" or a3b_triggered:
        a3b_class = "card-warning"
        a3b_state = "疑似"
        a3b_detail = f"观察证据已触发疑似告警；展示分数 {a3b_display:.3f}，观察分数 {a3b_observed:.3f}，确认置信度 {a3b_confidence:.3f}，来源 {a3b_source}。"
    elif a3b_has_record:
        a3b_class = "card-warning"
        a3b_state = "已记录"
        a3b_detail = f"最近 A3b 警告峰值 {recent_a3b_peak:.3f}；当前观察分数 {a3b_observed:.3f}，确认置信度 {a3b_confidence:.3f}。"
    elif a3b_machine_state == "observing" or a3b_observed >= 0.42:
        a3b_class = "card-idle"
        a3b_state = "观察中"
        suffix = f"；失败门控 {','.join(str(item) for item in failed_gates[:3])}" if failed_gates else ""
        a3b_detail = f"观察分数 {a3b_observed:.3f}，确认置信度 {a3b_confidence:.3f}，尚未形成确认{suffix}。"
    else:
        a3b_class = "card-idle"
        a3b_state = state_labels.get(a3b_machine_state, "OK")
        a3b_detail = f"未发现翻拍/假目标确认迹象；观察分数 {a3b_observed:.3f}，确认置信度 {a3b_confidence:.3f}。"

    return [
        {
            "branch": "p_adv",
            "title": "物理对抗扰动（p_adv）",
            "score": None if p_adv is None else _float(p_adv),
            "score_display": _score_display(p_adv),
            "score_bar_ratio": _bar_ratio(p_adv),
            "border_class": adv_class,
            "state": adv_state,
            "state_detail": adv_detail,
            "reason_text": status.get("reason") or "",
            "badges": ["模块A", "连续帧"] if p_adv_confirmed else ["模块A"],
        },
        {
            "branch": "p_safety",
            "title": "翻拍/假目标检测（A3b）",
            "score": float(a3b_display),
            "score_display": _score_display(a3b_display),
            "score_bar_ratio": _bar_ratio(a3b_display),
            "observed_score": float(a3b_observed),
            "confirmed_score": float(a3b_confirmed),
            "confidence": float(a3b_confidence),
            "display_score": float(a3b_display),
            "machine_state": a3b_machine_state,
            "border_class": a3b_class,
            "state": a3b_state,
            "state_detail": a3b_detail,
            "reason_text": recent_a3b_reason if a3b_source == "none" else a3b_source,
            "badges": ["A3b", "警告记录"] if (a3b_triggered or a3b_has_record) else ["A3b"],
        },
    ]

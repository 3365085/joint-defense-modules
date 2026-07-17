from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from defense.pipelines.video_decoder import DecodedFrameLease
from defense.module_a.ppe_postprocess import (
    PPEPostprocessConfig,
    is_bare_head_label,
    is_helmet_label,
    is_person_label,
)
from defense.module_a.postprocess import PPEDisplayTracker, merge_roi_detections
from defense.module_a.result_contract import adapt_a3b_result
from .a3b_soft_trigger import A3BSoftTriggerConfig, A3BSoftTriggerState
from .config import A3B_SENSITIVITY_PRESETS
from .pipeline_factory import PipelineBundle
from .ppe_business import evaluate_ppe_business
from .ppe_state import SafetyHelmetState

import json as _json
import os as _os

logger = logging.getLogger(__name__)

# 只读诊断探针: 环境变量 DEFENSE_FEATURE_PROBE 指向一个 JSONL 路径时, 每帧把 A1/运动/adv/
# 抑制门等关键数值追加一行。默认关闭, 全程 try/except 吞异常, 绝不影响检测主链路。
_PROBE_PATH = _os.environ.get("DEFENSE_FEATURE_PROBE") or ""


def _feature_probe(info: dict[str, Any], frame_idx: int, detections: Any = None) -> None:
    if not _PROBE_PATH:
        return
    try:
        # 原始 YOLO 检出计数(未过显示门), 区分"YOLO 没检出头" vs "后处理过滤掉头"
        raw_person = raw_head = raw_helmet = 0
        raw_head_confs: list[float] = []
        raw_helmet_confs: list[float] = []
        raw_max_area = 0.0
        if detections is not None:
            boxes = list(getattr(detections, "boxes", []) or [])
            classes = list(getattr(detections, "classes", []) or [])
            confs = list(getattr(detections, "confidences", []) or [])
            names = getattr(detections, "names", {}) or {}
            for b in boxes:
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    a = abs(float(b[2]) - float(b[0])) * abs(float(b[3]) - float(b[1])) / (640.0 * 640.0)
                    raw_max_area = max(raw_max_area, a)
            for cid, cf in zip(classes, confs):
                lbl = str(names.get(int(cid), "")) if isinstance(names, dict) else str(cid)
                if is_person_label(lbl):
                    raw_person += 1
                elif is_bare_head_label(lbl):
                    raw_head += 1
                    raw_head_confs.append(round(float(cf), 3))
                elif is_helmet_label(lbl):
                    raw_helmet += 1
                    raw_helmet_confs.append(round(float(cf), 3))
        details = info.get("details", {}) if isinstance(info.get("details"), dict) else {}
        scene = details.get("scene_context", {}) if isinstance(details.get("scene_context"), dict) else {}
        flow = details.get("flow_context", {}) if isinstance(details.get("flow_context"), dict) else {}
        joint = details.get("joint_decision", {}) if isinstance(details.get("joint_decision"), dict) else {}
        a1 = details.get("a1", {}) if isinstance(details.get("a1"), dict) else {}
        a2 = details.get("a2", {}) if isinstance(details.get("a2"), dict) else {}
        a3 = details.get("a3", {}) if isinstance(details.get("a3"), dict) else {}
        blinding = (
            details.get("blinding", {})
            if isinstance(details.get("blinding"), dict)
            else {}
        )
        timing = (
            details.get("timing", {})
            if isinstance(details.get("timing"), dict)
            else {}
        )
        latency = (
            info.get("latency_breakdown", {})
            if isinstance(info.get("latency_breakdown"), dict)
            else {}
        )
        row = {
            "frame": int(frame_idx),
            "frame_diff_global": round(float(scene.get("frame_diff_global", 0.0)), 4),
            "exposure_delta": round(float(scene.get("exposure_delta", 0.0)), 4),
            "overexp": round(float(scene.get("overexposure_ratio", 0.0)), 4),
            "underexp": round(float(scene.get("underexposed_ratio", 0.0)), 4),
            "global_motion_weight": round(float(flow.get("global_motion_weight", 0.0)), 4),
            "a1": round(float(a1.get("a1_feature_score", 0.0)), 4),
            "a1_target_related": bool(a1.get("target_related", False)),
            "a1_delta_h_global": round(float(a1.get("delta_h_global", 0.0)), 4),
            "a1_delta_h_local_max": round(float(a1.get("delta_h_local_max", 0.0)), 4),
            "a1_delta_h_roi_max": round(float(a1.get("delta_h_roi_max", 0.0)), 4),
            "a1_delta_h_roi_patch_max": round(
                float(a1.get("delta_h_roi_patch_max", 0.0)),
                4,
            ),
            "a1_delta_h_target_contrast": round(
                float(a1.get("delta_h_target_contrast", 0.0)),
                4,
            ),
            "a1_delta_h_spatial_concentration": round(
                float(a1.get("delta_h_spatial_concentration", 0.0)),
                4,
            ),
            "a1_delta_h_patch_concentration": round(
                float(a1.get("delta_h_patch_concentration", 0.0)),
                4,
            ),
            "a1_visibility_hold_active": bool(
                a1.get("a1_visibility_hold_active", False)
            ),
            "a2": round(float(a2.get("a2_feature_score", 0.0)), 4),
            "a2_target_related": bool(a2.get("target_related", False)),
            "a2_change_t_global": round(float(a2.get("change_t_global", 0.0)), 4),
            "a2_change_t_local_max": round(
                float(a2.get("change_t_local_max", 0.0)),
                4,
            ),
            "a2_change_t_roi_max": round(float(a2.get("change_t_roi_max", 0.0)), 4),
            "a2_change_t_context_mean": round(
                float(a2.get("change_t_context_mean", 0.0)),
                4,
            ),
            "a2_change_t_local_contrast": round(
                float(a2.get("change_t_local_contrast", 0.0)),
                4,
            ),
            "a2_change_t_without_motion_target": round(
                float(a2.get("change_t_without_motion_target", 0.0)),
                4,
            ),
            "a2_change_t_motion_aligned": round(
                float(a2.get("change_t_motion_aligned", 0.0)),
                4,
            ),
            "a2_change_t_motion_explain_score": round(
                float(a2.get("change_t_motion_explain_score", 0.0)),
                4,
            ),
            "a2_change_t_unexplained": round(
                float(a2.get("change_t_unexplained", 0.0)),
                4,
            ),
            "a2_change_t_burst": round(float(a2.get("change_t_burst", 0.0)), 4),
            "a2_flash_like": bool(a2.get("flash_like", False)),
            "a3": round(float(a3.get("a3_feature_score", 0.0)), 4),
            "a3_target_related": bool(a3.get("target_related", False)),
            "a3_flow_local_anomaly_ratio": round(
                float(a3.get("flow_local_anomaly_ratio", 0.0)),
                4,
            ),
            "a3_flow_max_magnitude_norm": round(
                float(a3.get("flow_max_magnitude_norm", 0.0)),
                4,
            ),
            "a3_flow_residual": round(float(a3.get("flow_residual", 0.0)), 4),
            "a3_flow_roi_residual": round(
                float(a3.get("flow_roi_residual", 0.0)),
                4,
            ),
            "a3_flow_context_residual": round(
                float(a3.get("flow_context_residual", 0.0)),
                4,
            ),
            "a3_flow_residual_contrast": round(
                float(a3.get("flow_residual_contrast", 0.0)),
                4,
            ),
            "a3_flow_roi_motion_gap": round(
                float(a3.get("flow_roi_motion_gap", 0.0)),
                4,
            ),
            "a3_flow_background_explain_score": round(
                float(a3.get("flow_background_explain_score", 0.0)),
                4,
            ),
            "a3_flow_shape_score": round(
                float(a3.get("flow_shape_score", 0.0)),
                4,
            ),
            "a3_flow_target_relation": round(
                float(a3.get("flow_target_relation", 0.0)),
                4,
            ),
            "a3_flow_background_coherence": round(
                float(a3.get("flow_background_coherence", 0.0)),
                4,
            ),
            "a3_flow_roi_coverage_ratio": round(
                float(a3.get("flow_roi_coverage_ratio", 0.0)),
                4,
            ),
            "a3_residual_hold_active": bool(
                a3.get("a3_residual_hold_active", False)
            ),
            "p_blind": round(float(blinding.get("p_blind", 0.0)), 4),
            "p_blind_triggered": bool(
                blinding.get("p_blind_triggered", False)
            ),
            "blind_ready": bool(blinding.get("blind_ready", False)),
            "blind_type": str(blinding.get("blind_type", "none")),
            "blind_sharpness": round(
                float(blinding.get("sharpness", 0.0)),
                4,
            ),
            "blind_ref_sharpness": round(
                float(blinding.get("ref_sharpness", 0.0)),
                4,
            ),
            "blind_contrast": round(
                float(blinding.get("contrast", 0.0)),
                4,
            ),
            "blind_ref_contrast": round(
                float(blinding.get("ref_contrast", 0.0)),
                4,
            ),
            "blind_det_strength": round(
                float(blinding.get("det_strength", 0.0)),
                4,
            ),
            "blind_ref_det": round(
                float(blinding.get("ref_det", 0.0)),
                4,
            ),
            "blind_sharp_drop": round(
                float(blinding.get("sharp_drop", 0.0)),
                4,
            ),
            "blind_sharp_drop_short": round(
                float(blinding.get("sharp_drop_short", 0.0)),
                4,
            ),
            "blind_contrast_drop": round(
                float(blinding.get("contrast_drop", 0.0)),
                4,
            ),
            "blind_det_drop": round(
                float(blinding.get("det_drop", 0.0)),
                4,
            ),
            "blind_glare": round(
                float(blinding.get("glare_blind", 0.0)),
                4,
            ),
            "blind_target_loss": round(
                float(blinding.get("target_loss", 0.0)),
                4,
            ),
            "blind_blur_detail_ratio": round(
                float(blinding.get("blur_detail_ratio", 0.0)),
                4,
            ),
            "blind_low_motion_target_loss_support": bool(
                blinding.get("low_motion_target_loss_support", False)
            ),
            "blind_motion_blur_scene_degradation_support": bool(
                blinding.get(
                    "motion_blur_scene_degradation_support",
                    False,
                )
            ),
            "blind_independent_support": bool(
                blinding.get("blind_independent_support", False)
            ),
            "runtime_frame_materialization_ms": round(
                float(latency.get("frame_materialization_ms", 0.0)),
                4,
            ),
            "runtime_previous_frame_materialization_ms": round(
                float(
                    latency.get("previous_frame_materialization_ms", 0.0)
                ),
                4,
            ),
            "runtime_detector_ms": round(
                float(latency.get("detector_ms", 0.0)),
                4,
            ),
            "runtime_module_a_total_ms": round(
                float(latency.get("module_a_total_ms", 0.0)),
                4,
            ),
            "runtime_module_a_reuse_hit": bool(
                latency.get("module_a_reuse_hit", False)
            ),
            "runtime_e2e_ms": round(
                float(latency.get("e2e_ms", info.get("timing_ms", 0.0))),
                4,
            ),
            "runtime_detector_reuse_hit": bool(
                latency.get("detector_reuse_hit", False)
            ),
            **{
                f"module_a_stage_{name}_ms": round(
                    float(timing.get(name, 0.0)),
                    4,
                )
                for name in (
                    "scene_context",
                    "lbp",
                    "flow",
                    "a1",
                    "a2",
                    "a3",
                    "a3b_schedule",
                    "a4",
                    "blinding",
                    "target_anchored",
                    "joint",
                    "result_build",
                    "state_update",
                    "total",
                )
            },
            "p_adv": round(float(info.get("p_adv") or 0.0), 4),
            "dominant": str(joint.get("dominant_input", "")),
            "adv_candidate_allowed": bool(
                joint.get("adv_candidate_allowed", False)
            ),
            "adv_allowed": bool(
                joint.get(
                    "adv_single_frame_candidate",
                    joint.get("adv_candidate_allowed", False),
                )
            ),
            "adv_physical_support": bool(
                joint.get("adv_physical_support", False)
            ),
            "a3_independent_attack_support": bool(
                joint.get("a3_independent_attack_support", False)
            ),
            "normal_target_motion_exclusion": bool(
                joint.get("normal_target_motion_exclusion", False)
            ),
            "normal_articulated_target_motion": bool(
                joint.get("normal_articulated_target_motion", False)
            ),
            "normal_high_contrast_target_texture_motion": bool(
                joint.get(
                    "normal_high_contrast_target_texture_motion",
                    False,
                )
            ),
            "normal_roi_flow_target_motion": bool(
                joint.get("normal_roi_flow_target_motion", False)
            ),
            "localized_a1_attack_support": bool(
                joint.get("localized_a1_attack_support", False)
            ),
            "photometric_attack_support": bool(
                joint.get("photometric_attack_support", False)
            ),
            "adv_explicit_suppression_reason": str(
                joint.get("adv_explicit_suppression_reason", "none")
            ),
            "joint_suppressed_reason": str(
                joint.get("suppressed_reason", "none")
            ),
            "adv_confirmed": bool(joint.get("adv_confirmed", False)),
            "alert_confirmation_source": str(
                joint.get("alert_confirmation_source", "none")
            ),
            "confirm_adv_count": int(
                (joint.get("confirm_window") or {}).get("adv_count", 0)
            ),
            "confirm_adv_required": int(
                (joint.get("confirm_window") or {}).get(
                    "adv_hit_required",
                    0,
                )
            ),
            "confirm_adv_support_count": int(
                (joint.get("confirm_window") or {}).get(
                    "adv_support_count",
                    0,
                )
            ),
            "blind_single_frame_candidate": bool(
                joint.get("blind_single_frame_candidate", False)
            ),
            "blind_confirmed": bool(joint.get("blind_confirmed", False)),
            "blind_explicitly_suppressed": bool(
                joint.get("blind_explicitly_suppressed", False)
            ),
            "blind_sustained_escalated": bool(
                joint.get("blind_sustained_escalated", False)
            ),
            "confirm_blind_count": int(
                (joint.get("confirm_window") or {}).get("blind_count", 0)
            ),
            "confirm_blind_required": int(
                (joint.get("confirm_window") or {}).get(
                    "blind_hit_required",
                    0,
                )
            ),
            "alert": bool(info.get("alert_confirmed", False)),
            "reasons": list(info.get("reason_codes") or joint.get("reason_codes") or []),
            # 抑制门(True=该门在抑制): 看晃动时是否失守
            "gate_normal_motion": bool(joint.get("normal_motion_texture_change", False)),
            "gate_scene_spike": bool(joint.get("nonlocal_a1_a3_scene_spike", False)),
            "gate_low_motion_bg": bool(joint.get("low_motion_background_like_adv", False)),
            "gate_scene_baseline": bool(joint.get("scene_baseline_normal", False)),
            "raw_person": raw_person,
            "raw_head": raw_head,
            "raw_helmet": raw_helmet,
            "raw_head_confs": raw_head_confs,
            "raw_helmet_confs": raw_helmet_confs,
            "raw_max_area": round(raw_max_area, 3),
        }
        with open(_PROBE_PATH, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


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


def _ppe_track_label_counts(ppe_tracks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in ppe_tracks:
        label = str(track.get("label") or "")
        if label not in {"person", "helmet", "head"}:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def info_reason(info: dict[str, Any]) -> str:
    reason_codes = info.get("reason_codes") or []
    if not reason_codes:
        reason_codes = info.get("details", {}).get("reason_codes") or []
    translated = [REASON_TEXT.get(str(code), str(code)) for code in reason_codes]
    return "，".join(translated[:5])


def _static_media_details(info: dict[str, Any]) -> dict[str, Any]:
    return adapt_a3b_result(info)


def _authoritative_a3b_confirmation(
    static_media: dict[str, Any],
) -> tuple[bool, float, Any, str]:
    """Return backend-confirmed A3b state without applying soft-trigger policy."""

    is_rebuilt = static_media.get("result_contract_source") == "rebuilt"
    confirmed = bool(
        static_media.get("media_confirmed")
        if is_rebuilt
        else (
            static_media.get("triggered")
            or static_media.get("static_image_triggered")
        )
    )
    if not confirmed:
        return False, 0.0, None, "none"

    legacy_static = (
        static_media.get("legacy_static_image")
        if isinstance(static_media.get("legacy_static_image"), dict)
        else {}
    )
    if is_rebuilt:
        score = max(
            _float(static_media.get("score")),
            _float(static_media.get("p_media_confirmed_score")),
        )
    else:
        score = max(
            _float(static_media.get("score")),
            _float(static_media.get("static_image_score")),
            _float(legacy_static.get("score")),
            _float(legacy_static.get("static_image_score")),
        )
        if score <= 0.0:
            score = max(
                _float(static_media.get("p_media")),
                _float(legacy_static.get("p_media")),
            )
    bbox = (
        static_media.get("p_media_bbox")
        or static_media.get("bbox")
        or static_media.get("candidate_bbox")
        or legacy_static.get("p_media_bbox")
        or legacy_static.get("bbox")
    )
    source = str(
        static_media.get("triggered_source")
        or static_media.get("static_image_triggered_source")
        or legacy_static.get("triggered_source")
        or legacy_static.get("static_image_triggered_source")
        or ("rebuilt_media_confirmed" if is_rebuilt else "legacy_static_media_triggered")
    )
    if source.strip().lower() in {"", "none"}:
        source = "rebuilt_media_confirmed" if is_rebuilt else "legacy_static_media_triggered"
    return True, float(score), bbox, source


def _a3b_backend_health(static_media: dict[str, Any]) -> dict[str, Any]:
    return {
        "a3b_background_enabled": bool(
            static_media.get("a3b_background_enabled", False)
        ),
        "a3b_generation": _int(static_media.get("a3b_generation")),
        "a3b_active_worker_count": _int(
            static_media.get("a3b_active_worker_count")
        ),
        "a3b_retired_worker_count": _int(
            static_media.get("a3b_retired_worker_count")
        ),
        "a3b_live_worker_count": _int(
            static_media.get("a3b_live_worker_count")
        ),
        "a3b_global_live_worker_count": _int(
            static_media.get("a3b_global_live_worker_count")
        ),
        "a3b_global_worker_limit": _int(
            static_media.get("a3b_global_worker_limit")
        ),
        "a3b_worker_limit_scope": str(
            static_media.get("a3b_worker_limit_scope") or "process"
        ),
        "a3b_worker_timeout_s": _float(
            static_media.get("a3b_worker_timeout_s")
        ),
        "a3b_max_retired_workers": _int(
            static_media.get("a3b_max_retired_workers")
        ),
        "a3b_active_worker_started_at": static_media.get(
            "a3b_active_worker_started_at"
        ),
        "a3b_active_worker_age_s": _float(
            static_media.get("a3b_active_worker_age_s")
        ),
        "a3b_active_worker_frame_idx": static_media.get(
            "a3b_active_worker_frame_idx"
        ),
        "a3b_active_worker_timestamp": static_media.get(
            "a3b_active_worker_timestamp"
        ),
        "a3b_timed_out_worker_count": _int(
            static_media.get("a3b_timed_out_worker_count")
        ),
        "a3b_worker_rejected_count": _int(
            static_media.get("a3b_worker_rejected_count")
        ),
        "a3b_last_worker_rejected_at": static_media.get(
            "a3b_last_worker_rejected_at"
        ),
        "a3b_schedule_blocked": bool(
            static_media.get("a3b_schedule_blocked", False)
        ),
        "a3b_schedule_blocked_reason": str(
            static_media.get("a3b_schedule_blocked_reason") or "none"
        ),
        "a3b_error_count": _int(static_media.get("a3b_error_count")),
        "a3b_last_error": static_media.get("a3b_last_error"),
        "a3b_last_error_at": static_media.get("a3b_last_error_at"),
        "a3b_last_success_at": static_media.get("a3b_last_success_at"),
        "a3b_source_frame_idx": static_media.get("a3b_source_frame_idx"),
        "a3b_source_timestamp": static_media.get("a3b_source_timestamp"),
        "a3b_source_fps": _float(static_media.get("a3b_source_fps")),
        "a3b_source_interval_frames": _int(
            static_media.get("a3b_source_interval_frames")
        ),
        "media_source_frame_units": _int(
            static_media.get("media_source_frame_units")
        ),
        "media_tighten_aspect_ratio": _float(
            static_media.get("media_tighten_aspect_ratio")
        ),
        "media_tighten_aspect_pass": bool(
            static_media.get("media_tighten_aspect_pass", False)
        ),
        "a3b_last_attempt_frame_idx": static_media.get(
            "a3b_last_attempt_frame_idx"
        ),
        "a3b_last_attempt_timestamp": static_media.get(
            "a3b_last_attempt_timestamp"
        ),
        "a3b_result_published_at": static_media.get(
            "a3b_result_published_at"
        ),
        "a3b_result_age_s": _float(static_media.get("a3b_result_age_s")),
        "a3b_result_lease_s": _float(static_media.get("a3b_result_lease_s")),
        "a3b_result_fresh": bool(
            static_media.get("a3b_result_fresh", False)
        ),
        "a3b_result_expired_count": _int(
            static_media.get("a3b_result_expired_count")
        ),
        "a3b_result_seq": _int(static_media.get("a3b_result_seq")),
    }


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
        inference_config = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
        runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
        business_min_confidence = float(
            ppe_config.get("business_min_confidence", inference_config.get("confidence", 0.25))
        )
        # person 仅作辅助显示、不进 PPE 报警证据；远距工人后端 conf 常低于 business 门被砍导致
        # "没人物框"。单独下调 person 显示门(默认 0.18)恢复远距人物框，不影响 p_adv/留出集口径。
        _person_display_min = ppe_config.get("person_display_min_confidence", 0.18)
        person_display_min_confidence = (
            float(_person_display_min) if _person_display_min is not None else None
        )
        candidate_min_confidence = ppe_config.get("temporal_candidate_min_confidence")
        self.ppe_postprocess_config = PPEPostprocessConfig(
            min_confidence=business_min_confidence,
            person_display_min_confidence=person_display_min_confidence,
            candidate_min_confidence=(
                float(candidate_min_confidence)
                if candidate_min_confidence is not None
                else None
            ),
            prefer_helmet_on_head_overlap=bool(
                ppe_config.get("prefer_helmet_on_head_overlap", True)
            ),
            head_helmet_mutex_iou=float(ppe_config.get("head_helmet_mutex_iou", 0.20)),
            head_helmet_mutex_center_distance=float(
                ppe_config.get("head_helmet_mutex_center_distance", 0.055)
            ),
            head_helmet_mutex_min_overlap=float(
                ppe_config.get("head_helmet_mutex_min_overlap", 0.18)
            ),
            head_helmet_mutex_min_helmet_confidence=float(
                ppe_config.get("head_helmet_mutex_min_helmet_confidence", 0.25)
            ),
        )
        hold_frames = int(ppe_config.get("max_missed_frames", 10 if ppe_config else 8))
        self.ppe_state = SafetyHelmetState(
            window=int(ppe_config.get("alert_window", 6)),
            trigger_count=int(ppe_config.get("alert_trigger_count", 3)),
            hold_frames=int(ppe_config.get("alert_hold_frames", 12)),
            event_hold_frames=int(ppe_config.get("event_hold_frames", 45)),
            fast_window=int(ppe_config.get("fast_alert_window", 3)),
            fast_trigger_count=int(ppe_config.get("fast_alert_trigger_count", 2)),
            fast_min_confidence=float(ppe_config.get("fast_alert_min_head_confidence", 0.65)),
        )
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
            weak_promotion_hits=int(ppe_config.get("weak_promotion_hits", 3)),
            weak_head_min_avg_confidence=float(ppe_config.get("weak_head_min_avg_confidence", 0.30)),
            weak_helmet_min_avg_confidence=float(ppe_config.get("weak_helmet_min_avg_confidence", 0.30)),
            weak_helmet_isolated_min_avg_confidence=float(
                ppe_config.get("weak_helmet_isolated_min_avg_confidence", 0.50)
            ),
            weak_edge_promotion_hits=int(ppe_config.get("weak_edge_promotion_hits", 4)),
            weak_edge_min_avg_confidence=float(ppe_config.get("weak_edge_min_avg_confidence", 0.45)),
        )
        self.ppe_tracking_enabled = bool(ppe_config.get("enabled", True))
        self.ppe_roi_redetect_enabled = bool(
            ppe_config.get("roi_redetect_enabled", runtime_config.get("ppe_roi_redetect_enabled", False))
        )
        self.ppe_file_realtime_max_render_misses = int(
            runtime_config.get("ppe_file_realtime_max_render_misses", 2) or 2
        )
        # 实时流(摄像头/网络)渲染 miss 宽限: 默认 2(与文件一致), 消除摄像头抖动导致的清框闪断。
        # 缺省键时用 2; 配置显式设为 null 则回退到原"实时流无宽限"行为。
        _stream_misses = runtime_config.get("ppe_stream_max_render_misses", 2)
        self.ppe_stream_max_render_misses = (
            None if _stream_misses is None else int(_stream_misses)
        )
        a3b_config = config.get("a3b", {}) if isinstance(config.get("a3b"), dict) else {}
        module_a_config = (
            config.get("module_a", {})
            if isinstance(config.get("module_a"), dict)
            else {}
        )
        soft_trigger_config = dict(a3b_config)
        soft_trigger_config.update(
            {
                "rebuilt_tighten_gate_enabled": bool(
                    module_a_config.get("rebuilt_a3b_tighten_gate", True)
                ),
                "rebuilt_gate_candidate_min": float(
                    module_a_config.get(
                        "rebuilt_a3b_gate_candidate_min",
                        0.70,
                    )
                ),
                "rebuilt_gate_edge_min": float(
                    module_a_config.get("rebuilt_a3b_gate_edge_min", 0.45)
                ),
                "rebuilt_gate_edge_max": float(
                    module_a_config.get("rebuilt_a3b_gate_edge_max", 0.58)
                ),
                "rebuilt_gate_border_contrast_min": float(
                    module_a_config.get(
                        "rebuilt_a3b_gate_border_contrast_min",
                        0.80,
                    )
                ),
                "rebuilt_gate_candidate_tolerance": float(
                    module_a_config.get(
                        "rebuilt_a3b_soft_gate_candidate_tolerance",
                        0.001,
                    )
                ),
                "rebuilt_gate_aspect_ratio_min": float(
                    module_a_config.get(
                        "rebuilt_a3b_soft_gate_aspect_ratio_min",
                        0.40,
                    )
                ),
                "rebuilt_gate_aspect_ratio_max": float(
                    module_a_config.get(
                        "rebuilt_a3b_soft_gate_aspect_ratio_max",
                        2.50,
                    )
                ),
            }
        )
        self.source_auth_media_suppression_threshold = float(a3b_config.get("observed_threshold", 0.42))
        self.a3b_soft = A3BSoftTriggerState(soft_trigger_config)
        self._a3b_authoritative_confirmed_once = False
        self._a3b_public_alert_hold_frames = max(
            0,
            int(module_a_config.get("rebuilt_a3b_alert_hold_frames", 90)),
        )
        self._a3b_public_alert_hold_remaining = 0
        self._a3b_public_alert_hold_bbox: Any = None
        self._a3b_public_alert_hold_score = 0.0
        self.processing_history: deque[float] = deque(maxlen=30)
        self.jpeg_quality = int(jpeg_quality)
        self._reported_a3b_error_identity: tuple[Any, ...] | None = None
        self._reported_a3b_schedule_block_identity: (
            tuple[Any, ...] | None
        ) = None

    def reset(self) -> None:
        self.pipeline.reset()
        self.ppe_state.reset()
        self.ppe_tracker.reset()
        self.a3b_soft.reset()
        self._a3b_authoritative_confirmed_once = False
        self._a3b_public_alert_hold_remaining = 0
        self._a3b_public_alert_hold_bbox = None
        self._a3b_public_alert_hold_score = 0.0
        self.processing_history.clear()
        self._reported_a3b_error_identity = None
        self._reported_a3b_schedule_block_identity = None

    def _module_a_effective_config(self) -> dict[str, Any]:
        bundle_config = self.bundle.config if isinstance(self.bundle.config, dict) else {}
        module_config = (
            bundle_config.get("module_a", {})
            if isinstance(bundle_config.get("module_a"), dict)
            else {}
        )
        runtime_config = (
            bundle_config.get("runtime", {})
            if isinstance(bundle_config.get("runtime"), dict)
            else {}
        )
        a3b_config = (
            bundle_config.get("a3b", {})
            if isinstance(bundle_config.get("a3b"), dict)
            else {}
        )
        soft_state = getattr(self, "a3b_soft", None)
        soft_config = getattr(soft_state, "config", None)
        soft_config_is_runtime = isinstance(
            soft_config,
            A3BSoftTriggerConfig,
        )
        if not isinstance(soft_config, A3BSoftTriggerConfig):
            soft_config = A3BSoftTriggerConfig.from_mapping(a3b_config)
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            pipeline = getattr(self.bundle, "pipeline", None)
        detector = getattr(pipeline, "detector", None)
        missing = object()

        def effective(
            config_key: str,
            *attribute_names: str,
            owner: Any = detector,
        ) -> Any:
            for attribute_name in attribute_names:
                value = getattr(owner, attribute_name, missing)
                if value is not missing:
                    return value
            return module_config.get(config_key)

        def detector_attribute(*attribute_names: str) -> Any:
            for attribute_name in attribute_names:
                value = getattr(detector, attribute_name, missing)
                if value is not missing:
                    return value
            return None

        detector_impl = getattr(pipeline, "detector_impl", missing)
        if detector_impl is missing:
            detector_impl = module_config.get("detector_impl")
        static_image_interval = effective(
            "static_image_interval",
            "_a3b_interval",
            "static_image_interval",
        )

        def infer_a3b_sensitivity() -> str | None:
            for sensitivity, preset in A3B_SENSITIVITY_PRESETS.items():
                matches = True
                for key, expected in preset.items():
                    if key == "static_image_interval":
                        actual = static_image_interval
                    else:
                        actual = getattr(soft_config, key, missing)
                        if actual is missing:
                            actual = a3b_config.get(key, missing)
                    if actual is missing or actual != expected:
                        matches = False
                        break
                if matches:
                    return sensitivity
            return None

        return {
            "detector_impl": detector_impl,
            "analysis_max_hz": effective(
                "analysis_max_hz",
                "_module_a_analysis_max_hz",
                owner=pipeline,
            ),
            "detector_process_fps_cap": runtime_config.get(
                "detector_process_fps_cap",
                runtime_config.get("process_fps_cap"),
            ),
            "a3b_sensitivity": infer_a3b_sensitivity(),
            "a3b_source_keyword_policy": "diagnostic_only",
            "a3b_source_keyword_match_required": False,
            "a3b_observed_only_source_keywords": list(
                soft_config.observed_only_source_keywords
            ),
            "a3b_trigger_source_keywords": list(
                soft_config.trigger_source_keywords
            ),
            "static_image_enabled": effective(
                "static_image_enabled",
                "static_image_enabled",
            ),
            "static_image_interval": static_image_interval,
            "static_image_worker_timeout_s": effective(
                "static_image_worker_timeout_s",
                "_a3b_worker_timeout_s",
            ),
            "static_image_result_lease_s": effective(
                "static_image_result_lease_s",
                "_a3b_result_lease_s",
            ),
            "static_image_max_retired_workers": effective(
                "static_image_max_retired_workers",
                "_a3b_max_retired_workers",
            ),
            "static_image_global_worker_limit": effective(
                "static_image_global_worker_limit",
                "_a3b_global_worker_limit",
            ),
            "rebuilt_theta_media_raw": effective(
                "rebuilt_theta_media_raw",
                "theta_media_raw",
            ),
            "rebuilt_theta_media": effective(
                "rebuilt_theta_media",
                "theta_media",
            ),
            "rebuilt_theta_adv": effective(
                "rebuilt_theta_adv",
                "theta_adv",
            ),
            "rebuilt_theta_blind": effective(
                "rebuilt_theta_blind",
                "theta_blind",
            ),
            "rebuilt_blind_confirm_ratio": effective(
                "rebuilt_blind_confirm_ratio",
                "_blind_confirm_ratio",
            ),
            "rebuilt_alert_hold_frames": effective(
                "rebuilt_alert_hold_frames",
                "_alert_hold_frames",
            ),
            "rebuilt_a3b_alert_hold_frames": effective(
                "rebuilt_a3b_alert_hold_frames",
                "_a3b_alert_hold_frames",
            ),
            "rebuilt_alert_hold_refresh_on_padv": effective(
                "rebuilt_alert_hold_refresh_on_padv",
                "_alert_hold_refresh_on_padv",
            ),
            "rebuilt_adv_candidate_bridge_frames": effective(
                "rebuilt_adv_candidate_bridge_frames",
                "_adv_cand_bridge_frames",
            ),
            "rebuilt_a4_classifier_rescue_underexposed_max": effective(
                "rebuilt_a4_classifier_rescue_underexposed_max",
                "_a4_classifier_rescue_underexposed_max",
            ),
            "rebuilt_sustained_adv_escalation": effective(
                "rebuilt_sustained_adv_escalation",
                "_sustained_adv_enabled",
            ),
            "rebuilt_sustained_adv_seconds": effective(
                "rebuilt_sustained_adv_seconds",
                "_sustained_adv_seconds",
            ),
            "rebuilt_sustained_adv_run_mult": effective(
                "rebuilt_sustained_adv_run_mult",
                "_sustained_adv_run_mult",
            ),
            "rebuilt_sustained_adv_benign_decay": effective(
                "rebuilt_sustained_adv_benign_decay",
                "_sustained_adv_benign_decay",
            ),
            "rebuilt_sustained_adv_require_target": effective(
                "rebuilt_sustained_adv_require_target",
                "_sustained_adv_require_target",
            ),
            "rebuilt_sustained_adv_require_physical_support": effective(
                "rebuilt_sustained_adv_require_physical_support",
                "_sustained_adv_require_physical_support",
            ),
            "rebuilt_sustained_adv_exclude_static_bg": effective(
                "rebuilt_sustained_adv_exclude_static_bg",
                "_sustained_adv_exclude_static_bg",
            ),
            "rebuilt_sustained_adv_recent_target_min": effective(
                "rebuilt_sustained_adv_recent_target_min",
                "_sustained_adv_recent_target_min",
            ),
            "rebuilt_blind_sustained_escalation": effective(
                "rebuilt_blind_sustained_escalation",
                "_blind_sustained_enabled",
            ),
            "rebuilt_blind_sustained_floor": effective(
                "rebuilt_blind_sustained_floor",
                "_blind_sustained_floor",
            ),
            "rebuilt_blind_sustained_degrade_min": effective(
                "rebuilt_blind_sustained_degrade_min",
                "_blind_sustained_degrade_min",
            ),
            "rebuilt_blind_sustained_established_min": effective(
                "rebuilt_blind_sustained_established_min",
                "_blind_sustained_established_min",
            ),
            "rebuilt_a3b_independent_trigger": effective(
                "rebuilt_a3b_independent_trigger",
                "_a3b_independent_trigger",
            ),
            "rebuilt_a3b_tighten_gate": effective(
                "rebuilt_a3b_tighten_gate",
                "_a3b_tighten_gate",
            ),
            "rebuilt_a3b_gate_candidate_min": effective(
                "rebuilt_a3b_gate_candidate_min",
                "_a3b_gate_candidate_min",
            ),
            "rebuilt_a3b_gate_edge_min": effective(
                "rebuilt_a3b_gate_edge_min",
                "_a3b_gate_edge_min",
            ),
            "rebuilt_a3b_gate_edge_max": effective(
                "rebuilt_a3b_gate_edge_max",
                "_a3b_gate_edge_max",
            ),
            "rebuilt_a3b_gate_border_contrast_min": effective(
                "rebuilt_a3b_gate_border_contrast_min",
                "_a3b_gate_border_contrast_min",
            ),
            "rebuilt_a3b_soft_gate_candidate_tolerance": (
                soft_config.rebuilt_gate_candidate_tolerance
                if soft_config_is_runtime
                else module_config.get(
                    "rebuilt_a3b_soft_gate_candidate_tolerance"
                )
            ),
            "rebuilt_a3b_soft_gate_aspect_ratio_min": (
                soft_config.rebuilt_gate_aspect_ratio_min
                if soft_config_is_runtime
                else module_config.get(
                    "rebuilt_a3b_soft_gate_aspect_ratio_min"
                )
            ),
            "rebuilt_a3b_soft_gate_aspect_ratio_max": (
                soft_config.rebuilt_gate_aspect_ratio_max
                if soft_config_is_runtime
                else module_config.get(
                    "rebuilt_a3b_soft_gate_aspect_ratio_max"
                )
            ),
            "rebuilt_a3b_media_run_floor": effective(
                "rebuilt_a3b_media_run_floor",
                "_a3b_media_run_floor",
            ),
            "rebuilt_a3b_media_run_gap_tol": effective(
                "rebuilt_a3b_media_run_gap_tol",
                "_a3b_media_run_gap_tol",
            ),
            "flow_requested_device": (
                detector_attribute("flow_requested_device")
                or module_config.get("device")
            ),
            "flow_effective_device": detector_attribute(
                "flow_effective_device",
                "effective_device",
            ),
            "flow_backend": detector_attribute(
                "flow_backend",
                "backend",
            ),
            "flow_fallback_reason": detector_attribute(
                "flow_fallback_reason",
                "fallback_reason",
            ),
            "flow_artifact_path": detector_attribute(
                "flow_artifact_path",
            ),
            "flow_artifact_sha256": detector_attribute(
                "flow_artifact_sha256",
            ),
            "flow_artifact_expected_sha256": detector_attribute(
                "flow_artifact_expected_sha256",
            ),
            "a4_classifier_configured": detector_attribute(
                "a4_classifier_configured"
            ),
            "a4_classifier_loaded": detector_attribute(
                "a4_classifier_loaded"
            ),
            "a4_classifier_error": detector_attribute(
                "a4_classifier_error"
            ),
            "a4_classifier_fallback_reason": detector_attribute(
                "a4_classifier_fallback_reason"
            ),
            "a4_classifier_alarm_window": detector_attribute(
                "a4_classifier_alarm_window"
            ),
            "a4_classifier_alarm_required_hits": detector_attribute(
                "a4_classifier_alarm_required_hits"
            ),
            "a4_classifier_path": detector_attribute(
                "a4_classifier_resolved_path"
            ),
            "a4_classifier_sha256": detector_attribute(
                "a4_classifier_sha256"
            ),
            "a4_classifier_expected_sha256": detector_attribute(
                "a4_classifier_expected_sha256"
            ),
            "native": detector_attribute("native_status"),
        }

    def _warn_new_a3b_backend_error(self, health: dict[str, Any]) -> None:
        error_count = _int(health.get("a3b_error_count"))
        if error_count > 0:
            identity = (
                health.get("a3b_generation"),
                error_count,
                health.get("a3b_last_error_at"),
                health.get("a3b_last_error"),
            )
            if identity != getattr(
                self,
                "_reported_a3b_error_identity",
                None,
            ):
                self._reported_a3b_error_identity = identity
                logger.warning(
                    "A3b background backend error count=%s generation=%s: %s",
                    error_count,
                    health.get("a3b_generation"),
                    health.get("a3b_last_error") or "unknown error",
                )

        if not bool(health.get("a3b_schedule_blocked", False)):
            self._reported_a3b_schedule_block_identity = None
            return
        blocked_identity = (
            health.get("a3b_generation"),
            health.get("a3b_schedule_blocked_reason"),
        )
        if blocked_identity == getattr(
            self,
            "_reported_a3b_schedule_block_identity",
            None,
        ):
            return
        self._reported_a3b_schedule_block_identity = blocked_identity
        logger.warning(
            "A3b background scheduling blocked generation=%s reason=%s "
            "local_live=%s global_live=%s global_limit=%s",
            health.get("a3b_generation"),
            health.get("a3b_schedule_blocked_reason") or "unknown",
            health.get("a3b_live_worker_count"),
            health.get("a3b_global_live_worker_count"),
            health.get("a3b_global_worker_limit"),
        )

    def process(
        self,
        frame: np.ndarray | None,
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
        temporal_previous_frame: np.ndarray | None = None,
        temporal_previous_frame_idx: int | None = None,
        temporal_previous_source_time_s: float | None = None,
        decoded_frame_lease: DecodedFrameLease | None = None,
        temporal_previous_decoded_frame: DecodedFrameLease | None = None,
    ) -> ProcessedFrame:
        started = time.perf_counter()
        frame_materialization_started = time.perf_counter()
        detector_input = None
        if decoded_frame_lease is not None:
            frame_640 = decoded_frame_lease.materialize_host_bgr(size=(640, 640))
            if str(decoded_frame_lease.storage).lower().startswith("cuda"):
                detector_input = decoded_frame_lease.cuda_tensor
        else:
            if frame is None:
                raise ValueError("frame or decoded_frame_lease is required")
            frame_640 = prepare_frame_640(frame)
        current_frame_materialization_ms = (
            time.perf_counter() - frame_materialization_started
        ) * 1000.0

        previous_frame_materialization_ms = 0.0
        effective_temporal_previous = temporal_previous_frame
        previous_frame_provider = None
        if (
            effective_temporal_previous is None
            and temporal_previous_decoded_frame is not None
        ):
            def materialize_temporal_previous() -> np.ndarray:
                nonlocal previous_frame_materialization_ms
                previous_materialization_started = time.perf_counter()
                try:
                    return temporal_previous_decoded_frame.materialize_host_bgr(
                        size=(640, 640)
                    )
                finally:
                    previous_frame_materialization_ms = (
                        time.perf_counter() - previous_materialization_started
                    ) * 1000.0

            previous_frame_provider = materialize_temporal_previous

        process_runtime_frame = getattr(self.pipeline, "process_runtime_frame", None)
        if callable(process_runtime_frame):
            _, detections, info = process_runtime_frame(
                frame_640,
                timestamp=float(video_time_s),
                previous_frame=effective_temporal_previous,
                current_source_frame_idx=int(frame_idx),
                previous_source_frame_idx=temporal_previous_frame_idx,
                previous_source_time_s=temporal_previous_source_time_s,
                detector_input=detector_input,
                previous_frame_provider=previous_frame_provider,
            )
        else:
            _, detections, info = self.pipeline.process_frame(frame_640)
        latency_breakdown = (
            info.setdefault("latency_breakdown", {})
            if isinstance(info, dict)
            else {}
        )
        if isinstance(latency_breakdown, dict):
            latency_breakdown["frame_materialization_ms"] = float(
                current_frame_materialization_ms
            )
            latency_breakdown["previous_frame_materialization_ms"] = float(
                previous_frame_materialization_ms
            )
        _feature_probe(info, frame_idx, detections)
        # Normalize the backend A3b contract once. PPE suppression and status
        # consume the same mapping so repeated result adaptation is avoided.
        static_media = dict(_static_media_details(info))
        static_media["source_path"] = source
        a3b_soft = self.a3b_soft.update(static_media)
        a3b_soft_triggered = bool(a3b_soft.get("triggered", False))
        authoritative_a3b_triggered, _, authoritative_a3b_bbox, _ = (
            _authoritative_a3b_confirmation(static_media)
        )
        effective_a3b_bbox = (
            a3b_soft.get("effective_bbox")
            if a3b_soft_triggered
            else authoritative_a3b_bbox or static_media.get("p_media_bbox")
        )
        redetect_ms = 0.0
        redetect_count = 0
        avg_processing_ms = (
            (sum(self.processing_history) / len(self.processing_history)) * 1000.0
            if self.processing_history
            else 0.0
        )
        budget_ok = (
            bool(self.ppe_roi_redetect_enabled)
            and (not self.processing_history or avg_processing_ms <= target_frame_budget_ms * 0.85)
        )
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
            max_render_misses=_ppe_max_render_misses(
                source_type=source_type,
                realtime=realtime,
                file_realtime_max_misses=self.ppe_file_realtime_max_render_misses,
                stream_max_misses=self.ppe_stream_max_render_misses,
            ),
            postprocess_config=self.ppe_postprocess_config,
            source_auth_media_bbox=effective_a3b_bbox,
            # Freshness and policy suppression are frame-visible state, so this
            # small CPU gate is intentionally evaluated once per processed frame
            # rather than cached across duplicate result_seq values.
            source_auth_suppression_active=_source_auth_media_suppression_active(
                static_media,
                threshold=self.source_auth_media_suppression_threshold,
                runtime_triggered=bool(
                    a3b_soft_triggered or authoritative_a3b_triggered
                ),
                bbox=effective_a3b_bbox,
            ),
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
            target_frame_budget_ms=target_frame_budget_ms,
            raw_boxes_count=len(getattr(detections, "boxes", []) or []),
            static_media=static_media,
            a3b_soft=a3b_soft,
        )
        status.update(
            {
                "detector_input_device": str(
                    latency_breakdown.get(
                        "detector_input_device",
                        getattr(detections, "input_device", "host"),
                    )
                    or "host"
                ),
                "detector_input_format": str(
                    latency_breakdown.get(
                        "detector_input_format",
                        getattr(detections, "input_format", "bgr24"),
                    )
                    or "bgr24"
                ),
                "detector_preprocess_ms": float(
                    latency_breakdown.get(
                        "detector_preprocess_ms",
                        getattr(detections, "preprocess_ms", 0.0),
                    )
                    or 0.0
                ),
                "frame_materialization_ms": float(
                    current_frame_materialization_ms
                ),
                "previous_frame_materialization_ms": float(
                    previous_frame_materialization_ms
                ),
            }
        )
        effective_config = status.get("module_a_effective_config")
        native = (
            effective_config.get("native")
            if isinstance(effective_config, dict)
            and isinstance(effective_config.get("native"), dict)
            else None
        )
        if isinstance(native, dict):
            status["native"] = {
                **native,
                "version": str(
                    native.get("crate_version")
                    or native.get("version")
                    or "unknown"
                ),
                "enabled_stages": list(
                    native.get("enabled_stages") or []
                ),
                "fallback_reason": str(
                    native.get("fallback_reason")
                    or native.get("load_error")
                    or "none"
                ),
            }
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
        target_frame_budget_ms: float,
        raw_boxes_count: int,
        static_media: dict[str, Any] | None = None,
        a3b_soft: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        static_media = (
            dict(static_media)
            if isinstance(static_media, dict)
            else dict(_static_media_details(info))
        )
        static_media["source_path"] = source
        latency = info.get("latency_breakdown", {}) if isinstance(info.get("latency_breakdown"), dict) else {}
        temporal_input = info.get("temporal_input", {}) if isinstance(info.get("temporal_input"), dict) else {}
        module_breakdown = latency.get("module_a_breakdown", {}) if isinstance(latency.get("module_a_breakdown"), dict) else {}
        module_details = (
            info.get("details", {})
            if isinstance(info.get("details"), dict)
            else {}
        )
        runtime_frame_lineage = (
            module_details.get("runtime_frame_lineage", {})
            if isinstance(
                module_details.get("runtime_frame_lineage"),
                dict,
            )
            else {}
        )
        joint_decision = (
            module_details.get("joint_decision", {})
            if isinstance(module_details.get("joint_decision"), dict)
            else {}
        )
        confirm_window = (
            joint_decision.get("confirm_window", {})
            if isinstance(joint_decision.get("confirm_window"), dict)
            else {}
        )
        blinding = (
            module_details.get("blinding", {})
            if isinstance(module_details.get("blinding"), dict)
            else {}
        )
        raw_module_a_primary_channel = str(
            joint_decision.get("primary_channel") or "none"
        )
        normalized_primary_channel = (
            raw_module_a_primary_channel.strip().lower() or "none"
        )
        media_primary = normalized_primary_channel in {
            "media",
            "p_media",
            "a3b",
            "static_media",
        }
        module_a_primary_channel = (
            "a3b" if media_primary else normalized_primary_channel
        )
        module_a_alert_held = bool(confirm_window.get("alert_held", False))
        module_a_alert_hold_remaining = _int(
            confirm_window.get("alert_hold_remaining")
        )
        backend_media_alert_held = bool(
            media_primary
            and module_a_alert_held
            and module_a_alert_hold_remaining > 0
        )
        p_adv = info.get("p_adv")
        # The rebuilt detector's legacy umbrella flags also cover its internal
        # ``primary_channel=media`` branch.  Runtime/Web legacy fields are a
        # physical-only contract, so media/A3b confirmations must be removed
        # from those fields and exposed only through the explicit A3b/Module A
        # union below.
        physical_alert_confirmed = bool(
            info.get("alert_confirmed", False)
        ) and not media_primary
        physical_attack_detected = bool(
            info.get("attack_detected", False)
        ) and not media_primary
        physical_attack_state_active = bool(
            info.get("attack_state_active", False)
        ) and not media_primary
        a3b_soft = (
            dict(a3b_soft)
            if isinstance(a3b_soft, dict)
            else self.a3b_soft.update(static_media)
        )
        a3b_triggered = bool(a3b_soft["triggered"])
        a3b_observed_score = _float(a3b_soft.get("observed_score"))
        a3b_smoothed_score = _float(
            static_media.get("live_score_display", static_media.get("score", a3b_observed_score))
        )
        # rebuilt 内核已对 media 做过 N-of-M 确认(media_confirmed), 是权威结果; 面板不应再经 a3b_soft
        # 二次确认导致 confirmed_score 时有时无(面板一直显 0)。已确认时用 rebuilt 的 p_media_confirmed_score
        # 直接回填面板置信度/卡片分数, 并标 confirmed。observed 仍走 a3b_soft(平滑显示)。纯显示, 不改检测。
        (
            _authoritative_confirmed,
            _authoritative_score,
            _authoritative_bbox,
            _authoritative_source,
        ) = _authoritative_a3b_confirmation(static_media)
        a3b_confirmed_score = _float(a3b_soft.get("confirmed_score"))
        if _authoritative_confirmed and _authoritative_score > a3b_confirmed_score:
            a3b_confirmed_score = _authoritative_score
        a3b_confidence = _float(a3b_soft.get("confidence", a3b_confirmed_score))
        if _authoritative_confirmed and _authoritative_score > a3b_confidence:
            a3b_confidence = _authoritative_score
        a3b_display_score = _float(a3b_soft.get("display_score"), a3b_confidence)
        if _authoritative_confirmed and _authoritative_score > a3b_display_score:
            a3b_display_score = _authoritative_score
        a3b_card_score = a3b_confidence
        a3b_event_score = a3b_confidence if a3b_confidence > 0 else a3b_observed_score
        a3b_state = str(a3b_soft.get("state") or ("confirmed" if a3b_soft.get("triggered") else "normal"))
        # Some contract tests and embedders construct FrameProcessor without
        # calling __init__.  Keep the public hold state backwards compatible.
        if not hasattr(self, "_a3b_authoritative_confirmed_once"):
            self._a3b_authoritative_confirmed_once = False
        if not hasattr(self, "_a3b_public_alert_hold_frames"):
            self._a3b_public_alert_hold_frames = 90
        if not hasattr(self, "_a3b_public_alert_hold_remaining"):
            self._a3b_public_alert_hold_remaining = 0
        if not hasattr(self, "_a3b_public_alert_hold_bbox"):
            self._a3b_public_alert_hold_bbox = None
        if not hasattr(self, "_a3b_public_alert_hold_score"):
            self._a3b_public_alert_hold_score = 0.0
        if _authoritative_confirmed:
            self._a3b_authoritative_confirmed_once = True
        a3b_soft_debug = dict(a3b_soft.get("debug") or {})
        a3b_reacquired_after_authoritative = bool(
            self._a3b_authoritative_confirmed_once
            and a3b_triggered
            and a3b_state.strip().lower() == "suspect"
            and bool(a3b_soft_debug.get("quality_gate_passed", False))
            and not list(
                a3b_soft_debug.get("current_explicit_guard_failures")
                or []
            )
            and a3b_soft.get("effective_bbox") is not None
        )
        a3b_public_fresh_confirmed = bool(
            _authoritative_confirmed
            or a3b_reacquired_after_authoritative
        )
        current_a3b_bbox = (
            _authoritative_bbox
            or a3b_soft.get("effective_bbox")
            or static_media.get("p_media_bbox")
        )
        current_a3b_score = max(
            a3b_confirmed_score,
            a3b_confidence,
            a3b_observed_score,
            _authoritative_score,
        )
        a3b_public_hold_active = False
        if a3b_public_fresh_confirmed:
            self._a3b_public_alert_hold_remaining = (
                self._a3b_public_alert_hold_frames
            )
            self._a3b_public_alert_hold_bbox = current_a3b_bbox
            self._a3b_public_alert_hold_score = current_a3b_score
        elif (
            self._a3b_authoritative_confirmed_once
            and self._a3b_public_alert_hold_remaining > 0
        ):
            self._a3b_public_alert_hold_remaining -= 1
            a3b_public_hold_active = True
            a3b_confirmed_score = max(
                a3b_confirmed_score,
                self._a3b_public_alert_hold_score * 0.85,
            )
            a3b_confidence = max(a3b_confidence, a3b_confirmed_score)
            a3b_display_score = max(a3b_display_score, a3b_confirmed_score)
            a3b_card_score = max(a3b_card_score, a3b_confirmed_score)
            a3b_event_score = max(a3b_event_score, a3b_confirmed_score)
        if (
            _authoritative_confirmed
            or backend_media_alert_held
            or a3b_reacquired_after_authoritative
            or a3b_public_hold_active
        ):
            a3b_state = "confirmed"
            a3b_triggered = True
        if a3b_public_hold_active:
            module_a_alert_held = True
            module_a_alert_hold_remaining = max(
                module_a_alert_hold_remaining,
                self._a3b_public_alert_hold_remaining,
            )
        a3b_confirmed_alert = bool(
            a3b_triggered and a3b_state.strip().lower() == "confirmed"
        )
        module_a_alert_confirmed = bool(
            physical_alert_confirmed or a3b_confirmed_alert
        )
        module_a_attack_detected = bool(
            physical_attack_detected or a3b_confirmed_alert
        )
        module_a_attack_state_active = bool(
            physical_attack_state_active or a3b_confirmed_alert
        )
        module_a_alert_channel = (
            module_a_primary_channel
            if physical_alert_confirmed or physical_attack_state_active
            else "a3b"
            if a3b_confirmed_alert
            else "none"
        )
        effective_a3b_bbox = (
            a3b_soft.get("effective_bbox")
            if bool(a3b_soft.get("triggered"))
            else static_media.get("p_media_bbox")
        )
        if _authoritative_confirmed and _authoritative_bbox is not None:
            effective_a3b_bbox = _authoritative_bbox
        elif a3b_public_hold_active and self._a3b_public_alert_hold_bbox is not None:
            effective_a3b_bbox = self._a3b_public_alert_hold_bbox
        if a3b_triggered and effective_a3b_bbox is None:
            effective_a3b_bbox = static_media.get("p_media_bbox")
        a3b_triggered_source = str(a3b_soft.get("triggered_source") or "none")
        if _authoritative_confirmed:
            a3b_triggered_source = _authoritative_source
        elif backend_media_alert_held:
            a3b_triggered_source = "rebuilt_media_hold"
        elif a3b_reacquired_after_authoritative:
            a3b_triggered_source = "rebuilt_media_reacquired"
        elif a3b_public_hold_active:
            a3b_triggered_source = "rebuilt_media_public_hold"
        a3b_health = _a3b_backend_health(static_media)
        self._warn_new_a3b_backend_error(a3b_health)
        a3b_debug = a3b_soft_debug
        a3b_debug.update(a3b_health)
        a3b_debug["rebuilt_backend_media_alert_held"] = bool(
            backend_media_alert_held
        )
        a3b_debug["rebuilt_reacquired_after_authoritative"] = bool(
            a3b_reacquired_after_authoritative
        )
        a3b_debug["rebuilt_public_alert_hold_active"] = bool(
            a3b_public_hold_active
        )
        a3b_debug["rebuilt_public_alert_hold_remaining"] = int(
            self._a3b_public_alert_hold_remaining
        )
        ppe_boxes_count = _int(ppe.get("person_count")) + _int(ppe.get("helmet_count")) + _int(ppe.get("head_count"))
        ppe_suppression = ppe.get("helmet_fp_suppression", {})
        ppe_weak_person_count = (
            len(ppe_suppression.get("weak_person_indices", []) or [])
            if isinstance(ppe_suppression, dict)
            else 0
        )
        ppe_weak_head_count = (
            len(ppe_suppression.get("weak_head_indices", []) or [])
            if isinstance(ppe_suppression, dict)
            else 0
        )
        ppe_weak_helmet_count = (
            len(ppe_suppression.get("weak_helmet_indices", []) or [])
            if isinstance(ppe_suppression, dict)
            else 0
        )
        source_auth_suppression = (
            ppe.get("source_auth_media_suppression", {})
            if isinstance(ppe.get("source_auth_media_suppression"), dict)
            else {}
        )
        source_auth_suppressed_labels = [
            str(label) for label in source_auth_suppression.get("suppressed_labels", []) or []
        ]
        tracked_boxes_count = len([track for track in ppe_tracks if str(track.get("source", "detected")) in {"tracked", "held"}])
        render_boxes_count = len(ppe_tracks)
        visible_ppe_counts = _ppe_track_label_counts(ppe_tracks)
        visible_person_count = visible_ppe_counts.get("person", 0)
        visible_helmet_count = visible_ppe_counts.get("helmet", 0)
        visible_head_count = visible_ppe_counts.get("head", 0)
        display_person_count = (
            visible_person_count
            if bool(ppe.get("has_person_class", False))
            else max(visible_person_count, _int(ppe.get("inferred_person_count", ppe.get("person_count"))))
        )
        ppe_boxes_count = visible_person_count + visible_helmet_count + visible_head_count
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
            "a3b_static_media_ms": _float(
                module_breakdown.get(
                    "a3b_static_media_ms",
                    module_breakdown.get("a3b_schedule"),
                )
            ),
            "target_frame_budget_ms": float(target_frame_budget_ms),
            "processing_budget_ok": bool(processing_ms <= target_frame_budget_ms),
            "latency_breakdown": latency,
            "detector_reuse_hit": bool(latency.get("detector_reuse_hit", False)),
            "detector_change_score": _float(latency.get("detector_change_score")),
            "source_frame_shape": latency.get("source_frame_shape", []),
            "detector_frame_shape": latency.get("detector_frame_shape", []),
            "temporal_input": dict(temporal_input),
            "module_a_processed_frame_idx": (
                runtime_frame_lineage.get("processed_frame_idx")
            ),
            "module_a_source_frame_idx": (
                runtime_frame_lineage.get("source_frame_idx")
            ),
            "module_a_input_frame_idx": (
                runtime_frame_lineage.get("module_a_input_frame_idx")
            ),
            "temporal_previous_frame_applied": bool(temporal_input.get("previous_frame_applied", False)),
            "temporal_strict_source_predecessor": bool(
                temporal_input.get("strict_source_predecessor", False)
            ),
            "temporal_source_gap_frames": temporal_input.get("source_gap_frames"),
            "p_adv": None if p_adv is None else _float(p_adv),
            "p_adv_display": _float(info.get("p_adv_display", p_adv or 0.0)),
            "p_adv_missing_reason": str(info.get("p_adv_missing_reason", "")),
            # Keep the legacy physical-channel fields stable for the p_adv /
            # p_blind card and physical-attack acceptance.  The explicit
            # module_a_* fields are the public umbrella state consumed by the
            # top-level Web alert: a confirmed A3b attack is still a Module A
            # alert, but must not masquerade as a physical-channel hit.
            "alert_confirmed": physical_alert_confirmed,
            "physical_alert_confirmed": physical_alert_confirmed,
            "module_a_alert_confirmed": module_a_alert_confirmed,
            "module_a_alert_channel": module_a_alert_channel,
            "single_frame_suspicious": bool(
                info.get("single_frame_suspicious", False)
            ),
            "attack_detected": physical_attack_detected,
            "physical_attack_detected": physical_attack_detected,
            "module_a_attack_detected": module_a_attack_detected,
            "attack_state_active": physical_attack_state_active,
            "physical_attack_state_active": physical_attack_state_active,
            "module_a_attack_state_active": module_a_attack_state_active,
            "module_a_primary_channel": module_a_primary_channel,
            "module_a_alert_held": module_a_alert_held,
            "module_a_alert_hold_remaining": module_a_alert_hold_remaining,
            "module_a_fresh_confirmed": bool(
                physical_alert_confirmed
                and not module_a_alert_held
            ),
            "p_blind": _float(blinding.get("p_blind")),
            "p_blind_triggered": bool(
                blinding.get("p_blind_triggered", False)
            ),
            "blind_type": str(blinding.get("blind_type") or "none"),
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
            "a3b_confirmed_alert": a3b_confirmed_alert,
            "a3b_p_media": _float(static_media.get("p_media")),
            "a3b_bbox": effective_a3b_bbox,
            "a3b_triggered_source": a3b_triggered_source,
            "a3b_reason": str(a3b_soft.get("reason") or ""),
            "a3b_debug": a3b_debug,
            **a3b_health,
            "module_a_effective_config": self._module_a_effective_config(),
            "ppe_warning": bool(ppe.get("warning", False)),
            "ppe_candidate": bool(ppe.get("candidate", False)),
            "ppe_confirmed": bool(ppe.get("confirmed", False)),
            "ppe_confirmed_source": str(ppe.get("confirmed_source", "")),
            "ppe_event_active": bool(ppe.get("event_active", False)),
            "ppe_event_hold_remaining": _int(ppe.get("event_hold_remaining")),
            "ppe_event_last_reason": str(ppe.get("event_last_reason", "")),
            "ppe_event_last_confirmed_source": str(ppe.get("event_last_confirmed_source", "")),
            "ppe_person_count": int(visible_person_count),
            "ppe_raw_person_count": _int(ppe.get("raw_person_count", ppe.get("person_count"))),
            "ppe_inferred_person_count": int(display_person_count),
            "ppe_person_context_count": int(visible_person_count),
            "ppe_weak_person_count": int(ppe_weak_person_count),
            "ppe_promoted_person_count": _int(ppe.get("promoted_person_count")),
            "ppe_effective_person_count": int(visible_person_count),
            "ppe_helmet_count": int(visible_helmet_count),
            "ppe_raw_helmet_count": _int(ppe.get("raw_helmet_count", ppe.get("helmet_count"))),
            "ppe_weak_helmet_count": int(ppe_weak_helmet_count),
            "ppe_promoted_helmet_count": _int(ppe.get("promoted_helmet_count")),
            "ppe_effective_helmet_count": int(visible_helmet_count),
            "ppe_head_count": int(visible_head_count),
            "ppe_raw_head_count": _int(ppe.get("raw_head_count", ppe.get("head_count"))),
            "ppe_weak_head_count": int(ppe_weak_head_count),
            "ppe_promoted_head_count": _int(ppe.get("promoted_head_count")),
            "ppe_effective_head_count": int(visible_head_count),
            "ppe_missing_helmet_count": _int(ppe.get("missing_helmet_count")),
            "ppe_has_person_class": bool(ppe.get("has_person_class", False)),
            "ppe_evidence_mode": str(ppe.get("evidence_mode", "")),
            "ppe_uncertain": bool(ppe.get("uncertain", False)),
            "ppe_reason": str(ppe.get("reason", "")),
            "ppe_source_auth_media_suppressed": bool(source_auth_suppression.get("active", False))
            and _int(source_auth_suppression.get("suppressed_count")) > 0,
            "ppe_source_auth_temporal_reset": bool(ppe.get("source_auth_temporal_reset", False)),
            "ppe_source_auth_media_bbox": source_auth_suppression.get("bbox"),
            "ppe_source_auth_media_suppressed_count": _int(source_auth_suppression.get("suppressed_count")),
            "ppe_source_auth_media_suppressed_person_count": sum(
                1 for label in source_auth_suppressed_labels if is_person_label(label)
            ),
            "ppe_source_auth_media_suppressed_head_count": sum(
                1 for label in source_auth_suppressed_labels if is_bare_head_label(label)
            ),
            "ppe_source_auth_media_suppressed_helmet_count": sum(
                1 for label in source_auth_suppressed_labels if is_helmet_label(label)
            ),
            "ppe_source_auth_media_suppression_reason": str(source_auth_suppression.get("reason") or ""),
            "ppe_window_positive": _int(ppe.get("window_positive")),
            "ppe_window": _int(ppe.get("window")),
            "ppe_fast_window_positive": _int(ppe.get("fast_window_positive")),
            "ppe_fast_window": _int(ppe.get("fast_window")),
            "ppe_fast_trigger_count": _int(ppe.get("fast_trigger_count")),
            "ppe_track_count": len(ppe_tracks),
            "ppe_tracks": [dict(track) for track in ppe_tracks],
            "ppe_class_counts": dict(visible_ppe_counts),
            "ppe_raw_class_counts": ppe.get("class_counts", {}),
            "raw_boxes_count": int(raw_boxes_count),
            "ppe_boxes_count": int(ppe_boxes_count),
            "tracked_boxes_count": int(tracked_boxes_count),
            "render_boxes_count": int(render_boxes_count),
            "ppe_file_realtime_max_render_misses": int(self.ppe_file_realtime_max_render_misses),
            "ppe_stream_max_render_misses": self.ppe_stream_max_render_misses,
            "ppe_roi_redetect_budget_ok": bool(redetect_budget_ok),
            "ppe_roi_redetect_triggered": bool(redetect_count > 0),
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


def _ppe_max_render_misses(
    *,
    source_type: str,
    realtime: bool,
    file_realtime_max_misses: int = 2,
    stream_max_misses: int | None = 2,
) -> int | None:
    """渲染 miss 宽限帧数(head/helmet 漏检若干帧内保持显示)。
    - 文件回放: 沿用 file_realtime_max_misses(默认2)。
    - 实时流(摄像头/网络): 之前返回 None(无宽限), 导致摄像头画面天然抖动时任一帧
      漏检/置信度跌破地板就立刻清框, 主观"很难稳定出框"。给实时流同样开宽限(默认2),
      纯显示层, 不影响检测/留出集离线口径(留出集走 VideoDefensePipeline 不经此)。
      设为 None 可回退到原无宽限行为。"""
    kind = str(source_type or "").lower()
    if kind == "file" and bool(realtime):
        return max(0, int(file_realtime_max_misses))
    if kind in {"camera", "rtsp"}:
        return None if stream_max_misses is None else max(0, int(stream_max_misses))
    return None


def _source_auth_media_suppression_active(
    static_media: dict[str, Any],
    *,
    threshold: float = 0.42,
    runtime_triggered: bool = False,
    bbox: Any = None,
) -> bool:
    effective_bbox = bbox if bbox is not None else static_media.get("p_media_bbox")
    if not effective_bbox:
        return False

    is_rebuilt = static_media.get("result_contract_source") == "rebuilt"
    if is_rebuilt and not bool(static_media.get("a3b_result_fresh", False)):
        return False

    policy = (
        static_media.get("policy")
        if isinstance(static_media.get("policy"), dict)
        else {}
    )
    suppression = (
        static_media.get("suppression")
        if isinstance(static_media.get("suppression"), dict)
        else {}
    )
    policy_state = (
        static_media.get("p_media_policy_state")
        if isinstance(static_media.get("p_media_policy_state"), dict)
        else {}
    )
    candidate_allowed_values = [
        static_media.get("media_candidate_allowed")
        if "media_candidate_allowed" in static_media
        else None,
        policy.get("media_candidate_allowed")
        if "media_candidate_allowed" in policy
        else None,
        suppression.get("media_candidate_allowed")
        if "media_candidate_allowed" in suppression
        else None,
        policy_state.get("media_candidate_allowed")
        if "media_candidate_allowed" in policy_state
        else None,
    ]
    if any(value is not None and not bool(value) for value in candidate_allowed_values):
        return False

    if any(
        bool(container.get("suppressed", False))
        for container in (policy, suppression, policy_state)
    ):
        return False

    no_suppression_reasons = {"", "none", "normal", "not_suppressed"}
    suppression_reasons = {
        str(static_media.get("suppressed_reason") or "").strip().lower(),
        str(policy.get("suppressed_reason") or "").strip().lower(),
        str(policy.get("reason") or "").strip().lower(),
        str(suppression.get("reason") or "").strip().lower(),
        str(policy_state.get("reason") or "").strip().lower(),
    }
    if any(
        reason not in no_suppression_reasons
        for reason in suppression_reasons
    ):
        return False
    if str(static_media.get("a3b_state") or "").strip().lower() == "suppressed":
        return False

    score_threshold = float(threshold)
    return bool(
        runtime_triggered
        or static_media.get("triggered")
        or static_media.get("p_media_triggered")
        or static_media.get("classifier_triggered")
        or _float(static_media.get("p_media")) >= score_threshold
        or _float(static_media.get("live_score")) >= score_threshold
        or _float(static_media.get("live_score_display")) >= score_threshold
        or _float(static_media.get("score")) >= score_threshold
    )


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
    module_a_primary_channel = str(
        status.get("module_a_primary_channel") or "none"
    )
    module_a_alert_held = bool(status.get("module_a_alert_held", False))
    blind_primary = module_a_primary_channel == "blind"
    module_a_score = status.get("p_blind") if blind_primary else p_adv
    module_a_score_missing = module_a_score is None
    p_adv_confirmed = bool(status.get("alert_confirmed"))
    p_adv_active = bool(status.get("attack_state_active") or status.get("attack_detected"))
    if blind_primary:
        module_a_title = "致盲/去信号攻击（p_blind）"
        module_a_badges = ["模块A", "致盲"]
    else:
        module_a_title = "物理对抗扰动（p_adv）"
        module_a_badges = ["模块A"]
    if module_a_score_missing:
        adv_class = "card-missing"
        adv_state = "待检测"
        adv_detail = (
            "尚未产生致盲检测结果。"
            if blind_primary
            else status.get("p_adv_missing_reason")
            or "尚未产生物理扰动检测结果。"
        )
    elif module_a_alert_held and p_adv_confirmed:
        adv_class = "card-warning"
        adv_state = "告警保持"
        remaining = _int(status.get("module_a_alert_hold_remaining"))
        channel_text = "致盲/去信号" if blind_primary else "物理扰动"
        adv_detail = (
            f"最近一次{channel_text}确认仍在保持窗口"
            + (f"，剩余 {remaining} 帧。" if remaining > 0 else "。")
        )
    elif p_adv_confirmed:
        adv_class = "card-confirmed"
        adv_state = "确认告警"
        adv_detail = (
            "连续帧满足致盲/去信号告警条件。"
            if blind_primary
            else "连续帧满足模块A物理扰动告警条件。"
        )
    elif p_adv_active:
        adv_class = "card-warning"
        adv_state = "疑似扰动"
        adv_detail = (
            "当前帧存在致盲/去信号迹象，等待连续帧确认。"
            if blind_primary
            else "当前帧存在物理扰动迹象，等待连续帧确认。"
        )
    else:
        adv_class = "card-idle"
        adv_state = "OK"
        adv_detail = (
            "未触发致盲/去信号检测。"
            if blind_primary
            else "未触发物理扰动检测。"
        )

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
            "title": module_a_title,
            "score": (
                None
                if module_a_score is None
                else _float(module_a_score)
            ),
            "score_display": _score_display(module_a_score),
            "score_bar_ratio": _bar_ratio(module_a_score),
            "border_class": adv_class,
            "state": adv_state,
            "state_detail": adv_detail,
            "reason_text": status.get("reason") or "",
            "badges": (
                module_a_badges
                + (
                    ["告警保持"]
                    if module_a_alert_held and p_adv_confirmed
                    else ["连续帧"]
                    if p_adv_confirmed
                    else []
                )
            ),
            "primary_channel": module_a_primary_channel,
            "score_source": "p_blind" if blind_primary else "p_adv",
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

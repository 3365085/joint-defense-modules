from __future__ import annotations

from collections import deque
import os
from typing import Any

import torch

from .alert_state import AlertState
from .artifacts import _resolve_artifact_path
from .features.a3.static_media import GPUStaticMediaSpoofDetector
from .features.blur_degradation import GPUBlurDegradationDetector
from .features.lbp_texture import GPULBPTextureAnalyzer
from .features.light_flow import GPULightOpticalFlowDetector
from .features.motion_artifact import GPUMotionArtifactDetector
from .features.overexposure import GPUOverexposureDetector
from .features.temporal_texture import GPUTemporalTextureAnalyzer
from .features.track_consistency import TrackConsistencyAnalyzer
from .fusion.classifier_fusion import TorchLogisticFusion
from .fusion.rule_fusion import GPURuleFusion
from .fusion.target_anchored import TargetAnchoredAnalyzer
from .scheduler import ModuleAScheduler


def initialize_detector(detector: Any, config: dict[str, Any] | None = None) -> None:
    config = config or {}
    module_config = config.get("module_a", config)
    runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    realtime_profile = str(runtime_config.get("profile") or "").lower()
    default_multiscale_fallback = realtime_profile not in {"desktop_rtx", "edge_fast", "low_power"}
    detector.require_gpu = bool(module_config.get("require_gpu", True))
    detector.profile_cuda_sync = bool(module_config.get("profile_cuda_sync", False))
    strict_gpu = os.environ.get("MODULE_A_STRICT_GPU", "0") == "1"
    explicit_device = module_config.get("device")
    if explicit_device:
        detector.device = str(explicit_device)
    elif torch.cuda.is_available():
        detector.device = "cuda:0"
    else:
        detector.device = "cpu"
    if detector.device.startswith("cuda") and not torch.cuda.is_available():
        if strict_gpu:
            raise RuntimeError(f"Configured CUDA device {detector.device} but CUDA is not available")
        detector.device = "cpu"
    if detector.require_gpu and not detector.device.startswith("cuda"):
        if strict_gpu:
            raise RuntimeError(f"Module A GPU mode requires a CUDA device, got {detector.device}")
        detector.require_gpu = False

    detector.frame_size = int(module_config.get("frame_size", 640))
    detector.grid_roi_count = int(module_config.get("grid_roi_count", 4))
    detector.use_grid_when_no_roi = bool(module_config.get("use_grid_when_no_roi", True))
    detector.emit_roi_details = bool(module_config.get("emit_roi_details", False))
    detector.emit_roi_texture_details = bool(module_config.get("emit_roi_texture_details", False))
    detector.emit_roi_temporal_details = bool(
        module_config.get("emit_roi_temporal_details", detector.emit_roi_details)
    )
    detector.emit_roi_motion_details = bool(module_config.get("emit_roi_motion_details", False))
    detector.roi_temporal_burst_window = max(
        1, int(module_config.get("roi_temporal_burst_window", 5))
    )
    detector.roi_temporal_burst_trigger_count = max(
        1,
        min(
            int(module_config.get("roi_temporal_burst_trigger_count", 2)),
            detector.roi_temporal_burst_window,
        ),
    )
    detector.roi_temporal_history: deque[int] = deque(maxlen=detector.roi_temporal_burst_window)
    detector.benign_global_motion_filter_enabled = bool(
        module_config.get("benign_global_motion_filter_enabled", True)
    )
    detector.benign_global_motion_region_count_min = int(
        module_config.get("benign_global_motion_region_count_min", 500)
    )
    detector.benign_global_motion_overexposure_max = float(
        module_config.get("benign_global_motion_overexposure_max", 0.02)
    )
    detector.light_flow_enabled = bool(module_config.get("light_flow_enabled", True))
    detector.light_flow_interval = max(1, int(module_config.get("light_flow_interval", 3)))
    detector.light_flow_temporal_candidate = float(
        module_config.get("light_flow_temporal_candidate", 0.025)
    )
    detector.light_flow_local_temporal_candidate = float(
        module_config.get("light_flow_local_temporal_candidate", 0.18)
    )
    detector.static_image_enabled = bool(module_config.get("static_image_enabled", True))
    detector.static_image_interval = max(
        1,
        int(
            module_config.get(
                "static_image_interval", 3
            )
        ),
    )
    detector.static_image_temporal_candidate = float(
        module_config.get("static_image_temporal_candidate", 0.015)
    )
    detector.static_image_local_temporal_candidate = float(
        module_config.get("static_image_local_temporal_candidate", 0.10)
    )
    detector.static_image_hold_frames = max(
        0, int(module_config.get("static_image_hold_frames", 5))
    )
    detector.static_image_hold_remaining = 0
    detector.static_image_hold_score = 0.0
    # Full-field carryover for non-execution frames (2026-06-11 架构修复)
    # Holds p_media, p_media_scores, candidate_count, warp_residual, flow_gap
    # so A3BSoftTrigger.quality_gate survives interval skips.
    detector.static_image_hold_carryover_enabled = bool(
        module_config.get("static_image_hold_carryover_enabled", True)
    )
    detector.static_image_hold_state: dict[str, Any] = {}
    # Dynamic interval: when hold_score > high_score_threshold, force interval=1
    detector.static_image_dynamic_interval_enabled = bool(
        module_config.get("static_image_dynamic_interval_enabled", True)
    )
    detector.static_image_high_score_threshold = float(
        module_config.get("static_image_high_score_threshold", 0.50)
    )
    # Max consecutive temporal reuse frames before forcing re-detection
    detector.temporal_reuse_max_consecutive = max(
        1, int(module_config.get("temporal_reuse_max_consecutive", 3))
    )
    detector.a3b_display_alpha = float(module_config.get("a3b_display_alpha", 0.35))
    detector.a3b_display_score = 0.0
    detector._a3b_display_hold = deque(maxlen=5)
    detector.p_adv_display_alpha = float(module_config.get("p_adv_display_alpha", 0.35))
    detector.p_adv_display_score = 0.0
    detector._p_adv_display_hold = deque(maxlen=5)
    detector.strong_temporal_trigger = float(module_config.get("strong_temporal_trigger", 0.10))
    detector.strong_local_temporal_trigger = float(
        module_config.get("strong_local_temporal_trigger", 0.50)
    )
    detector.strong_evidence_hold_frames = max(
        0, int(module_config.get("strong_evidence_hold_frames", 0))
    )
    detector.strong_evidence_min_p_adv = float(
        module_config.get("strong_evidence_min_p_adv", 0.40)
    )
    detector.strong_evidence_min_track_score = float(
        module_config.get("strong_evidence_min_track_score", 0.75)
    )
    detector.strong_evidence_min_conf_drop = float(
        module_config.get("strong_evidence_min_conf_drop", 0.35)
    )
    detector.strong_evidence_hold_remaining = 0
    detector.strong_evidence_hold_score = 0.0
    detector.strong_evidence_hold_reason = "none"
    detector.blur_hold_frames = max(0, int(module_config.get("blur_hold_frames", 0)))
    detector.blur_hold_temporal_threshold = float(
        module_config.get("blur_hold_temporal_threshold", 0.33)
    )
    detector.blur_hold_score_threshold = float(
        module_config.get("blur_hold_score_threshold", 0.30)
    )
    detector.blur_hold_remaining = 0
    detector.track_context_temporal_threshold = float(
        module_config.get("track_context_temporal_threshold", 0.28)
    )
    detector.static_media_replay_window = max(
        1, int(module_config.get("static_media_replay_window", 30))
    )
    detector.static_media_replay_min_p_media = float(
        module_config.get("static_media_replay_min_p_media", 0.58)
    )
    detector.static_media_replay_max_bbox_area = float(
        module_config.get("static_media_replay_max_bbox_area", 12000.0)
    )
    detector.static_media_free_candidate_max_bbox_area = float(
        module_config.get("static_media_free_candidate_max_bbox_area", 2500.0)
    )
    detector.static_media_replay_min_temporal = float(
        module_config.get("static_media_replay_min_temporal", 0.03)
    )
    detector.static_media_replay_min_blur = float(
        module_config.get("static_media_replay_min_blur", 0.45)
    )
    detector.static_media_replay_min_warp_residual = float(
        module_config.get("static_media_replay_min_warp_residual", 0.18)
    )
    detector.static_media_replay_min_flow_gap = float(
        module_config.get("static_media_replay_min_flow_gap", 0.35)
    )
    detector.static_media_replay_max_center_span = float(
        module_config.get("static_media_replay_max_center_span", 80.0)
    )
    detector.static_media_replay_max_area_ratio = float(
        module_config.get("static_media_replay_max_area_ratio", 3.0)
    )
    detector.static_media_replay_votes: deque[int] = deque(
        maxlen=detector.static_media_replay_window
    )
    detector.static_media_replay_bboxes: deque[tuple[float, float, float] | None] = deque(
        maxlen=detector.static_media_replay_window
    )
    detector.static_media_fast_window = max(
        1, int(module_config.get("static_media_fast_window", 6))
    )
    detector.static_media_fast_trigger_count = max(
        1, int(module_config.get("static_media_fast_trigger_count", 1))
    )
    detector.static_media_fast_min_p_media = float(
        module_config.get("static_media_fast_min_p_media", 0.62)
    )
    detector.static_media_fast_min_replay_signal = float(
        module_config.get("static_media_fast_min_replay_signal", 0.30)
    )
    detector.static_media_fast_alt_min_p_media = float(
        module_config.get("static_media_fast_alt_min_p_media", 0.60)
    )
    detector.static_media_fast_alt_min_warp_residual = float(
        module_config.get("static_media_fast_alt_min_warp_residual", 0.18)
    )
    detector.static_media_fast_alt_min_flow_gap = float(
        module_config.get("static_media_fast_alt_min_flow_gap", 0.80)
    )
    detector.static_media_fast_edge_min_replay_signal = float(
        module_config.get("static_media_fast_edge_min_replay_signal", 0.18)
    )
    detector.static_media_fast_min_edge_score = float(
        module_config.get("static_media_fast_min_edge_score", 0.18)
    )
    detector.static_media_fast_min_yolo_context = float(
        module_config.get("static_media_fast_min_yolo_context", 0.08)
    )
    detector.static_media_replay_min_support_flow_gap = float(
        module_config.get("static_media_replay_min_support_flow_gap", 0.35)
    )
    detector.static_media_legacy_direct_alert_enabled = bool(
        module_config.get("static_media_legacy_direct_alert_enabled", False)
    )
    detector.static_media_camera_motion_suppress_enabled = bool(
        module_config.get("static_media_camera_motion_suppress_enabled", True)
    )
    detector.static_media_camera_motion_min_valid_ratio = float(
        module_config.get("static_media_camera_motion_min_valid_ratio", 0.22)
    )
    detector.static_media_camera_motion_min_p_media = float(
        module_config.get("static_media_camera_motion_min_p_media", 0.55)
    )
    detector.static_media_camera_motion_max_yolo_context = float(
        module_config.get("static_media_camera_motion_max_yolo_context", 0.08)
    )
    detector.static_media_camera_motion_score_cap = float(
        module_config.get("static_media_camera_motion_score_cap", 0.40)
    )
    detector.static_media_camera_motion_max_flow_ratio = float(
        module_config.get("static_media_camera_motion_max_flow_ratio", 0.42)
    )
    detector.static_media_exposure_motion_suppress_enabled = bool(
        module_config.get("static_media_exposure_motion_suppress_enabled", True)
    )
    detector.static_media_exposure_motion_min_ratio = float(
        module_config.get("static_media_exposure_motion_min_ratio", 0.001)
    )
    detector.static_media_exposure_motion_min_temporal = float(
        module_config.get("static_media_exposure_motion_min_temporal", 0.20)
    )
    detector.static_media_fast_edge_margin = float(
        module_config.get("static_media_fast_edge_margin", 3.0)
    )
    detector.static_media_border_suppress_enabled = bool(
        module_config.get("static_media_border_suppress_enabled", True)
    )
    detector.static_media_border_margin = float(
        module_config.get("static_media_border_margin", 8.0)
    )
    detector.static_media_border_max_area_ratio = float(
        module_config.get("static_media_border_max_area_ratio", 0.08)
    )
    detector.static_media_fast_votes: deque[int] = deque(
        maxlen=detector.static_media_fast_window
    )
    detector.static_media_fast_bboxes: deque[tuple[float, float, float] | None] = deque(
        maxlen=detector.static_media_fast_window
    )
    detector.static_media_occlusion_hold_frames = max(
        0, int(module_config.get("static_media_occlusion_hold_frames", 1500))
    )
    detector.static_media_occlusion_min_score = float(
        module_config.get("static_media_occlusion_min_score", 0.68)
    )
    detector.static_media_occlusion_reacquire_min_p_media = float(
        module_config.get("static_media_occlusion_reacquire_min_p_media", 0.55)
    )
    detector.physical_media_motion_min_p_adv = float(
        module_config.get("physical_media_motion_min_p_adv", 0.62)
    )
    detector.physical_media_motion_score_cap = float(
        module_config.get("physical_media_motion_score_cap", 0.38)
    )
    detector.static_media_occlusion_hold_remaining = 0
    detector.static_media_occlusion_hold_score = 0.0
    detector.static_media_occlusion_last_reason = "none"
    detector.static_media_display_alpha = float(module_config.get("static_media_display_alpha", 0.35))
    detector.a3b_display_score = 0.0
    detector.p_adv_display_score = 0.0
    detector.static_media_display_score = 0.0
    detector._p_adv_display_hold = deque(maxlen=5)
    detector._a3b_display_hold = deque(maxlen=5)
    detector.legacy_source_auth_disabled = False

    detector.fusion_backend = str(module_config.get("fusion_backend", "rule")).lower()
    if detector.fusion_backend not in {"rule", "classifier", "rule_or_classifier"}:
        raise ValueError(f"Unsupported Module A fusion backend: {detector.fusion_backend}")

    detector.scheduler = ModuleAScheduler(module_config.get("keyframe_interval", 3))
    detector.alert_state = AlertState(
        window=module_config.get("alert_window", 7),
        trigger_count=module_config.get("alert_trigger_count", 3),
        hold_frames=module_config.get("attack_state_hold_frames", 4),
    )
    detector.glare_ratio_threshold = float(module_config.get("glare_ratio_threshold", 0.06))
    detector.a3b_glare_suppress_frames = max(
        0, int(module_config.get("a3b_glare_suppress_frames", 30))
    )
    detector.a3b_glare_suppress_remaining = 0
    detector.a3b_physical_suppress_frames = max(
        0, int(module_config.get("a3b_physical_suppress_frames", 180))
    )
    detector.a3b_physical_suppress_remaining = 0
    detector.glare_hold_frames = max(0, int(module_config.get("glare_hold_frames", 0)))
    detector.glare_hold_remaining = 0
    detector.overexposure = GPUOverexposureDetector(
        threshold=detector.glare_ratio_threshold,
        flash_diff_threshold=float(module_config.get("glare_flash_diff_threshold", 30.0)),
        flash_ratio_threshold=float(module_config.get("glare_flash_ratio_threshold", 0.08)),
        flash_min_polarity=float(module_config.get("glare_flash_min_polarity", 0.35)),
        flash_min_abs_mean=float(module_config.get("glare_flash_min_abs_mean", 8.0)),
    )
    # A3b thresholds (tunable via tuning tool)
    detector.a3b_high_score_bypass_threshold = float(
        module_config.get("a3b_high_score_bypass_threshold", 0.70)
    )
    # A4 flow_local anomaly threshold
    detector.flow_local_anomaly_threshold = float(
        module_config.get("flow_local_anomaly_threshold", 0.68)
    )
    detector.texture = GPULBPTextureAnalyzer(
        radius=module_config.get("lbp_radius", 3),
        grid_size=module_config.get("texture_grid_size", 16),
        emit_roi_details=detector.emit_roi_texture_details,
    )
    detector.temporal = GPUTemporalTextureAnalyzer(
        threshold=module_config.get("lbp_temporal_change_threshold", 0.25),
        grid_size=module_config.get("texture_grid_size", 16),
        emit_roi_details=detector.emit_roi_temporal_details,
        persistence_frames=int(module_config.get("temporal_persistence_frames", 1)),
        adaptive_baseline=bool(module_config.get("temporal_adaptive_baseline", True)),
        adaptive_ema_alpha=float(module_config.get("temporal_adaptive_ema_alpha", 0.02)),
        adaptive_multiplier=float(module_config.get("temporal_adaptive_multiplier", 2.0)),
        adaptive_floor=float(module_config.get("temporal_adaptive_floor", 0.015)),
    )
    detector.motion = GPUMotionArtifactDetector(
        diff_threshold=module_config.get("motion_diff_threshold", 25.0),
        grid_size=module_config.get("motion_grid_size", 16),
        emit_roi_details=detector.emit_roi_motion_details,
    )
    detector.blur = GPUBlurDegradationDetector(
        roi_energy_ratio_trigger=module_config.get("blur_roi_energy_ratio_trigger", 0.55),
        low_energy_ratio_trigger=module_config.get("blur_low_energy_ratio_trigger", 0.55),
        low_energy_global_factor=module_config.get("blur_low_energy_global_factor", 0.35),
        min_roi_area=module_config.get("blur_min_roi_area", 900),
        emit_roi_details=detector.emit_roi_motion_details,
    )
    detector.track = TrackConsistencyAnalyzer(
        labels=tuple(module_config.get("track_labels", ("person", "helmet", "head"))),
        iou_threshold=module_config.get("track_iou_threshold", 0.12),
        center_distance_ratio=module_config.get("track_center_distance_ratio", 0.55),
        high_confidence=module_config.get("track_high_confidence", 0.45),
        confidence_drop_trigger=module_config.get("track_confidence_drop_trigger", 0.25),
        max_missing=module_config.get("track_max_missing", 4),
        score_normalizer=module_config.get("track_score_normalizer", 2.0),
        max_candidates_per_label=module_config.get("track_max_candidates_per_label", 4),
        max_tracks_per_label=module_config.get("track_max_tracks_per_label", 8),
    )
    detector.light_flow = GPULightOpticalFlowDetector(
        flow_size=module_config.get("light_flow_size", 160),
        window_size=module_config.get("light_flow_window_size", 9),
        grid_size=module_config.get(
            "light_flow_grid_size", module_config.get("motion_grid_size", 16)
        ),
        residual_threshold=module_config.get("light_flow_residual_threshold", 0.75),
        min_magnitude=module_config.get("light_flow_min_magnitude", 0.25),
        cell_ratio_threshold=module_config.get("light_flow_cell_ratio_threshold", 0.20),
        score_region_normalizer=module_config.get("light_flow_score_region_normalizer", 8.0),
        emit_roi_details=detector.emit_roi_motion_details,
    )
    detector.static_image = GPUStaticMediaSpoofDetector(
        target_labels=tuple(module_config.get("static_image_target_labels", ("person",))),
        screen_labels=tuple(
            module_config.get("static_image_screen_labels", ("helmet", "head"))
        ),
        patch_size=module_config.get("static_image_patch_size", 64),
        min_similarity=module_config.get("static_image_min_similarity", 0.94),
        trigger_stable_count=module_config.get("static_image_trigger_stable_count", 2),
        min_edge_mean=module_config.get("static_image_min_edge_mean", 0.038),
        screen_min_edge_mean=module_config.get("static_image_screen_min_edge_mean", 0.018),
        min_center_motion=module_config.get("static_image_min_center_motion", 0.0012),
        context_motion_threshold=module_config.get(
            "static_image_context_motion_threshold", 0.010
        ),
        context_contrast_threshold=module_config.get(
            "static_image_context_contrast_threshold", 1.6
        ),
        min_roi_area=module_config.get("static_image_min_roi_area", 1200),
        screen_min_roi_area=module_config.get("static_image_screen_min_roi_area", 450),
        screen_max_roi_area=module_config.get("static_image_screen_max_roi_area", 8000),
        screen_context_expand_ratio=module_config.get(
            "static_image_screen_context_expand_ratio", 2.4
        ),
        screen_min_context_edge_mean=module_config.get(
            "static_image_screen_min_context_edge_mean", 0.004
        ),
        screen_min_context_std=module_config.get("static_image_screen_min_context_std", 0.22),
        screen_min_line_score=module_config.get("static_image_screen_min_line_score", 0.10),
        screen_max_roi_context_area_ratio=module_config.get(
            "static_image_screen_max_roi_context_area_ratio", 0.42
        ),
        screen_person_containment_threshold=module_config.get(
            "static_image_screen_person_containment_threshold", 0.72
        ),
        min_roi_confidence=module_config.get("static_image_min_roi_confidence", 0.50),
        score_trigger=module_config.get("static_image_score_trigger", 0.68),
        expand_ratio=module_config.get("static_image_expand_ratio", 0.35),
        edge_margin_px=module_config.get("static_image_edge_margin_px", 6),
        min_same_label_count=module_config.get("static_image_min_same_label_count", 2),
        max_person_area_ratio=module_config.get("static_image_max_person_area_ratio", 0.65),
        max_context_iou=module_config.get("static_image_max_context_iou", 0.20),
        max_tracks=module_config.get("static_image_max_tracks", 64),
        emit_roi_details=detector.emit_roi_motion_details,
        multiscale_fallback_enabled=module_config.get(
            "static_image_multiscale_fallback_enabled", default_multiscale_fallback
        ),
        multiscale_trigger_count=module_config.get(
            "static_image_multiscale_trigger_count", 1
        ),
        backend=module_config.get("static_image_backend", "legacy"),
    )
    detector.fusion = GPURuleFusion(
        device=detector.device,
        weights=module_config.get("module_a_fusion_weights", (0.20, 0.30, 0.20, 0.10, 0.20)),
        threshold=module_config.get("p_adv_threshold", 0.55),
        temporal_trigger=module_config.get("temporal_trigger", 0.03),
        local_temporal_trigger=module_config.get("local_temporal_trigger", 0.045),
        local_flow_ratio_trigger=module_config.get("local_flow_ratio_trigger", 0.42),
        strong_temporal_trigger=module_config.get("strong_temporal_trigger", 0.10),
        strong_local_temporal_trigger=module_config.get("strong_local_temporal_trigger", 0.50),
        paired_local_temporal_trigger=module_config.get("paired_local_temporal_trigger", 0.50),
        paired_local_flow_trigger=module_config.get("paired_local_flow_trigger", 0.45),
        light_flow_anomaly_trigger=module_config.get("light_flow_anomaly_trigger", 0.22),
        light_flow_score_trigger=module_config.get("light_flow_score_trigger", 0.35),
        paired_light_flow_temporal_trigger=module_config.get(
            "paired_light_flow_temporal_trigger", 0.35
        ),
        blur_score_trigger=module_config.get("blur_score_trigger", 0.45),
        paired_blur_temporal_trigger=module_config.get("paired_blur_temporal_trigger", 0.18),
        track_score_trigger=module_config.get("track_score_trigger", 0.50),
        paired_track_temporal_trigger=module_config.get("paired_track_temporal_trigger", 0.18),
        paired_track_blur_trigger=module_config.get("paired_track_blur_trigger", 0.25),
        static_image_score_trigger=module_config.get("static_image_score_trigger", 0.68),
    )
    detector.classifier_fusion: TorchLogisticFusion | None = None
    classifier_artifact = module_config.get("classifier_artifact")
    if detector.fusion_backend in {"classifier", "rule_or_classifier"}:
        if not classifier_artifact:
            raise ValueError(
                "classifier_artifact is required when fusion_backend uses classifier"
            )
        artifact_path = _resolve_artifact_path(str(classifier_artifact))
        detector.classifier_fusion = TorchLogisticFusion(
            artifact_path,
            detector.device,
            calibration_model=module_config.get("classifier_calibration_model"),
            threshold_override=module_config.get("classifier_threshold_override"),
        )

    # --- Target-anchored analyzer (2026-05-13) ---
    detector.target_anchored = TargetAnchoredAnalyzer(
        target_labels=tuple(module_config.get("roi_target_labels", ("person", "helmet", "head"))),
        roi_blur_threshold=float(module_config.get("target_anchored_blur_threshold", 0.40)),
        roi_overexposure_threshold=float(
            module_config.get("target_anchored_overexposure_threshold", 0.15)
        ),
        roi_confidence_drop_threshold=float(
            module_config.get("target_anchored_confidence_drop_threshold", 0.25)
        ),
        roi_texture_anomaly_threshold=float(
            module_config.get("target_anchored_texture_anomaly_threshold", 0.12)
        ),
        track_drop_threshold=float(
            module_config.get("target_anchored_track_drop_threshold", 0.40)
        ),
        track_confidence_drop_threshold=float(
            module_config.get("target_anchored_track_confidence_drop_threshold", 0.20)
        ),
        motion_score_threshold=float(
            module_config.get("target_anchored_motion_score_threshold", 0.35)
        ),
        light_flow_score_threshold=float(
            module_config.get("target_anchored_light_flow_score_threshold", 0.45)
        ),
        paired_temporal_motion_threshold=float(
            module_config.get("target_anchored_paired_temporal_motion_threshold", 0.18)
        ),
        global_fallback_overexposure_threshold=float(
            module_config.get("target_anchored_global_fallback_overexposure_threshold", 0.20)
        ),
        natural_exposure_suppression=bool(
            module_config.get("natural_exposure_suppression_enabled", True)
        ),
        natural_exposure_max_ratio=float(
            module_config.get("natural_exposure_max_ratio", 0.18)
        ),
        natural_exposure_max_light_flow=float(
            module_config.get("natural_exposure_max_light_flow", 0.35)
        ),
        natural_exposure_max_motion_score=float(
            module_config.get("natural_exposure_max_motion_score", 1.01)
        ),
        no_target_fallback_window_frames=int(
            module_config.get("no_target_fallback_window_frames", 45)
        ),
        no_target_fallback_max_exposure_ratio=float(
            module_config.get("no_target_fallback_max_exposure_ratio", 0.0005)
        ),
        no_target_blur_score_threshold=float(
            module_config.get("no_target_blur_score_threshold", 0.15)
        ),
        no_target_blur_low_energy_threshold=float(
            module_config.get("no_target_blur_low_energy_threshold", 0.64)
        ),
        no_target_occlusion_local_ratio_threshold=float(
            module_config.get("no_target_occlusion_local_ratio_threshold", 0.55)
        ),
        strong_static_glare_ratio_threshold=float(
            module_config.get("strong_static_glare_ratio_threshold", 0.10)
        ),
        flow_local_anomaly_threshold=float(
            module_config.get("flow_local_anomaly_threshold", 0.68)
        ),
    )

    # --- Static_Media_Classifier (A3b sub-feature, Task 5.4 / Req 7.3+7.5) ---
    # Independent of the main A4 ``classifier_fusion``: this artifact scores
    # only the per-frame ``static_media`` feature block and feeds its output
    # back via ``classifier_score`` / ``classifier_triggered`` under
    # ``module_a_features.static_media``. Not configured -> not constructed
    # -> current heuristic behaviour is preserved bit-for-bit.
    #
    # The ``static_media_classifier_enabled`` switch (default False) is the
    # rollout gate: training may land before real hand-held LAN footage is
    # available, in which case the artifact is loaded for shadow scoring
    # but must NOT push ``static_image_triggered`` (Req 6.3 hard rule).
    detector.static_media_classifier_enabled = bool(
        module_config.get("static_media_classifier_enabled", False)
    )
    detector.static_media_classifier: TorchLogisticFusion | None = None
    static_media_classifier_artifact = module_config.get("static_media_classifier_artifact")
    if static_media_classifier_artifact:
        sm_artifact_path = _resolve_artifact_path(str(static_media_classifier_artifact))
        detector.static_media_classifier = TorchLogisticFusion(
            sm_artifact_path,
            detector.device,
            calibration_model=module_config.get("static_media_classifier_calibration_model"),
            threshold_override=module_config.get(
                "static_media_classifier_threshold_override"
            ),
        )

    detector.prev_gray: torch.Tensor | None = None
    detector.prev_lbp: torch.Tensor | None = None
    detector.frame_idx = 0
    detector._last_light_flow_score: float = 0.0
    detector._last_light_flow_ratio: float = 0.0


def reset_detector_state(detector: Any) -> None:
    detector.prev_gray = None
    detector.prev_lbp = None
    detector.frame_idx = 0
    detector._last_light_flow_score = 0.0
    detector._last_light_flow_ratio = 0.0
    detector.roi_temporal_history.clear()
    detector.alert_state.reset()
    detector.track.reset()
    detector.static_image.reset()
    detector.temporal.reset()
    detector.target_anchored.reset()
    detector.overexposure.reset()
    detector.static_image_hold_remaining = 0
    detector.static_image_hold_score = 0.0
    detector.static_image_hold_state.clear()
    detector.static_media_replay_votes.clear()
    detector.static_media_replay_bboxes.clear()
    detector.static_media_fast_votes.clear()
    detector.static_media_fast_bboxes.clear()
    detector.static_media_occlusion_hold_remaining = 0
    detector.static_media_occlusion_hold_score = 0.0
    detector.static_media_occlusion_last_reason = "none"
    detector.a3b_display_score = 0.0
    detector.p_adv_display_score = 0.0
    detector.strong_evidence_hold_remaining = 0
    detector.strong_evidence_hold_score = 0.0
    detector.strong_evidence_hold_reason = "none"
    detector.blur_hold_remaining = 0
    detector.static_media_display_score = 0.0
    detector._p_adv_display_hold.clear()
    detector._a3b_display_hold.clear()
    detector.a3b_glare_suppress_remaining = 0
    detector.a3b_physical_suppress_remaining = 0
    detector.glare_hold_remaining = 0

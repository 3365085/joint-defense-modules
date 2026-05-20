from __future__ import annotations

from typing import Any

from .types import ROI


def build_p_media_extras(motion: dict[str, Any]) -> dict[str, Any]:
    """A3+ PR6: Build p_media diagnostics for the extras field.

    Embeds p_media_scores into the details extras so Web display can
    show A3+ diagnostics without adding a new card. Only populates
    when p_media > 0.0 to avoid noise on normal frames.
    """
    extras: dict[str, Any] = {}
    p_media = float(motion.get("p_media", 0.0))
    if p_media > 0.0:
        extras["p_media"] = p_media
        extras["p_media_type"] = str(motion.get("p_media_type", "normal"))
        extras["p_media_scores"] = motion.get("p_media_scores", {})
    return extras

def build_details(
    detector,
    overexposure: dict[str, Any],
    texture: dict[str, Any],
    temporal: dict[str, Any],
    motion: dict[str, Any],
    blur: dict[str, Any],
    track: dict[str, Any],
    fusion: dict[str, Any],
    source_auth: dict[str, Any],
    rois: list[ROI],
    roi_results: list[dict[str, Any]],
    frame_idx: int,
) -> dict[str, Any]:
    feature_details = {
        "overexposure": {
            "ratio": float(overexposure["ratio"]),
            "underexposed_ratio": float(overexposure["underexposed_ratio"]),
            "is_glare": bool(overexposure["is_glare"]),
            "threshold": float(overexposure["threshold"]),
        },
        "lbp": {
            "anomaly_ratio": float(texture["delta_h"]),
            "local_max": float(texture["local_max"]),
            "global_mean": float(texture["global_mean"]),
            "global_std": float(texture["global_std"]),
            "backend": "gpu_lbp",
        },
        "temporal": {
            "change_rate": float(temporal["change_t"]),
            "local_max": float(temporal["local_max"]),
            "threshold": float(temporal["threshold"]),
            "backend": "gpu_lbp_temporal",
        },
        "flow": {
            "region_count": int(motion["region_count"]),
            "max_magnitude": float(motion["max_magnitude"]),
            "local_max_ratio": float(motion["local_max_ratio"]),
            "motion_score": float(motion["motion_score"]),
            "backend": motion["backend"],
            "light_flow_available": bool(motion.get("light_flow_available", False)),
            "light_flow_backend": str(motion.get("light_flow_backend", "disabled")),
            "light_flow_region_count": int(motion.get("light_flow_region_count", 0)),
            "light_flow_max_magnitude": float(motion.get("light_flow_max_magnitude", 0.0)),
            "light_flow_mean_magnitude": float(motion.get("light_flow_mean_magnitude", 0.0)),
            "light_flow_dominant_magnitude": float(
                motion.get("light_flow_dominant_magnitude", 0.0)
            ),
            "light_flow_max_residual": float(motion.get("light_flow_max_residual", 0.0)),
            "light_flow_local_anomaly_ratio": float(
                motion.get("light_flow_local_anomaly_ratio", 0.0)
            ),
            "light_flow_score": float(motion.get("light_flow_score", 0.0)),
            "light_flow_valid_ratio": float(motion.get("light_flow_valid_ratio", 0.0)),
        },
        "static_image": {
            "score": float(motion.get("static_image_score", 0.0)),
            "live_score": float(motion.get("static_image_live_score_raw", motion.get("static_image_score", 0.0))),
            "live_score_display": float(motion.get("static_image_live_score_display", motion.get("static_image_live_score", motion.get("static_image_score", 0.0)))),
            "triggered": bool(motion.get("static_image_triggered", False)),
            "trigger_count": int(motion.get("static_image_trigger_count", 0)),
            "patch_similarity": float(motion.get("static_image_patch_similarity", 0.0)),
            "stable_count": int(motion.get("static_image_stable_count", 0)),
            "center_motion": float(motion.get("static_image_center_motion", 0.0)),
            "roi_motion": float(motion.get("static_image_roi_motion", 0.0)),
            "context_motion": float(motion.get("static_image_context_motion", 0.0)),
            "motion_contrast": float(motion.get("static_image_motion_contrast", 0.0)),
            "edge_mean": float(motion.get("static_image_edge_mean", 0.0)),
            "screen_like": bool(motion.get("static_image_screen_like", False)),
            "screen_context_edge_mean": float(
                motion.get("static_image_screen_context_edge_mean", 0.0)
            ),
            "screen_context_std": float(motion.get("static_image_screen_context_std", 0.0)),
            "screen_line_score": float(motion.get("static_image_screen_line_score", 0.0)),
            "screen_roi_context_area_ratio": float(
                motion.get("static_image_screen_roi_context_area_ratio", 0.0)
            ),
            "screen_inside_person": bool(
                motion.get("static_image_screen_inside_person", False)
            ),
            "backend": str(motion.get("static_image_backend", "gpu_static_media_spoof")),
            "p_media": float(motion.get("p_media", 0.0)),
            "p_media_triggered": bool(motion.get("p_media_triggered", False)),
            "p_media_type": str(motion.get("p_media_type", "normal")),
            "p_media_bbox": motion.get("p_media_bbox"),
            "p_media_target_related": bool(
                motion.get("p_media_target_related", False)
            ),
            "p_media_strong_evidence": bool(
                motion.get("p_media_strong_evidence", False)
            ),
            "p_media_background_static_suppressed": bool(
                motion.get("p_media_background_static_suppressed", False)
            ),
            "p_media_scores": dict(motion.get("p_media_scores", {})),
            "p_media_replay_state": dict(motion.get("p_media_replay_state", {})),
            "p_media_fast_state": dict(motion.get("p_media_fast_state", {})),
            "p_media_occlusion_state": dict(motion.get("p_media_occlusion_state", {})),
            "p_media_border_state": dict(motion.get("p_media_border_state", {})),
            "p_media_physical_motion_state": dict(
                motion.get("p_media_physical_motion_state", {})
            ),
            "triggered_source": str(
                motion.get("static_image_triggered_source", "none")
            ),
            # Task 5.4 / Req 7.3 — expose Static_Media_Classifier shadow
            # scores on every frame so offline replay and event evidence
            # can reason about classifier vs. heuristic attribution even
            # when the rollout gate (``classifier_enabled``) is closed.
            "classifier_score": float(motion.get("static_image_classifier_score", 0.0)),
            "classifier_triggered": bool(
                motion.get("static_image_classifier_triggered", False)
            ),
            "classifier_threshold": float(motion.get("static_image_classifier_threshold", 0.0)),
            "classifier_artifact": str(motion.get("static_image_classifier_artifact", "")),
            "classifier_enabled": bool(detector.static_media_classifier_enabled),
            "classifier_forced_trigger": bool(
                motion.get("static_image_classifier_forced_trigger", False)
            ),
        },
        "blur": {
            "score": float(blur.get("blur_score", 0.0)),
            "roi_energy_ratio": float(blur.get("blur_roi_energy_ratio", 1.0)),
            "low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
            "global_mean": float(blur.get("blur_global_mean", 0.0)),
            "global_max": float(blur.get("blur_global_max", 0.0)),
            "best_roi_is_grid": bool(blur.get("blur_best_roi_is_grid", False)),
            "backend": blur.get("backend", "gpu_laplacian_blur"),
        },
        "track": {
            "score": float(track.get("track_score", 0.0)),
            "drop_score": float(track.get("track_drop_score", 0.0)),
            "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
            "missing_track_count": int(track.get("missing_track_count", 0)),
            "confidence_drop_count": int(track.get("confidence_drop_count", 0)),
            "matched_track_count": int(track.get("matched_track_count", 0)),
            "active_track_count": int(track.get("active_track_count", 0)),
            "candidate_roi_count": int(track.get("candidate_roi_count", 0)),
            "backend": track.get("backend", "roi_track_consistency"),
        },
        "fusion": {
            "p_adv": float(fusion["p_adv"]),
            "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
            "p_adv_raw": float(fusion["p_adv"]),
            "threshold": float(fusion["threshold"]),
            "backend": detector.fusion_backend,
            "reason": ",".join(fusion["reason_codes"]),
            "classifier_p_adv": float(fusion.get("classifier_p_adv", 0.0)),
            "classifier_threshold": float(fusion.get("classifier_threshold", 0.0)),
            "classifier_triggered": bool(fusion.get("classifier_triggered", False)),
            "classifier_kind": str(fusion.get("classifier_kind", "")),
            "classifier_transform_mode": str(fusion.get("classifier_transform_mode", "")),
            "classifier_calibration_model": str(fusion.get("classifier_calibration_model", "")),
            "overexposure_triggered": bool(fusion.get("overexposure_triggered", False)),
            "temporal_triggered": bool(fusion["temporal_triggered"]),
            "flow_triggered": bool(fusion["local_flow_triggered"]),
            "roi_temporal_triggered": bool(fusion.get("roi_temporal_triggered", False)),
            "sustained_roi_temporal_triggered": bool(
                fusion.get("sustained_roi_temporal_triggered", False)
            ),
            "benign_global_motion_suppressed": bool(
                fusion.get("benign_global_motion_suppressed", False)
            ),
            "texture_high_triggered": False,
            "texture_low_triggered": False,
            "texture_very_low_triggered": False,
            "local_texture_triggered": False,
            "local_temporal_triggered": bool(fusion["local_temporal_triggered"]),
            "local_flow_triggered": bool(fusion["local_flow_triggered"]),
            "p_adv_triggered": bool(fusion["p_adv_triggered"]),
            "strong_temporal_triggered": bool(fusion["strong_temporal_triggered"]),
            "paired_temporal_flow_triggered": bool(fusion["paired_temporal_flow_triggered"]),
            "light_flow_triggered": bool(fusion.get("light_flow_triggered", False)),
            "paired_temporal_light_flow_triggered": bool(
                fusion.get("paired_temporal_light_flow_triggered", False)
            ),
            "blur_triggered": bool(fusion.get("blur_triggered", False)),
            "paired_temporal_blur_triggered": bool(
                fusion.get("paired_temporal_blur_triggered", False)
            ),
            "track_triggered": bool(fusion.get("track_triggered", False)),
            "paired_track_triggered": bool(fusion.get("paired_track_triggered", False)),
            "static_image_triggered": bool(fusion.get("static_image_triggered", False)),
            "static_image_hold_active": bool(fusion.get("static_image_hold_active", False)),
            "strong_flow_triggered": False,
            "fft_triggered": False,
        },
        "scheduler": {
            "frame_idx": int(frame_idx),
            "is_keyframe": detector.scheduler.is_keyframe(frame_idx),
            "slow_path_requested": detector.scheduler.should_run_slow_path(
                frame_idx,
                bool(fusion["is_suspicious"]),
            ),
        },
        "rois": [roi.to_dict() for roi in rois],
        "roi_results": roi_results,
        "emit_roi_details": detector.emit_roi_details,
        "roi_detail_modes": {
            "texture": detector.emit_roi_texture_details,
            "temporal": detector.emit_roi_temporal_details,
            "motion": detector.emit_roi_motion_details,
            "static_image": detector.emit_roi_motion_details,
        },
        "gpu": {
            "device": str(detector.device),
            "required": bool(detector.require_gpu),
        },
    }
    static_image_details = feature_details.get("static_image", {})
    feature_details["static_media"] = {
        "score": float(static_image_details.get("score", 0.0)),
        "live_score": float(
            max(
                static_image_details.get("live_score", 0.0),
                static_image_details.get("score", 0.0),
                static_image_details.get("p_media", 0.0),
                static_image_details.get("classifier_score", 0.0),
            )
        ),
        "triggered": bool(static_image_details.get("triggered", False)),
        "trigger_count": int(static_image_details.get("trigger_count", 0)),
        "media_type": "screen_or_paper"
        if static_image_details.get("screen_like", False)
        else "flat_media_candidate",
        "inner_target_locked_to_plane": bool(static_image_details.get("triggered", False)),
        "screen_or_paper_like": bool(static_image_details.get("screen_like", False)),
        "patch_similarity": float(static_image_details.get("patch_similarity", 0.0)),
        "stable_count": int(static_image_details.get("stable_count", 0)),
        "center_motion": float(static_image_details.get("center_motion", 0.0)),
        "context_motion": float(static_image_details.get("context_motion", 0.0)),
        "line_score": float(static_image_details.get("screen_line_score", 0.0)),
        "context_std": float(static_image_details.get("screen_context_std", 0.0)),
        "legacy_static_image": static_image_details,
        "backend": str(static_image_details.get("backend", "gpu_static_media_spoof")),
        "p_media": float(static_image_details.get("p_media", 0.0)),
        "p_media_triggered": bool(static_image_details.get("p_media_triggered", False)),
        "p_media_type": str(static_image_details.get("p_media_type", "normal")),
        "p_media_bbox": static_image_details.get("p_media_bbox"),
        "p_media_target_related": bool(
            static_image_details.get("p_media_target_related", False)
        ),
        "p_media_strong_evidence": bool(
            static_image_details.get("p_media_strong_evidence", False)
        ),
        "p_media_background_static_suppressed": bool(
            static_image_details.get("p_media_background_static_suppressed", False)
        ),
        "p_media_scores": dict(static_image_details.get("p_media_scores", {})),
        "p_media_replay_state": dict(
            static_image_details.get("p_media_replay_state", {})
        ),
        "p_media_fast_state": dict(
            static_image_details.get("p_media_fast_state", {})
        ),
        "p_media_occlusion_state": dict(
            static_image_details.get("p_media_occlusion_state", {})
        ),
        "p_media_border_state": dict(
            static_image_details.get("p_media_border_state", {})
        ),
        "p_media_physical_motion_state": dict(
            static_image_details.get("p_media_physical_motion_state", {})
        ),
        "triggered_source": str(static_image_details.get("triggered_source", "none")),
        # Task 5.4 / Req 7.3 — mirror the classifier fields from
        # ``static_image`` so ``module_a_features.static_media`` is the
        # single source of truth for Static_Media_Classifier attribution.
        "classifier_score": float(static_image_details.get("classifier_score", 0.0)),
        "classifier_triggered": bool(static_image_details.get("classifier_triggered", False)),
        "classifier_threshold": float(static_image_details.get("classifier_threshold", 0.0)),
        "classifier_artifact": str(static_image_details.get("classifier_artifact", "")),
        "classifier_enabled": bool(static_image_details.get("classifier_enabled", False)),
        "classifier_forced_trigger": bool(
            static_image_details.get("classifier_forced_trigger", False)
        ),
        "static_image_triggered": bool(static_image_details.get("triggered", False)),
    }
    flow_details = feature_details.get("flow", {})
    feature_details["a3"] = {
        "motion_artifact": {
            "score": float(flow_details.get("motion_score", 0.0)),
            "region_count": int(flow_details.get("region_count", 0)),
            "local_flow_ratio": float(flow_details.get("local_max_ratio", 0.0)),
            "light_flow_available": bool(flow_details.get("light_flow_available", False)),
            "light_flow_score": float(flow_details.get("light_flow_score", 0.0)),
            "light_flow_local_anomaly_ratio": float(
                flow_details.get("light_flow_local_anomaly_ratio", 0.0)
            ),
        },
        "static_media": feature_details["static_media"],
        "physical_channel": {
            "p_adv": float(fusion["p_adv"]),
            "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
            "p_adv_raw": float(fusion.get("p_adv_raw", fusion["p_adv"])),
            "suspicious": bool(fusion["is_suspicious"]),
            "reason_codes": list(fusion["reason_codes"]),
        },
    }
    return {
        "module_a_features": feature_details,
        "module_a": {
            "p_adv": float(fusion["p_adv"]),
            "p_adv_display": float(fusion.get("p_adv_display", fusion["p_adv"])),
            "p_adv_raw": float(fusion.get("p_adv_raw", fusion["p_adv"])),
            "reason_codes": list(fusion["reason_codes"]),
            "single_frame_suspicious": bool(fusion["is_suspicious"]),
        },
        # A3+ PR6: embed p_media diagnostics in extras for Web display
        "extras": build_p_media_extras(motion),
    }

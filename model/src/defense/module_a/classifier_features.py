from __future__ import annotations

from typing import Any


def build_classifier_features(
    overexposure: dict[str, Any],
    texture: dict[str, Any],
    temporal: dict[str, Any],
    motion: dict[str, Any],
    blur: dict[str, Any],
    track: dict[str, Any],
    fusion: dict[str, Any],
    roi_count: int,
) -> dict[str, float]:
    flow_region_count = float(motion.get("region_count", 0.0))
    flow_max = float(motion.get("max_magnitude", 0.0))
    return {
        "overexposure_ratio": float(overexposure.get("ratio", 0.0)),
        "underexposed_ratio": float(overexposure.get("underexposed_ratio", 0.0)),
        "texture_score": float(texture.get("delta_h", 0.0)),
        "local_texture_score": float(texture.get("local_max", 0.0)),
        "texture_global_mean": float(texture.get("global_mean", 0.0)),
        "texture_global_std": float(texture.get("global_std", 0.0)),
        "temporal_change": float(temporal.get("change_t", 0.0)),
        "local_temporal_change": float(temporal.get("local_max", 0.0)),
        "flow_region_ratio": min(1.0, flow_region_count / 6400.0),
        "flow_max_magnitude_norm": min(1.0, flow_max / 255.0),
        "local_flow_ratio": float(motion.get("local_max_ratio", 0.0)),
        "motion_score": float(motion.get("motion_score", 0.0)),
        "light_flow_score": float(motion.get("light_flow_score", 0.0)),
        "light_flow_local_anomaly_ratio": float(
            motion.get("light_flow_local_anomaly_ratio", 0.0)
        ),
        "light_flow_max_residual_norm": min(
            1.0, float(motion.get("light_flow_max_residual", 0.0)) / 16.0
        ),
        "static_image_score": float(motion.get("static_image_score", 0.0)),
        "static_image_patch_similarity": float(
            motion.get("static_image_patch_similarity", 0.0)
        ),
        "static_image_center_motion": float(motion.get("static_image_center_motion", 0.0)),
        "static_image_screen_like": 1.0
        if motion.get("static_image_screen_like", False)
        else 0.0,
        "blur_score": float(blur.get("blur_score", 0.0)),
        "blur_roi_energy_ratio": min(4.0, float(blur.get("blur_roi_energy_ratio", 1.0))) / 4.0,
        "blur_low_energy_ratio": float(blur.get("blur_low_energy_ratio", 0.0)),
        "track_score": float(track.get("track_score", 0.0)),
        "track_drop_score": float(track.get("track_drop_score", 0.0)),
        "confidence_drop_score": float(track.get("confidence_drop_score", 0.0)),
        "track_missing_count_norm": min(1.0, float(track.get("missing_track_count", 0)) / 5.0),
        "track_confidence_drop_count_norm": min(
            1.0, float(track.get("confidence_drop_count", 0)) / 5.0
        ),
        "roi_count_norm": min(1.0, float(roi_count) / 20.0),
        "rule_p_adv": float(fusion.get("p_adv", 0.0)),
        "rule_temporal_triggered": 1.0 if fusion.get("temporal_triggered", False) else 0.0,
        "rule_local_temporal_triggered": 1.0
        if fusion.get("local_temporal_triggered", False)
        else 0.0,
        "rule_roi_temporal_triggered": 1.0
        if fusion.get("roi_temporal_triggered", False)
        else 0.0,
        "rule_sustained_roi_temporal_triggered": 1.0
        if fusion.get("sustained_roi_temporal_triggered", False)
        else 0.0,
        "rule_local_flow_triggered": 1.0 if fusion.get("local_flow_triggered", False) else 0.0,
        "rule_strong_temporal_triggered": 1.0
        if fusion.get("strong_temporal_triggered", False)
        else 0.0,
        "rule_paired_temporal_flow_triggered": 1.0
        if fusion.get("paired_temporal_flow_triggered", False)
        else 0.0,
        "rule_light_flow_triggered": 1.0 if fusion.get("light_flow_triggered", False) else 0.0,
        "rule_paired_temporal_light_flow_triggered": 1.0
        if fusion.get("paired_temporal_light_flow_triggered", False)
        else 0.0,
        "rule_blur_triggered": 1.0 if fusion.get("blur_triggered", False) else 0.0,
        "rule_paired_temporal_blur_triggered": 1.0
        if fusion.get("paired_temporal_blur_triggered", False)
        else 0.0,
        "rule_track_triggered": 1.0 if fusion.get("track_triggered", False) else 0.0,
        "rule_paired_track_triggered": 1.0
        if fusion.get("paired_track_triggered", False)
        else 0.0,
        "rule_static_image_triggered": 1.0
        if fusion.get("static_image_triggered", False)
        else 0.0,
        "rule_overexposure_triggered": 1.0
        if fusion.get("overexposure_triggered", False)
        else 0.0,
        "rule_benign_global_motion_suppressed": 1.0
        if fusion.get("benign_global_motion_suppressed", False)
        else 0.0,
    }

def build_static_media_classifier_features(
    motion: dict[str, Any],
) -> dict[str, float]:
    """Build the 16-dim feature vector consumed by Static_Media_Classifier.

    Mirrors :func:`tools.train_static_media_classifier.extract_classifier_features`
    so the production inference path and the offline training path score
    the same feature definitions. ``roi_count_norm`` is intentionally fixed
    at 0.5 to match the training extractor (the per-frame ``static_media``
    block has no equivalent count; the training dataset uses the same
    placeholder). ``best_stable_count_norm`` and ``trigger_count_norm``
    share the training-side divisor of 5.0.

    All lookups default to ``0.0``/``False`` so that frames on which
    ``_should_run_static_image`` skipped the A3b block (and therefore did
    not populate ``motion["static_image_*"]``) still produce a well-defined
    16-dim vector; those frames score out to the classifier's "no signal"
    region and do not spuriously trigger.
    """
    stable_count = float(motion.get("static_image_stable_count", 0))
    trigger_count = float(motion.get("static_image_trigger_count", 0))
    return {
        "best_static_image_score": float(motion.get("static_image_score", 0.0)),
        "best_patch_similarity": float(motion.get("static_image_patch_similarity", 0.0)),
        "best_stable_count_norm": min(1.0, stable_count / 5.0),
        "best_center_motion": float(motion.get("static_image_center_motion", 0.0)),
        "best_roi_motion": float(motion.get("static_image_roi_motion", 0.0)),
        "best_context_motion": float(motion.get("static_image_context_motion", 0.0)),
        "best_motion_contrast": float(motion.get("static_image_motion_contrast", 0.0)),
        "best_edge_mean": float(motion.get("static_image_edge_mean", 0.0)),
        "best_screen_like": 1.0 if motion.get("static_image_screen_like", False) else 0.0,
        "best_screen_context_edge_mean": float(
            motion.get("static_image_screen_context_edge_mean", 0.0)
        ),
        "best_screen_context_std": float(motion.get("static_image_screen_context_std", 0.0)),
        "best_screen_line_score": float(motion.get("static_image_screen_line_score", 0.0)),
        "best_screen_roi_context_area_ratio": float(
            motion.get("static_image_screen_roi_context_area_ratio", 0.0)
        ),
        "best_screen_inside_person": 1.0
        if motion.get("static_image_screen_inside_person", False)
        else 0.0,
        "trigger_count_norm": min(1.0, trigger_count / 5.0),
        # Fixed placeholder matching tools/train_static_media_classifier.py
        # (_DEFAULT_ROI_COUNT_NORM). Kept in sync with the training
        # extractor so production/training scoring stays aligned.
        "roi_count_norm": 0.5,
    }


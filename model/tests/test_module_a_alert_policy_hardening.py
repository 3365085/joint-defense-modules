from __future__ import annotations

import copy

import numpy as np
import pytest

from defense.module_a.rebuilt.detector import ModuleADetector
from defense.module_a.types import ModuleAInput, ROI


def _detector(monkeypatch: pytest.MonkeyPatch) -> ModuleADetector:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        lambda self, _path: None,
    )
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        lambda self: None,
    )
    return ModuleADetector(
        {
            "module_a": {
                "frame_size": 64,
                "static_image_enabled": False,
                "light_flow_enabled": False,
                "rebuilt_alert_hold_frames": 3,
                "rebuilt_alert_hold_refresh_on_padv": True,
                "rebuilt_sustained_adv_escalation": True,
                "rebuilt_sustained_adv_seconds": 0.2,
                "rebuilt_sustained_adv_run_mult": 1.0,
            }
        }
    )


def _templates(
    detector: ModuleADetector,
) -> tuple[
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
]:
    result = detector.process(
        ModuleAInput(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            frame_idx=0,
            timestamp=0.0,
            rois=[],
        )
    )
    details = result.details
    detector.reset()
    return (
        copy.deepcopy(details["a1"]),
        copy.deepcopy(details["a2"]),
        copy.deepcopy(details["a3"]),
        copy.deepcopy(details["a4"]),
        copy.deepcopy(details["a3b"]),
        copy.deepcopy(details["scene_context"]),
        copy.deepcopy(details["flow_context"]),
    )


def _normal_scene_raw_high_inputs(
    detector: ModuleADetector,
) -> tuple[
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
]:
    a1, a2, a3, a4, a3b, exposure, flow = _templates(detector)
    a1.update(
        a1_feature_score=0.20,
        target_related=True,
        delta_h_roi_patch_max=0.10,
        delta_h_patch_concentration=0.10,
    )
    a2.update(
        a2_feature_score=0.20,
        target_related=True,
        flash_like=False,
    )
    a3.update(
        a3_feature_score=0.0,
        target_related=True,
        a3_residual_hold_active=False,
        flow_residual_contrast=0.0,
    )
    a4.update(
        p_adv=0.80,
        p_adv_triggered=True,
        dominant_adv_input="A4_MIXED",
        a4_multi_evidence=0.02,
    )
    a3b.update(
        media_candidate_allowed=False,
        p_media_target_related=False,
        p_media_strong_evidence=False,
        p_media_policy=0.0,
        suppressed_reason="natural_scene_texture_plane",
    )
    exposure.update(
        high_false_positive_scene=False,
        overexposure_ratio=0.0,
        underexposed_ratio=0.0,
        exposure_delta=0.0,
        frame_diff_global=0.01,
    )
    flow.update(
        global_motion_weight=0.0,
    )
    blinding = {
        "p_blind": 0.0,
        "p_blind_triggered": False,
        "blind_type": "none",
        "sharp_drop": 0.0,
        "glare_blind": 0.0,
    }
    return a1, a2, a3, a4, a3b, exposure, flow, blinding


def test_sustained_adv_cannot_override_normal_scene_without_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector.process_fps = 10.0
        detector.recent_target_presence.extend([1] * 8)
        detector._scene_baseline_min = 1
        detector._sb_maxfeat.extend([0.20] * 8)
        rois = [ROI("person", (8, 8, 40, 56), "person", 0.9)]

        decisions = [
            detector._joint_decision(
                a1,
                a2,
                a3,
                a4,
                a3b,
                rois,
                exposure,
                flow,
                blinding=blinding,
            )
            for _ in range(5)
        ]

        assert all(
            decision["scene_baseline_normal"]
            for decision in decisions
        )
        assert all(
            not decision["adv_candidate_allowed"]
            for decision in decisions
        )
        assert all(
            not decision["adv_physical_support"]
            for decision in decisions
        )
        assert all(
            not decision["sustained_adv_has_independent_support"]
            for decision in decisions
        )
        assert all(
            not decision["sustained_adv_escalated"]
            for decision in decisions
        )
        assert all(
            not decision["alert_confirmed"]
            for decision in decisions
        )
    finally:
        detector.close()


def test_hold_refresh_requires_candidate_from_original_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._sustained_adv_enabled = False
        detector._alert_hold_channel = "blind"
        detector._alert_hold_remaining = 2

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [],
            exposure,
            flow,
            blinding=blinding,
        )

        assert decision["alert_confirmed"] is True
        assert decision["confirm_window"]["alert_held"] is True
        assert decision["alert_hold_refresh_signal"] is False
        assert decision["alert_hold_refresh_source"] == (
            "blind_candidate"
        )
        assert (
            decision["confirm_window"]["alert_hold_remaining"]
            == 1
        )
    finally:
        detector.close()


def test_adv_hold_does_not_refresh_across_current_explicit_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._sustained_adv_enabled = False
        detector._alert_hold_channel = "adv"
        detector._alert_hold_remaining = 2
        detector._adv_cand_bridge_frames = 1
        detector._adv_cand_bridge_remaining = 1
        a1.update(
            a1_feature_score=0.82,
            delta_h_roi_patch_max=0.70,
            delta_h_patch_concentration=0.80,
        )
        rois = [ROI("person", (8, 8, 40, 56), "person", 0.9)]

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding=blinding,
        )

        assert decision["cold_start_low_motion_adv"] is True
        assert (
            decision["adv_candidate_bridge_explicit_suppression"]
            is True
        )
        assert decision["adv_candidate_bridge_blocked"] is True
        assert decision["adv_single_frame_candidate"] is False
        assert decision["adv_explicitly_suppressed"] is True
        assert decision["alert_confirmed"] is False
        assert decision["alert_hold_refresh_signal"] is False
        assert (
            decision["alert_hold_blocked_reason"]
            == "adv_candidate_policy_suppressed"
        )
        assert (
            decision["confirm_window"]["alert_hold_remaining"]
            == 0
        )
    finally:
        detector.close()


def test_normal_target_motion_exclusion_blocks_head_only_a3_n_of_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        a1.update(
            a1_feature_score=0.42,
            target_related=True,
        )
        a2.update(
            a2_feature_score=0.18,
            target_related=True,
        )
        a3.update(
            a3_feature_score=0.944,
            target_related=True,
            flow_local_anomaly_ratio=0.056,
            flow_roi_coverage_ratio=0.20,
            flow_shape_score=0.944,
            flow_residual_contrast=1.52,
            flow_roi_motion_gap=1.70,
            flow_target_relation=1.0,
        )
        a4.update(
            p_adv=0.95,
            p_adv_triggered=True,
            dominant_adv_input="A3_FLOW_ARTIFACT",
            a4_multi_evidence=0.02,
        )
        exposure.update(
            frame_diff_global=0.0107,
            exposure_delta=0.005,
        )
        flow.update(global_motion_weight=0.0)
        detector.recent_target_presence.extend([1] * 8)
        detector._adv_cand_bridge_frames = 4
        detector._adv_cand_bridge_remaining = 4
        detector._adv_cand_bridge_has_physical_support = True
        rois = [
            ROI("helmet-1", (18, 10, 34, 28), "helmet", 0.9),
        ]

        decisions = [
            detector._joint_decision(
                a1,
                a2,
                a3,
                a4,
                a3b,
                rois,
                exposure,
                flow,
                blinding=blinding,
            )
            for _ in range(8)
        ]

        assert all(
            decision["normal_target_motion_exclusion"]
            for decision in decisions
        )
        assert all(
            not decision["adv_candidate_allowed"]
            for decision in decisions
        )
        assert all(
            not decision["adv_physical_support"]
            for decision in decisions
        )
        assert all(
            not decision["adv_single_frame_candidate"]
            for decision in decisions
        )
        assert all(
            not decision["adv_candidate_bridge_eligible"]
            for decision in decisions
        )
        assert all(
            not decision["adv_candidate_bridged"]
            for decision in decisions
        )
        assert decisions[-1]["adv_candidate_bridge_remaining"] == 0
        assert all(
            not decision["alert_confirmed"]
            for decision in decisions
        )
    finally:
        detector.close()


def test_articulated_target_motion_overrides_residual_hold_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._scene_baseline_enabled = False
        a1.update(
            a1_feature_score=1.0,
            target_related=True,
        )
        a2.update(
            a2_feature_score=0.32,
            target_related=True,
            change_t_motion_aligned=1.0,
            change_t_motion_explain_score=0.4583,
        )
        a3.update(
            a3_feature_score=0.92,
            target_related=True,
            flow_local_anomaly_ratio=0.076,
            flow_max_magnitude_norm=0.93,
            flow_roi_coverage_ratio=0.22,
            flow_residual_contrast=1.74,
            flow_background_explain_score=0.70,
            a3_residual_hold_active=True,
        )
        a4.update(
            p_adv=0.96,
            p_adv_triggered=True,
            dominant_adv_input="A3_FLOW_ARTIFACT",
            a4_multi_evidence=0.50,
        )
        a3b.update(
            suppressed_reason=(
                "low_display_target_plane_prefers_A1_A2_A3"
            ),
        )
        exposure.update(
            frame_diff_global=0.016,
            exposure_delta=0.001,
            overexposure_ratio=0.022,
            underexposed_ratio=0.246,
        )
        detector.recent_target_presence.extend([1] * 8)

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [ROI("helmet-1", (18, 10, 34, 28), "helmet", 0.9)],
            exposure,
            flow,
            blinding=blinding,
        )

        assert decision["a3_residual_fallback"] is True
        assert decision["normal_articulated_target_motion"] is True
        assert decision["normal_target_motion_exclusion"] is True
        assert decision["adv_candidate_allowed"] is False
        assert decision["adv_physical_support"] is False
        assert decision["adv_single_frame_candidate"] is False
        assert (
            decision["adv_explicit_suppression_reason"]
            == "normal_articulated_target_motion"
        )
        assert decision["alert_confirmed"] is False
    finally:
        detector.close()


def test_adv_patch_residual_support_is_not_articulated_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._scene_baseline_enabled = False
        a1.update(
            a1_feature_score=0.42,
            target_related=True,
        )
        a2.update(
            a2_feature_score=0.32,
            target_related=True,
            change_t_motion_aligned=1.0,
            change_t_motion_explain_score=0.4583,
        )
        a3.update(
            a3_feature_score=0.95,
            target_related=True,
            flow_local_anomaly_ratio=0.05,
            flow_max_magnitude_norm=0.80,
            flow_roi_coverage_ratio=0.40,
            flow_residual_contrast=2.66,
            flow_background_explain_score=0.70,
            a3_residual_hold_active=True,
        )
        a4.update(
            p_adv=0.92,
            p_adv_triggered=True,
            dominant_adv_input="A3_FLOW_ARTIFACT",
            a4_multi_evidence=0.50,
        )
        a3b.update(
            suppressed_reason=(
                "low_display_target_plane_prefers_A1_A2_A3"
            ),
        )
        exposure.update(
            frame_diff_global=0.030,
            exposure_delta=0.001,
            overexposure_ratio=0.02,
            underexposed_ratio=0.05,
        )
        detector.recent_target_presence.extend([1] * 8)

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [ROI("helmet-1", (18, 10, 34, 28), "helmet", 0.9)],
            exposure,
            flow,
            blinding=blinding,
        )

        assert decision["a3_residual_fallback"] is True
        assert decision["normal_articulated_target_motion"] is False
        assert decision["normal_target_motion_exclusion"] is False
        assert decision["adv_candidate_allowed"] is True
        assert decision["adv_physical_support"] is True
        assert decision["adv_single_frame_candidate"] is True
    finally:
        detector.close()


@pytest.mark.parametrize(
    (
        "sharpness",
        "ref_sharpness",
        "contrast",
        "ref_contrast",
        "underexposed_ratio",
        "frame_diff",
        "global_motion",
        "expected_support",
    ),
    [
        pytest.param(
            57.0,
            146.0,
            10.0,
            12.3,
            0.02,
            0.005,
            0.10,
            False,
            id="high-detail-construction-scene",
        ),
        pytest.param(
            0.5,
            100.0,
            10.0,
            10.0,
            0.02,
            0.025,
            1.00,
            False,
            id="foreground-object-motion",
        ),
        pytest.param(
            10.0,
            100.0,
            10.0,
            10.0,
            0.004,
            0.005,
            0.27,
            True,
            id="authoritative-like-motion-blur",
        ),
        pytest.param(
            10.0,
            100.0,
            10.0,
            10.0,
            0.19,
            0.005,
            0.27,
            False,
            id="low-light-target-disappearance",
        ),
    ],
)
def test_motion_blur_independent_support_requires_low_detail_stable_scene(
    monkeypatch: pytest.MonkeyPatch,
    sharpness: float,
    ref_sharpness: float,
    contrast: float,
    ref_contrast: float,
    underexposed_ratio: float,
    frame_diff: float,
    global_motion: float,
    expected_support: bool,
) -> None:
    detector = _detector(monkeypatch)
    try:
        monkeypatch.setattr(
            detector,
            "_native_call",
            lambda *_args, **_kwargs: sharpness,
        )
        detector._sb_sharp.extend(
            [ref_sharpness] * detector._scene_baseline_min
        )
        detector._sb_contrast.extend(
            [ref_contrast] * detector._scene_baseline_min
        )
        detector._sb_detstr.extend(
            [1.0] * detector._scene_baseline_min
        )
        detector.recent_target_presence.extend([1] * 8)
        detector._prev_sharp = sharpness

        result = detector._compute_blinding(
            np.zeros((64, 64), dtype=np.uint8),
            [],
            {
                "brightness_std": contrast,
                "overexposure_ratio": 0.0,
                "underexposed_ratio": underexposed_ratio,
                "exposure_delta": 0.0,
                "frame_diff_global": frame_diff,
            },
            {"global_motion_weight": global_motion},
        )

        assert result["blind_type"] == "motion_blur"
        assert (
            result["motion_blur_scene_degradation_support"]
            is expected_support
        )
        assert result["blind_independent_support"] is expected_support
        assert result["p_blind_triggered"] is expected_support
    finally:
        detector.close()


def test_glare_support_rejects_high_contrast_scene_with_deep_shadows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._scene_baseline_enabled = False
        a1.update(a1_feature_score=0.90, target_related=True)
        a2.update(a2_feature_score=0.82, target_related=True)
        a4.update(p_adv=0.92, p_adv_triggered=True)
        exposure.update(
            overexposure_ratio=0.261,
            underexposed_ratio=0.435,
            frame_diff_global=0.030,
        )

        high_contrast = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [ROI("person-1", (8, 8, 48, 60), "person", 0.9)],
            exposure,
            flow,
            blinding=blinding,
        )

        exposure.update(
            overexposure_ratio=0.150,
            underexposed_ratio=0.004,
        )
        real_glare = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [ROI("person-1", (8, 8, 48, 60), "person", 0.9)],
            exposure,
            flow,
            blinding=blinding,
        )

        assert high_contrast["glare_attack_support"] is False
        assert high_contrast["photometric_attack_support"] is False
        assert real_glare["glare_attack_support"] is True
        assert real_glare["photometric_attack_support"] is True
    finally:
        detector.close()


def test_high_contrast_target_texture_motion_is_explicitly_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._scene_baseline_enabled = False
        a1.update(
            a1_feature_score=0.90,
            target_related=True,
            delta_h_roi_patch_max=0.70,
            delta_h_patch_concentration=0.82,
        )
        a2.update(
            a2_feature_score=0.82,
            target_related=True,
            change_t_motion_aligned=1.0,
            change_t_motion_explain_score=0.40,
        )
        a3.update(
            a3_feature_score=0.40,
            target_related=True,
            flow_local_anomaly_ratio=0.10,
            flow_roi_coverage_ratio=0.05,
        )
        a4.update(
            p_adv=0.95,
            p_adv_triggered=True,
            a4_multi_evidence=0.45,
        )
        a3b.update(
            suppressed_reason="target_attached_patch_prefers_A1_A2_A3",
        )
        exposure.update(
            overexposure_ratio=0.02,
            underexposed_ratio=0.35,
            exposure_delta=0.005,
            frame_diff_global=0.030,
        )
        flow.update(global_motion_weight=0.20)
        rois = [ROI("person-1", (8, 8, 48, 60), "person", 0.9)]

        normal_motion = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding=blinding,
        )

        exposure["underexposed_ratio"] = 0.05
        physical_attack = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding=blinding,
        )

        assert normal_motion["localized_a1_attack_support"] is True
        assert normal_motion["photometric_attack_support"] is False
        assert (
            normal_motion["normal_high_contrast_target_texture_motion"]
            is True
        )
        assert normal_motion["normal_target_motion_exclusion"] is True
        assert normal_motion["adv_candidate_allowed"] is False
        assert normal_motion["adv_physical_support"] is False
        assert (
            normal_motion["adv_explicit_suppression_reason"]
            == "normal_high_contrast_target_texture_motion"
        )
        assert (
            physical_attack["normal_high_contrast_target_texture_motion"]
            is False
        )
        assert physical_attack["normal_target_motion_exclusion"] is False
        assert physical_attack["adv_candidate_allowed"] is True
        assert physical_attack["adv_physical_support"] is True
    finally:
        detector.close()


def test_roi_flow_target_motion_overrides_residual_hold_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            blinding,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._scene_baseline_enabled = False
        a1.update(
            a1_feature_score=0.42,
            target_related=True,
        )
        a2.update(
            a2_feature_score=0.28,
            target_related=True,
            change_t_roi_max=0.16,
            change_t_motion_aligned=0.77,
            change_t_motion_explain_score=0.32,
        )
        a3.update(
            a3_feature_score=0.93,
            target_related=True,
            flow_local_anomaly_ratio=0.06,
            flow_max_magnitude_norm=0.50,
            flow_roi_coverage_ratio=0.39,
            flow_residual_contrast=1.30,
            a3_residual_hold_active=True,
        )
        a4.update(
            p_adv=0.95,
            p_adv_triggered=True,
            dominant_adv_input="A3_FLOW_ARTIFACT",
            a4_multi_evidence=0.45,
        )
        a3b.update(
            suppressed_reason=(
                "low_display_target_plane_prefers_A1_A2_A3"
            ),
        )
        exposure.update(
            overexposure_ratio=0.0,
            underexposed_ratio=0.005,
            exposure_delta=0.001,
            frame_diff_global=0.004,
        )
        flow.update(global_motion_weight=0.0)
        rois = [ROI("helmet-1", (18, 10, 34, 28), "helmet", 0.9)]

        normal_motion = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding=blinding,
        )

        a2["change_t_roi_max"] = 0.30
        physical_attack = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding=blinding,
        )

        assert normal_motion["a3_residual_fallback"] is True
        assert normal_motion["normal_roi_flow_target_motion"] is True
        assert normal_motion["normal_target_motion_exclusion"] is True
        assert normal_motion["adv_candidate_allowed"] is False
        assert normal_motion["adv_physical_support"] is False
        assert (
            normal_motion["adv_explicit_suppression_reason"]
            == "normal_roi_flow_target_motion"
        )
        assert physical_attack["normal_roi_flow_target_motion"] is False
        assert physical_attack["normal_target_motion_exclusion"] is False
        assert physical_attack["adv_candidate_allowed"] is True
        assert physical_attack["adv_physical_support"] is True
    finally:
        detector.close()


def test_unsupported_motion_blur_cannot_enter_blind_candidate_or_sustained_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        (
            a1,
            a2,
            a3,
            a4,
            a3b,
            exposure,
            flow,
            _,
        ) = _normal_scene_raw_high_inputs(detector)
        detector._sustained_adv_enabled = False
        detector._blind_target_established = True
        detector.recent_target_presence.extend([1] * 8)
        a4.update(
            p_adv=0.10,
            p_adv_triggered=False,
            dominant_adv_input="A4_MIXED",
        )
        blinding = {
            "p_blind": 0.90,
            "p_blind_triggered": True,
            "blind_type": "motion_blur",
            "sharp_drop": 0.80,
            "glare_blind": 0.0,
            "blind_independent_support": False,
        }

        decisions = [
            detector._joint_decision(
                a1,
                a2,
                a3,
                a4,
                a3b,
                [],
                exposure,
                flow,
                blinding=blinding,
            )
            for _ in range(16)
        ]

        assert all(
            not decision["blind_independent_support"]
            for decision in decisions
        )
        assert all(
            not decision["blind_single_frame_candidate"]
            for decision in decisions
        )
        assert detector._blind_run == 0
        assert all(
            not decision["blind_degrade_evidence"]
            for decision in decisions
        )
        assert all(
            not decision["blind_sustained_escalated"]
            for decision in decisions
        )
        assert all(
            not decision["alert_confirmed"]
            for decision in decisions
        )
    finally:
        detector.close()

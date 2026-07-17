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
                "rebuilt_a3b_media_run_floor": 6,
                "rebuilt_a3b_media_run_gap_tol": 3,
                "rebuilt_alert_hold_frames": 3,
                "rebuilt_a3b_alert_hold_frames": 90,
                "rebuilt_sustained_adv_escalation": False,
            }
        }
    )


def _templates(detector: ModuleADetector) -> tuple[dict, ...]:
    result = detector.process(
        ModuleAInput(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            frame_idx=0,
            timestamp=0.0,
            rois=[],
        )
    )
    detector.reset()
    details = result.details
    return tuple(
        copy.deepcopy(details[key])
        for key in (
            "a1",
            "a2",
            "a3",
            "a4",
            "a3b",
            "scene_context",
            "flow_context",
            "blinding",
        )
    )


def _robust_display_scores() -> dict[str, float]:
    return {
        "candidate_score": 0.62,
        "edge": 0.44,
        "border_contrast": 0.95,
        "display_frame": 0.80,
        "area_ratio": 0.13,
        "boundary": 0.60,
        "rect": 0.70,
        "source_score": 0.78,
        "target_iou": 0.11,
        "target_proximity": 0.66,
        "flow_gap": 0.10,
        "warp_residual": 0.05,
        "yolo_context": 0.66,
        "plane": 0.55,
        "track": 0.50,
    }


def test_robust_phone_display_is_not_suppressed_as_target_near_scene_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, _, _, exposure, flow, _ = _templates(detector)
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.0,
            frame_diff_global=0.01,
        )
        flow.update(global_motion_weight=0.0)

        policy = detector._apply_media_policy(
            p_media_raw=0.75,
            p_media_type="static_image_spoof",
            bbox=(40, 20, 180, 100),
            target_related=True,
            strong_evidence=True,
            scores=_robust_display_scores(),
            flow=flow,
            exposure=exposure,
            a1=a1,
            a2=a2,
            a3=a3,
            width=256,
            height=256,
        )

        assert policy["suppressed_reason"] == "none"
        assert policy["media_candidate_allowed"] is True
        assert policy["p_media_policy"] >= detector.theta_media
    finally:
        detector.close()


def test_robust_phone_display_can_confirm_below_legacy_candidate_and_edge_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow, blinding = _templates(
            detector
        )
        a1.update(a1_feature_score=0.10, target_related=False)
        a2.update(a2_feature_score=0.10, target_related=False)
        a3.update(a3_feature_score=0.10, target_related=False)
        a4.update(
            p_adv=0.10,
            p_adv_triggered=False,
            a4_rule_triggered=False,
            a4_classifier_triggered=False,
            dominant_adv_input="A4_MIXED",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.0,
            frame_diff_global=0.01,
        )
        flow.update(global_motion_weight=0.0)
        blinding.update(
            p_blind=0.0,
            p_blind_triggered=False,
            blind_independent_support=False,
        )

        decisions = []
        for seq in range(1, 5):
            payload = copy.deepcopy(a3b)
            payload.update(
                p_media_raw=0.75,
                p_media_policy=0.75,
                p_media_triggered=True,
                p_media_type="static_image_spoof",
                p_media_bbox=[40, 20, 180, 100],
                p_media_target_related=True,
                p_media_strong_evidence=True,
                media_candidate_allowed=True,
                suppressed_reason="none",
                p_media_scores=_robust_display_scores(),
                a3b_result_fresh=True,
                a3b_result_seq=seq,
                a3b_source_frame_idx=seq * 2,
                a3b_source_timestamp=seq * 2 / 30.0,
                a3b_source_fps=30.0,
                a3b_source_interval_frames=2,
            )
            decisions.append(
                detector._joint_decision(
                    a1,
                    a2,
                    a3,
                    a4,
                    payload,
                    [],
                    exposure,
                    flow,
                    blinding=blinding,
                )
            )

        assert all(
            decision["media_tighten_candidate_pass"] is False
            for decision in decisions
        )
        assert all(
            decision["media_tighten_edge_pass"] is False
            for decision in decisions
        )
        assert all(
            decision["media_tighten_robust_display_pass"] is True
            for decision in decisions
        )
        assert all(decision["media_gate_ok"] is True for decision in decisions)
        assert decisions[-1]["media_confirmed"] is True
        assert decisions[-1]["alert_confirmed"] is True
        assert decisions[-1]["primary_channel"] == "media"

        gap_payload = copy.deepcopy(a3b)
        gap_payload.update(
            p_media_raw=0.0,
            p_media_policy=0.0,
            p_media_triggered=False,
            p_media_bbox=None,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            media_candidate_allowed=False,
            suppressed_reason="no_media_candidate",
            p_media_scores={},
            a3b_result_fresh=False,
        )
        held = []
        for _ in range(40):
            held.append(
                detector._joint_decision(
                    a1,
                    a2,
                    a3,
                    a4,
                    gap_payload,
                    [],
                    exposure,
                    flow,
                    blinding=blinding,
                )
            )
        assert all(decision["alert_confirmed"] for decision in held)
        assert all(decision["primary_channel"] == "media" for decision in held)
        assert held[-1]["confirm_window"]["alert_hold_window_frames"] == 90
    finally:
        detector.close()


def test_motion_aligned_people_in_fixed_camera_scene_do_not_confirm_adv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow, blinding = _templates(
            detector
        )
        detector._scene_baseline_enabled = False
        detector.lbp_baseline_samples = 8
        detector.recent_target_presence.extend([1] * 8)
        a1.update(
            a1_feature_score=0.42,
            target_related=True,
            delta_h_roi_patch_max=0.37,
            delta_h_patch_concentration=1.0,
        )
        a2.update(
            a2_feature_score=0.83,
            target_related=True,
            flash_like=False,
            change_t_motion_aligned=0.90,
            change_t_motion_explain_score=0.40,
        )
        a3.update(
            a3_feature_score=0.92,
            target_related=True,
            flow_local_anomaly_ratio=0.08,
            flow_max_magnitude_norm=0.90,
            flow_roi_coverage_ratio=0.24,
            flow_background_explain_score=0.30,
            flow_residual_contrast=1.50,
            a3_residual_hold_active=True,
        )
        a4.update(
            p_adv=0.95,
            p_adv_triggered=True,
            a4_rule_triggered=True,
            a4_classifier_triggered=False,
            dominant_adv_input="A4_MIXED",
            a4_multi_evidence=0.45,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=False,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="base_adv_evidence_prefers_A1_A2_A3",
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.001,
            underexposed_ratio=0.001,
            exposure_delta=0.001,
            frame_diff_global=0.018,
        )
        flow.update(global_motion_weight=0.0)
        blinding.update(
            p_blind=0.0,
            p_blind_triggered=False,
            blind_independent_support=False,
        )

        decision = detector._joint_decision(
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

        assert decision["normal_articulated_target_motion"] is True
        assert decision["normal_target_motion_exclusion"] is True
        assert decision["adv_single_frame_candidate"] is False
        assert decision["adv_confirmation_blocked_reason"] == (
            "normal_articulated_target_motion"
        )
        assert decision["alert_confirmed"] is False
    finally:
        detector.close()

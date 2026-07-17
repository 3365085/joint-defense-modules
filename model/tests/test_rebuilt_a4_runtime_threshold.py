from __future__ import annotations

import copy

import numpy as np
import pytest

from defense.module_a.rebuilt.detector import A4_FEATURE_NAMES, ModuleADetector
from defense.module_a.types import ModuleAInput, ROI


class _ProbabilityClassifier:
    n_features_in_ = len(A4_FEATURE_NAMES)
    feature_importances_ = np.full(
        len(A4_FEATURE_NAMES),
        1.0 / len(A4_FEATURE_NAMES),
        dtype=np.float32,
    )

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, rows: list[list[float]]) -> np.ndarray:
        return np.asarray(
            [[1.0 - self.probability, self.probability] for _ in rows],
            dtype=np.float64,
        )


def _a4_inputs() -> tuple[dict, dict, dict]:
    a1 = {
        "delta_h": 0.10,
        "delta_h_roi_max": 0.20,
        "delta_h_local_max": 0.30,
        "delta_h_target_contrast": 0.25,
        "a1_feature_score": 0.40,
    }
    a2 = {
        "change_t": 0.15,
        "change_t_roi_max": 0.25,
        "change_t_local_max": 0.30,
        "change_t_without_motion_target": 0.35,
        "a2_feature_score": 0.45,
    }
    a3 = {
        "f_flow": 0.20,
        "flow_local_anomaly_ratio": 0.10,
        "flow_residual": 1.10,
        "flow_shape_score": 0.30,
        "flow_target_relation": 0.60,
        "a3_feature_score": 0.50,
    }
    return a1, a2, a3


def test_classifier_threshold_does_not_replace_rule_threshold() -> None:
    detector = object.__new__(ModuleADetector)
    detector.theta_adv = 0.65
    detector.a4_classifier_decision_threshold = 0.75
    detector._classifier = _ProbabilityClassifier(0.70)
    detector.a4_classifier_loaded = True
    detector._a4_classifier_runtime_disabled = False
    detector.a4_classifier_error = None
    detector.a4_classifier_fallback_reason = "none"
    detector.a4_classifier_configured = True
    detector.a4_classifier_path = "bound.pkl"
    detector.a4_classifier_resolved_path = "bound.pkl"
    detector.a4_classifier_metadata = {"selected_threshold": 0.75}

    out = detector._compute_a4(*_a4_inputs())

    assert out["p_adv"] > 0.70
    assert out["a4_classifier_p_adv"] == pytest.approx(0.70)
    assert out["a4_classifier_triggered"] is False
    assert out["a4_rule_triggered"] is True
    assert out["p_adv_triggered"] is True
    assert out["theta_adv"] == pytest.approx(0.65)
    assert out["a4_decision_threshold"] == pytest.approx(0.75)
    assert out["a4_decision_threshold_source"] == "classifier_metadata"


def _detector(monkeypatch: pytest.MonkeyPatch) -> ModuleADetector:
    monkeypatch.setattr(ModuleADetector, "_load_classifier", lambda self, _path: None)
    monkeypatch.setattr(ModuleADetector, "_load_flownet", lambda self: None)
    return ModuleADetector(
        {
            "module_a": {
                "frame_size": 64,
                "static_image_enabled": False,
                "light_flow_enabled": False,
                "rebuilt_scene_baseline": False,
                "rebuilt_sustained_adv_escalation": True,
                "rebuilt_sustained_adv_seconds": 0.2,
                "rebuilt_sustained_adv_run_mult": 1.0,
            }
        }
    )


def _normal_motion_joint_inputs(detector: ModuleADetector) -> tuple[dict, ...]:
    initial = detector.process(
        ModuleAInput(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            frame_idx=0,
            timestamp=0.0,
            rois=[],
        )
    )
    details = initial.details
    detector.reset()
    a1 = copy.deepcopy(details["a1"])
    a2 = copy.deepcopy(details["a2"])
    a3 = copy.deepcopy(details["a3"])
    a4 = copy.deepcopy(details["a4"])
    a3b = copy.deepcopy(details["a3b"])
    exposure = copy.deepcopy(details["scene_context"])
    flow = copy.deepcopy(details["flow_context"])
    a1.update(a1_feature_score=0.40, target_related=False)
    a2.update(
        a2_feature_score=0.70,
        target_related=False,
        change_t_global=0.30,
        change_t_local_max=0.30,
    )
    a3.update(
        a3_feature_score=0.30,
        target_related=False,
        flow_local_anomaly_ratio=0.05,
    )
    exposure.update(
        high_false_positive_scene=False,
        overexposure_ratio=0.0,
        underexposed_ratio=0.0,
        exposure_delta=0.02,
        frame_diff_global=0.03,
    )
    flow.update(global_motion_weight=0.0)
    return a1, a2, a3, a4, a3b, exposure, flow


def _no_blinding() -> dict:
    return {
        "p_blind": 0.0,
        "p_blind_triggered": False,
        "blind_independent_support": False,
        "blind_type": "none",
        "sharp_drop": 0.0,
        "glare_blind": 0.0,
    }


def test_patch_baseline_becomes_ready_after_twelve_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3 = _a4_inputs()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        outputs = [
            detector._compute_a4(a1, a2, a3, frame=frame, frame_idx=frame_idx)
            for frame_idx in range(12)
        ]

        assert outputs[10]["a4_patch_baseline_ready"] is False
        assert outputs[11]["a4_patch_baseline_ready"] is True
        assert outputs[11]["a4_patch_baseline_samples"] == 12
    finally:
        detector.close()


def test_classifier_below_sidecar_threshold_does_not_rescue_normal_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow = _normal_motion_joint_inputs(detector)
        a4.update(
            p_adv=0.93,
            p_adv_triggered=False,
            a4_classifier_used=True,
            a4_patch_baseline_ready=True,
            a4_classifier_p_adv=0.93,
            a4_classifier_triggered=False,
            a4_decision_threshold=0.94,
        )

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [],
            exposure,
            flow,
            blinding=_no_blinding(),
        )

        assert decision["normal_motion_texture_change"] is True
        assert decision["classifier_adv_rescue"] is False
        assert decision["adv_candidate_allowed"] is False
    finally:
        detector.close()


def test_classifier_rescue_bypasses_normal_motion_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow = _normal_motion_joint_inputs(detector)
        a4.update(
            p_adv=0.97,
            p_adv_triggered=True,
            a4_classifier_used=True,
            a4_patch_baseline_ready=True,
            a4_classifier_p_adv=0.97,
            a4_classifier_triggered=True,
            a4_decision_threshold=0.94,
        )

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [],
            exposure,
            flow,
            blinding=_no_blinding(),
        )

        assert decision["normal_motion_texture_change"] is True
        assert decision["classifier_adv_rescue"] is True
        assert decision["adv_candidate_allowed"] is True
        assert decision["adv_single_frame_candidate"] is True
        assert decision["adv_explicit_suppression_reason"] == "none"
    finally:
        detector.close()


def test_classifier_rescue_does_not_bypass_severe_underexposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow = _normal_motion_joint_inputs(detector)
        exposure["underexposed_ratio"] = 0.73
        a4.update(
            p_adv=0.97,
            p_adv_triggered=True,
            a4_classifier_used=True,
            a4_patch_baseline_ready=True,
            a4_classifier_p_adv=0.97,
            a4_classifier_triggered=True,
            a4_decision_threshold=0.94,
        )

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [],
            exposure,
            flow,
            blinding=_no_blinding(),
        )

        assert decision["classifier_adv_rescue_requested"] is True
        assert decision["classifier_adv_rescue_dark_scene_blocked"] is True
        assert decision["classifier_adv_rescue"] is False
        assert decision["adv_candidate_allowed"] is False
    finally:
        detector.close()


def test_classifier_confirmation_uses_sidecar_window_not_rule_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        detector.process_fps = 4.0
        detector.a4_classifier_alarm_window = 8
        detector.a4_classifier_alarm_required_hits = 5
        a1, a2, a3, a4, a3b, exposure, flow = _normal_motion_joint_inputs(detector)
        detector.process_fps = 4.0
        exposure["high_false_positive_scene"] = True
        a4.update(
            p_adv=0.97,
            p_adv_triggered=True,
            a4_rule_triggered=False,
            a4_classifier_used=True,
            a4_patch_baseline_ready=True,
            a4_classifier_p_adv=0.97,
            a4_classifier_triggered=True,
            a4_decision_threshold=0.94,
        )

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
                blinding=_no_blinding(),
            )
            for _ in range(5)
        ]

        assert decisions[-1]["confirm_window"]["window_frames"] == 3
        assert decisions[-1]["confirm_window"]["adv_hit_required"] == 3
        assert decisions[-1]["confirm_window"]["classifier_adv_window"] == 8
        assert (
            decisions[-1]["confirm_window"]["classifier_adv_hit_required"]
            == 5
        )
        assert all(not item["adv_confirmed"] for item in decisions[:4])
        assert decisions[4]["adv_confirmed"] is True
        assert decisions[4]["confirm_window"]["rule_adv_confirmed"] is False
        assert (
            decisions[4]["confirm_window"]["classifier_adv_confirmed"]
            is True
        )
    finally:
        detector.close()


def test_severe_underexposure_clears_classifier_confirmation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        a1, a2, a3, a4, a3b, exposure, flow = _normal_motion_joint_inputs(detector)
        a4.update(
            p_adv=0.97,
            p_adv_triggered=True,
            a4_rule_triggered=False,
            a4_classifier_used=True,
            a4_patch_baseline_ready=True,
            a4_classifier_p_adv=0.97,
            a4_classifier_triggered=True,
            a4_decision_threshold=0.94,
        )
        bright = [
            detector._joint_decision(
                a1,
                a2,
                a3,
                a4,
                a3b,
                [],
                exposure,
                flow,
                blinding=_no_blinding(),
            )
            for _ in range(5)
        ]
        assert bright[-1]["confirm_window"]["classifier_adv_confirmed"] is True

        exposure["underexposed_ratio"] = 0.73
        dark = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            [],
            exposure,
            flow,
            blinding=_no_blinding(),
        )

        assert dark["classifier_adv_rescue_dark_scene_blocked"] is True
        assert dark["confirm_window"]["classifier_adv_count"] == 0
        assert dark["confirm_window"]["classifier_adv_confirmed"] is False
        assert dark["adv_confirmed"] is False
    finally:
        detector.close()


def test_joint_sustained_path_uses_effective_a4_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    try:
        first = detector.process(
            ModuleAInput(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                frame_idx=0,
                timestamp=0.0,
                rois=[],
            )
        )
        details = first.details
        detector.reset()
        a1 = copy.deepcopy(details["a1"])
        a2 = copy.deepcopy(details["a2"])
        a3 = copy.deepcopy(details["a3"])
        a4 = copy.deepcopy(details["a4"])
        a3b = copy.deepcopy(details["a3b"])
        exposure = copy.deepcopy(details["scene_context"])
        flow = copy.deepcopy(details["flow_context"])

        a1.update(
            a1_feature_score=0.82,
            target_related=True,
            delta_h_roi_patch_max=0.62,
            delta_h_patch_concentration=0.82,
        )
        a2.update(a2_feature_score=0.60, target_related=True, flash_like=False)
        a3.update(
            a3_feature_score=0.35,
            target_related=True,
            a3_residual_hold_active=False,
            flow_residual_contrast=0.5,
        )
        a4.update(
            p_adv=0.80,
            p_adv_triggered=False,
            a4_decision_threshold=0.85,
            dominant_adv_input="A4_MIXED",
            a4_multi_evidence=0.60,
        )
        a3b.update(
            media_candidate_allowed=False,
            p_media_target_related=True,
            p_media_strong_evidence=False,
            p_media_policy=0.0,
            suppressed_reason="target_attached_patch_prefers_A1_A2_A3",
            p_media_scores={
                "display_frame": 0.20,
                "boundary": 0.05,
                "area_ratio": 0.05,
            },
        )
        exposure.update(
            high_false_positive_scene=False,
            overexposure_ratio=0.0,
            underexposed_ratio=0.0,
            exposure_delta=0.03,
            frame_diff_global=0.02,
        )
        flow.update(global_motion_weight=0.0)
        detector.process_fps = 10.0
        detector.recent_target_presence.extend([1] * 8)
        rois = [ROI("helmet", (8, 8, 40, 40), "helmet", 0.9)]

        decision = detector._joint_decision(
            a1,
            a2,
            a3,
            a4,
            a3b,
            rois,
            exposure,
            flow,
            blinding={
                "p_blind": 0.0,
                "p_blind_triggered": False,
                "blind_independent_support": False,
                "blind_type": "none",
                "sharp_drop": 0.0,
                "glare_blind": 0.0,
            },
        )

        assert decision["adv_candidate_allowed"] is False
        assert decision["sustained_adv_run"] == 0
        assert decision["adv_confirmation_blocked_reason"] == "score_below_threshold"
    finally:
        detector.close()

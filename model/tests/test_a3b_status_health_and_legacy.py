from __future__ import annotations

from types import SimpleNamespace

import pytest

from defense.runtime.frame_processor import FrameProcessor


MODULE_A_OBSERVABILITY_DEFAULTS = {
    "flow_artifact_path": None,
    "flow_artifact_sha256": None,
    "flow_artifact_expected_sha256": None,
    "a4_classifier_path": None,
    "a4_classifier_sha256": None,
    "a4_classifier_expected_sha256": None,
    "native": None,
}


class _Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _processor(
    *,
    detector: object | None = None,
    detector_impl: str = "rebuilt",
    module_config: dict[str, object] | None = None,
) -> FrameProcessor:
    pipeline = _Resettable()
    pipeline.detector = detector if detector is not None else SimpleNamespace()
    pipeline.detector_impl = detector_impl
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={"module_a": dict(module_config or {})},
        backend="fake",
        model_family="fake",
        artifact_path="",
    )
    return FrameProcessor(bundle)


def _status(
    processor: FrameProcessor,
    *,
    source_type: str = "file",
    source: str = "ordinary-source",
    static_media: dict[str, object] | None = None,
    info: dict[str, object] | None = None,
) -> dict[str, object]:
    return processor._build_status(
        source_type=source_type,
        source=source,
        profile="test",
        realtime=source_type != "file",
        frame_idx=7,
        video_time_s=1.25,
        source_fps=30.0,
        fps=20.0,
        dropped_frames=0,
        info=dict(info or {}),
        ppe={},
        ppe_tracks=[],
        display_options={},
        feature_options={},
        custom_model={},
        redetect_budget_ok=False,
        redetect_count=0,
        redetect_ms=0.0,
        processing_ms=5.0,
        target_frame_budget_ms=33.0,
        raw_boxes_count=0,
        static_media=dict(static_media or {}),
    )


@pytest.mark.parametrize("source_type", ["file", "camera", "rtsp"])
@pytest.mark.parametrize("trigger_field", ["triggered", "static_image_triggered"])
def test_legacy_confirmation_is_authoritative_without_source_keyword(
    source_type: str,
    trigger_field: str,
) -> None:
    processor = _processor(detector_impl="legacy")
    bbox = [10, 20, 210, 260]

    status = _status(
        processor,
        source_type=source_type,
        source=f"{source_type}-source-without-trigger-keyword",
        static_media={
            trigger_field: True,
            "score": 0.67,
            "p_media": 0.74,
            "p_media_bbox": bbox,
            "triggered_source": "legacy_fast_state",
        },
    )

    assert status["a3b_triggered"] is True
    assert status["a3b_state"] == "confirmed"
    assert status["a3b_score"] == pytest.approx(0.67)
    assert status["a3b_confirmed_score"] == pytest.approx(0.67)
    assert status["a3b_display_score"] == pytest.approx(0.67)
    assert status["a3b_bbox"] == bbox
    assert status["a3b_triggered_source"] == "legacy_fast_state"


def test_rebuilt_authority_still_depends_on_media_confirmed() -> None:
    processor = _processor()
    candidate = {
        "result_contract_source": "rebuilt",
        "triggered": True,
        "static_image_triggered": True,
        "media_confirmed": False,
        "score": 0.67,
        "p_media": 0.67,
        "p_media_bbox": [1, 2, 30, 40],
    }

    unconfirmed = _status(processor, static_media=candidate)
    confirmed = _status(
        processor,
        static_media={
            **candidate,
            "media_confirmed": True,
            "p_media_confirmed_score": 0.66,
        },
    )

    assert unconfirmed["a3b_triggered"] is False
    assert unconfirmed["a3b_state"] != "confirmed"
    assert confirmed["a3b_triggered"] is True
    assert confirmed["a3b_state"] == "confirmed"
    assert confirmed["a3b_triggered_source"] == "rebuilt_media_confirmed"


def test_rebuilt_health_is_exposed_and_new_errors_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = _processor()
    health = {
        "result_contract_source": "rebuilt",
        "a3b_background_enabled": True,
        "a3b_generation": 3,
        "a3b_error_count": 1,
        "a3b_last_error": "RuntimeError: exploded",
        "a3b_last_error_at": 100.5,
        "a3b_last_success_at": 90.25,
        "a3b_source_frame_idx": 41,
        "a3b_source_timestamp": 1.75,
        "a3b_result_seq": 8,
    }

    with caplog.at_level("WARNING", logger="defense.runtime.frame_processor"):
        first = _status(processor, static_media=health)
        second = _status(processor, static_media=health)
        third = _status(
            processor,
            static_media={
                **health,
                "a3b_error_count": 2,
                "a3b_last_error_at": 101.5,
                "a3b_last_error": "ValueError: failed again",
            },
        )

    assert first["a3b_background_enabled"] is True
    assert first["a3b_generation"] == 3
    assert first["a3b_error_count"] == 1
    assert first["a3b_last_error"] == "RuntimeError: exploded"
    assert first["a3b_last_error_at"] == pytest.approx(100.5)
    assert first["a3b_last_success_at"] == pytest.approx(90.25)
    assert first["a3b_source_frame_idx"] == 41
    assert first["a3b_source_timestamp"] == pytest.approx(1.75)
    assert first["a3b_result_seq"] == 8
    assert first["a3b_debug"]["a3b_error_count"] == 1
    assert second["a3b_error_count"] == 1
    assert third["a3b_error_count"] == 2
    warnings = [
        record
        for record in caplog.records
        if "A3b background backend error" in record.getMessage()
    ]
    assert len(warnings) == 2

    processor.reset()
    with caplog.at_level("WARNING", logger="defense.runtime.frame_processor"):
        _status(processor, static_media=health)
    warnings = [
        record
        for record in caplog.records
        if "A3b background backend error" in record.getMessage()
    ]
    assert len(warnings) == 3


def test_schedule_block_warning_is_emitted_once_per_blocked_episode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = _processor()
    blocked = {
        "result_contract_source": "rebuilt",
        "a3b_generation": 4,
        "a3b_schedule_blocked": True,
        "a3b_schedule_blocked_reason": "global_worker_limit",
        "a3b_live_worker_count": 0,
        "a3b_global_live_worker_count": 2,
        "a3b_global_worker_limit": 2,
    }

    with caplog.at_level("WARNING", logger="defense.runtime.frame_processor"):
        _status(processor, static_media=blocked)
        _status(processor, static_media=blocked)
        _status(
            processor,
            static_media={
                **blocked,
                "a3b_schedule_blocked": False,
                "a3b_schedule_blocked_reason": "none",
            },
        )
        _status(processor, static_media=blocked)

    warnings = [
        record
        for record in caplog.records
        if "A3b background scheduling blocked" in record.getMessage()
    ]
    assert len(warnings) == 2


def test_module_a_effective_config_prefers_detector_then_bundle_config() -> None:
    detector = SimpleNamespace(
        static_image_enabled=True,
        _a3b_interval=5,
        _a3b_worker_timeout_s=3.5,
        _a3b_result_lease_s=5.5,
        _a3b_max_retired_workers=2,
        _a3b_global_worker_limit=3,
        theta_media_raw=0.51,
        theta_media=0.57,
        theta_adv=0.66,
        theta_blind=0.56,
        _alert_hold_frames=12,
        _alert_hold_refresh_on_padv=True,
        _adv_cand_bridge_frames=4,
        _sustained_adv_enabled=True,
        _sustained_adv_seconds=2.0,
        _sustained_adv_run_mult=1.6,
        _sustained_adv_benign_decay=0.9,
        _sustained_adv_require_target=False,
        _sustained_adv_require_physical_support=False,
        _sustained_adv_exclude_static_bg=True,
        _sustained_adv_recent_target_min=3,
        _blind_sustained_enabled=True,
        _blind_sustained_floor=12,
        _blind_sustained_degrade_min=0.30,
        _blind_sustained_established_min=3,
        _a4_classifier_rescue_underexposed_max=0.54,
        _a3b_independent_trigger=True,
        _a3b_tighten_gate=True,
        _a3b_gate_candidate_min=0.71,
        _a3b_gate_edge_min=0.46,
        _a3b_gate_edge_max=0.59,
        _a3b_gate_border_contrast_min=0.81,
        _a3b_media_run_floor=16,
        _a3b_media_run_gap_tol=4,
    )
    config = {
        "detector_impl": "legacy",
        "static_image_enabled": False,
        "static_image_interval": 99,
        "static_image_worker_timeout_s": 8.0,
        "static_image_result_lease_s": 9.0,
        "static_image_max_retired_workers": 5,
        "static_image_global_worker_limit": 6,
        "rebuilt_theta_media_raw": 0.11,
        "rebuilt_theta_media": 0.12,
        "rebuilt_theta_adv": 0.13,
        "rebuilt_theta_blind": 0.14,
        "rebuilt_alert_hold_frames": 99,
        "rebuilt_alert_hold_refresh_on_padv": False,
        "rebuilt_adv_candidate_bridge_frames": 98,
        "rebuilt_a4_classifier_rescue_underexposed_max": 0.91,
        "rebuilt_sustained_adv_escalation": False,
        "rebuilt_sustained_adv_seconds": 9.0,
        "rebuilt_sustained_adv_run_mult": 9.1,
        "rebuilt_sustained_adv_benign_decay": 0.1,
        "rebuilt_sustained_adv_require_target": True,
        "rebuilt_sustained_adv_require_physical_support": True,
        "rebuilt_sustained_adv_exclude_static_bg": False,
        "rebuilt_sustained_adv_recent_target_min": 9,
        "rebuilt_blind_sustained_escalation": False,
        "rebuilt_blind_sustained_floor": 97,
        "rebuilt_blind_sustained_degrade_min": 0.9,
        "rebuilt_blind_sustained_established_min": 8,
        "rebuilt_a3b_independent_trigger": False,
        "rebuilt_a3b_tighten_gate": False,
        "rebuilt_a3b_gate_candidate_min": 0.21,
        "rebuilt_a3b_gate_edge_min": 0.22,
        "rebuilt_a3b_gate_edge_max": 0.23,
        "rebuilt_a3b_gate_border_contrast_min": 0.24,
        "rebuilt_a3b_media_run_floor": 25,
        "rebuilt_a3b_media_run_gap_tol": 6,
    }
    effective = _status(
        _processor(detector=detector, detector_impl="rebuilt", module_config=config)
    )["module_a_effective_config"]

    expected_stable = {
        "detector_impl": "rebuilt",
        "detector_process_fps_cap": None,
        "a3b_sensitivity": None,
        "a3b_source_keyword_policy": "diagnostic_only",
        "a3b_source_keyword_match_required": False,
        "a3b_observed_only_source_keywords": [],
        "a3b_trigger_source_keywords": [],
        "static_image_enabled": True,
        "static_image_interval": 5,
        "static_image_worker_timeout_s": 3.5,
        "static_image_result_lease_s": 5.5,
        "static_image_max_retired_workers": 2,
        "static_image_global_worker_limit": 3,
        "rebuilt_theta_media_raw": 0.51,
        "rebuilt_theta_media": 0.57,
        "rebuilt_theta_adv": 0.66,
        "rebuilt_theta_blind": 0.56,
        "rebuilt_blind_confirm_ratio": None,
        "rebuilt_alert_hold_frames": 12,
        "rebuilt_alert_hold_refresh_on_padv": True,
        "rebuilt_adv_candidate_bridge_frames": 4,
        "rebuilt_a4_classifier_rescue_underexposed_max": 0.54,
        "rebuilt_sustained_adv_escalation": True,
        "rebuilt_sustained_adv_seconds": 2.0,
        "rebuilt_sustained_adv_run_mult": 1.6,
        "rebuilt_sustained_adv_benign_decay": 0.9,
        "rebuilt_sustained_adv_require_target": False,
        "rebuilt_sustained_adv_require_physical_support": False,
        "rebuilt_sustained_adv_exclude_static_bg": True,
        "rebuilt_sustained_adv_recent_target_min": 3,
        "rebuilt_blind_sustained_escalation": True,
        "rebuilt_blind_sustained_floor": 12,
        "rebuilt_blind_sustained_degrade_min": 0.30,
        "rebuilt_blind_sustained_established_min": 3,
        "rebuilt_a3b_independent_trigger": True,
        "rebuilt_a3b_tighten_gate": True,
        "rebuilt_a3b_gate_candidate_min": 0.71,
        "rebuilt_a3b_gate_edge_min": 0.46,
        "rebuilt_a3b_gate_edge_max": 0.59,
        "rebuilt_a3b_gate_border_contrast_min": 0.81,
        "rebuilt_a3b_soft_gate_candidate_tolerance": 0.001,
        "rebuilt_a3b_soft_gate_aspect_ratio_min": 0.40,
        "rebuilt_a3b_soft_gate_aspect_ratio_max": 2.50,
        "rebuilt_a3b_media_run_floor": 16,
        "rebuilt_a3b_media_run_gap_tol": 4,
        "flow_requested_device": None,
        "flow_effective_device": None,
        "flow_backend": None,
        "flow_fallback_reason": None,
        "a4_classifier_configured": None,
        "a4_classifier_loaded": None,
        "a4_classifier_error": None,
        "a4_classifier_fallback_reason": None,
        "a4_classifier_alarm_window": None,
        "a4_classifier_alarm_required_hits": None,
    }
    assert {
        key: effective[key]
        for key in expected_stable
    } == expected_stable
    assert {
        key: effective[key]
        for key in MODULE_A_OBSERVABILITY_DEFAULTS
    } == MODULE_A_OBSERVABILITY_DEFAULTS

    fallback = _status(
        _processor(
            detector=SimpleNamespace(),
            detector_impl="rebuilt",
            module_config=config,
        )
    )["module_a_effective_config"]
    assert fallback["static_image_enabled"] is False
    assert fallback["static_image_interval"] == 99
    assert fallback["static_image_worker_timeout_s"] == pytest.approx(8.0)
    assert fallback["static_image_result_lease_s"] == pytest.approx(9.0)
    assert fallback["static_image_max_retired_workers"] == 5
    assert fallback["static_image_global_worker_limit"] == 6
    assert fallback["rebuilt_theta_media_raw"] == pytest.approx(0.11)
    assert fallback["rebuilt_a3b_independent_trigger"] is False
    assert fallback["rebuilt_a3b_media_run_gap_tol"] == 6


def test_rebuilt_a3b_schedule_timing_populates_legacy_status_alias() -> None:
    status = _status(
        _processor(),
        info={
            "latency_breakdown": {
                "module_a_breakdown": {
                    "a3b_schedule": 1.75,
                }
            }
        },
    )

    assert status["a3b_static_media_ms"] == pytest.approx(1.75)

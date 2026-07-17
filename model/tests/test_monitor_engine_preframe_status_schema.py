from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from defense.runtime.frame_processor import FrameProcessor, _a3b_backend_health
from defense.runtime.runner import MonitorEngine


EXPECTED_A3B_HEALTH = {
    "a3b_background_enabled": False,
    "a3b_generation": 0,
    "a3b_active_worker_count": 0,
    "a3b_retired_worker_count": 0,
    "a3b_live_worker_count": 0,
    "a3b_global_live_worker_count": 0,
    "a3b_global_worker_limit": 0,
    "a3b_worker_limit_scope": "process",
    "a3b_worker_timeout_s": 0.0,
    "a3b_max_retired_workers": 0,
    "a3b_active_worker_started_at": None,
    "a3b_active_worker_age_s": 0.0,
    "a3b_active_worker_frame_idx": None,
    "a3b_active_worker_timestamp": None,
    "a3b_timed_out_worker_count": 0,
    "a3b_worker_rejected_count": 0,
    "a3b_last_worker_rejected_at": None,
    "a3b_schedule_blocked": False,
    "a3b_schedule_blocked_reason": "none",
    "a3b_error_count": 0,
    "a3b_last_error": None,
    "a3b_last_error_at": None,
    "a3b_last_success_at": None,
    "a3b_source_frame_idx": None,
    "a3b_source_timestamp": None,
    "a3b_last_attempt_frame_idx": None,
    "a3b_last_attempt_timestamp": None,
    "a3b_result_published_at": None,
    "a3b_result_age_s": 0.0,
    "a3b_result_lease_s": 0.0,
    "a3b_result_fresh": False,
    "a3b_result_expired_count": 0,
    "a3b_result_seq": 0,
}

EXPECTED_MODULE_A_EFFECTIVE_CONFIG = {
    "detector_impl": None,
    "detector_process_fps_cap": None,
    "a3b_sensitivity": None,
    "a3b_source_keyword_policy": "diagnostic_only",
    "a3b_source_keyword_match_required": False,
    "a3b_observed_only_source_keywords": [],
    "a3b_trigger_source_keywords": [],
    "static_image_enabled": None,
    "static_image_interval": None,
    "static_image_worker_timeout_s": None,
    "static_image_result_lease_s": None,
    "static_image_max_retired_workers": None,
    "static_image_global_worker_limit": None,
    "rebuilt_theta_media_raw": None,
    "rebuilt_theta_media": None,
    "rebuilt_theta_adv": None,
    "rebuilt_theta_blind": None,
    "rebuilt_blind_confirm_ratio": None,
    "rebuilt_alert_hold_frames": None,
    "rebuilt_alert_hold_refresh_on_padv": None,
    "rebuilt_adv_candidate_bridge_frames": None,
    "rebuilt_a4_classifier_rescue_underexposed_max": None,
    "rebuilt_sustained_adv_escalation": None,
    "rebuilt_sustained_adv_seconds": None,
    "rebuilt_sustained_adv_run_mult": None,
    "rebuilt_sustained_adv_benign_decay": None,
    "rebuilt_sustained_adv_require_target": None,
    "rebuilt_sustained_adv_require_physical_support": None,
    "rebuilt_sustained_adv_exclude_static_bg": None,
    "rebuilt_sustained_adv_recent_target_min": None,
    "rebuilt_blind_sustained_escalation": None,
    "rebuilt_blind_sustained_floor": None,
    "rebuilt_blind_sustained_degrade_min": None,
    "rebuilt_blind_sustained_established_min": None,
    "rebuilt_a3b_independent_trigger": None,
    "rebuilt_a3b_tighten_gate": None,
    "rebuilt_a3b_gate_candidate_min": None,
    "rebuilt_a3b_gate_edge_min": None,
    "rebuilt_a3b_gate_edge_max": None,
    "rebuilt_a3b_gate_border_contrast_min": None,
    "rebuilt_a3b_soft_gate_candidate_tolerance": None,
    "rebuilt_a3b_soft_gate_aspect_ratio_min": None,
    "rebuilt_a3b_soft_gate_aspect_ratio_max": None,
    "rebuilt_a3b_media_run_floor": None,
    "rebuilt_a3b_media_run_gap_tol": None,
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


EXPECTED_MODULE_A_OBSERVABILITY_DEFAULTS = {
    "flow_artifact_path": None,
    "flow_artifact_sha256": None,
    "flow_artifact_expected_sha256": None,
    "a4_classifier_path": None,
    "a4_classifier_sha256": None,
    "a4_classifier_expected_sha256": None,
    "native": None,
}

EXPECTED_AUTHORITATIVE_MODEL_DEFAULTS = {
    "model_id": "mask_bd_v4_clean_baseline",
    "locked": True,
    "metadata_valid": False,
    "source": None,
    "engine": None,
    "onnx": None,
}

EXPECTED_DECODER_DEFAULTS = {
    "requested_backend": "nvdec",
    "backend": "not_started",
    "effective_backend": "not_started",
    "gpu_device": "cuda:0",
    "frame_device": "not_started",
    "frames_decoded": 0,
    "bytes_decoded": 0,
    "fallback_count": 0,
    "fallback_reason": "not_started",
    "fallback_reasons": [],
    "closed": False,
    "eof": False,
}


def _assert_mapping_subset(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert {
        key: actual[key]
        for key in expected
    } == expected


def _assert_preframe_schema(status: dict[str, Any]) -> None:
    _assert_mapping_subset(status, EXPECTED_A3B_HEALTH)
    _assert_mapping_subset(status["a3b_debug"], EXPECTED_A3B_HEALTH)
    _assert_mapping_subset(
        status["module_a_effective_config"],
        EXPECTED_MODULE_A_EFFECTIVE_CONFIG,
    )
    assert "artifact" in status
    assert status["artifact"] is None or isinstance(status["artifact"], str)
    _assert_mapping_subset(
        status["authoritative_model"],
        EXPECTED_AUTHORITATIVE_MODEL_DEFAULTS,
    )
    assert isinstance(status["authoritative_model"]["backend"], str)
    _assert_mapping_subset(status["decoder"], EXPECTED_DECODER_DEFAULTS)


class _BlockingCache:
    def __init__(self) -> None:
        self.get_entered = threading.Event()
        self.release_get = threading.Event()
        self.bundle = SimpleNamespace(
            backend="fake",
            model_family="fake",
            artifact_path="fake.pt",
            config={"runtime": {"detector_thread_warmup_timeout_s": 2.0}},
            cache_hit=False,
            cache_get_ms=0.0,
            config_load_ms=0.0,
            backend_create_ms=0.0,
            pipeline_construct_ms=0.0,
            warmup_ms=0.0,
            warmup_frames=0,
            pipeline_reset_ms=0.0,
            warmup_error="",
        )

    def get(self, **_kwargs: Any) -> SimpleNamespace:
        self.get_entered.set()
        assert self.release_get.wait(timeout=2.0)
        return self.bundle

    def clear(self) -> None:
        pass


def test_initial_status_exposes_stable_preframe_a3b_schema() -> None:
    engine = MonitorEngine(_BlockingCache())

    status = engine.get_status()

    processor = object.__new__(FrameProcessor)
    processor.pipeline = SimpleNamespace(detector=SimpleNamespace())
    processor.bundle = SimpleNamespace(config={"module_a": {}}, pipeline=processor.pipeline)

    _assert_mapping_subset(_a3b_backend_health({}), EXPECTED_A3B_HEALTH)
    effective_config = processor._module_a_effective_config()
    _assert_mapping_subset(effective_config, EXPECTED_MODULE_A_EFFECTIVE_CONFIG)
    _assert_mapping_subset(
        effective_config,
        EXPECTED_MODULE_A_OBSERVABILITY_DEFAULTS,
    )
    assert status["running"] is False
    assert status["initializing"] is False
    assert status["prewarming"] is False
    assert status["first_detection_ready"] is False
    assert status["artifact"] is None
    assert status["authoritative_model"]["backend"] == "tensorrt"
    assert status["source_frames_skipped_for_realtime"] == 0
    assert status["capture_frames_published"] == 0
    assert status["detector_submission_count"] == 0
    assert status["processed_detection_frames"] == 0
    assert status["detection_source_coverage_ratio"] == 0.0
    _assert_preframe_schema(status)


def test_status_reports_processed_detection_source_coverage() -> None:
    engine = MonitorEngine(_BlockingCache())
    engine.overlay_seq = 50
    engine.status.update(
        {
            "source_fps": 30.0,
            "source_time_s": 5.0,
            "frame_idx": 100,
        }
    )

    status = engine.get_status()

    assert status["processed_detection_frames"] == 50
    assert status["detection_source_coverage_ratio"] == 50 / 151


def test_start_preserves_preframe_a3b_schema_through_initializing_and_prewarm() -> None:
    cache = _BlockingCache()
    engine = MonitorEngine(cache)
    process_entered = threading.Event()
    allow_detector_ready = threading.Event()
    start_result: dict[str, Any] = {}

    def fake_process_loop(run_id: int, *_args: Any) -> None:
        process_entered.set()
        assert allow_detector_ready.wait(timeout=2.0)
        with engine.condition:
            if run_id == engine.run_id:
                engine.status.update(
                    {
                        "detector_ready": True,
                        "initializing": False,
                        "prewarming": False,
                        "preview_mode": "detector_ready_wait_first_frame",
                    }
                )
                engine.condition.notify_all()

    engine._backend_process_loop = fake_process_loop  # type: ignore[method-assign]
    engine._backend_capture_loop = lambda *_args: None  # type: ignore[method-assign]
    engine._preview_render_loop = lambda *_args: None  # type: ignore[method-assign]

    def run_start() -> None:
        start_result["run_id"] = engine.start(source_type="camera", source="0")

    start_thread = threading.Thread(target=run_start, name="test-monitor-start")
    start_thread.start()
    try:
        assert cache.get_entered.wait(timeout=2.0)
        initializing = engine.get_status()
        assert initializing["initializing"] is True
        assert initializing["prewarming"] is True
        assert initializing["first_detection_ready"] is False
        _assert_preframe_schema(initializing)

        cache.release_get.set()
        assert process_entered.wait(timeout=2.0)
        prewarming = engine.get_status()
        assert prewarming["initializing"] is False
        assert prewarming["prewarming"] is True
        assert prewarming["first_detection_ready"] is False
        _assert_preframe_schema(prewarming)

        allow_detector_ready.set()
        start_thread.join(timeout=2.0)
        assert not start_thread.is_alive()

        before_first_frame = engine.get_status()
        assert before_first_frame["detector_ready"] is True
        assert before_first_frame["prewarming"] is False
        assert before_first_frame["first_detection_ready"] is False
        assert start_result["run_id"] == before_first_frame["run_id"]
        _assert_preframe_schema(before_first_frame)
    finally:
        cache.release_get.set()
        allow_detector_ready.set()
        start_thread.join(timeout=2.0)
        engine.stop(release_pipeline_cache=False)


def test_seek_reset_path_does_not_drop_preframe_a3b_schema() -> None:
    engine = MonitorEngine(_BlockingCache())
    engine.run_id = 7
    engine.status.update(
        {
            "run_id": 7,
            "running": True,
            "source_type": "file",
            "source_duration_s": 10.0,
            "source_epoch": 1,
        }
    )

    status = engine.control_run(7, "seek", source_time_s=3.0)

    assert status["source_epoch"] == 2
    assert status["first_detection_ready"] is False
    _assert_preframe_schema(status)

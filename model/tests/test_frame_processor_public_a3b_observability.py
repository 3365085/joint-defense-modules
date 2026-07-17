from __future__ import annotations

from types import SimpleNamespace

import pytest

from defense.runtime.frame_processor import FrameProcessor


class _Pipeline:
    def __init__(self, detector: object) -> None:
        self.detector = detector
        self.detector_impl = "rebuilt"

    def reset(self) -> None:
        pass


def _processor(
    *,
    detector: object,
    module_config: dict[str, object] | None = None,
) -> FrameProcessor:
    pipeline = _Pipeline(detector)
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
    static_media: dict[str, object],
) -> dict[str, object]:
    return processor._build_status(
        source_type="file",
        source="test-source",
        profile="test",
        realtime=False,
        frame_idx=7,
        video_time_s=1.25,
        source_fps=30.0,
        fps=20.0,
        dropped_frames=0,
        info={},
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
        static_media=static_media,
    )


def test_public_status_exposes_a3b_worker_and_last_attempt_health() -> None:
    status = _status(
        _processor(detector=SimpleNamespace()),
        {
            "a3b_active_worker_count": 1,
            "a3b_retired_worker_count": 2,
            "a3b_live_worker_count": 3,
            "a3b_global_live_worker_count": 4,
            "a3b_global_worker_limit": 5,
            "a3b_worker_limit_scope": "process",
            "a3b_worker_timeout_s": 3.0,
            "a3b_max_retired_workers": 2,
            "a3b_active_worker_started_at": 100.0,
            "a3b_active_worker_age_s": 0.25,
            "a3b_active_worker_frame_idx": 40,
            "a3b_active_worker_timestamp": 1.5,
            "a3b_timed_out_worker_count": 1,
            "a3b_worker_rejected_count": 2,
            "a3b_last_worker_rejected_at": 101.0,
            "a3b_schedule_blocked": True,
            "a3b_schedule_blocked_reason": "retired_worker_limit",
            "a3b_last_attempt_frame_idx": 41,
            "a3b_last_attempt_timestamp": 1.75,
            "a3b_result_published_at": 99.0,
            "a3b_result_age_s": 0.5,
            "a3b_result_lease_s": 5.0,
            "a3b_result_fresh": True,
            "a3b_result_expired_count": 3,
        },
    )

    assert status["a3b_active_worker_count"] == 1
    assert status["a3b_retired_worker_count"] == 2
    assert status["a3b_live_worker_count"] == 3
    assert status["a3b_global_live_worker_count"] == 4
    assert status["a3b_global_worker_limit"] == 5
    assert status["a3b_worker_limit_scope"] == "process"
    assert status["a3b_worker_timeout_s"] == pytest.approx(3.0)
    assert status["a3b_max_retired_workers"] == 2
    assert status["a3b_active_worker_started_at"] == pytest.approx(100.0)
    assert status["a3b_active_worker_age_s"] == pytest.approx(0.25)
    assert status["a3b_active_worker_frame_idx"] == 40
    assert status["a3b_active_worker_timestamp"] == pytest.approx(1.5)
    assert status["a3b_timed_out_worker_count"] == 1
    assert status["a3b_worker_rejected_count"] == 2
    assert status["a3b_last_worker_rejected_at"] == pytest.approx(101.0)
    assert status["a3b_schedule_blocked"] is True
    assert status["a3b_schedule_blocked_reason"] == "retired_worker_limit"
    assert status["a3b_last_attempt_frame_idx"] == 41
    assert status["a3b_last_attempt_timestamp"] == pytest.approx(1.75)
    assert status["a3b_result_published_at"] == pytest.approx(99.0)
    assert status["a3b_result_age_s"] == pytest.approx(0.5)
    assert status["a3b_result_lease_s"] == pytest.approx(5.0)
    assert status["a3b_result_fresh"] is True
    assert status["a3b_result_expired_count"] == 3
    assert status["a3b_debug"]["a3b_active_worker_count"] == 1
    assert status["a3b_debug"]["a3b_retired_worker_count"] == 2
    assert status["a3b_debug"]["a3b_last_attempt_frame_idx"] == 41
    assert status["a3b_debug"]["a3b_last_attempt_timestamp"] == pytest.approx(
        1.75
    )


def test_effective_config_exposes_static_image_and_independent_trigger() -> None:
    module_config = {
        "static_image_enabled": False,
        "static_image_worker_timeout_s": 9.0,
        "static_image_result_lease_s": 11.0,
        "static_image_max_retired_workers": 4,
        "static_image_global_worker_limit": 6,
        "rebuilt_a3b_independent_trigger": False,
    }
    detector = SimpleNamespace(
        static_image_enabled=True,
        _a3b_worker_timeout_s=3.0,
        _a3b_result_lease_s=5.0,
        _a3b_max_retired_workers=2,
        _a3b_global_worker_limit=2,
        _a3b_independent_trigger=True,
    )

    effective = _status(
        _processor(detector=detector, module_config=module_config),
        {},
    )["module_a_effective_config"]
    fallback = _status(
        _processor(detector=SimpleNamespace(), module_config=module_config),
        {},
    )["module_a_effective_config"]

    assert effective["static_image_enabled"] is True
    assert effective["static_image_worker_timeout_s"] == pytest.approx(3.0)
    assert effective["static_image_result_lease_s"] == pytest.approx(5.0)
    assert effective["static_image_max_retired_workers"] == 2
    assert effective["static_image_global_worker_limit"] == 2
    assert effective["rebuilt_a3b_independent_trigger"] is True
    assert fallback["static_image_enabled"] is False
    assert fallback["static_image_worker_timeout_s"] == pytest.approx(9.0)
    assert fallback["static_image_result_lease_s"] == pytest.approx(11.0)
    assert fallback["static_image_max_retired_workers"] == 4
    assert fallback["static_image_global_worker_limit"] == 6
    assert fallback["rebuilt_a3b_independent_trigger"] is False

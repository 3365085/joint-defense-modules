from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline


FRAME = np.zeros((640, 640, 3), dtype=np.uint8)


def _pipeline() -> VideoDefensePipeline:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline._offline_default_source_fps = 30.0
    pipeline._a3b_suppression_hold_s = 6.0
    pipeline._a3b_suppression_stale_bridge_s = 0.5
    pipeline._a3b_suppress_remaining = 0
    pipeline._a3b_suppress_bbox = None
    pipeline._a3b_suppress_result_seq = None
    pipeline._a3b_suppress_lease_expires_at_s = None
    pipeline._a3b_suppress_clock_s = None
    pipeline._a3b_suppress_clock_basis = None
    pipeline._a3b_suppress_last_source_time_s = None
    return pipeline


def _detections(*boxes: list[int]) -> DetectionFrameResult:
    return DetectionFrameResult(
        image=FRAME,
        boxes=[list(box) for box in boxes],
        classes=[0] * len(boxes),
        confidences=[0.9] * len(boxes),
        names={0: "person"},
        backend="fake",
        artifact_path="",
        inference_ms=0.0,
    )


def _rebuilt_info(
    *,
    bbox: list[int],
    seq: int,
    confirmed: bool = True,
    fresh: bool | None = True,
    candidate_allowed: bool = True,
    policy_suppressed: bool = False,
) -> dict:
    a3b = {
        "media_confirmed": confirmed,
        "p_media_triggered": True,
        "a3b_state": "confirmed" if confirmed else "candidate",
        "p_media_bbox": bbox,
        "a3b_result_seq": seq,
        "media_candidate_allowed": candidate_allowed,
        "suppression": {
            "suppressed": policy_suppressed,
            "media_candidate_allowed": candidate_allowed,
        },
    }
    if fresh is not None:
        a3b["a3b_result_fresh"] = fresh
    return {
        "details": {
            "a3b": a3b,
        }
    }


def _apply(
    pipeline: VideoDefensePipeline,
    info: dict,
    *boxes: list[int],
    timestamp_s: float | None,
) -> tuple[DetectionFrameResult, list]:
    return pipeline._apply_a3b_suppression(
        FRAME,
        _detections(*boxes),
        [],
        info,
        source_timestamp_s=timestamp_s,
    )


def _legacy_info(*, bbox: list[int], confirmed: bool = True) -> dict:
    return {
        "details": {
            "module_a_features": {
                "static_media": {
                    "static_image_triggered": confirmed,
                    "p_media_bbox": bbox,
                }
            }
        }
    }


def test_confirmed_new_seq_and_moving_bbox_refresh_suppression_immediately() -> None:
    pipeline = _pipeline()
    old_bbox = [100, 100, 200, 200]
    new_bbox = [300, 300, 400, 400]

    _apply(
        pipeline,
        _rebuilt_info(bbox=old_bbox, seq=1),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )

    info = _rebuilt_info(bbox=new_bbox, seq=2)
    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        [320, 320, 360, 360],
        timestamp_s=11.0,
    )

    assert pipeline._a3b_suppress_bbox == (300, 300, 400, 400)
    assert pipeline._a3b_suppress_result_seq == 2
    assert pipeline._a3b_suppress_lease_expires_at_s == 17.0
    assert pipeline._a3b_suppress_remaining == 180
    assert filtered.boxes == [[120, 120, 160, 160]]
    assert info["a3b_suppression_filtered"] is True
    assert info["a3b_suppression_hold_s"] == 6.0
    assert info["a3b_suppression_remaining_s"] == 6.0


def test_same_seq_and_bbox_refreshes_only_when_result_is_fresh() -> None:
    pipeline = _pipeline()
    info = _rebuilt_info(bbox=[100, 100, 200, 200], seq=7)

    _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    assert pipeline._a3b_suppress_lease_expires_at_s == 16.0

    refreshed_info = _rebuilt_info(bbox=[100, 100, 200, 200], seq=7)
    _apply(
        pipeline,
        refreshed_info,
        [120, 120, 160, 160],
        timestamp_s=12.0,
    )

    assert pipeline._a3b_suppress_lease_expires_at_s == 18.0
    assert pipeline._a3b_suppress_result_seq == 7
    assert refreshed_info["a3b_suppression_refreshed"] is True


def test_new_seq_refreshes_hold_even_when_bbox_is_unchanged() -> None:
    pipeline = _pipeline()

    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=7),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=8),
        [120, 120, 160, 160],
        timestamp_s=12.0,
    )

    assert pipeline._a3b_suppress_lease_expires_at_s == 18.0
    assert pipeline._a3b_suppress_result_seq == 8


def test_confirmed_cached_result_renews_at_expiry_without_release_gap() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=7),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    info = _rebuilt_info(bbox=[100, 100, 200, 200], seq=7)

    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=16.0,
    )

    assert filtered.boxes == []
    assert pipeline._a3b_suppress_lease_expires_at_s == 22.0
    assert info["a3b_suppression_active"] is True
    assert "a3b_suppression_released" not in info


def test_unconfirmed_result_after_backend_failure_does_not_renew_expired_hold() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=7),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=7,
        confirmed=False,
    )
    info["details"]["a3b"].update(
        {
            "a3b_error_count": 1,
            "a3b_last_error": "RuntimeError: newer a3b attempt failed",
        }
    )

    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=16.0,
    )

    assert filtered.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert info["a3b_suppression_released"] is True


def test_expired_rebuilt_result_does_not_renew_confirmed_hold() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=7),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=7,
        confirmed=True,
        fresh=False,
    )

    bridged, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=12.0,
    )

    assert bridged.boxes == []
    assert pipeline._a3b_suppress_lease_expires_at_s == 12.5
    assert info["a3b_suppression_remaining_s"] == 0.5
    assert info["a3b_suppression_lease_clamped"] is True
    assert info["a3b_suppression_refresh_blocked_reason"] == "stale_result"

    released, _ = _apply(
        pipeline,
        _rebuilt_info(
            bbox=[100, 100, 200, 200],
            seq=7,
            fresh=False,
        ),
        [120, 120, 160, 160],
        timestamp_s=12.5,
    )

    assert released.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None


def test_missing_rebuilt_freshness_cannot_renew_same_cached_result() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=7),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=7,
        fresh=None,
    )

    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=12.0,
    )

    assert filtered.boxes == []
    assert pipeline._a3b_suppress_lease_expires_at_s == 12.5
    assert info["a3b_suppression_refresh_blocked_reason"] == "freshness_missing"
    assert "a3b_suppression_refreshed" not in info


def test_legacy_confirmed_bbox_change_refreshes_without_result_seq() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _legacy_info(bbox=[100, 100, 200, 200]),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )

    filtered, _ = _apply(
        pipeline,
        _legacy_info(bbox=[300, 300, 400, 400]),
        [120, 120, 160, 160],
        [320, 320, 360, 360],
        timestamp_s=11.0,
    )

    assert pipeline._a3b_suppress_bbox == (300, 300, 400, 400)
    assert pipeline._a3b_suppress_result_seq is None
    assert pipeline._a3b_suppress_lease_expires_at_s == 17.0
    assert filtered.boxes == [[120, 120, 160, 160]]


def test_unconfirmed_candidate_does_not_start_pipeline_suppression() -> None:
    pipeline = _pipeline()
    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=1,
        confirmed=False,
    )

    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=1.0,
    )

    assert filtered.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert pipeline._a3b_suppress_result_seq is None
    assert "a3b_suppression_active" not in info
    assert info["a3b_suppression_refresh_blocked_reason"] == "not_confirmed"


def test_legacy_remaining_counter_does_not_drive_real_release() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=3),
        [120, 120, 160, 160],
        timestamp_s=10.0,
    )
    pipeline._a3b_suppress_remaining = 0

    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=3,
        confirmed=False,
        fresh=True,
    )
    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=11.0,
    )

    assert filtered.boxes == []
    assert info["a3b_suppression_active"] is True
    assert info["a3b_suppression_remaining_s"] == 5.0
    assert pipeline._a3b_suppress_remaining == 150


def test_missing_source_timestamp_uses_deterministic_cadence_fallback() -> None:
    pipeline = _pipeline()
    first_info = _rebuilt_info(bbox=[100, 100, 200, 200], seq=4)
    _apply(
        pipeline,
        first_info,
        [120, 120, 160, 160],
        timestamp_s=None,
    )

    assert first_info["a3b_suppression_clock_basis"] == "cadence_fallback"
    assert first_info["a3b_suppression_remaining_s"] == 6.0

    stale_info: dict = {}
    filtered = None
    for _ in range(16):
        stale_info = _rebuilt_info(
            bbox=[100, 100, 200, 200],
            seq=4,
            fresh=False,
        )
        filtered, _ = _apply(
            pipeline,
            stale_info,
            [120, 120, 160, 160],
            timestamp_s=None,
        )

    assert filtered is not None
    assert filtered.boxes == [[120, 120, 160, 160]]
    assert stale_info["a3b_suppression_released"] is True
    assert pipeline._a3b_suppress_bbox is None


def test_source_timestamp_rewind_clears_old_source_lease() -> None:
    pipeline = _pipeline()
    _apply(
        pipeline,
        _rebuilt_info(bbox=[100, 100, 200, 200], seq=5),
        [120, 120, 160, 160],
        timestamp_s=100.0,
    )

    info = _rebuilt_info(
        bbox=[100, 100, 200, 200],
        seq=5,
        fresh=False,
    )
    filtered, _ = _apply(
        pipeline,
        info,
        [120, 120, 160, 160],
        timestamp_s=1.0,
    )

    assert filtered.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_bbox is None
    assert pipeline._a3b_suppress_lease_expires_at_s is None
    assert info["a3b_suppression_clock_reset_reason"] == (
        "source_timestamp_rewind"
    )


def test_reset_clears_suppression_result_seq() -> None:
    reset_calls: list[str] = []
    pipeline = _pipeline()
    pipeline.detector = SimpleNamespace(
        reset=lambda: reset_calls.append("detector"),
    )
    pipeline.frame_idx = 9
    pipeline._last_small_gray = object()
    pipeline._last_detections = object()
    pipeline._last_rois = [object()]
    pipeline._last_detector_frame_idx = 8
    pipeline._temporal_reuse_target_state = {1: {}}
    pipeline._temporal_reuse_consecutive = 3
    pipeline._a3b_suppress_remaining = 12
    pipeline._a3b_suppress_bbox = (100, 100, 200, 200)
    pipeline._a3b_suppress_result_seq = 8
    pipeline._a3b_suppress_lease_expires_at_s = 16.0
    pipeline._a3b_suppress_clock_s = 10.0
    pipeline._a3b_suppress_clock_basis = "source_timestamp"
    pipeline._a3b_suppress_last_source_time_s = 10.0

    pipeline.reset()

    assert reset_calls == ["detector"]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert pipeline._a3b_suppress_result_seq is None
    assert pipeline._a3b_suppress_lease_expires_at_s is None
    assert pipeline._a3b_suppress_clock_s is None
    assert pipeline._a3b_suppress_clock_basis is None
    assert pipeline._a3b_suppress_last_source_time_s is None


def test_close_closes_detector_before_backend() -> None:
    close_order: list[str] = []
    pipeline = _pipeline()
    pipeline.detector = SimpleNamespace(
        close=lambda: close_order.append("detector"),
    )
    pipeline.detector_backend = SimpleNamespace(
        close=lambda: close_order.append("backend"),
    )
    pipeline._last_small_gray = object()
    pipeline._last_detections = object()
    pipeline._last_rois = [object()]
    pipeline._temporal_reuse_target_state = {1: {}}
    pipeline._temporal_reuse_consecutive = 3
    pipeline._a3b_suppress_bbox = (100, 100, 200, 200)
    pipeline._a3b_suppress_lease_expires_at_s = 16.0
    pipeline._a3b_suppress_clock_s = 10.0
    pipeline._a3b_suppress_clock_basis = "source_timestamp"
    pipeline._a3b_suppress_last_source_time_s = 10.0

    pipeline.close()

    assert close_order == ["detector", "backend"]
    assert pipeline._last_small_gray is None
    assert pipeline._last_detections is None
    assert pipeline._last_rois is None
    assert pipeline._temporal_reuse_target_state == {}
    assert pipeline._temporal_reuse_consecutive == 0
    assert pipeline._a3b_suppress_bbox is None
    assert pipeline._a3b_suppress_lease_expires_at_s is None
    assert pipeline._a3b_suppress_clock_s is None

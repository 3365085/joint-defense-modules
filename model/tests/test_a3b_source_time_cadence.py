from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from defense.module_a.rebuilt.detector import ModuleADetector
from defense.module_a.types import ModuleAInput
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState


SOURCE_FPS = 30.0


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    **module_config: object,
) -> ModuleADetector:
    monkeypatch.setattr(
        ModuleADetector,
        "_load_flownet",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        ModuleADetector,
        "_load_classifier",
        staticmethod(lambda _path: None),
    )
    config = {
        "frame_size": 64,
        "static_image_enabled": True,
        "static_image_interval": 1000,
        "light_flow_enabled": False,
        "rebuilt_a3b_media_run_floor": 15,
        **module_config,
    }
    detector = ModuleADetector({"module_a": config})
    detector.process_fps = SOURCE_FPS
    return detector


def _item(frame_idx: int, timestamp: float) -> ModuleAInput:
    return ModuleAInput(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        frame_idx=frame_idx,
        timestamp=timestamp,
        rois=[],
    )


def _media_payload(
    detector: ModuleADetector,
    *,
    interval: int,
) -> dict[str, Any]:
    payload = detector._empty_a3b()
    payload.update(
        {
            "p_media_raw": 0.82,
            "p_media_raw_triggered": True,
            "p_media": 0.82,
            "p_media_policy": 0.82,
            "p_media_triggered": True,
            "p_media_type": "screen_replay",
            "p_media_bbox": [8, 8, 56, 56],
            "p_media_target_related": False,
            "p_media_strong_evidence": True,
            "media_candidate_allowed": True,
            "suppressed_reason": "none",
            "a3b_state": "candidate",
            "a3b_source_fps": SOURCE_FPS,
            "a3b_source_interval_frames": interval,
            "p_media_scores": {
                "candidate_score": 0.80,
                "edge": 0.50,
                "border_contrast": 0.90,
                "display_frame": 0.80,
                "area_ratio": 0.12,
                "boundary": 0.30,
            },
        }
    )
    return payload


def _publish_media_result(
    detector: ModuleADetector,
    *,
    payload: dict[str, Any],
    seq: int,
    source_frame_idx: int,
    source_timestamp: float,
) -> None:
    published_at = time.time()
    with detector._a3b_bg_lock:
        detector._a3b_result_seq = seq
        detector._a3b_last_success_at = published_at
        detector._a3b_source_frame_idx = source_frame_idx
        detector._a3b_source_timestamp = source_timestamp
        detector._a3b_result_published_at = published_at
        detector._a3b_result_published_monotonic = time.monotonic()
        detector._a3b_bg_result = dict(payload)


def _detector_confirmation_time(
    monkeypatch: pytest.MonkeyPatch,
    *,
    interval: int,
    source_frame_indices: list[int] | None = None,
    source_timestamps: list[float] | None = None,
) -> float:
    detector = _detector(monkeypatch)
    try:
        payload = _media_payload(detector, interval=interval)
        if source_frame_indices is None:
            source_frame_indices = [
                interval * seq - 1 for seq in range(1, 25)
            ]
        if source_timestamps is None:
            source_timestamps = [
                frame_idx / SOURCE_FPS
                for frame_idx in source_frame_indices
            ]
        for seq, (source_frame_idx, source_timestamp) in enumerate(
            zip(source_frame_indices, source_timestamps, strict=True),
            start=1,
        ):
            _publish_media_result(
                detector,
                payload=payload,
                seq=seq,
                source_frame_idx=source_frame_idx,
                source_timestamp=source_timestamp,
            )
            result = detector.process(
                _item(seq, source_timestamp + 1.0 / SOURCE_FPS)
            )
            joint = result.details["joint_decision"]
            if joint["media_confirmed"]:
                return source_timestamp
        raise AssertionError("media confirmation was not reached")
    finally:
        detector.close()


def _soft_payload(
    *,
    seq: int,
    interval: int,
    source_frame_idx: int,
    source_timestamp: float,
    quality: bool = True,
) -> dict[str, object]:
    return {
        "result_contract_source": "rebuilt",
        "source_path": "D:/neutral/cadence-case.mp4",
        "a3b_result_seq": seq,
        "a3b_result_fresh": True,
        "a3b_source_frame_idx": source_frame_idx,
        "a3b_source_timestamp": source_timestamp,
        "a3b_source_fps": SOURCE_FPS,
        "a3b_source_interval_frames": interval,
        "p_media": 0.67,
        "live_score": 0.67,
        "p_media_candidate_count": 1,
        "p_media_bbox": [120, 120, 420, 420],
        "p_media_strong_evidence": True,
        "media_candidate_allowed": True,
        "policy": {
            "media_candidate_allowed": True,
            "suppressed": False,
        },
        "suppression": {
            "media_candidate_allowed": True,
            "suppressed": False,
        },
        "p_media_scores": {
            "candidate_score": 0.72 if quality else 0.60,
            "edge": 0.50,
            "border_contrast": 0.90,
            "track": 0.80,
            "yolo_context": 0.80,
        },
    }


@pytest.mark.parametrize(
    ("source_fps", "expected_interval"),
    [
        (15.0, 6),
        (25.0, 6),
        (30.0, 6),
        (60.0, 12),
    ],
)
def test_a3b_analysis_interval_preserves_30fps_source_time_cadence(
    monkeypatch: pytest.MonkeyPatch,
    source_fps: float,
    expected_interval: int,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_interval=6,
    )
    try:
        detector.process_fps = source_fps

        assert detector._effective_a3b_interval() == expected_interval
    finally:
        detector.close()


def test_a3b_worker_reports_real_source_interval_when_analysis_is_30hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch, static_image_interval=6)
    try:
        detector.process_fps = 30.0
        detector.source_fps = 60.0

        assert detector._a3b_source_cadence(6) == (60.0, 12)
    finally:
        detector.close()


@pytest.mark.parametrize("interval", [1, 3, 6])
def test_media_confirmation_uses_source_frame_equivalent_cadence(
    monkeypatch: pytest.MonkeyPatch,
    interval: int,
) -> None:
    confirmed_at = _detector_confirmation_time(
        monkeypatch,
        interval=interval,
    )

    assert confirmed_at == pytest.approx(17.0 / SOURCE_FPS)


def test_media_confirmation_tolerates_worker_delay_and_small_source_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _detector_confirmation_time(monkeypatch, interval=3)
    worker_delayed = _detector_confirmation_time(
        monkeypatch,
        interval=3,
        source_frame_indices=[2, 5, 8, 14, 17, 20],
    )
    small_drop = _detector_confirmation_time(
        monkeypatch,
        interval=3,
        source_frame_indices=[2, 5, 8, 11, 14, 17, 20],
        source_timestamps=[
            value / SOURCE_FPS
            for value in [2, 5, 9, 12, 15, 18, 21]
        ],
    )

    assert abs(worker_delayed - baseline) <= 3.0 / SOURCE_FPS
    assert abs(small_drop - baseline) <= 2.0 / SOURCE_FPS


@pytest.mark.parametrize("interval", [1, 3, 6])
def test_soft_confirmation_and_hold_use_source_frame_units(
    interval: int,
) -> None:
    state = A3BSoftTriggerState(
        {
            "allow_single_strong_trigger": False,
            "allow_window_accumulated_trigger": True,
            "window_size": 12,
            "min_window_hits": 6,
            "min_consecutive_hits": 6,
            "trigger_threshold": 0.60,
            "trigger_hold_frames": 6,
        }
    )
    result = None
    seq = 0
    for seq in range(1, 8):
        source_frame_idx = interval * seq - 1
        result = state.update(
            _soft_payload(
                seq=seq,
                interval=interval,
                source_frame_idx=source_frame_idx,
                source_timestamp=source_frame_idx / SOURCE_FPS,
            )
        )
        if result["triggered"]:
            break

    assert result is not None
    assert result["triggered"] is True
    assert result["triggered_source"] == "window_accumulated"
    assert result["debug"]["source_frame_units"] == interval
    expected_trigger_frame = 11 if interval == 6 else 5
    assert interval * seq - 1 == expected_trigger_frame
    assert result["debug"]["quality_window_result_hits"] >= 2

    duplicate = state.update(
        _soft_payload(
            seq=seq,
            interval=interval,
            source_frame_idx=interval * seq - 1,
            source_timestamp=(interval * seq - 1) / SOURCE_FPS,
        )
    )
    assert duplicate["debug"]["source_frame_units"] == 0
    assert (
        duplicate["debug"]["quality_window_hits"]
        == result["debug"]["quality_window_hits"]
    )

    failing_frame_idx = interval * (seq + 1) - 1
    held = state.update(
        _soft_payload(
            seq=seq + 1,
            interval=interval,
            source_frame_idx=failing_frame_idx,
            source_timestamp=failing_frame_idx / SOURCE_FPS,
            quality=False,
        )
    )
    assert held["triggered"] is True
    assert held["debug"]["trigger_hold_remaining"] == max(0, 6 - interval)

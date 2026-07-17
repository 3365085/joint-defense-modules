from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from defense.module_a.rebuilt import detector as rebuilt_detector_module
from defense.module_a.rebuilt.detector import ModuleADetector
from defense.module_a.types import ModuleAInput


def _detector(monkeypatch: pytest.MonkeyPatch, **module_config: object) -> ModuleADetector:
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
        "static_image_interval": 1,
        **module_config,
    }
    return ModuleADetector({"module_a": config})


def _item(frame_idx: int = 1, timestamp: float = 1.0) -> ModuleAInput:
    return ModuleAInput(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        frame_idx=frame_idx,
        timestamp=timestamp,
        rois=[],
    )


def _media_payload(detector: ModuleADetector) -> dict[str, object]:
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
            "p_media_strong_evidence": True,
            "media_candidate_allowed": True,
            "suppressed_reason": "none",
            "a3b_state": "candidate",
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


def _completed_thread(detector: ModuleADetector) -> threading.Thread:
    thread = detector._a3b_bg_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    return thread


def test_static_image_disabled_never_schedules_a3b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch, static_image_enabled=False)
    called = threading.Event()

    def compute_a3b(*_args: object) -> dict[str, object]:
        called.set()
        return {"unexpected": True}

    monkeypatch.setattr(detector, "_compute_a3b", compute_a3b)

    result = detector.process(_item())

    assert detector._a3b_bg_thread is None
    assert not called.wait(timeout=0.05)
    assert result.details["a3b"]["a3b_background_enabled"] is False
    assert result.details["a3b"]["a3b_active_worker_count"] == 0
    assert result.details["a3b"]["a3b_retired_worker_count"] == 0
    assert result.details["a3b"]["a3b_state"] == "disabled"
    assert result.details["a3b"]["suppressed_reason"] == "disabled"


@pytest.mark.parametrize("worker_outcome", ["result", "error"])
def test_reset_rejects_pre_reset_worker_publication(
    monkeypatch: pytest.MonkeyPatch,
    worker_outcome: str,
) -> None:
    detector = _detector(monkeypatch)
    started = threading.Event()
    release = threading.Event()

    def delayed_compute(*_args: object) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=2.0)
        if worker_outcome == "error":
            raise RuntimeError("stale worker failure")
        return {"worker_marker": "stale"}

    monkeypatch.setattr(detector, "_compute_a3b", delayed_compute)

    detector.process(_item(frame_idx=7, timestamp=1.25))
    assert started.wait(timeout=2.0)
    old_thread = detector._a3b_bg_thread
    old_generation = detector._a3b_generation

    detector.reset()
    assert detector._a3b_generation == old_generation + 1

    release.set()
    assert old_thread is not None
    old_thread.join(timeout=2.0)
    assert not old_thread.is_alive()

    payload = detector._a3b_result_snapshot()
    assert detector._a3b_bg_result is None
    assert detector.media_track is None
    assert payload["a3b_generation"] == old_generation + 1
    assert payload["a3b_result_seq"] == 0
    assert payload["a3b_error_count"] == 0
    assert payload["a3b_last_error"] is None
    assert payload["a3b_last_success_at"] is None
    assert payload["a3b_source_frame_idx"] is None


def test_repeated_reset_source_switch_retains_workers_and_rejects_stale_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    started = [threading.Event(), threading.Event()]
    releases = [threading.Event(), threading.Event()]
    call_lock = threading.Lock()
    call_count = 0

    def delayed_compute(*_args: object) -> dict[str, object]:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        started[call_index].set()
        assert releases[call_index].wait(timeout=2.0)
        return {"worker_marker": f"stale-{call_index}"}

    monkeypatch.setattr(detector, "_compute_a3b", delayed_compute)

    detector.process(_item(frame_idx=41, timestamp=7.0))
    assert started[0].wait(timeout=2.0)
    first_worker = detector._a3b_bg_thread
    assert first_worker is not None

    detector.reset()
    first_reset = detector._a3b_result_snapshot()
    assert first_reset["a3b_active_worker_count"] == 0
    assert first_reset["a3b_retired_worker_count"] == 1
    assert first_worker in detector._a3b_retired_threads

    detector.process(_item(frame_idx=42, timestamp=7.5))
    assert started[1].wait(timeout=2.0)
    second_worker = detector._a3b_bg_thread
    assert second_worker is not None
    assert second_worker is not first_worker

    detector.reset()
    second_reset = detector._a3b_result_snapshot()
    assert second_reset["a3b_active_worker_count"] == 0
    assert second_reset["a3b_retired_worker_count"] == 2
    assert detector._a3b_retired_threads == [first_worker, second_worker]

    for release in releases:
        release.set()
    for worker in (first_worker, second_worker):
        worker.join(timeout=2.0)
        assert not worker.is_alive()

    payload = detector._a3b_result_snapshot()
    assert payload["a3b_active_worker_count"] == 0
    assert payload["a3b_retired_worker_count"] == 0
    assert detector._a3b_retired_threads == []
    assert detector._a3b_bg_result is None
    assert detector.media_track is None
    assert payload["a3b_result_seq"] == 0
    assert "worker_marker" not in payload


def test_reset_clears_source_scoped_lifecycle_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    detector._a3b_timed_out_worker_count = 2
    detector._a3b_worker_rejected_count = 3
    detector._a3b_last_worker_rejected_at = 100.0
    detector._a3b_result_expired_count = 4

    detector.reset()
    payload = detector._a3b_result_snapshot()

    assert payload["a3b_timed_out_worker_count"] == 0
    assert payload["a3b_worker_rejected_count"] == 0
    assert payload["a3b_last_worker_rejected_at"] is None
    assert payload["a3b_result_expired_count"] == 0


def test_background_exception_is_visible_in_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)

    def failing_compute(*_args: object) -> dict[str, object]:
        raise RuntimeError("a3b exploded")

    monkeypatch.setattr(detector, "_compute_a3b", failing_compute)

    detector.process(_item(frame_idx=11, timestamp=2.75))
    _completed_thread(detector)
    payload = detector._a3b_result_snapshot()

    assert payload["a3b_error_count"] == 1
    assert payload["a3b_last_error"] == "RuntimeError: a3b exploded"
    assert isinstance(payload["a3b_last_error_at"], float)
    assert payload["a3b_last_success_at"] is None
    assert payload["a3b_last_attempt_frame_idx"] == 11
    assert payload["a3b_last_attempt_timestamp"] == pytest.approx(2.75)
    assert payload["a3b_source_frame_idx"] is None
    assert payload["a3b_source_timestamp"] is None
    assert payload["a3b_result_seq"] == 0


def test_background_exception_invalidates_previous_success_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    previous = _media_payload(detector)
    previous.update(
        {
            "worker_marker": "previous-success",
            "p_media_bbox": [8, 8, 56, 56],
        }
    )
    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: previous)

    detector.process(_item(frame_idx=21, timestamp=4.0))
    _completed_thread(detector)
    successful = detector._a3b_result_snapshot()
    assert successful["worker_marker"] == "previous-success"
    assert successful["a3b_result_seq"] == 1

    def failing_compute(*_args: object) -> dict[str, object]:
        raise RuntimeError("newer a3b attempt failed")

    monkeypatch.setattr(detector, "_compute_a3b", failing_compute)
    detector.process(_item(frame_idx=22, timestamp=4.25))
    _completed_thread(detector)
    failed = detector._a3b_result_snapshot()

    assert detector._a3b_bg_result is None
    assert "worker_marker" not in failed
    assert failed["p_media_bbox"] is None
    assert failed["a3b_error_count"] == 1
    assert failed["a3b_last_error"] == "RuntimeError: newer a3b attempt failed"
    assert failed["a3b_last_success_at"] == successful["a3b_last_success_at"]
    assert failed["a3b_source_frame_idx"] == 21
    assert failed["a3b_last_attempt_frame_idx"] == 22
    assert failed["a3b_result_seq"] == 1


def test_non_mapping_result_does_not_relabel_previous_payload_as_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    previous = _media_payload(detector)
    previous["worker_marker"] = "good"
    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: previous)

    detector.process(_item(frame_idx=20, timestamp=3.75))
    _completed_thread(detector)
    before = detector._a3b_result_snapshot()
    assert before["worker_marker"] == "good"
    assert before["a3b_result_seq"] == 1
    assert before["a3b_result_fresh"] is True

    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: None)
    detector.process(_item(frame_idx=21, timestamp=4.0))
    _completed_thread(detector)
    after = detector._a3b_result_snapshot()

    assert detector._a3b_bg_result is None
    assert "worker_marker" not in after
    assert after["a3b_result_seq"] == 1
    assert after["a3b_result_fresh"] is False
    assert after["a3b_error_count"] == 1
    assert after["a3b_last_error"].startswith("TypeError:")
    assert after["a3b_last_attempt_frame_idx"] == 21
    assert after["a3b_global_live_worker_count"] == 0


def test_hung_worker_timeout_invalidates_cache_and_rejects_late_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_worker_timeout_s=0.05,
        static_image_max_retired_workers=2,
    )
    previous = _media_payload(detector)
    previous["worker_marker"] = "previous-success"
    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: previous)

    detector.process(_item(frame_idx=21, timestamp=4.0))
    _completed_thread(detector)
    successful = detector._a3b_result_snapshot()
    previous_generation = successful["a3b_generation"]
    assert successful["worker_marker"] == "previous-success"
    assert successful["a3b_result_seq"] == 1

    started = threading.Event()
    release = threading.Event()

    def hanging_compute(*_args: object) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=2.0)
        return {"worker_marker": "late-timeout-result"}

    monkeypatch.setattr(detector, "_compute_a3b", hanging_compute)
    detector.process(_item(frame_idx=22, timestamp=4.25))
    assert started.wait(timeout=2.0)
    worker = detector._a3b_bg_thread
    assert worker is not None

    active = detector._a3b_result_snapshot()
    assert active["worker_marker"] == "previous-success"
    assert active["a3b_active_worker_count"] == 1
    assert active["a3b_active_worker_frame_idx"] == 22
    assert active["a3b_active_worker_timestamp"] == pytest.approx(4.25)
    assert active["a3b_last_attempt_frame_idx"] == 22

    time.sleep(0.07)
    timed_out = detector._a3b_result_snapshot()

    assert detector._a3b_bg_result is None
    assert "worker_marker" not in timed_out
    assert timed_out["a3b_generation"] == previous_generation + 1
    assert timed_out["a3b_result_seq"] == 1
    assert timed_out["a3b_error_count"] == 1
    assert timed_out["a3b_timed_out_worker_count"] == 1
    assert timed_out["a3b_last_error"].startswith(
        "TimeoutError: A3b background worker exceeded"
    )
    assert timed_out["a3b_active_worker_count"] == 0
    assert timed_out["a3b_retired_worker_count"] == 1
    assert timed_out["a3b_schedule_blocked"] is False

    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    final = detector._a3b_result_snapshot()
    assert final["a3b_retired_worker_count"] == 0
    assert detector._a3b_bg_result is None
    assert "late-timeout-result" not in final.values()


def test_successful_result_is_not_timed_out_while_worker_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_worker_timeout_s=0.01,
        static_image_result_lease_s=5.0,
    )
    compute_started = threading.Event()
    allow_compute = threading.Event()
    release_started = threading.Event()
    allow_release = threading.Event()
    original_release = (
        rebuilt_detector_module._release_a3b_global_worker_token
    )

    def compute_a3b(*_args: object) -> dict[str, object]:
        compute_started.set()
        assert allow_compute.wait(timeout=2.0)
        return {"worker_marker": "published-before-thread-exit"}

    def blocking_release(token: int) -> None:
        release_started.set()
        try:
            assert allow_release.wait(timeout=2.0)
        finally:
            original_release(token)

    monkeypatch.setattr(detector, "_compute_a3b", compute_a3b)
    monkeypatch.setattr(
        rebuilt_detector_module,
        "_release_a3b_global_worker_token",
        blocking_release,
    )

    detector.process(_item(frame_idx=24, timestamp=4.75))
    assert compute_started.wait(timeout=2.0)
    worker = detector._a3b_bg_thread
    assert worker is not None
    time.sleep(0.02)
    allow_compute.set()
    assert release_started.wait(timeout=2.0)
    assert worker.is_alive()

    payload = detector._a3b_result_snapshot()
    assert payload["worker_marker"] == "published-before-thread-exit"
    assert payload["a3b_active_worker_count"] == 0
    assert payload["a3b_live_worker_count"] == 1
    assert payload["a3b_error_count"] == 0
    assert payload["a3b_timed_out_worker_count"] == 0
    assert payload["a3b_result_fresh"] is True

    allow_release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert detector._a3b_result_snapshot()["a3b_live_worker_count"] == 0


def test_success_result_expires_by_monotonic_lease_without_active_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_result_lease_s=0.05,
        static_image_worker_timeout_s=60.0,
    )
    payload = _media_payload(detector)
    payload["worker_marker"] = "leased-result"
    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: payload)

    detector.process(_item(frame_idx=25, timestamp=5.0))
    _completed_thread(detector)
    fresh = detector._a3b_result_snapshot()
    assert fresh["worker_marker"] == "leased-result"
    assert fresh["a3b_result_fresh"] is True
    assert fresh["a3b_result_age_s"] < fresh["a3b_result_lease_s"]

    time.sleep(0.07)
    expired = detector._a3b_result_snapshot()
    assert detector._a3b_bg_result is None
    assert "worker_marker" not in expired
    assert expired["a3b_result_fresh"] is False
    assert expired["a3b_result_expired_count"] == 1
    assert expired["a3b_result_seq"] == 1
    assert expired["a3b_error_count"] == 0


def test_repeated_reset_stops_scheduling_at_retired_worker_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_worker_timeout_s=60.0,
        static_image_max_retired_workers=2,
    )
    started = [threading.Event(), threading.Event(), threading.Event()]
    releases = [threading.Event(), threading.Event()]
    call_lock = threading.Lock()
    call_count = 0

    def hanging_compute(*_args: object) -> dict[str, object]:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        started[call_index].set()
        assert releases[call_index].wait(timeout=2.0)
        return {"worker_marker": call_index}

    monkeypatch.setattr(detector, "_compute_a3b", hanging_compute)

    detector.process(_item(frame_idx=31, timestamp=5.0))
    assert started[0].wait(timeout=2.0)
    first = detector._a3b_bg_thread
    assert first is not None
    detector.reset()

    detector.process(_item(frame_idx=32, timestamp=5.25))
    assert started[1].wait(timeout=2.0)
    second = detector._a3b_bg_thread
    assert second is not None and second is not first
    detector.reset()

    blocked = detector.process(_item(frame_idx=33, timestamp=5.5))
    blocked_health = blocked.details["a3b"]
    assert call_count == 2
    assert not started[2].wait(timeout=0.05)
    assert detector._a3b_bg_thread is None
    assert blocked_health["a3b_retired_worker_count"] == 2
    assert blocked_health["a3b_max_retired_workers"] == 2
    assert blocked_health["a3b_schedule_blocked"] is True
    assert (
        blocked_health["a3b_schedule_blocked_reason"]
        == "retired_worker_limit"
    )

    detector.reset()
    detector.process(_item(frame_idx=34, timestamp=5.75))
    assert call_count == 2
    assert detector._a3b_result_snapshot()["a3b_retired_worker_count"] == 2

    for release in releases:
        release.set()
    for worker in (first, second):
        worker.join(timeout=2.0)
        assert not worker.is_alive()
    final = detector._a3b_result_snapshot()
    assert final["a3b_retired_worker_count"] == 0
    assert final["a3b_schedule_blocked"] is False


def test_consecutive_worker_timeouts_allow_one_replacement_then_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_worker_timeout_s=0.05,
        static_image_max_retired_workers=2,
    )
    started = [threading.Event(), threading.Event(), threading.Event()]
    releases = [threading.Event(), threading.Event()]
    call_lock = threading.Lock()
    call_count = 0

    def hanging_compute(*_args: object) -> dict[str, object]:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        started[call_index].set()
        assert releases[call_index].wait(timeout=2.0)
        return {"worker_marker": call_index}

    monkeypatch.setattr(detector, "_compute_a3b", hanging_compute)

    detector.process(_item(frame_idx=41, timestamp=6.0))
    assert started[0].wait(timeout=2.0)
    first = detector._a3b_bg_thread
    assert first is not None

    time.sleep(0.07)
    detector.process(_item(frame_idx=42, timestamp=6.25))
    assert started[1].wait(timeout=2.0)
    second = detector._a3b_bg_thread
    assert second is not None and second is not first
    after_first_timeout = detector._a3b_result_snapshot()
    assert after_first_timeout["a3b_timed_out_worker_count"] == 1
    assert after_first_timeout["a3b_retired_worker_count"] == 1
    assert after_first_timeout["a3b_active_worker_count"] == 1

    time.sleep(0.07)
    blocked = detector.process(_item(frame_idx=43, timestamp=6.5))
    health = blocked.details["a3b"]
    assert call_count == 2
    assert not started[2].wait(timeout=0.05)
    assert detector._a3b_bg_thread is None
    assert health["a3b_timed_out_worker_count"] == 2
    assert health["a3b_error_count"] == 2
    assert health["a3b_retired_worker_count"] == 2
    assert health["a3b_schedule_blocked"] is True
    assert health["a3b_schedule_blocked_reason"] == "retired_worker_limit"

    for release in releases:
        release.set()
    for worker in (first, second):
        worker.join(timeout=2.0)
        assert not worker.is_alive()
    assert detector._a3b_result_snapshot()["a3b_retired_worker_count"] == 0


def test_global_worker_budget_survives_detector_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detectors = [
        _detector(
            monkeypatch,
            static_image_worker_timeout_s=60.0,
            static_image_global_worker_limit=10,
        )
        for _ in range(3)
    ]
    started = [threading.Event(), threading.Event()]
    releases = [threading.Event(), threading.Event()]
    workers: list[threading.Thread] = []

    for index, detector in enumerate(detectors[:2]):
        def hanging_compute(
            *_args: object,
            worker_index: int = index,
        ) -> dict[str, object]:
            started[worker_index].set()
            assert releases[worker_index].wait(timeout=2.0)
            return {"worker_marker": worker_index}

        monkeypatch.setattr(detector, "_compute_a3b", hanging_compute)
        detector.process(_item(frame_idx=51 + index, timestamp=8.0 + index))
        assert started[index].wait(timeout=2.0)
        worker = detector._a3b_bg_thread
        assert worker is not None
        workers.append(worker)

    third_called = threading.Event()
    monkeypatch.setattr(
        detectors[2],
        "_compute_a3b",
        lambda *_args: third_called.set() or {"unexpected": True},
    )
    detectors[2].process(_item(frame_idx=53, timestamp=10.0))
    blocked = detectors[2]._a3b_result_snapshot()

    assert not third_called.wait(timeout=0.05)
    assert detectors[2]._a3b_bg_thread is None
    assert blocked["a3b_global_live_worker_count"] == 2
    assert blocked["a3b_global_worker_limit"] == 2
    assert blocked["a3b_worker_limit_scope"] == "process"
    assert blocked["a3b_worker_rejected_count"] == 1
    assert blocked["a3b_schedule_blocked"] is True
    assert blocked["a3b_schedule_blocked_reason"] == "global_worker_limit"

    for release in releases:
        release.set()
    for worker in workers:
        worker.join(timeout=2.0)
        assert not worker.is_alive()
    assert (
        detectors[2]._a3b_result_snapshot()[
            "a3b_global_live_worker_count"
        ]
        == 0
    )


def test_success_payload_exposes_source_and_result_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    monkeypatch.setattr(detector, "_compute_a3b", lambda *_args: _media_payload(detector))

    detector.process(_item(frame_idx=23, timestamp=4.5))
    _completed_thread(detector)
    payload = detector._a3b_result_snapshot()

    assert payload["a3b_error_count"] == 0
    assert payload["a3b_last_error"] is None
    assert isinstance(payload["a3b_last_success_at"], float)
    assert payload["a3b_source_frame_idx"] == 23
    assert payload["a3b_source_timestamp"] == pytest.approx(4.5)
    assert payload["a3b_source_fps"] == pytest.approx(15.0)
    assert payload["a3b_source_interval_frames"] == 1
    assert payload["a3b_last_attempt_frame_idx"] == 23
    assert payload["a3b_last_attempt_timestamp"] == pytest.approx(4.5)
    assert payload["a3b_result_seq"] == 1
    assert payload["a3b_active_worker_count"] == 0
    assert payload["a3b_retired_worker_count"] == 0


def test_close_invalidates_pre_close_cached_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    monkeypatch.setattr(
        detector,
        "_compute_a3b",
        lambda *_args: {"worker_marker": "pre-close"},
    )

    detector.process(_item(frame_idx=49, timestamp=7.75))
    _completed_thread(detector)
    assert detector._a3b_result_snapshot()["worker_marker"] == "pre-close"

    previous_generation = detector._a3b_generation
    detector.close()
    payload = detector._a3b_result_snapshot()

    assert payload["a3b_generation"] == previous_generation + 1
    assert payload["a3b_result_seq"] == 0
    assert payload["a3b_source_frame_idx"] is None
    assert detector._a3b_bg_result is None
    assert "worker_marker" not in payload


def test_close_joins_current_and_retired_workers_with_one_total_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def delayed_compute(*_args: object) -> dict[str, object]:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        started[call_index].set()
        assert release.wait(timeout=2.0)
        return {"worker_marker": call_index}

    monkeypatch.setattr(detector, "_compute_a3b", delayed_compute)

    detector.process(_item(frame_idx=51, timestamp=8.0))
    assert started[0].wait(timeout=2.0)
    first_worker = detector._a3b_bg_thread
    assert first_worker is not None
    detector.reset()

    detector.process(_item(frame_idx=52, timestamp=8.5))
    assert started[1].wait(timeout=2.0)
    second_worker = detector._a3b_bg_thread
    assert second_worker is not None

    release_timer = threading.Timer(0.05, release.set)
    release_timer.start()
    started_at = time.monotonic()
    detector.close()
    elapsed = time.monotonic() - started_at
    release_timer.join(timeout=1.0)

    assert elapsed < 1.2
    assert not first_worker.is_alive()
    assert not second_worker.is_alive()
    payload = detector._a3b_result_snapshot()
    assert payload["a3b_active_worker_count"] == 0
    assert payload["a3b_retired_worker_count"] == 0
    assert detector._a3b_bg_thread is None
    assert detector._a3b_retired_threads == []
    assert detector._a3b_bg_result is None

    repeated_at = time.monotonic()
    detector.close()
    assert time.monotonic() - repeated_at < 0.1


def test_close_returns_within_budget_when_worker_remains_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(monkeypatch)
    monkeypatch.setattr(detector, "_A3B_CLOSE_JOIN_BUDGET_SECONDS", 0.10)
    started = threading.Event()
    release = threading.Event()

    def hanging_compute(*_args: object) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=2.0)
        return {"worker_marker": "released"}

    monkeypatch.setattr(detector, "_compute_a3b", hanging_compute)

    detector.process(_item(frame_idx=61, timestamp=9.0))
    assert started.wait(timeout=2.0)
    worker = detector._a3b_bg_thread
    assert worker is not None

    started_at = time.monotonic()
    detector.close()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert worker.is_alive()
    payload = detector._a3b_result_snapshot()
    assert payload["a3b_active_worker_count"] == 0
    assert payload["a3b_retired_worker_count"] == 1
    assert detector._a3b_retired_threads == [worker]

    repeated_at = time.monotonic()
    detector.close()
    assert time.monotonic() - repeated_at < 0.5
    assert detector._a3b_retired_threads == [worker]

    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    final_payload = detector._a3b_result_snapshot()
    assert final_payload["a3b_retired_worker_count"] == 0
    assert detector._a3b_retired_threads == []
    assert detector._a3b_bg_result is None


def test_joint_decision_exposes_media_gate_and_temporal_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_interval=1000,
        rebuilt_a3b_media_run_floor=3,
    )
    media_payload = _media_payload(detector)
    with detector._a3b_bg_lock:
        detector._a3b_bg_result = media_payload

    result = detector.process(_item(frame_idx=31, timestamp=6.0))
    joint = result.details["joint_decision"]

    assert joint["media_tighten_gate_enabled"] is True
    assert joint["media_tighten_candidate_score"] == pytest.approx(0.80)
    assert joint["media_tighten_edge"] == pytest.approx(0.50)
    assert joint["media_tighten_border_contrast"] == pytest.approx(0.90)
    assert joint["media_tighten_candidate_pass"] is True
    assert joint["media_tighten_edge_pass"] is True
    assert joint["media_tighten_border_pass"] is True
    assert joint["media_tighten_aspect_pass"] is True
    assert joint["media_gate_ok"] is True
    assert joint["media_run"] == 1
    assert joint["media_run_gap"] == 0
    assert joint["media_run_floor"] == 3
    assert joint["media_count"] == 0
    assert joint["media_hit_required"] >= 2


@pytest.mark.parametrize(
    "bbox",
    [
        [2, 20, 62, 35],
        [20, 2, 35, 62],
    ],
    ids=["too-wide", "too-narrow"],
)
def test_media_confirmation_requires_current_aspect_gate(
    monkeypatch: pytest.MonkeyPatch,
    bbox: list[int],
) -> None:
    detector = _detector(
        monkeypatch,
        static_image_interval=1000,
        rebuilt_a3b_media_run_floor=1,
    )
    try:
        for seq in range(1, 5):
            payload = _media_payload(detector)
            published_at = time.time()
            with detector._a3b_bg_lock:
                detector._a3b_result_seq = seq
                detector._a3b_source_frame_idx = seq
                detector._a3b_source_timestamp = seq / 30.0
                detector._a3b_last_success_at = published_at
                detector._a3b_result_published_at = published_at
                detector._a3b_result_published_monotonic = time.monotonic()
                detector._a3b_bg_result = payload
            valid = detector.process(
                _item(frame_idx=seq, timestamp=seq / 30.0)
            )

        assert valid.details["joint_decision"]["media_confirmed"] is True

        invalid_payload = _media_payload(detector)
        invalid_payload["p_media_bbox"] = bbox
        published_at = time.time()
        with detector._a3b_bg_lock:
            detector._a3b_result_seq = 5
            detector._a3b_source_frame_idx = 5
            detector._a3b_source_timestamp = 5 / 30.0
            detector._a3b_last_success_at = published_at
            detector._a3b_result_published_at = published_at
            detector._a3b_result_published_monotonic = time.monotonic()
            detector._a3b_bg_result = invalid_payload

        invalid = detector.process(
            _item(frame_idx=5, timestamp=5 / 30.0)
        ).details["joint_decision"]

        assert invalid["media_tighten_aspect_pass"] is False
        assert invalid["media_gate_ok"] is False
        assert invalid["media_count"] >= invalid["media_hit_required"]
        assert invalid["media_confirmed"] is False
    finally:
        detector.close()

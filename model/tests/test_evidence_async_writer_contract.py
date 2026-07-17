from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from defense.runtime import evidence as evidence_module
from defense.runtime.evidence import EvidenceSession


def _frame(value: int = 0) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _disable_clip_writer(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_module,
        "_write_browser_mp4_from_frames",
        lambda *args, **kwargs: (
            None,
            {
                "evidence_clip_status": "disabled_for_test",
                "evidence_clip_browser_playable": False,
            },
        ),
    )


def _update_a3b(
    session: EvidenceSession,
    frame_idx: int,
    *,
    active: bool,
) -> list[dict]:
    return session.update(
        frame_idx=frame_idx,
        frame=_frame(frame_idx),
        info={},
        ppe={},
        status={
            "a3b_triggered": active,
            "a3b_event_score": 0.9 if active else 0.0,
            "a3b_triggered_source": "contract" if active else "",
        },
    )


def test_update_only_enqueues_and_finalize_stays_fifo_behind_frames(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_imwrite = evidence_module.cv2.imwrite
    writer_entered = threading.Event()
    release_writer = threading.Event()
    call_order: list[str] = []

    def slow_imwrite(path, frame, params):
        call_order.append("frame")
        writer_entered.set()
        assert release_writer.wait(2.0)
        return original_imwrite(path, frame, params)

    def clip_writer(*args, **kwargs):
        call_order.append("clip")
        return None, {
            "evidence_clip_status": "disabled_for_test",
            "evidence_clip_browser_playable": False,
        }

    monkeypatch.setattr(evidence_module.cv2, "imwrite", slow_imwrite)
    monkeypatch.setattr(
        evidence_module,
        "_write_browser_mp4_from_frames",
        clip_writer,
    )
    session = EvidenceSession(
        source_type="file",
        source="async.mp4",
        profile="test",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
        writer_drain_timeout_s=2.0,
    )

    started = time.perf_counter()
    assert _update_a3b(session, 1, active=True) == []
    assert (time.perf_counter() - started) < 0.25
    assert writer_entered.wait(1.0)

    started = time.perf_counter()
    completed = _update_a3b(session, 2, active=False)
    assert (time.perf_counter() - started) < 0.25
    assert len(completed) == 1
    event = completed[0]
    assert event["evidence_persistence_status"] == "pending"
    assert call_order == ["frame"]

    release_writer.set()
    assert session.close() == []

    assert call_order == ["frame", "clip"]
    assert event["evidence_persistence_status"] == "complete"
    assert event["evidence_persistence_pending"] is False
    assert (Path(event["event_dir"]) / "event.json").is_file()
    status = session.writer_status()
    assert status["alive"] is False
    assert status["pending"] == 0


def test_bounded_queue_pressure_is_fail_visible_and_marks_partial_event(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    original_imwrite = evidence_module.cv2.imwrite
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def slow_first_imwrite(path, frame, params):
        writer_entered.set()
        assert release_writer.wait(2.0)
        return original_imwrite(path, frame, params)

    monkeypatch.setattr(
        evidence_module.cv2,
        "imwrite",
        slow_first_imwrite,
    )
    session = EvidenceSession(
        source_type="file",
        source="queue-pressure.mp4",
        profile="test",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
        max_frames_per_event=20,
        writer_queue_capacity=8,
        writer_enqueue_timeout_s=0.01,
        writer_drain_timeout_s=2.0,
    )

    _update_a3b(session, 1, active=True)
    assert writer_entered.wait(1.0)
    for frame_idx in range(2, 10):
        _update_a3b(session, frame_idx, active=True)

    pressured = session.writer_status()
    assert pressured["queue_capacity"] == 8
    assert pressured["queue_full"] >= 1
    assert pressured["failed"] >= 1
    assert "queue_full" in pressured["last_error"]

    release_writer.set()
    event = session.close()[0]

    assert event["evidence_write_attempt_count"] == 9
    assert event["evidence_write_error_count"] >= 1
    assert event["evidence_partial"] is True
    assert event["evidence_complete"] is False
    assert event["evidence_persistence_status"] == "partial"


def test_close_timeout_is_bounded_and_retry_drains_without_false_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    original_imwrite = evidence_module.cv2.imwrite
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def blocked_imwrite(path, frame, params):
        writer_entered.set()
        assert release_writer.wait(2.0)
        return original_imwrite(path, frame, params)

    monkeypatch.setattr(evidence_module.cv2, "imwrite", blocked_imwrite)
    session = EvidenceSession(
        source_type="file",
        source="drain-timeout.mp4",
        profile="test",
        root=tmp_path,
        pre_frames=0,
        post_frames=1,
        sample_every=1,
        writer_drain_timeout_s=0.05,
    )
    _update_a3b(session, 1, active=True)
    assert writer_entered.wait(1.0)

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="drain_timeout"):
        session.close()
    assert (time.perf_counter() - started) < 0.5
    timed_out = session.writer_status()
    assert timed_out["alive"] is True
    assert timed_out["pending"] > 0
    assert "drain_timeout" in timed_out["last_error"]

    release_writer.set()
    assert session.close() == []
    assert session.saved_event_count == 1
    assert session.writer_status()["pending"] == 0


def test_module_a_requires_confirmation_to_start_but_hold_keeps_event_active(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    session = EvidenceSession(
        source_type="file",
        source="physical-confirmation.mp4",
        profile="test",
        root=tmp_path,
        pre_frames=0,
        post_frames=2,
        sample_every=1,
    )

    def update(frame_idx: int, status: dict) -> list[dict]:
        return session.update(
            frame_idx=frame_idx,
            frame=_frame(frame_idx),
            info={},
            ppe={},
            status=status,
        )

    assert update(
        1,
        {
            "physical_alert_confirmed": False,
            "physical_attack_state_active": True,
            "alert_confirmed": True,
            "attack_state_active": True,
            "p_adv": 0.7,
            "reason": "suspect_only",
        },
    ) == []
    assert update(
        2,
        {
            "physical_alert_confirmed": True,
            "physical_attack_state_active": True,
            "p_adv": 0.9,
            "reason": "confirmed",
        },
    ) == []
    assert update(
        3,
        {
            "physical_alert_confirmed": False,
            "physical_attack_state_active": True,
            "p_adv": 0.4,
            "reason": "confirmed_hold",
        },
    ) == []
    assert update(
        4,
        {
            "physical_alert_confirmed": False,
            "physical_attack_state_active": False,
        },
    ) == []
    completed = update(
        5,
        {
            "physical_alert_confirmed": False,
            "physical_attack_state_active": False,
        },
    )
    assert len(completed) == 1
    event = completed[0]

    assert session.close() == []
    assert event["channel"] == "module_a"
    assert event["started_frame"] == 2
    assert event["last_active_frame"] == 3
    assert event["reason"] == "confirmed;confirmed_hold"
    assert session.saved_event_count == 1


def test_a3b_requires_final_confirmation_to_start_but_trigger_hold_keeps_event_active(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_clip_writer(monkeypatch)
    session = EvidenceSession(
        source_type="file",
        source="a3b-confirmation.mp4",
        profile="test",
        root=tmp_path,
        pre_frames=0,
        post_frames=2,
        sample_every=1,
    )

    def update(frame_idx: int, status: dict) -> list[dict]:
        return session.update(
            frame_idx=frame_idx,
            frame=_frame(frame_idx),
            info={},
            ppe={},
            status=status,
        )

    assert update(
        1,
        {
            "a3b_triggered": True,
            "a3b_confirmed_alert": False,
            "a3b_state": "suspect",
            "a3b_event_score": 0.8,
            "a3b_triggered_source": "single_strong",
        },
    ) == []
    assert update(
        2,
        {
            "a3b_triggered": True,
            "a3b_confirmed_alert": True,
            "a3b_state": "confirmed",
            "a3b_event_score": 0.9,
            "a3b_triggered_source": "window_accumulated",
        },
    ) == []
    assert update(
        3,
        {
            "a3b_triggered": True,
            "a3b_confirmed_alert": False,
            "a3b_state": "suspect",
            "a3b_event_score": 0.7,
            "a3b_triggered_source": "trigger_hold",
        },
    ) == []
    assert update(
        4,
        {
            "a3b_triggered": False,
            "a3b_confirmed_alert": False,
            "a3b_state": "normal",
        },
    ) == []
    completed = update(
        5,
        {
            "a3b_triggered": False,
            "a3b_confirmed_alert": False,
            "a3b_state": "normal",
        },
    )

    assert len(completed) == 1
    event = completed[0]
    assert session.close() == []
    assert event["channel"] == "a3b"
    assert event["started_frame"] == 2
    assert event["last_active_frame"] == 3
    assert event["reason"] == "trigger_hold;window_accumulated"

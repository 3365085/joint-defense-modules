from __future__ import annotations

import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import defense.runtime.runner as runner_module
from defense.runtime.backend_pipeline import FramePacket
from defense.runtime.frame_processor import ProcessedFrame
from defense.runtime.runner import MonitorEngine


class _FakePreviewBus:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def clear(self) -> None:
        return None


class _FakeDetectionBus:
    """Deterministic queue that keeps pending packets available after close."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.packets: deque[FramePacket] = deque()
        self.closed = False
        self.dropped = 0

    def push(self, packet: FramePacket) -> None:
        with self.condition:
            if self.closed:
                raise RuntimeError("cannot push to a closed fake detection bus")
            self.packets.append(packet)
            self.condition.notify_all()

    def pop_latest(
        self,
        last_seq: int,
        timeout: float = 0.2,
    ) -> FramePacket | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.condition:
            while not self.packets:
                if self.closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.condition.wait(timeout=min(0.05, remaining))
            while self.packets:
                packet = self.packets.popleft()
                if packet.seq > last_seq:
                    return packet
            return None

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def clear(self) -> None:
        with self.condition:
            self.packets.clear()
            self.condition.notify_all()


class _FakePipeline:
    warmup_frames = 0

    def warmup(self, _frames: int) -> None:
        return None

    def reset(self) -> None:
        return None


class _FakeCache:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            pipeline=_FakePipeline(),
            config={
                "runtime": {
                    "detector_process_fps_cap": 30,
                    "evidence_enabled": True,
                    "evidence_fps": 30,
                    "release_pipeline_cache_on_file_end": False,
                }
            },
            backend="fake",
            model_family="fake",
            artifact_path="fake.engine",
            warmup_error="",
        )

    def get(self, **_kwargs: Any) -> Any:
        return self.bundle

    def clear(self) -> None:
        return None


class _ProcessHarness:
    def __init__(
        self,
        *,
        block_processor: bool = False,
        block_evidence_close: bool = False,
        writer_failed: bool = False,
    ) -> None:
        self.processor_started = threading.Event()
        self.allow_processor = threading.Event()
        self.evidence_close_started = threading.Event()
        self.allow_evidence_close = threading.Event()
        self.evidence_closed = threading.Event()
        self.processed_frames: list[int] = []
        self.evidence_updated_frames: list[int] = []
        self.writer_failed = bool(writer_failed)
        if not block_processor:
            self.allow_processor.set()
        if not block_evidence_close:
            self.allow_evidence_close.set()


def _packet(seq: int, *, epoch: int = 1) -> FramePacket:
    frame = np.full((8, 8, 3), seq, dtype=np.uint8)
    return FramePacket(
        seq=seq,
        frame_idx=seq,
        source_time_s=seq / 30.0,
        wall_time_ms=0.0,
        epoch=epoch,
        frame=frame,
        width=8,
        height=8,
        fps=30.0,
        flags={},
    )


def _install_process_fakes(
    monkeypatch: pytest.MonkeyPatch,
    harness: _ProcessHarness,
) -> None:
    class FakeProcessor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def process(
            self,
            frame: np.ndarray,
            **kwargs: Any,
        ) -> ProcessedFrame:
            frame_idx = int(kwargs["frame_idx"])
            harness.processor_started.set()
            if not harness.allow_processor.wait(timeout=2.0):
                raise RuntimeError("test processor release timed out")
            harness.processed_frames.append(frame_idx)
            source_time_s = float(kwargs["video_time_s"])
            return ProcessedFrame(
                frame_idx=frame_idx,
                frame_640=frame,
                rendered_frame=frame,
                info={},
                ppe={},
                ppe_tracks=[],
                status={
                    "frame_idx": frame_idx,
                    "source_type": str(kwargs["source_type"]),
                    "source_time_s": source_time_s,
                    "video_time_s": source_time_s,
                    "a3b_triggered": False,
                },
            )

    class FakeEvidence:
        enabled = True
        session_dir = None
        manifest_path = None
        saved_event_count = 0

        def __init__(self, **_kwargs: Any) -> None:
            return None

        def update(self, **kwargs: Any) -> list[dict[str, Any]]:
            harness.evidence_updated_frames.append(int(kwargs["frame_idx"]))
            return []

        def close(self) -> list[dict[str, Any]]:
            harness.evidence_close_started.set()
            if not harness.allow_evidence_close.wait(timeout=2.0):
                raise RuntimeError("test evidence close release timed out")
            harness.evidence_closed.set()
            return []

        def writer_status(self) -> dict[str, Any]:
            closed = harness.evidence_closed.is_set()
            return {
                "enabled": True,
                "alive": not closed,
                "queue_capacity": 32,
                "pending": 0,
                "completed": 3 if closed else len(
                    harness.evidence_updated_frames
                ),
                "failed": 1 if harness.writer_failed else 0,
                "queue_full": 0,
                "drain_ms": 0.25 if closed else 0.0,
                "last_error": (
                    "simulated_writer_failure"
                    if harness.writer_failed
                    else ""
                ),
            }

    monkeypatch.setattr(runner_module, "FrameProcessor", FakeProcessor)
    monkeypatch.setattr(runner_module, "EvidenceSession", FakeEvidence)


def _prepare_engine(
    *,
    run_id: int,
    cache: _FakeCache | None = None,
) -> MonitorEngine:
    engine = MonitorEngine(cache or _FakeCache())  # type: ignore[arg-type]
    engine.run_id = run_id
    engine._source_epoch = 1
    engine.process_done_event.clear()
    engine.status.update(
        {
            "run_id": run_id,
            "running": True,
            "source_type": "file",
            "source": "fake.mp4",
            "source_epoch": 1,
            "source_eof_reached": False,
            "source_ended": False,
        }
    )
    return engine


def _run_process_loop(
    engine: MonitorEngine,
    detection_bus: _FakeDetectionBus,
) -> None:
    engine._backend_process_loop(
        engine.run_id,
        _FakePreviewBus(),  # type: ignore[arg-type]
        detection_bus,  # type: ignore[arg-type]
        "file",
        "fake.mp4",
        "default",
        True,
        {},
        {},
        {},
    )


def _wait_for_status(
    engine: MonitorEngine,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    with engine.condition:
        while not predicate(engine.status):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                pytest.fail(f"timed out waiting for status: {engine.status}")
            engine.condition.wait(timeout=min(0.05, remaining))
        return dict(engine.status)


def test_file_eof_waits_for_last_packet_and_evidence_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ProcessHarness(
        block_processor=True,
        block_evidence_close=True,
    )
    _install_process_fakes(monkeypatch, harness)
    engine = _prepare_engine(run_id=7)
    engine.detector_drain_timeout_s = 2.0
    preview_bus = _FakePreviewBus()
    detection_bus = _FakeDetectionBus()
    detection_bus.push(_packet(1))

    process_thread = threading.Thread(
        target=_run_process_loop,
        args=(engine, detection_bus),
        name="drain-contract-process",
        daemon=True,
    )
    finalize_thread = threading.Thread(
        target=engine._finalize_capture_run,
        kwargs={
            "run_id": 7,
            "preview_bus": preview_bus,
            "detection_bus": detection_bus,
            "source_type": "file",
            "source_ended_candidate": True,
        },
        name="drain-contract-finalize",
        daemon=True,
    )

    process_thread.start()
    assert harness.processor_started.wait(timeout=1.0)
    finalize_thread.start()
    try:
        draining = _wait_for_status(
            engine,
            lambda status: bool(
                status.get("source_eof_reached")
                and status.get("detector_drain_active")
            ),
        )
        assert draining["source_ended"] is False
        assert draining["running"] is True
        assert draining["process_done"] is False

        harness.allow_processor.set()
        assert harness.evidence_close_started.wait(timeout=1.0)
        evidence_draining = _wait_for_status(
            engine,
            lambda status: bool(status.get("evidence_drain_active")),
        )
        assert evidence_draining["source_ended"] is False
        assert evidence_draining["detector_drain_active"] is True
        assert evidence_draining["process_done"] is False

        harness.allow_evidence_close.set()
        process_thread.join(timeout=1.0)
        finalize_thread.join(timeout=1.0)
        assert not process_thread.is_alive()
        assert not finalize_thread.is_alive()
    finally:
        harness.allow_processor.set()
        harness.allow_evidence_close.set()
        process_thread.join(timeout=1.0)
        finalize_thread.join(timeout=1.0)

    status = engine.get_status()
    assert harness.processed_frames == [1]
    assert harness.evidence_updated_frames == [1]
    assert harness.evidence_closed.is_set()
    assert status["source_eof_reached"] is True
    assert status["source_ended"] is True
    assert status["running"] is False
    assert status["process_done"] is True
    assert status["detector_drain_active"] is False
    assert status["detector_drain_completed"] is True
    assert status["detector_drain_timed_out"] is False
    assert status["detector_drain_ms"] >= 0.0
    assert status["detector_drain_failed_reason"] == ""
    assert status["evidence_drain_active"] is False
    assert status["evidence_drain_completed"] is True
    assert status["evidence_drain_failed"] is False
    assert status["evidence_drain_ms"] >= 0.0
    assert status["evidence_drain_error"] == ""
    assert status["evidence_writer_enabled"] is True
    assert status["evidence_writer_alive"] is False
    assert status["evidence_writer_queue_capacity"] == 32
    assert status["evidence_writer_pending"] == 0
    assert status["evidence_writer_failed"] == 0
    assert status["evidence_writer_queue_full"] == 0
    assert status["evidence_writer_last_error"] == ""
    assert preview_bus.closed is True
    assert detection_bus.closed is True


def test_file_eof_drain_timeout_is_fail_visible() -> None:
    engine = _prepare_engine(run_id=8)
    engine.detector_drain_timeout_s = 0.01
    preview_bus = _FakePreviewBus()
    detection_bus = _FakeDetectionBus()

    started = time.perf_counter()
    engine._finalize_capture_run(
        run_id=8,
        preview_bus=preview_bus,  # type: ignore[arg-type]
        detection_bus=detection_bus,  # type: ignore[arg-type]
        source_type="file",
        source_ended_candidate=True,
    )
    elapsed = time.perf_counter() - started

    status = engine.get_status()
    assert elapsed < 0.75
    assert status["source_eof_reached"] is True
    assert status["source_ended"] is False
    assert status["running"] is False
    assert status["process_done"] is False
    assert status["detector_drain_active"] is False
    assert status["detector_drain_completed"] is False
    assert status["detector_drain_timed_out"] is True
    assert status["detector_drain_ms"] >= 0.0
    assert status["detector_drain_failed_reason"].startswith(
        "detector_drain_timeout:"
    )
    assert status["warning"] == "detector_drain_timeout"
    assert status["detector_drain_failed_reason"] in status["secondary_errors"]
    assert status["evidence_drain_active"] is False
    assert status["evidence_drain_completed"] is False
    assert status["preview_mode"] == "source_drain_failed"
    assert status["detector_pipeline_mode"] == "drain_failed"


def test_detector_cycle_metrics_publish_non_negative_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ProcessHarness()
    _install_process_fakes(monkeypatch, harness)
    engine = _prepare_engine(run_id=9)
    detection_bus = _FakeDetectionBus()
    for seq in range(1, 4):
        detection_bus.push(_packet(seq))
    detection_bus.close()

    _run_process_loop(engine, detection_bus)

    status = engine.get_status()
    assert harness.processed_frames == [1, 2, 3]
    assert harness.evidence_updated_frames == [1, 2, 3]
    assert status["process_done"] is True
    for metric in (
        "detector_cycle_ms",
        "evidence_update_ms",
        "overlay_status_publish_ms",
    ):
        assert float(status[metric]) >= 0.0
        distribution = status[f"{metric}_distribution"]
        assert distribution["count"] == 3
        for statistic in ("latest", "mean", "p50", "p95", "max"):
            assert float(distribution[statistic]) >= 0.0
        assert float(distribution["p95"]) >= float(distribution["p50"])

    assert status["detector_cycle_ms"] >= status["evidence_update_ms"]
    assert (
        status["detector_cycle_ms"]
        >= status["overlay_status_publish_ms"]
    )
    assert status["evidence_writer_enabled"] is True
    assert status["evidence_writer_queue_capacity"] == 32
    assert status["evidence_writer_failed"] == 0
    assert status["evidence_writer_queue_full"] == 0


def test_explicit_stop_is_not_reported_as_normal_eof() -> None:
    engine = _prepare_engine(run_id=10)
    preview_bus = _FakePreviewBus()
    detection_bus = _FakeDetectionBus()
    engine.preview_bus = preview_bus  # type: ignore[assignment]
    engine.detection_bus = detection_bus  # type: ignore[assignment]

    engine.stop(release_pipeline_cache=False)

    status = engine.get_status()
    assert engine.stop_event.is_set()
    assert status["source_eof_reached"] is False
    assert status["source_ended"] is False
    assert status["running"] is False
    assert status["preview_mode"] == "stopped"
    assert status["detector_pipeline_mode"] == "idle"
    assert preview_bus.closed is True
    assert detection_bus.closed is True


def test_evidence_writer_failure_is_fail_visible_after_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ProcessHarness(writer_failed=True)
    _install_process_fakes(monkeypatch, harness)
    engine = _prepare_engine(run_id=11)
    detection_bus = _FakeDetectionBus()
    detection_bus.push(_packet(1))
    detection_bus.close()

    _run_process_loop(engine, detection_bus)

    status = engine.get_status()
    assert status["process_done"] is True
    assert status["source_ended"] is False
    assert status["evidence_drain_completed"] is False
    assert status["evidence_drain_failed"] is True
    assert status["evidence_writer_pending"] == 0
    assert status["evidence_writer_failed"] == 1
    assert status["evidence_writer_last_error"] == "simulated_writer_failure"
    assert status["evidence_drain_error"].startswith(
        "evidence_writer_unhealthy_after_drain:"
    )
    assert "evidence writer drain failed" in status["error"]

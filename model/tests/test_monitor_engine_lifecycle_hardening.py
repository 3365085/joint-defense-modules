from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import defense.runtime.runner as runner_module
from defense.runtime.backend_pipeline import DetectionBus, FramePacket, PreviewBus
from defense.runtime.config import load_runtime_config
from defense.runtime.runner import MonitorEngine


def _packet(seq: int, *, epoch: int, value: int = 0) -> FramePacket:
    frame = np.full((8, 8, 3), value, dtype=np.uint8)
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


class _GuardedCache:
    def __init__(self) -> None:
        self.get_count = 0
        self.clear_count = 0

    def get(self, **_kwargs: Any) -> Any:
        self.get_count += 1
        raise AssertionError("a blocked restart must not touch the pipeline cache")

    def clear(self) -> None:
        self.clear_count += 1


def test_stop_timeout_retains_thread_and_blocks_cache_reuse_until_exit() -> None:
    cache = _GuardedCache()
    engine = MonitorEngine(cache)  # type: ignore[arg-type]
    engine.thread_join_timeout_s = 0.01
    release = threading.Event()
    started = threading.Event()

    def worker() -> None:
        started.set()
        release.wait(timeout=2.0)

    thread = threading.Thread(target=worker, name="blocked-detector")
    thread.start()
    assert started.wait(timeout=1.0)
    engine.process_thread = thread
    engine.status["running"] = True

    engine.stop()

    assert engine.process_thread is thread
    assert engine.stop_event.is_set()
    assert cache.clear_count == 0
    assert engine.status["stop_threads_pending"] == ["blocked-detector"]
    assert engine.status["pipeline_cache_release_deferred"] is True

    with pytest.raises(RuntimeError, match="previous worker threads are still running"):
        engine.start(source_type="camera", source="0")

    assert engine.stop_event.is_set()
    assert engine.process_thread is thread
    assert cache.get_count == 0

    release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    engine.stop(release_pipeline_cache=False)

    assert engine.process_thread is None
    assert cache.clear_count == 1
    assert engine.status["stop_threads_pending"] == []
    assert engine.status["pipeline_cache_release_deferred"] is False


def test_set_error_preserves_root_cause_and_wakes_waiting_peers() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    preview_bus = PreviewBus()
    detection_bus = DetectionBus()
    engine.run_id = 4
    engine.preview_bus = preview_bus
    engine.detection_bus = detection_bus
    engine.status.update({"run_id": 4, "running": True, "error": ""})

    awakened: list[str] = []

    def wait_preview() -> None:
        preview_bus.wait_for_frame(0, timeout=5.0)
        awakened.append("preview")

    def wait_detection() -> None:
        detection_bus.pop_latest(0, timeout=5.0)
        awakened.append("detection")

    threads = [
        threading.Thread(target=wait_preview),
        threading.Thread(target=wait_detection),
    ]
    for thread in threads:
        thread.start()

    engine._set_error("root failure", 4)
    engine._set_error("cleanup failure", 4)

    for thread in threads:
        thread.join(timeout=0.5)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(awakened) == ["detection", "preview"]
    assert engine.stop_event.is_set()
    assert preview_bus.closed is True
    assert detection_bus.closed is True
    assert engine.status["running"] is False
    assert engine.status["error"] == "root failure"
    assert engine.status["secondary_errors"] == ["cleanup failure"]


def test_seek_clears_preview_and_detection_buffers_without_resetting_sequences() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    preview_bus = PreviewBus()
    detection_bus = DetectionBus()
    preview_bus.publish(_packet(5, epoch=1))
    detection_bus.push(_packet(5, epoch=1))
    engine.preview_bus = preview_bus
    engine.detection_bus = detection_bus
    engine.run_id = 2
    engine._source_epoch = 1
    engine.status.update(
        {
            "run_id": 2,
            "running": True,
            "source_type": "file",
            "source_epoch": 1,
            "source_duration_s": 20.0,
        }
    )

    status = engine.control_run(2, "seek", source_time_s=3.0)

    assert status["source_epoch"] == 2
    assert preview_bus.latest_packet() is None
    assert detection_bus.latest is None
    assert preview_bus.latest_seq == 5
    assert detection_bus.latest_seq == 5


def test_seek_drops_inflight_old_epoch_capture_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingSeekCapture:
        def __init__(self) -> None:
            self.read_started = threading.Event()
            self.release_first_read = threading.Event()
            self.read_count = 0
            self.position = 0.0
            self.released = False

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.read_count += 1
            if self.read_count == 1:
                self.read_started.set()
                assert self.release_first_read.wait(timeout=2.0)
                self.position = 1.0
                return True, np.full((8, 8, 3), 10, dtype=np.uint8)
            self.position = 91.0
            return True, np.full((8, 8, 3), 20, dtype=np.uint8)

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 300.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 8.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 8.0
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return self.position
            if prop == cv2.CAP_PROP_POS_MSEC:
                return max(0.0, self.position - 1.0) / 30.0 * 1000.0
            return 0.0

        def set(self, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                self.position = float(value) / 1000.0 * 30.0
            elif prop == cv2.CAP_PROP_POS_FRAMES:
                self.position = float(value)
            return True

        def release(self) -> None:
            self.released = True

    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 2
    engine._source_epoch = 1
    engine.status.update(
        {
            "run_id": 2,
            "running": True,
            "source_type": "file",
            "source_epoch": 1,
            "source_duration_s": 10.0,
        }
    )
    class StopAfterNewPreviewBus(PreviewBus):
        def publish(self, packet: FramePacket) -> None:
            super().publish(packet)
            if int(packet.frame[0, 0, 0]) == 20:
                engine.stop_event.set()

    preview_bus = StopAfterNewPreviewBus()
    detection_bus = DetectionBus()
    engine.preview_bus = preview_bus
    engine.detection_bus = detection_bus
    capture = BlockingSeekCapture()
    monkeypatch.setattr(runner_module, "open_capture", lambda *_args: capture)

    thread = threading.Thread(
        target=engine._backend_capture_loop,
        args=(
            2,
            preview_bus,
            detection_bus,
            "file",
            "sample.mp4",
            "default",
            False,
            {},
            {},
            25.0,
            30.0,
            1280,
            0.0,
        ),
    )
    thread.start()
    assert capture.read_started.wait(timeout=1.0)

    seek_status = engine.control_run(2, "seek", source_time_s=3.0)
    capture.release_first_read.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert seek_status["source_epoch"] == 2
    latest_preview = preview_bus.latest_packet()
    latest_detection = detection_bus.latest
    assert latest_preview is not None
    assert latest_detection is not None
    assert latest_preview.epoch == 2
    assert latest_detection.epoch == 2
    assert int(latest_preview.frame[0, 0, 0]) == 20
    assert int(latest_detection.frame[0, 0, 0]) == 20
    assert engine.status["source_epoch"] == 2
    assert capture.released is True


def test_preview_loop_forwards_packet_epoch_to_publish_guard() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 3
    engine.status.update({"run_id": 3, "running": True, "source_epoch": 7})
    packet = _packet(1, epoch=7)

    class OneShotPreviewBus:
        closed = False

        def wait_for_frame(self, _last_seq: int, timeout: float) -> FramePacket | None:
            del timeout
            self.closed = True
            return packet

    published_epochs: list[int | None] = []
    engine._select_preview_overlay = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    engine._render_backend_preview = lambda *_args, **_kwargs: packet.frame  # type: ignore[method-assign]

    def publish(
        _jpeg: bytes,
        _run_id: int,
        *,
        source_epoch: int | None = None,
        source_time_s: float | None = None,
        frame_idx: int | None = None,
    ) -> None:
        del source_time_s, frame_idx
        published_epochs.append(source_epoch)
        engine.stop_event.set()

    engine._publish_preview = publish  # type: ignore[method-assign]

    engine._preview_render_loop(3, OneShotPreviewBus(), preview_render_fps=10.0)  # type: ignore[arg-type]

    assert published_epochs == [7]


def test_preview_snapshot_binds_jpeg_to_packet_time_and_frame() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 3
    engine.status.update(
        {
            "run_id": 3,
            "running": True,
            "source_epoch": 7,
            "source_time_s": 99.0,
            "frame_idx": 999,
        }
    )

    engine._publish_preview(
        b"jpeg",
        3,
        source_epoch=7,
        source_time_s=1.25,
        frame_idx=42,
    )
    engine.status.update({"source_time_s": 2.0, "frame_idx": 60})

    seq, jpeg, running, metadata = engine.wait_latest_jpeg_snapshot(
        last_seq=0,
        timeout=0.01,
    )

    assert seq == 1
    assert jpeg == b"jpeg"
    assert running is True
    assert metadata == {
        "preview_seq": 1,
        "source_epoch": 7,
        "source_time_s": 1.25,
        "frame_idx": 42,
    }


def test_preview_loop_applies_render_fps_cadence_without_waiting_for_detection() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 3
    engine.status.update(
        {
            "run_id": 3,
            "running": True,
            "source_epoch": 7,
            "preview_never_wait_for_detection": True,
            "first_detection_ready": False,
        }
    )
    packet = _packet(1, epoch=7)

    class OneShotPreviewBus:
        closed = False

        def wait_for_frame(self, _last_seq: int, timeout: float) -> FramePacket | None:
            del timeout
            return packet

    class StopAfterCadenceWait:
        def __init__(self) -> None:
            self.wait_timeouts: list[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            self.wait_timeouts.append(timeout)
            return True

        def set(self) -> None:
            return None

    stop_event = StopAfterCadenceWait()
    engine.stop_event = stop_event  # type: ignore[assignment]
    engine._select_preview_overlay = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    engine._render_backend_preview = lambda *_args, **_kwargs: packet.frame  # type: ignore[method-assign]
    engine._publish_preview = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    engine._preview_render_loop(3, OneShotPreviewBus(), preview_render_fps=10.0)  # type: ignore[arg-type]

    assert stop_event.wait_timeouts
    assert 0.07 <= stop_event.wait_timeouts[-1] <= 0.1


class _FakeCapture:
    def __init__(
        self,
        reads: list[tuple[bool, np.ndarray | None]],
        *,
        stop_on_success: threading.Event | None = None,
    ) -> None:
        self.reads = list(reads)
        self.stop_on_success = stop_on_success
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.reads:
            return False, None
        ok, frame = self.reads.pop(0)
        if ok and frame is not None and self.stop_on_success is not None:
            self.stop_on_success.set()
        return ok, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 8.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 8.0
        return 0.0

    def release(self) -> None:
        self.released = True


def test_stream_reconnect_advances_epoch_and_drops_old_temporal_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 1
    engine._source_epoch = 1
    engine.status.update({"run_id": 1, "running": True, "source_epoch": 1})
    preview_bus = PreviewBus()
    detection_bus = DetectionBus()
    engine.preview_bus = preview_bus
    engine.detection_bus = detection_bus

    frame_a = np.full((8, 8, 3), 10, dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 20, dtype=np.uint8)
    first = _FakeCapture([(True, frame_a), (False, None)])
    second = _FakeCapture([(True, frame_b)])
    captures = iter([first, second])
    monkeypatch.setattr(runner_module, "open_capture", lambda *_args: next(captures))

    class StopAfterReconnectPreviewBus(PreviewBus):
        def publish(self, packet: FramePacket) -> None:
            super().publish(packet)
            if packet.epoch == 2:
                engine.stop_event.set()

    preview_bus = StopAfterReconnectPreviewBus()
    engine.preview_bus = preview_bus

    clock = {"value": 0.0}

    def perf_counter() -> float:
        clock["value"] += 10.0
        return clock["value"]

    monkeypatch.setattr(runner_module.time, "perf_counter", perf_counter)

    engine._backend_capture_loop(
        1,
        preview_bus,
        detection_bus,
        "camera",
        "0",
        "default",
        True,
        {},
        {},
        25.0,
        5.0,
        1280,
        0.0,
    )

    latest = detection_bus.latest
    assert latest is not None
    assert latest.seq == 2
    assert latest.epoch == 2
    assert latest.previous_frame is None
    assert latest.previous_frame_idx is None
    assert preview_bus.latest_packet() is latest
    assert engine.status["source_epoch"] == 2
    assert engine.status["stream_reconnects"] == 1
    assert first.released is True


def test_capture_loop_throttles_detection_pushes_to_effective_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingDetectionBus(DetectionBus):
        def __init__(self) -> None:
            super().__init__()
            self.push_count = 0

        def push(self, packet: FramePacket) -> None:
            self.push_count += 1
            super().push(packet)

    def run_with_cap(cap: float) -> int:
        engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
        engine.run_id = 1
        engine._source_epoch = 1
        engine.status.update({"run_id": 1, "running": True, "source_epoch": 1})
        frames = [
            (True, np.full((8, 8, 3), idx, dtype=np.uint8))
            for idx in range(12)
        ]
        capture = _FakeCapture(frames, stop_on_success=None)
        original_read = capture.read

        def read() -> tuple[bool, np.ndarray | None]:
            ok, frame = original_read()
            if len(capture.reads) == 0:
                engine.stop_event.set()
            return ok, frame

        capture.read = read  # type: ignore[method-assign]
        monkeypatch.setattr(runner_module, "open_capture", lambda *_args: capture)

        clock = {"value": 0.0}

        def perf_counter() -> float:
            clock["value"] += 0.05
            return clock["value"]

        monkeypatch.setattr(runner_module.time, "perf_counter", perf_counter)
        detection_bus = CountingDetectionBus()
        engine._backend_capture_loop(
            1,
            PreviewBus(),
            detection_bus,
            "camera",
            "0",
            "default",
            True,
            {},
            {},
            25.0,
            cap,
            1280,
            0.0,
        )
        return detection_bus.push_count

    low_cap_pushes = run_with_cap(2.0)
    high_cap_pushes = run_with_cap(10.0)

    assert 1 <= low_cap_pushes <= 4
    assert high_cap_pushes >= 10
    assert low_cap_pushes < high_cap_pushes


def test_completed_events_are_scoped_to_run_and_source_epoch() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 8
    engine.status["source_epoch"] = 3
    event = {"channel": "module_a", "event_id": "old"}

    engine._merge_completed_events(event, run_id=7)
    engine._merge_completed_events(event, run_id=8, source_epoch=2)

    assert list(engine.recent_events) == []

    engine._merge_completed_events(event, run_id=8, source_epoch=3)

    assert list(engine.recent_events) == [event]


def test_process_epoch_reset_finalizes_evidence_and_resets_processor() -> None:
    engine = MonitorEngine(_GuardedCache())  # type: ignore[arg-type]
    engine.run_id = 8
    engine.status["source_epoch"] = 3

    class Processor:
        reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

    class Evidence:
        reset_reasons: list[str] = []

        def reset(
            self,
            *,
            reason: str,
            source_epoch: int,
        ) -> list[dict[str, Any]]:
            self.reset_reasons.append(reason)
            assert source_epoch == 4
            return [{"channel": "module_a", "event_id": "epoch-1"}]

    processor = Processor()
    evidence = Evidence()

    engine._reset_process_epoch_state(
        run_id=8,
        processor=processor,  # type: ignore[arg-type]
        evidence=evidence,  # type: ignore[arg-type]
        source_epoch=4,
    )

    assert processor.reset_count == 1
    assert evidence.reset_reasons == ["source_epoch_changed"]
    assert list(engine.recent_events) == [
        {"channel": "module_a", "event_id": "epoch-1"}
    ]


def test_process_loop_snapshot_uses_full_config_and_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Pipeline:
        warmup_frames = 0

        def warmup(self, _frames: int) -> None:
            return None

        def reset(self) -> None:
            return None

    config = {
        "runtime": {
            "evidence_enabled": True,
            "evidence_fps": 15,
            "detector_process_fps_cap": 9,
        },
        "module_a": {"marker": "full-config"},
        "inference": {"device": "cpu"},
    }
    bundle = SimpleNamespace(
        pipeline=Pipeline(),
        config=config,
        backend="fake",
        model_family="fake",
        artifact_path="fake.engine",
        warmup_error="",
    )

    class Cache:
        def get(self, **_kwargs: Any) -> Any:
            return bundle

        def clear(self) -> None:
            return None

    class Evidence:
        enabled = True
        session_dir = tmp_path
        manifest_path = tmp_path / "manifest.json"
        saved_event_count = 0

        def __init__(self, **kwargs: Any) -> None:
            captured["evidence_kwargs"] = kwargs

        def close(self) -> list[dict[str, Any]]:
            return []

        def update(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured["evidence_status"] = kwargs["status"]
            return []

    captured: dict[str, Any] = {}

    def write_snapshot(snapshot: dict[str, Any], target_dir: str | Path) -> Path:
        captured["snapshot"] = snapshot
        captured["target_dir"] = Path(target_dir)
        return Path(target_dir) / "config_snapshot.json"

    monkeypatch.setattr(runner_module, "EvidenceSession", Evidence)
    class Processor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def process(self, frame: np.ndarray, **kwargs: Any) -> Any:
            captured["process_kwargs"] = kwargs
            return SimpleNamespace(
                frame_idx=int(kwargs["frame_idx"]),
                frame_640=frame,
                rendered_frame=frame,
                info={},
                ppe={},
                ppe_tracks=[],
                status={"a3b_triggered": False},
            )

    monkeypatch.setattr(runner_module, "FrameProcessor", Processor)
    monkeypatch.setattr(runner_module, "write_config_snapshot", write_snapshot)

    engine = MonitorEngine(Cache())  # type: ignore[arg-type]
    engine.run_id = 6
    engine._source_epoch = 4
    engine.status.update({"run_id": 6, "running": True, "source_epoch": 4})
    detection_bus = DetectionBus()
    detection_bus.push(_packet(1, epoch=4))
    detection_bus.close()

    engine._backend_process_loop(
        6,
        PreviewBus(),
        detection_bus,
        "camera",
        "0",
        "default",
        True,
        {},
        {},
        {},
    )

    assert captured["snapshot"]["module_a"] == {"marker": "full-config"}
    assert captured["snapshot"]["inference"] == {"device": "cpu"}
    assert captured["snapshot"]["_runtime_context"] == {
        "run_id": 6,
        "source_epoch": 4,
    }
    assert captured["target_dir"] == tmp_path
    assert captured["evidence_kwargs"]["run_id"] == 6
    assert captured["evidence_kwargs"]["source_epoch"] == 4
    assert captured["evidence_kwargs"]["sample_every"] == 2
    assert captured["process_kwargs"]["source_fps"] == pytest.approx(30.0)
    assert captured["process_kwargs"]["target_frame_budget_ms"] == pytest.approx(
        1000.0 / 9.0
    )
    assert captured["evidence_status"]["run_id"] == 6
    assert captured["evidence_status"]["source_epoch"] == 4
    assert captured["evidence_status"]["source_time_s"] == pytest.approx(1.0 / 30.0)


def test_process_loop_disabled_evidence_skips_snapshot_and_keeps_null_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipeline:
        warmup_frames = 0

        def warmup(self, _frames: int) -> None:
            return None

        def reset(self) -> None:
            return None

    bundle = SimpleNamespace(
        pipeline=Pipeline(),
        config={
            "runtime": {
                "evidence_enabled": False,
                "detector_process_fps_cap": 9,
            }
        },
        backend="fake",
        model_family="fake",
        artifact_path="fake.engine",
        warmup_error="",
    )

    class Cache:
        def get(self, **_kwargs: Any) -> Any:
            return bundle

        def clear(self) -> None:
            return None

    class Evidence:
        enabled = False
        session_dir = None
        manifest_path = None
        saved_event_count = 0

        def __init__(self, **_kwargs: Any) -> None:
            return None

        def update(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

        def close(self) -> list[dict[str, Any]]:
            return []

    class Processor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def process(self, frame: np.ndarray, **kwargs: Any) -> Any:
            return SimpleNamespace(
                frame_idx=int(kwargs["frame_idx"]),
                frame_640=frame,
                rendered_frame=frame,
                info={},
                ppe={},
                ppe_tracks=[],
                status={"a3b_triggered": False},
            )

    monkeypatch.setattr(runner_module, "EvidenceSession", Evidence)
    monkeypatch.setattr(runner_module, "FrameProcessor", Processor)
    monkeypatch.setattr(
        runner_module,
        "write_config_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled evidence must not write a config snapshot"
        ),
    )

    engine = MonitorEngine(Cache())  # type: ignore[arg-type]
    engine.run_id = 9
    engine._source_epoch = 2
    engine.status.update({"run_id": 9, "running": True, "source_epoch": 2})
    detection_bus = DetectionBus()
    detection_bus.push(_packet(1, epoch=2))
    detection_bus.close()

    engine._backend_process_loop(
        9,
        PreviewBus(),
        detection_bus,
        "camera",
        "0",
        "default",
        True,
        {},
        {},
        {},
    )

    status = engine.get_status()
    assert status["evidence_session_dir"] is None
    assert status["evidence_manifest_path"] is None


def test_implicit_feature_options_preserve_profile_and_explicit_options_apply() -> None:
    cache_options, public_options = MonitorEngine._normalize_start_feature_options(None)
    profile_config = load_runtime_config(
        profile="desktop_rtx",
        feature_options=cache_options,
    )

    assert public_options == {}
    assert profile_config["module_a"]["static_image_enabled"] is True

    cache_options, public_options = MonitorEngine._normalize_start_feature_options(
        {"static_image_enabled": True, "a3b_sensitivity": "high"}
    )
    explicit_config = load_runtime_config(
        profile="desktop_rtx",
        feature_options=cache_options,
    )

    assert public_options == {
        "static_image_enabled": True,
        "a3b_sensitivity": "high",
    }
    assert explicit_config["module_a"]["static_image_enabled"] is True

    cache_options, public_options = MonitorEngine._normalize_start_feature_options(
        {"static_image_enabled": True}
    )
    partial_config = load_runtime_config(
        profile="desktop_rtx",
        feature_options=cache_options,
    )

    assert cache_options == {"static_image_enabled": True}
    assert public_options == {"static_image_enabled": True}
    assert "a3b_sensitivity" not in cache_options
    assert partial_config["module_a"]["static_image_enabled"] is True

    cache_options, public_options = MonitorEngine._normalize_start_feature_options(
        {"a3b_sensitivity": "high"}
    )
    partial_config = load_runtime_config(
        profile="desktop_rtx",
        feature_options=cache_options,
    )

    assert cache_options == {"a3b_sensitivity": "high"}
    assert public_options == {"a3b_sensitivity": "high"}
    assert "static_image_enabled" not in cache_options
    assert partial_config["module_a"]["static_image_enabled"] is True


def test_initial_effective_config_schema_matches_runtime_diagnostics() -> None:
    effective = MonitorEngine(_GuardedCache()).get_status()[
        "module_a_effective_config"
    ]

    assert effective["detector_process_fps_cap"] is None
    assert effective["a3b_sensitivity"] is None
    assert effective["a3b_source_keyword_policy"] == "diagnostic_only"
    assert effective["a3b_source_keyword_match_required"] is False
    assert effective["a3b_observed_only_source_keywords"] == []
    assert effective["a3b_trigger_source_keywords"] == []
    assert effective["flow_requested_device"] is None
    assert effective["flow_effective_device"] is None
    assert effective["flow_backend"] is None
    assert effective["flow_fallback_reason"] is None
    assert effective["a4_classifier_configured"] is None
    assert effective["a4_classifier_loaded"] is None
    assert effective["a4_classifier_error"] is None
    assert effective["a4_classifier_fallback_reason"] is None


def test_start_uses_effective_detector_cap_without_coupling_preview_to_detection() -> None:
    bundle = SimpleNamespace(
        config={
            "runtime": {
                "detector_process_fps_cap": 7,
                "preview_render_fps": 19,
                "capture_max_side": 640,
                "detector_thread_warmup_timeout_s": 1.0,
            }
        },
        backend="fake",
        model_family="fake",
        artifact_path="fake.engine",
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

    class Cache:
        def __init__(self) -> None:
            self.feature_options: list[dict[str, Any]] = []

        def get(self, **kwargs: Any) -> Any:
            self.feature_options.append(dict(kwargs.get("feature_options") or {}))
            return bundle

        def clear(self) -> None:
            return None

    cache = Cache()
    engine = MonitorEngine(cache)  # type: ignore[arg-type]
    capture_args: list[tuple[Any, ...]] = []

    def process_loop(run_id: int, *_args: Any) -> None:
        with engine.condition:
            engine.status.update(
                {
                    "detector_ready": True,
                    "initializing": False,
                    "prewarming": False,
                }
            )
            engine.condition.notify_all()

    def capture_loop(*args: Any) -> None:
        capture_args.append(args)

    engine._backend_process_loop = process_loop  # type: ignore[method-assign]
    engine._backend_capture_loop = capture_loop  # type: ignore[method-assign]
    engine._preview_render_loop = lambda *_args: None  # type: ignore[method-assign]

    run_id = engine.start(source_type="camera", source="0", feature_options=None)
    assert engine.capture_thread is not None
    engine.capture_thread.join(timeout=0.5)

    status = engine.get_status()
    assert run_id == 1
    assert status["detector_process_fps_cap"] == 7
    assert status["preview_render_fps"] == 19
    assert status["preview_never_wait_for_detection"] is True
    assert capture_args
    assert capture_args[0][10] == 7.0
    assert capture_args[0][7] == {}
    assert cache.feature_options[0] == {}

    engine.stop(release_pipeline_cache=False)


def test_latest_only_and_idle_waits_do_not_busy_poll() -> None:
    detection_bus = DetectionBus()
    detection_bus.push(_packet(1, epoch=1))
    detection_bus.push(_packet(2, epoch=1))
    detection_bus.push(_packet(3, epoch=1))

    latest = detection_bus.pop_latest(0, timeout=0.01)

    assert latest is not None
    assert latest.seq == 3
    assert detection_bus.dropped == 2

    preview_bus = PreviewBus()
    started = time.perf_counter()
    assert preview_bus.wait_for_frame(0, timeout=0.04) is None
    preview_elapsed = time.perf_counter() - started

    empty_detection_bus = DetectionBus()
    started = time.perf_counter()
    assert empty_detection_bus.pop_latest(0, timeout=0.04) is None
    detection_elapsed = time.perf_counter() - started

    assert preview_elapsed >= 0.025
    assert detection_elapsed >= 0.025

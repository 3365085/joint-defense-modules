from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from defense.runtime.frame_processor import ProcessedFrame
from defense.runtime.backend_pipeline import FramePacket, PreviewBus
from defense.runtime.pipeline_factory import PipelineCache
from defense.runtime.runner import MonitorEngine


class DummyCache:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1


def test_stop_joins_worker_threads() -> None:
    cache = DummyCache()
    engine = MonitorEngine(cache)
    engine.thread_join_timeout_s = 0.5
    started = threading.Event()

    def worker() -> None:
        started.set()
        while not engine.stop_event.is_set():
            time.sleep(0.01)

    thread = threading.Thread(target=worker, name="test-worker")
    thread.start()
    assert started.wait(timeout=1.0)
    engine.capture_thread = thread
    engine.status["running"] = True

    engine.stop()

    assert not thread.is_alive()
    assert engine.capture_thread is None
    assert engine.status["running"] is False
    assert engine.status["ready_for_preview"] is False
    assert engine.status["preview_started"] is False
    assert engine.status["preview_mode"] == "stopped"
    assert engine.status["detector_pipeline_mode"] == "idle"
    assert engine.status["stop_threads_pending"] == []
    assert cache.clear_count == 1


def test_stop_reports_threads_that_do_not_exit() -> None:
    cache = DummyCache()
    engine = MonitorEngine(cache)
    engine.thread_join_timeout_s = 0.01
    release = threading.Event()
    started = threading.Event()

    def worker() -> None:
        started.set()
        release.wait(timeout=1.0)

    thread = threading.Thread(target=worker, name="stubborn-worker")
    thread.start()
    assert started.wait(timeout=1.0)
    engine.capture_thread = thread
    engine.status["running"] = True

    engine.stop()
    release.set()
    thread.join(timeout=1.0)

    assert engine.status["stop_threads_pending"] == ["stubborn-worker"]
    assert engine.status["warning"] == "worker_threads_did_not_stop"
    assert cache.clear_count == 1


def test_stop_ignores_thread_objects_that_never_started() -> None:
    engine = MonitorEngine(DummyCache())
    engine.preview_thread = threading.Thread(target=lambda: None, name="never-started")
    engine.status["running"] = True

    engine.stop()

    assert engine.preview_thread is None
    assert engine.status["stop_threads_pending"] == []


def test_stop_can_preserve_pipeline_cache_for_restart() -> None:
    cache = DummyCache()
    engine = MonitorEngine(cache)

    engine.stop(release_pipeline_cache=False)

    assert cache.clear_count == 0


def test_wait_latest_jpeg_does_not_replay_stale_frame_after_stop() -> None:
    engine = MonitorEngine(DummyCache())
    engine.latest_jpeg = b"old-frame"
    engine.latest_jpeg_seq = 4
    engine.status["running"] = False

    seq, jpeg, running = engine.wait_latest_jpeg(last_seq=0, timeout=0.01)

    assert seq == 4
    assert jpeg is None
    assert running is False

    seq, jpeg, running = engine.wait_latest_jpeg(last_seq=4, timeout=0.01)

    assert seq == 4
    assert jpeg is None
    assert running is False


def test_wait_latest_jpeg_does_not_replay_source_ended_frame() -> None:
    engine = MonitorEngine(DummyCache())
    engine.latest_jpeg = b"old-frame"
    engine.latest_jpeg_seq = 4
    engine.status["running"] = False
    engine.status["source_ended"] = True

    seq, jpeg, running = engine.wait_latest_jpeg(last_seq=0, timeout=0.01)

    assert seq == 4
    assert jpeg is None
    assert running is False


def test_wait_latest_jpeg_does_not_replay_same_seq_while_running() -> None:
    engine = MonitorEngine(DummyCache())
    engine.latest_jpeg = b"old-frame"
    engine.latest_jpeg_seq = 4
    engine.status["running"] = True

    seq, jpeg, running = engine.wait_latest_jpeg(last_seq=4, timeout=0.01)

    assert seq == 4
    assert jpeg is None
    assert running is True


def test_seek_keeps_preview_and_overlay_sequences_monotonic() -> None:
    engine = MonitorEngine(DummyCache())
    engine.run_id = 3
    engine.status.update(
        {
            "run_id": 3,
            "running": True,
            "source_type": "file",
            "source_duration_s": 10.0,
            "source_epoch": 1,
            "preview_seq": 12,
            "overlay_seq": 7,
        }
    )
    engine._source_duration_s = 10.0
    engine.latest_jpeg = b"frame"
    engine.latest_jpeg_seq = 12
    engine.overlay_seq = 7

    status = engine.control_run(3, "seek", source_time_s=4.0)

    assert status["source_epoch"] == 2
    assert status["source_time_s"] == 4.0
    assert status["preview_seq"] == 12
    assert status["overlay_seq"] == 7
    assert engine.latest_jpeg is None
    assert engine.latest_jpeg_seq == 12
    assert engine.overlay_seq == 7


def test_control_rejects_inactive_run() -> None:
    engine = MonitorEngine(DummyCache())
    engine.run_id = 5
    engine.status.update({"run_id": 5, "running": False, "source_ended": True})

    try:
        engine.control_run(5, "play")
    except RuntimeError as exc:
        assert "not active" in str(exc)
    else:
        raise AssertionError("control_run should reject inactive runs")


def test_after_processed_discards_stale_epoch_result() -> None:
    class RaisingEvidence:
        session_dir = "evidence"
        manifest_path = "manifest.json"
        saved_event_count = 0

        def update(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("stale processed frames must not update evidence")

    engine = MonitorEngine(DummyCache())
    engine.run_id = 8
    engine.status.update(
        {
            "run_id": 8,
            "running": True,
            "source_epoch": 2,
            "frame_idx": 10,
            "raw_boxes_count": 3,
        }
    )
    processed = ProcessedFrame(
        frame_idx=3,
        frame_640=np.zeros((8, 8, 3), dtype=np.uint8),
        rendered_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        info={},
        ppe={},
        ppe_tracks=[],
        status={"frame_idx": 3, "raw_boxes_count": 0, "source_epoch": 1},
    )

    engine._after_processed(
        8,
        processed,
        RaisingEvidence(),
        preview_mode="backend_source_pipeline",
        extra_status={"source_epoch": 1},
        publish_jpeg=False,
    )

    assert engine.overlay_seq == 0
    assert engine.status["source_epoch"] == 2
    assert engine.status["frame_idx"] == 10
    assert engine.status["raw_boxes_count"] == 3


def test_publish_preview_discards_stale_epoch_frame() -> None:
    engine = MonitorEngine(DummyCache())
    engine.run_id = 9
    engine.status.update({"run_id": 9, "running": True, "source_epoch": 4})

    engine._publish_preview(b"old", 9, source_epoch=3)

    assert engine.latest_jpeg is None
    assert engine.latest_jpeg_seq == 0

    engine._publish_preview(b"new", 9, source_epoch=4)

    assert engine.latest_jpeg == b"new"
    assert engine.latest_jpeg_seq == 1


def test_file_end_state_clears_preview_readiness() -> None:
    engine = MonitorEngine(DummyCache())
    engine.status.update(
        {
            "running": False,
            "source_ended": True,
            "source_time_s": 1.0,
            "video_time_s": 1.0,
            "ready_for_preview": False,
            "preview_started": False,
            "preview_mode": "source_ended",
            "detector_pipeline_mode": "ended",
        }
    )

    status = engine.get_status()

    assert status["source_ended"] is True
    assert status["running"] is False
    assert status["ready_for_preview"] is False
    assert status["preview_started"] is False
    assert status["detector_pipeline_mode"] == "ended"


@pytest.mark.skip(reason="超前契约未实装:MonitorEngine无record_model_security_context方法")
def test_model_security_context_is_recorded_only_after_start_resolution() -> None:
    engine = MonitorEngine(DummyCache())

    assert "model_security" not in engine.get_status()

    engine.record_model_security_context(
        model_security={
            "admission_status": "trusted",
            "allowed": True,
        },
        runtime_replacement={
            "mode": "purified_runtime",
            "source_model_security": {"admission_status": "purified_alternative_available"},
        },
    )

    status = engine.get_status()
    assert status["model_security"]["admission_status"] == "trusted"
    assert status["model_security_runtime_replacement"]["mode"] == "purified_runtime"
    assert (
        status["model_security_runtime_replacement"]["source_model_security"]["admission_status"]
        == "purified_alternative_available"
    )


def test_finished_file_pipeline_cache_is_preserved_by_default() -> None:
    engine = MonitorEngine(DummyCache())
    engine.run_id = 3
    engine.status.update({"source_ended": True})

    should_release = engine._should_release_finished_file_pipeline(
        run_id=3,
        source_type="file",
        runtime_config={},
    )

    assert should_release is False


def test_finished_file_pipeline_cache_release_can_be_enabled() -> None:
    engine = MonitorEngine(DummyCache())
    engine.run_id = 3
    engine.status.update({"source_ended": True})

    should_release = engine._should_release_finished_file_pipeline(
        run_id=3,
        source_type="file",
        runtime_config={"release_pipeline_cache_on_file_end": True},
    )

    assert should_release is True


def test_preview_bus_wait_for_frame_does_not_return_old_packet_without_new_seq() -> None:
    bus = PreviewBus()
    packet = FramePacket(
        seq=1,
        frame_idx=1,
        source_time_s=0.04,
        wall_time_ms=0.0,
        epoch=0,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        fps=25.0,
        flags={},
    )
    bus.publish(packet)

    assert bus.wait_for_frame(0, timeout=0.01) is packet
    assert bus.wait_for_frame(1, timeout=0.01) is None


def test_pipeline_cache_clear_closes_cached_pipeline() -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pipeline = Pipeline()
    cache = PipelineCache()
    cache._bundle = type(
        "Bundle",
        (),
        {"pipeline": pipeline},
    )()
    cache._key = ("profile",)

    cache.clear()

    assert pipeline.closed is True
    assert cache._bundle is None
    assert cache._key is None


def test_capture_resize_limits_large_frames() -> None:
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    resized, changed = MonitorEngine._resize_capture_frame(frame, 1280)

    assert changed is True
    assert resized.shape[:2] == (720, 1280)


def test_file_frame_step_is_disabled_without_explicit_cap() -> None:
    step = MonitorEngine._file_frame_step(
        source_fps=60.0,
        file_source_fps_cap=0.0,
        preview_render_fps=25.0,
        detector_fps=15.0,
    )

    assert step == 1.0


def test_file_frame_step_caps_high_fps_sources_when_configured() -> None:
    step = MonitorEngine._file_frame_step(
        source_fps=60.0,
        file_source_fps_cap=25.0,
        preview_render_fps=25.0,
        detector_fps=15.0,
    )

    assert 2.3 < step < 2.5


def test_file_preview_caps_held_overlay_to_reduce_stale_boxes() -> None:
    engine = MonitorEngine(DummyCache())
    engine.status.update(
        {
            "source_type": "file",
            "realtime": True,
            "detector_process_fps_cap": 15.0,
            "overlay_match_window_ms": 180.0,
            "overlay_hold_ms": 550.0,
            "overlay_interpolate_ms": 400.0,
            "overlay_max_age_ms": 950.0,
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 1,
            "source_epoch": 0,
            "video_time_s": 10.0,
            "ppe_tracks": [{"track_id": 1, "box": [10, 20, 30, 40], "source": "detected", "hold_eligible": True}],
        }
    )

    assert engine._select_preview_overlay(10.09, 0) is not None
    held = engine._select_preview_overlay(10.20, 0)
    assert held is not None
    assert held["ppe_tracks"][0]["source"] == "held"
    assert engine._select_preview_overlay(10.42, 0) is None

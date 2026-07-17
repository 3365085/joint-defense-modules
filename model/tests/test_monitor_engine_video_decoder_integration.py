from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import defense.pipelines.video_decoder_factory as decoder_factory_module
import defense.runtime.runner as runner_module
from defense.pipelines.video_decoder import DecodedFrameLease, VideoStreamInfo
from defense.runtime.backend_pipeline import FramePacket
from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config
from defense.runtime.runner import MonitorEngine


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    message: str,
) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), message


def test_file_realtime_wait_uses_absolute_source_clock() -> None:
    wait_s = runner_module._file_realtime_wait_s(
        playback_anchor_wall=100.0,
        playback_anchor_frame=0.0,
        next_frame=1.0,
        fps=30.0,
        speed=1.0,
        now=100.01,
    )

    assert wait_s == pytest.approx((1.0 / 30.0) - 0.01)


def test_file_realtime_wait_does_not_accumulate_prior_delay() -> None:
    assert (
        runner_module._file_realtime_wait_s(
            playback_anchor_wall=100.0,
            playback_anchor_frame=0.0,
            next_frame=1.0,
            fps=30.0,
            speed=1.0,
            now=100.05,
        )
        == 0.0
    )
    assert runner_module._file_realtime_wait_s(
        playback_anchor_wall=100.0,
        playback_anchor_frame=0.0,
        next_frame=2.0,
        fps=30.0,
        speed=2.0,
        now=100.02,
    ) == pytest.approx((2.0 / 60.0) - 0.02)


def test_detector_completion_fps_uses_cycle_aligned_stable_window() -> None:
    engine = MonitorEngine(object())  # type: ignore[arg-type]
    assert engine.detect_times.maxlen == 240
    assert engine.detect_times.maxlen == engine.detector_cycle_samples.maxlen

    completion_times = [frame_idx / 25.0 for frame_idx in range(300)]
    completion_times[-1] += 0.019
    short_window = deque(completion_times, maxlen=60)
    stable_window = deque(
        completion_times,
        maxlen=runner_module._DETECTOR_FPS_WINDOW_FRAMES,
    )

    assert round(
        runner_module._detector_completion_fps(short_window),
        1,
    ) == 24.8
    assert round(
        runner_module._detector_completion_fps(stable_window),
        1,
    ) == 25.0


def test_detector_submit_every_frame_tolerates_nominal_60_fps_metadata() -> None:
    assert runner_module._detector_can_follow_file_source(60.0, 60.49)
    assert runner_module._detector_can_follow_file_source(60.0, 60.5)
    assert not runner_module._detector_can_follow_file_source(30.0, 59.94)


class _Pipeline:
    warmup_frames = 0

    def warmup(self, _frames: int) -> None:
        return None

    def reset(self) -> None:
        return None


class _Cache:
    def __init__(self, config: dict[str, Any]) -> None:
        self.clear_count = 0
        self.bundle = SimpleNamespace(
            pipeline=_Pipeline(),
            config=config,
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

    def get(self, **_kwargs: Any) -> Any:
        return self.bundle

    def clear(self) -> None:
        self.clear_count += 1


def _install_decoder_factory(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[..., Any],
) -> None:
    # Cover both supported import styles: direct symbol import in runner.py and
    # module-qualified calls through video_decoder_factory.
    monkeypatch.setattr(
        decoder_factory_module,
        "create_video_decoder",
        factory,
    )
    monkeypatch.setattr(
        runner_module,
        "create_video_decoder",
        factory,
        raising=False,
    )


def _make_ready_process_loop(engine: MonitorEngine) -> Callable[..., None]:
    def process_loop(
        run_id: int,
        _preview_bus: Any,
        detection_bus: Any,
        *_args: Any,
    ) -> None:
        with engine.condition:
            if run_id == engine.run_id:
                engine.status.update(
                    {
                        "detector_ready": True,
                        "initializing": False,
                        "prewarming": False,
                    }
                )
                engine.condition.notify_all()
        try:
            last_seq = 0
            while (
                run_id == engine.run_id
                and not engine.stop_event.is_set()
                and not detection_bus.closed
            ):
                packet = detection_bus.pop_latest(last_seq, timeout=0.01)
                if packet is None:
                    continue
                last_seq = packet.seq
                packet.release_lease_refs()
        finally:
            with engine.condition:
                if run_id == engine.run_id:
                    engine.status.update(
                        {
                            "evidence_drain_active": False,
                            "evidence_drain_completed": True,
                            "evidence_drain_failed": False,
                            "process_done": True,
                        }
                    )
                    engine.condition.notify_all()
            engine.process_done_event.set()

    return process_loop


def _prepare_file_engine(config: dict[str, Any]) -> MonitorEngine:
    engine = MonitorEngine(_Cache(config))  # type: ignore[arg-type]
    engine._backend_process_loop = _make_ready_process_loop(engine)  # type: ignore[method-assign]
    engine._preview_render_loop = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    return engine


def _lease(
    value: int,
    *,
    frame_idx: int = 0,
    pts_s: float = 0.0,
    released: threading.Event | None = None,
) -> DecodedFrameLease:
    stable_owner = object()
    frame = np.full((8, 12, 3), value, dtype=np.uint8)

    def materialize(
        size: tuple[int, int] | None,
        roi: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        output = frame
        if roi is not None:
            x1, y1, x2, y2 = roi
            output = output[y1:y2, x1:x2]
        if size is not None:
            width, height = size
            output = np.full((height, width, 3), value, dtype=np.uint8)
        return np.ascontiguousarray(output)

    return DecodedFrameLease(
        frame_idx=frame_idx,
        pts_s=pts_s,
        width=12,
        height=8,
        pixel_format="rgbp",
        storage="cuda:0",
        decode_ms=1.5,
        d2d_copy_ms=0.25,
        owner=stable_owner,
        cuda_tensor=stable_owner,
        metadata={
            "backend": "nvdec",
            "surface_cloned": True,
            "surface_ownership": "stable_owned_copy",
        },
        _host_materializer=materialize,
        _release_callback=(released.set if released is not None else None),
    )


class _FakeDecoder:
    def __init__(
        self,
        *,
        backend: str = "nvdec",
        requested_backend: str = "nvdec",
        fallback_reason: str = "",
        frames: list[DecodedFrameLease] | None = None,
        block_first_read: bool = False,
        close_error: str = "",
        raise_on_close: bool = False,
    ) -> None:
        self.info = VideoStreamInfo(
            source="fake.mp4",
            backend=backend,
            codec="h264",
            width=12,
            height=8,
            fps=30.0,
            frame_count=len(frames or []),
            duration_s=len(frames or []) / 30.0,
            gpu_device="cuda:0" if backend == "nvdec" else None,
            output_format="rgbp" if backend == "nvdec" else "bgr24",
            frame_device="cuda:0" if backend == "nvdec" else "host",
        )
        self.requested_backend = requested_backend
        self.backend = backend
        self.fallback_reason = fallback_reason
        self.frames = list(frames or [])
        self.frames_decoded = 0
        self.bytes_decoded = 0
        self.closed = False
        self.close_error = ""
        self.configured_close_error = close_error
        self.raise_on_close = raise_on_close
        self.block_first_read = block_first_read
        self.read_started = threading.Event()
        self.release_first_read = threading.Event()
        self.close_called = threading.Event()
        self.seek_times: list[float] = []
        self.seek_frames: list[int] = []

    def read(self) -> DecodedFrameLease | None:
        if self.block_first_read and not self.read_started.is_set():
            self.read_started.set()
            assert self.release_first_read.wait(timeout=2.0)
        if not self.frames:
            return None
        lease = self.frames.pop(0)
        self.frames_decoded += 1
        self.bytes_decoded += lease.width * lease.height * 3
        return lease

    def seek_time(self, seconds: float) -> None:
        self.seek_times.append(float(seconds))

    def seek_frame(self, frame_idx: int) -> None:
        self.seek_frames.append(int(frame_idx))

    def status_snapshot(self) -> dict[str, Any]:
        fallback_count = 1 if self.fallback_reason else 0
        return {
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "effective_backend": self.backend,
            "codec": self.info.codec,
            "gpu_device": self.info.gpu_device,
            "output_format": self.info.output_format,
            "frame_device": self.info.frame_device,
            # Backends currently expose these metric names. MonitorEngine owns
            # the public status.decoder aliases asserted below.
            "decode_ms_p50": 1.25,
            "decode_ms_p95": 2.5,
            "d2d_copy_ms_p50": 0.2,
            "d2d_copy_ms_p95": 0.4,
            "d2h_copy_ms_p50": 0.7,
            "d2h_copy_ms_p95": 1.1,
            "frames_decoded": self.frames_decoded,
            "bytes_decoded": self.bytes_decoded,
            "fallback_count": fallback_count,
            "fallback_reason": self.fallback_reason,
            "close_error": self.close_error,
            "closed": self.closed,
            "source_alias_cleaned": self.closed,
            "source_alias_cleanup_error": "",
        }

    def close(self) -> None:
        self.closed = True
        self.close_error = self.configured_close_error
        self.close_called.set()
        if self.raise_on_close:
            raise RuntimeError(self.configured_close_error or "synthetic_close_failure")


class _StreamingDecoder(_FakeDecoder):
    def __init__(self, *, value: int) -> None:
        super().__init__(frames=[])
        self.value = value
        self.first_frame = threading.Event()
        self.created_leases: list[DecodedFrameLease] = []

    def read(self) -> DecodedFrameLease | None:
        if self.closed:
            return None
        release_event = threading.Event()
        lease = _lease(
            self.value,
            frame_idx=self.frames_decoded,
            pts_s=self.frames_decoded / self.info.fps,
            released=release_event,
        )
        # Keep the event attached to the lease for a deterministic ownership
        # assertion without relying on Python garbage collection timing.
        lease.metadata["test_release_event"] = release_event
        self.created_leases.append(lease)
        self.frames_decoded += 1
        self.bytes_decoded += lease.width * lease.height * 3
        self.first_frame.set()
        time.sleep(0.01)
        return lease


def _assert_public_decoder_status(
    decoder_status: dict[str, Any],
    *,
    effective_backend: str,
) -> None:
    assert decoder_status["requested_backend"] == "nvdec"
    assert decoder_status["backend"] == effective_backend
    assert decoder_status["effective_backend"] == effective_backend
    assert decoder_status["codec"] == "h264"
    assert decoder_status["output_format"]
    assert decoder_status["frame_device"]
    assert decoder_status["decode_p50_ms"] == pytest.approx(1.25)
    assert decoder_status["decode_p95_ms"] == pytest.approx(2.5)
    assert decoder_status["d2d_copy_p50_ms"] == pytest.approx(0.2)
    assert decoder_status["d2d_copy_p95_ms"] == pytest.approx(0.4)
    assert decoder_status["gpu_to_cpu_copy_p50_ms"] == pytest.approx(0.7)
    assert decoder_status["gpu_to_cpu_copy_p95_ms"] == pytest.approx(1.1)


def test_file_default_path_uses_factory_and_preserves_visible_fallback_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "factory-route.mp4"
    source.write_bytes(b"decoder factory test")
    calls: list[tuple[Path, dict[str, Any]]] = []
    created_decoders: list[_FakeDecoder] = []

    def factory(path: str | Path, **kwargs: Any) -> _FakeDecoder:
        calls.append((Path(path), dict(kwargs)))
        decoder = _FakeDecoder(
            backend="opencv",
            requested_backend="nvdec",
            fallback_reason="nvdec_init_failed:synthetic",
            frames=[_lease(17)],
        )
        created_decoders.append(decoder)
        return decoder

    _install_decoder_factory(monkeypatch, factory)
    monkeypatch.setattr(
        runner_module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail(
            "file MonitorEngine path must not instantiate cv2.VideoCapture"
        ),
    )
    engine = _prepare_file_engine(config)

    try:
        run_id = engine.start(source_type="file", source=str(source))
        assert run_id == 1
        _wait_until(
            lambda: engine.capture_thread is not None
            and not engine.capture_thread.is_alive(),
            message="file capture thread did not finish after decoder EOF",
        )

        assert calls
        assert all(decoder.close_called.is_set() for decoder in created_decoders)
        called_path, kwargs = calls[-1]
        assert called_path.resolve() == source.resolve()
        assert kwargs["preference"] == "nvdec"
        assert kwargs["allow_cpu_fallback"] is True

        status = engine.get_status()
        assert status["source_ended"] is True
        assert status["running"] is False
        decoder_status = status["decoder"]
        _assert_public_decoder_status(
            decoder_status,
            effective_backend="opencv",
        )
        assert decoder_status["frames_decoded"] == 1
        assert decoder_status["bytes_decoded"] > 0
        assert decoder_status["fallback_count"] == 1
        assert decoder_status["fallback_reason"] == "nvdec_init_failed:synthetic"
        assert decoder_status["closed"] is True
        assert decoder_status["source_alias_cleaned"] is True
        assert decoder_status["source_alias_cleanup_error"] == ""
        assert decoder_status["close_error"] in {"", "none"}
        assert status["error"] == ""
    finally:
        engine.stop(release_pipeline_cache=False)


def test_zero_frame_decoder_is_error_not_normal_source_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "zero-frame.mp4"
    source.write_bytes(b"zero frame")
    decoder = _FakeDecoder(frames=[])
    _install_decoder_factory(monkeypatch, lambda *_args, **_kwargs: decoder)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    engine = _prepare_file_engine(config)

    try:
        engine.start(source_type="file", source=str(source))
        assert decoder.close_called.wait(timeout=2.0)
        status = engine.get_status()

        assert status["source_ended"] is False
        assert "decoder_zero_frame_eof" in status["error"]
        assert "backend=nvdec" in status["error"]
        assert "codec=h264" in status["error"]
        assert "frames_decoded=0" in status["error"]
    finally:
        engine.stop(release_pipeline_cache=False)


def test_file_prevalidation_uses_effective_decoder_fallback_and_gpu_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "  video_decoder_allow_cpu_fallback: true",
        "  video_decoder_allow_cpu_fallback: false",
    ).replace(
        "  video_decoder_gpu_id: 0",
        "  video_decoder_gpu_id: 2",
    )
    config_path = tmp_path / "decoder-probe.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_runtime_config(
        config_path=config_path,
        profile="default",
    )
    source = tmp_path / "probe-effective-config.mp4"
    source.write_bytes(b"probe effective config")
    calls: list[dict[str, Any]] = []

    def factory(_path: str | Path, **kwargs: Any) -> _FakeDecoder:
        calls.append(dict(kwargs))
        return _FakeDecoder(frames=[_lease(len(calls))])

    _install_decoder_factory(monkeypatch, factory)
    engine = _prepare_file_engine(config)
    engine.cache.config_path = config_path  # type: ignore[attr-defined]

    try:
        engine.start(source_type="file", source=str(source))
        _wait_until(
            lambda: engine.capture_thread is not None
            and not engine.capture_thread.is_alive(),
            message="file capture thread did not finish",
        )

        assert len(calls) >= 2
        assert all(call["preference"] == "nvdec" for call in calls)
        assert all(
            call["allow_cpu_fallback"] is False
            for call in calls
        )
        assert all(call["gpu_id"] == 2 for call in calls)
    finally:
        engine.stop(release_pipeline_cache=False)


def test_default_file_runtime_never_skips_source_frames_to_catch_wall_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "no-wall-clock-skip.mp4"
    source.write_bytes(b"no wall clock skip")

    class SlowDecoder(_FakeDecoder):
        def read(self) -> DecodedFrameLease | None:
            time.sleep(0.06)
            return super().read()

    decoder = SlowDecoder(
        frames=[
            _lease(index, frame_idx=index, pts_s=index / 30.0)
            for index in range(5)
        ]
    )
    _install_decoder_factory(monkeypatch, lambda *_args, **_kwargs: decoder)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    engine = _prepare_file_engine(config)

    try:
        engine.start(source_type="file", source=str(source), realtime=True)
        assert decoder.close_called.wait(timeout=2.0)
        status = engine.get_status()

        assert status["file_source_fps_cap"] == 0.0
        assert status["source_frame_skip_enabled"] is False
        assert status["detector_submit_every_file_frame"] is True
        assert status["source_frames_skipped_for_realtime"] == 0
        assert status["capture_frames_published"] == 5
        assert status["detector_submission_count"] == 5
        assert status["dropped_detection_frames"] == 0
        assert status["decoder"]["frames_decoded"] == 5
    finally:
        engine.stop(release_pipeline_cache=False)


def test_file_capture_publishes_stable_lease_without_eager_host_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "lease-publication.mp4"
    source.write_bytes(b"lease publication test")
    materialize_calls: list[
        tuple[tuple[int, int] | None, tuple[int, int, int, int] | None]
    ] = []
    lease_released = threading.Event()
    stable_owner = object()

    def materialize(
        size: tuple[int, int] | None,
        roi: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        materialize_calls.append((size, roi))
        width, height = size or (12, 8)
        return np.zeros((height, width, 3), dtype=np.uint8)

    lease = DecodedFrameLease(
        frame_idx=0,
        pts_s=0.0,
        width=12,
        height=8,
        pixel_format="rgbp",
        storage="cuda:0",
        decode_ms=1.0,
        d2d_copy_ms=0.2,
        owner=stable_owner,
        cuda_tensor=stable_owner,
        metadata={"surface_cloned": True},
        _host_materializer=materialize,
        _release_callback=lease_released.set,
    )

    class HoldAfterFirstDecoder(_FakeDecoder):
        def __init__(self) -> None:
            super().__init__(frames=[lease])
            self.second_read_started = threading.Event()
            self.finish = threading.Event()
            self.read_count = 0

        def read(self) -> DecodedFrameLease | None:
            self.read_count += 1
            if self.read_count == 1:
                return super().read()
            self.second_read_started.set()
            assert self.finish.wait(timeout=2.0)
            return None

    decoder = HoldAfterFirstDecoder()
    _install_decoder_factory(monkeypatch, lambda *_args, **_kwargs: decoder)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    monkeypatch.setattr(
        runner_module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail(
            "file capture must not use cv2.VideoCapture"
        ),
    )
    engine = _prepare_file_engine(config)

    try:
        engine.start(source_type="file", source=str(source))
        assert decoder.second_read_started.wait(timeout=1.0)
        assert engine.preview_bus is not None
        assert engine.detection_bus is not None
        preview_packet = engine.preview_bus.latest_packet()
        detection_packet = engine.detection_bus.latest

        assert preview_packet is not None
        assert detection_packet is preview_packet
        assert preview_packet.frame is None
        assert preview_packet.decoder_lease is lease
        assert preview_packet.decoder_lease.owner is stable_owner
        assert preview_packet.decoder_lease.metadata["surface_cloned"] is True
        assert materialize_calls == []
        assert lease.released is False

        decoder.finish.set()
        assert decoder.close_called.wait(timeout=1.0)
        _wait_until(
            lambda: engine.capture_thread is not None
            and not engine.capture_thread.is_alive(),
            message="file capture did not finish after synthetic EOF",
        )
        assert lease.released is True
        assert lease_released.is_set()
    finally:
        decoder.finish.set()
        engine.stop(release_pipeline_cache=False)


@pytest.mark.parametrize(
    ("source_type", "source"),
    [
        ("camera", "0"),
        ("rtsp", "rtsp://example.invalid/stream"),
    ],
)
def test_camera_and_rtsp_keep_existing_capture_path(
    source_type: str,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    engine = _prepare_file_engine(config)
    open_calls: list[tuple[str, str]] = []

    class Capture:
        released = False
        read_count = 0

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.read_count += 1
            engine.stop_event.set()
            return True, np.zeros((8, 12, 3), dtype=np.uint8)

        def get(self, _prop: int) -> float:
            return 30.0

        def release(self) -> None:
            self.released = True

    capture = Capture()

    def open_capture(kind: str, value: str, *_args: Any, **_kwargs: Any) -> Capture:
        open_calls.append((kind, value))
        return capture

    monkeypatch.setattr(runner_module, "open_capture", open_capture)
    _install_decoder_factory(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail(
            "camera/rtsp sources must not use the file decoder factory"
        ),
    )

    try:
        engine.start(source_type=source_type, source=source)
        _wait_until(
            lambda: engine.capture_thread is not None
            and not engine.capture_thread.is_alive(),
            message=f"{source_type} capture thread did not finish",
        )
        assert open_calls == [(source_type, source)]
        assert capture.released is True
        decoder_status = engine.get_status()["decoder"]
        assert decoder_status["requested_backend"] == "nvdec"
        assert decoder_status["effective_backend"] == "opencv"
        assert decoder_status["frame_device"] == "host"
        assert decoder_status["fallback_count"] == 1
        assert decoder_status["fallback_reason"].startswith(
            f"{source_type}_nvdec_adapter_unavailable:"
        )
    finally:
        engine.stop(release_pipeline_cache=False)


def test_seek_closes_old_decoder_reopens_at_target_and_releases_stale_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "seek.mp4"
    source.write_bytes(b"seek lifecycle test")
    stale_released = threading.Event()
    stale_lease = _lease(10, frame_idx=0, pts_s=0.0, released=stale_released)
    old_decoder = _FakeDecoder(
        frames=[stale_lease],
        block_first_read=True,
    )
    old_decoder.info = VideoStreamInfo(
        source="fake.mp4",
        backend="nvdec",
        codec="h264",
        width=12,
        height=8,
        fps=30.0,
        frame_count=300,
        duration_s=10.0,
        gpu_device="cuda:0",
        output_format="rgbp",
        frame_device="cuda:0",
    )
    new_decoder = _FakeDecoder(
        frames=[_lease(20, frame_idx=60, pts_s=2.0)],
    )
    decoders = iter([old_decoder, new_decoder])
    factory_calls: list[Path] = []

    def factory(path: str | Path, **_kwargs: Any) -> _FakeDecoder:
        factory_calls.append(Path(path))
        return next(decoders)

    _install_decoder_factory(monkeypatch, factory)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    monkeypatch.setattr(
        runner_module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail(
            "seek must stay on the unified file decoder path"
        ),
    )
    engine = _prepare_file_engine(config)

    try:
        run_id = engine.start(source_type="file", source=str(source))
        assert old_decoder.read_started.wait(timeout=1.0)

        seek_status = engine.control_run(run_id, "seek", source_time_s=2.0)
        old_decoder.release_first_read.set()

        _wait_until(
            lambda: len(factory_calls) >= 2,
            message="seek did not replace the active file decoder",
        )
        assert old_decoder.close_called.wait(timeout=1.0)
        assert stale_released.wait(timeout=1.0)
        assert new_decoder.close_called.wait(timeout=2.0)

        assert seek_status["source_epoch"] >= 2
        assert len(factory_calls) == 2
        assert all(path.resolve() == source.resolve() for path in factory_calls)
        assert new_decoder.seek_times == pytest.approx([2.0])
        assert old_decoder.closed is True
        assert new_decoder.closed is True

        _wait_until(
            lambda: bool(
                engine.get_status().get("source_ended")
                or engine.get_status().get("error")
            ),
            message=(
                "seek decoder closed before the EOF completion barrier "
                "published its final status"
            ),
        )
        status = engine.get_status()
        assert status["source_epoch"] >= 2
        assert status["source_ended"] is True
        assert status["decoder"]["closed"] is True
        latest = engine.preview_bus.latest_packet() if engine.preview_bus is not None else None
        if latest is not None:
            assert latest.epoch == status["source_epoch"]
            assert latest.frame_idx == 60
    finally:
        old_decoder.release_first_read.set()
        engine.stop(release_pipeline_cache=False)


def test_stop_cancels_inflight_file_seek_without_leaving_pending_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "cancel-seek.mp4"
    source.write_bytes(b"cancel seek lifecycle test")
    old_decoder = _FakeDecoder(
        frames=[_lease(10, frame_idx=0, pts_s=0.0)],
        block_first_read=True,
    )

    class CancellableSeekDecoder(_FakeDecoder):
        def __init__(self) -> None:
            super().__init__(frames=[])
            self.seek_started = threading.Event()
            self.cancel_requested = threading.Event()

        def seek_time(self, seconds: float) -> None:
            self.seek_times.append(float(seconds))
            self.seek_started.set()
            assert self.cancel_requested.wait(timeout=2.0)
            raise RuntimeError("synthetic_seek_cancelled")

        def request_cancel(self) -> None:
            self.cancel_requested.set()

    seek_decoder = CancellableSeekDecoder()
    decoders = iter([old_decoder, seek_decoder])
    _install_decoder_factory(
        monkeypatch,
        lambda *_args, **_kwargs: next(decoders),
    )
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    engine = _prepare_file_engine(config)

    run_id = engine.start(source_type="file", source=str(source))
    try:
        assert old_decoder.read_started.wait(timeout=1.0)
        engine.control_run(run_id, "seek", source_time_s=8.0)
        old_decoder.release_first_read.set()
        assert seek_decoder.seek_started.wait(timeout=1.0)

        engine.stop(release_pipeline_cache=False)

        assert seek_decoder.cancel_requested.is_set()
        assert seek_decoder.close_called.wait(timeout=1.0)
        assert engine.capture_thread is None
        status = engine.get_status()
        assert status["stop_threads_pending"] == []
        assert status["error"] == ""
        assert status.get("restart_blocked_reason", "") == ""
    finally:
        old_decoder.release_first_read.set()
        seek_decoder.cancel_requested.set()
        engine.stop(release_pipeline_cache=False)


def test_stop_and_restart_close_decoders_and_release_gpu_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "restart.mp4"
    source.write_bytes(b"restart lifecycle test")
    first = _StreamingDecoder(value=31)
    second = _StreamingDecoder(value=47)
    decoders = iter([first, second])
    factory_count = 0

    def factory(_path: str | Path, **_kwargs: Any) -> _StreamingDecoder:
        nonlocal factory_count
        factory_count += 1
        return next(decoders)

    _install_decoder_factory(monkeypatch, factory)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    monkeypatch.setattr(
        runner_module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail(
            "restart must stay on the unified file decoder path"
        ),
    )
    engine = _prepare_file_engine(config)

    first_run = engine.start(source_type="file", source=str(source))
    assert first.first_frame.wait(timeout=1.0)
    engine.stop(release_pipeline_cache=False)

    assert first.closed is True
    assert first.created_leases
    assert all(
        lease.released
        and lease.metadata["test_release_event"].is_set()
        for lease in first.created_leases
    )

    second_run = engine.start(source_type="file", source=str(source))
    assert second.first_frame.wait(timeout=1.0)
    engine.stop(release_pipeline_cache=False)

    assert second_run == first_run + 1
    assert factory_count == 2
    assert second.closed is True
    assert second.created_leases
    assert all(
        lease.released
        and lease.metadata["test_release_event"].is_set()
        for lease in second.created_leases
    )
    assert engine.get_status()["stop_threads_pending"] == []


def test_decoder_close_failure_is_visible_in_status_and_engine_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_runtime_config(profile="default")
    source = tmp_path / "close-error.mp4"
    source.write_bytes(b"close failure test")
    decoder = _FakeDecoder(
        frames=[],
        close_error="synthetic_alias_cleanup_failure",
        raise_on_close=True,
    )
    _install_decoder_factory(monkeypatch, lambda *_args, **_kwargs: decoder)
    monkeypatch.setattr(
        runner_module,
        "validate_file_source",
        lambda _source, **_kwargs: source.resolve(),
    )
    monkeypatch.setattr(
        runner_module.cv2,
        "VideoCapture",
        lambda *_args, **_kwargs: pytest.fail(
            "close-error test must stay on the unified file decoder path"
        ),
    )
    engine = _prepare_file_engine(config)

    try:
        engine.start(source_type="file", source=str(source))
        assert decoder.close_called.wait(timeout=2.0)
        _wait_until(
            lambda: engine.capture_thread is not None
            and not engine.capture_thread.is_alive(),
            message="capture thread did not terminate after decoder close failure",
        )

        status = engine.get_status()
        visible_errors = " | ".join(
            [
                str(status.get("error") or ""),
                *[
                    str(item)
                    for item in status.get("secondary_errors") or []
                ],
            ]
        )
        assert "synthetic_alias_cleanup_failure" in visible_errors
        assert (
            "synthetic_alias_cleanup_failure"
            in status["decoder"]["close_error"]
        )
        assert status["decoder"]["closed"] is True
    finally:
        engine.stop(release_pipeline_cache=False)


def test_frame_packet_carries_stable_lease_for_branch_specific_materialization() -> None:
    field_names = {field.name for field in fields(FramePacket)}
    lease_field = next(
        (
            name
            for name in (
                "decoder_lease",
                "decoded_frame",
                "frame_lease",
                "lease",
            )
            if name in field_names
        ),
        None,
    )
    assert lease_field is not None, (
        "FramePacket must carry a DecodedFrameLease instead of exposing a "
        "decoder-recycled PyNv surface"
    )

    calls: list[
        tuple[tuple[int, int] | None, tuple[int, int, int, int] | None]
    ] = []
    lock = threading.Lock()
    stable_owner = object()
    released = threading.Event()

    def materialize(
        size: tuple[int, int] | None,
        roi: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        with lock:
            calls.append((size, roi))
        width, height = size or (12, 8)
        return np.zeros((height, width, 3), dtype=np.uint8)

    lease = DecodedFrameLease(
        frame_idx=4,
        pts_s=0.16,
        width=12,
        height=8,
        pixel_format="rgbp",
        storage="cuda:0",
        decode_ms=1.0,
        d2d_copy_ms=0.2,
        owner=stable_owner,
        cuda_tensor=stable_owner,
        metadata={"surface_cloned": True},
        _host_materializer=materialize,
        _release_callback=released.set,
    )
    kwargs: dict[str, Any] = {
        "seq": 1,
        "frame_idx": 4,
        "source_time_s": 0.16,
        "wall_time_ms": 0.0,
        "epoch": 2,
        "frame": None,
        "width": 12,
        "height": 8,
        "fps": 25.0,
        "flags": {},
        lease_field: lease,
    }
    packet = FramePacket(**kwargs)
    carried = getattr(packet, lease_field)

    results: dict[str, np.ndarray] = {}

    def preview_materialize() -> None:
        results["preview"] = carried.materialize_host_bgr(size=(6, 4))

    def detection_materialize() -> None:
        results["detection"] = carried.materialize_host_bgr(size=(12, 8))

    threads = [
        threading.Thread(target=preview_materialize),
        threading.Thread(target=detection_materialize),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert results["preview"].shape == (4, 6, 3)
    assert results["detection"].shape == (8, 12, 3)
    assert sorted(calls) == [((6, 4), None), ((12, 8), None)]
    assert carried.owner is stable_owner
    assert carried.metadata["surface_cloned"] is True
    assert carried.released is False

    carried.release()
    assert carried.released is True
    assert released.is_set()

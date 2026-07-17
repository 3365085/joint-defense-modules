from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import defense.runtime.runner as runner_module
from defense.pipelines.video_decoder import DecodedFrameLease, VideoStreamInfo
from defense.runtime.backend_pipeline import DetectionBus, PreviewBus
from defense.runtime.runner import (
    MonitorEngine,
    _adaptive_file_overlay_bridge_s,
)


class _Cache:
    def clear(self) -> None:
        return None


def _file_preview_engine() -> MonitorEngine:
    engine = MonitorEngine(_Cache())  # type: ignore[arg-type]
    engine.status.update(
        {
            "source_type": "file",
            "realtime": True,
            "source_epoch": 0,
            "detector_process_fps_cap": 60.0,
            "preview_render_fps": 25.0,
            "overlay_match_window_ms": 180.0,
            "overlay_hold_ms": 550.0,
            "overlay_interpolate_ms": 400.0,
            "overlay_max_age_ms": 950.0,
            "file_realtime_overlay_bridge_frames": 3.2,
            "file_realtime_overlay_bridge_min_s": 0.20,
            "file_realtime_overlay_bridge_max_s": 0.36,
        }
    )
    return engine


def _track(track_id: int = 1) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "label": "head",
        "box": [10, 20, 30, 40],
        "source": "detected",
        "hold_eligible": True,
    }


def test_file_overlay_bridge_uses_measured_cycle_not_configured_cap() -> None:
    status = {
        "detector_process_fps_cap": 60.0,
        "preview_render_fps": 25.0,
        "playback_speed": 1.0,
        "fps": 10.0,
        "detector_cycle_ms_distribution": {"p95": 180.0},
        "overlay_max_age_ms": 950.0,
        "file_realtime_overlay_bridge_frames": 3.2,
        "file_realtime_overlay_bridge_min_s": 0.20,
        "file_realtime_overlay_bridge_max_s": 0.36,
    }
    records = [
        {"video_time_s": 1.0},
        {"video_time_s": 1.1},
        {"video_time_s": 1.2},
    ]

    bridge_s = _adaptive_file_overlay_bridge_s(status, records)

    assert bridge_s == pytest.approx(0.616)
    assert bridge_s > status["file_realtime_overlay_bridge_max_s"]
    assert bridge_s <= status["overlay_max_age_ms"] / 1000.0


def test_slow_detector_overlay_is_held_without_periodic_empty_frame() -> None:
    engine = _file_preview_engine()
    engine.status.update(
        {
            "fps": 5.0,
            "detector_cycle_ms_distribution": {"p95": 200.0},
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 1,
            "source_epoch": 0,
            "frame_idx": 300,
            "video_time_s": 10.0,
            "ppe_tracks": [_track()],
        }
    )

    assert engine._select_preview_overlay(10.0, 0) is not None
    held = engine._select_preview_overlay(10.50, 0)

    assert held is not None
    assert held["held"] is True
    assert held["ppe_tracks"][0]["source"] == "held"
    assert engine._select_preview_overlay(11.0, 0) is None


def test_interpolation_keeps_discrete_alarm_state_causal() -> None:
    engine = _file_preview_engine()
    engine.overlay_timeline.extend(
        [
            {
                "overlay_seq": 1,
                "source_epoch": 0,
                "video_time_s": 1.0,
                "module_a_alert_confirmed": True,
                "module_a_attack_state_active": True,
                "reason_codes": ["A3B_MEDIA_CONFIRMED"],
                "a3b_state": "confirmed",
                "a3b_triggered": True,
                "ppe_tracks": [_track()],
            },
            {
                "overlay_seq": 2,
                "source_epoch": 0,
                "video_time_s": 1.2,
                "module_a_alert_confirmed": False,
                "module_a_attack_state_active": False,
                "reason_codes": [],
                "a3b_state": "normal",
                "a3b_triggered": False,
                "ppe_tracks": [_track()],
            },
        ]
    )

    selected = engine._select_preview_overlay(1.15, 0)

    assert selected is not None
    assert selected["interpolated"] is True
    assert selected["module_a_alert_confirmed"] is True
    assert selected["a3b_state"] == "confirmed"
    assert selected["reason_codes"] == ["A3B_MEDIA_CONFIRMED"]


def test_interpolation_does_not_show_future_alarm_early() -> None:
    engine = _file_preview_engine()
    engine.overlay_timeline.extend(
        [
            {
                "overlay_seq": 1,
                "source_epoch": 0,
                "video_time_s": 1.0,
                "module_a_alert_confirmed": False,
                "a3b_state": "normal",
                "ppe_tracks": [],
            },
            {
                "overlay_seq": 2,
                "source_epoch": 0,
                "video_time_s": 1.2,
                "module_a_alert_confirmed": True,
                "a3b_state": "confirmed",
                "ppe_tracks": [],
            },
        ]
    )

    selected = engine._select_preview_overlay(1.15, 0)

    assert selected is not None
    assert selected["module_a_alert_confirmed"] is False
    assert selected["a3b_state"] == "normal"


def test_interpolation_keeps_unmatched_track_until_explicit_suppression() -> None:
    engine = _file_preview_engine()
    engine.overlay_timeline.extend(
        [
            {
                "overlay_seq": 1,
                "source_epoch": 0,
                "video_time_s": 1.0,
                "ppe_tracks": [_track(7)],
            },
            {
                "overlay_seq": 2,
                "source_epoch": 0,
                "video_time_s": 1.1,
                "ppe_tracks": [],
            },
        ]
    )

    selected = engine._select_preview_overlay(1.05, 0)

    assert selected is not None
    assert [track["track_id"] for track in selected["ppe_tracks"]] == [7]


def test_held_overlay_never_drags_boxes_across_scene_cut() -> None:
    engine = _file_preview_engine()
    engine.status.update(
        {
            "fps": 5.0,
            "detector_cycle_ms_distribution": {"p95": 200.0},
        }
    )
    engine.overlay_timeline.append(
        {
            "overlay_seq": 1,
            "source_epoch": 0,
            "frame_idx": 100,
            "video_time_s": 2.0,
            "module_a_alert_confirmed": True,
            "a3b_bbox": [100, 100, 200, 200],
            "ppe_tracks": [_track()],
        }
    )
    assert engine._select_preview_overlay(2.0, 0) is not None

    held = engine._select_preview_overlay(
        2.5,
        0,
        display_frame_idx=130,
        display_scene_cut_frame_idx=120,
    )

    assert held is not None
    assert held["ppe_tracks"] == []
    assert held["a3b_bbox"] is None
    assert held["module_a_alert_confirmed"] is True


def test_detection_coverage_uses_displayed_source_progress() -> None:
    engine = _file_preview_engine()
    engine.overlay_seq = 10
    engine.status.update(
        {
            "source_fps": 30.0,
            "source_time_s": 1.0,
            "frame_idx": 30,
        }
    )
    engine.latest_jpeg_meta = {
        "source_time_s": 2.0,
        "frame_idx": 60,
    }

    status = engine.get_status()

    assert status["processed_detection_frames"] == 10
    assert status["detection_source_coverage_ratio"] == pytest.approx(10 / 61)


def _lease(frame_idx: int) -> DecodedFrameLease:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    return DecodedFrameLease(
        frame_idx=frame_idx,
        pts_s=frame_idx / 30.0,
        width=12,
        height=8,
        pixel_format="bgr24",
        storage="host",
        decode_ms=0.1,
        host_array=frame,
    )


class _FakeDecoder:
    def __init__(self) -> None:
        self.frames = [_lease(index) for index in range(3)]
        self.info = VideoStreamInfo(
            source="fake.mp4",
            backend="opencv",
            codec="h264",
            width=12,
            height=8,
            fps=30.0,
            frame_count=3,
            duration_s=0.1,
        )
        self.closed = False
        self.frames_decoded = 0

    def read(self) -> DecodedFrameLease | None:
        if not self.frames:
            return None
        self.frames_decoded += 1
        return self.frames.pop(0)

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "requested_backend": "opencv",
            "effective_backend": "opencv",
            "codec": "h264",
            "frame_device": "host",
            "output_format": "bgr24",
            "frames_decoded": self.frames_decoded,
            "fallback_reason": "none",
            "closed": self.closed,
        }

    def close(self) -> None:
        self.closed = True


class _NeverConsumedDetectionBus(DetectionBus):
    def wait_until_consumed(self, seq: int, timeout: float = 0.2) -> bool:
        del seq, timeout
        raise AssertionError("file capture must not wait for detector consumption")


def test_file_capture_publishes_independently_of_detector_consumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decoder = _FakeDecoder()
    monkeypatch.setattr(
        runner_module,
        "create_video_decoder",
        lambda *_args, **_kwargs: decoder,
    )
    engine = MonitorEngine(_Cache())  # type: ignore[arg-type]
    engine.run_id = 1
    engine._source_epoch = 1
    engine.status.update(
        {
            "run_id": 1,
            "running": True,
            "source_epoch": 1,
            "source_type": "file",
            "realtime": False,
        }
    )
    engine.process_done_event.set()
    preview_bus = PreviewBus()
    detection_bus = _NeverConsumedDetectionBus()
    source = tmp_path / "independent-preview.mp4"
    source.write_bytes(b"fake")

    engine._backend_file_decoder_loop(
        run_id=1,
        preview_bus=preview_bus,
        detection_bus=detection_bus,
        source=str(source),
        realtime=False,
        preview_render_fps=25.0,
        detector_process_fps_cap=60.0,
        capture_max_side=960,
        file_source_fps_cap=0.0,
        decoder_preference="opencv",
        allow_cpu_fallback=True,
        gpu_id=0,
    )

    assert engine.status["error"] == ""
    assert engine.status["capture_frames_published"] == 3
    assert engine.status["detector_submission_count"] == 3
    assert detection_bus.dropped == 2


from __future__ import annotations

import numpy as np

from defense.pipelines.video_defense_pipeline import VideoDefensePipeline
from defense.runtime.backend_pipeline import FramePacket


class _TemporalDetector:
    def __init__(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        return gray + 1


def test_frame_packet_temporal_predecessor_defaults_are_compatible() -> None:
    packet = FramePacket(
        seq=1,
        frame_idx=4,
        source_time_s=0.16,
        wall_time_ms=1.0,
        epoch=0,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        width=4,
        height=4,
        fps=25.0,
        flags={},
    )

    assert packet.previous_frame is None
    assert packet.previous_frame_idx is None
    assert packet.previous_source_time_s is None


def test_runtime_frame_primes_detector_with_strict_source_predecessor() -> None:
    pipeline = VideoDefensePipeline.__new__(VideoDefensePipeline)
    pipeline.detector = _TemporalDetector()
    captured: dict[str, object] = {}

    def _run_detection(frame: np.ndarray, *, timestamp: float):
        captured["frame"] = frame
        captured["timestamp"] = timestamp
        captured["prev_gray"] = pipeline.detector.prev_gray.copy()
        captured["prev_lbp"] = pipeline.detector.prev_lbp.copy()
        return frame, object(), {"latency_breakdown": {}}, 0.0, 0.0

    pipeline._run_detection = _run_detection
    previous = np.full((8, 8, 3), 7, dtype=np.uint8)
    current = np.full((8, 8, 3), 9, dtype=np.uint8)

    _, _, info = pipeline.process_runtime_frame(
        current,
        timestamp=0.4,
        previous_frame=previous,
        current_source_frame_idx=10,
        previous_source_frame_idx=9,
        previous_source_time_s=0.36,
    )

    assert captured["timestamp"] == 0.4
    assert np.asarray(captured["prev_gray"]).shape == (640, 640)
    assert np.all(np.asarray(captured["prev_gray"]) == 7)
    assert np.all(np.asarray(captured["prev_lbp"]) == 8)
    assert info["temporal_input"] == {
        "previous_frame_applied": True,
        "current_source_frame_idx": 10,
        "previous_source_frame_idx": 9,
        "source_gap_frames": 1,
        "strict_source_predecessor": True,
        "current_source_time_s": 0.4,
        "previous_source_time_s": 0.36,
    }


def test_runtime_frame_marks_non_adjacent_predecessor_as_not_strict() -> None:
    pipeline = VideoDefensePipeline.__new__(VideoDefensePipeline)
    pipeline.detector = _TemporalDetector()
    pipeline._run_detection = lambda frame, *, timestamp: (
        frame,
        object(),
        {"latency_breakdown": {}},
        0.0,
        0.0,
    )

    _, _, info = pipeline.process_runtime_frame(
        np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp=1.0,
        previous_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        current_source_frame_idx=20,
        previous_source_frame_idx=18,
        previous_source_time_s=0.8,
    )

    assert info["temporal_input"]["previous_frame_applied"] is True
    assert info["temporal_input"]["source_gap_frames"] == 2
    assert info["temporal_input"]["strict_source_predecessor"] is False

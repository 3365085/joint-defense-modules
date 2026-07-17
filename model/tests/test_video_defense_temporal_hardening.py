from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline


FRAME = np.zeros((640, 640, 3), dtype=np.uint8)


class _ROIProvider:
    target_labels = {"person", "head", "helmet"}

    @staticmethod
    def from_detections(
        boxes: list[list[int]],
        classes: list[int],
        confidences: list[float],
    ) -> list[Any]:
        return []

    @staticmethod
    def normalize_label(label: str) -> str:
        return str(label).strip().lower()


class _Backend:
    def __init__(self, *, class_id: int = 9, label: str = "background") -> None:
        self.names = {class_id: label}
        self.class_id = class_id
        self.predict_calls = 0

    def predict(self, frame: np.ndarray) -> DetectionFrameResult:
        self.predict_calls += 1
        return DetectionFrameResult(
            image=frame,
            boxes=[],
            classes=[self.class_id],
            confidences=[0.9],
            names=self.names,
            backend="fake",
            artifact_path="",
            inference_ms=1.0,
        )


class _ModuleDetector:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.process_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def process(self, item: Any) -> Any:
        self.process_calls += 1
        return _ModuleResult(frame_idx=int(item.frame_idx))


class _ModuleResult:
    def __init__(self, *, frame_idx: int) -> None:
        self.frame_idx = frame_idx
        self.p_adv = 0.0
        self.reason_codes: list[str] = []
        self.timing_ms = 0.0
        self.details = {
            "module_a": {},
            "module_a_breakdown": {},
            "a3b": {},
            "timing": {
                "scene_context": 0.1,
                "flow": 0.2,
                "total": 0.5,
            },
        }

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "timing_ms": self.timing_ms,
            "details": self.details,
        }


class _TemporalDetector:
    def __init__(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None
        self.prev_brightness = 200.0
        self.prev_timestamp = 123.0

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        return gray + 1


def _pipeline_for_detection(
    *,
    class_id: int = 9,
    label: str = "background",
) -> tuple[VideoDefensePipeline, _Backend]:
    pipeline = object.__new__(VideoDefensePipeline)
    backend = _Backend(class_id=class_id, label=label)
    pipeline.detector_backend = backend
    pipeline.class_names = backend.names
    pipeline.roi_provider = _ROIProvider()
    pipeline.detector = _ModuleDetector()
    pipeline.frame_idx = 0
    pipeline._last_small_gray = None
    pipeline._last_detections = None
    pipeline._last_rois = None
    pipeline._last_detector_frame_idx = -1
    pipeline._last_detector_source_frame_idx = None
    pipeline._last_detector_source_time_s = None
    pipeline._current_reuse_source_frame_idx = None
    pipeline._current_reuse_source_time_s = None
    pipeline._temporal_reuse_threshold = 0.01
    pipeline._temporal_reuse_ppe_change_threshold = 0.002
    pipeline._temporal_reuse_max_gap = 2
    pipeline._temporal_reuse_max_source_time_gap_s = 0.04
    pipeline._temporal_reuse_consecutive = 0
    pipeline._temporal_reuse_max_consecutive = 3
    pipeline._temporal_reuse_target_state = {}
    pipeline._last_reuse_decision = {
        "hit": False,
        "reason": "not_evaluated",
    }
    pipeline._detector_reuse_attempt_count = 0
    pipeline._detector_reuse_hit_count = 0
    pipeline._detector_backend_predict_count = 0
    pipeline._detector_reuse_miss_reasons = {}
    pipeline._module_a_analysis_max_hz = 25.0
    pipeline._last_module_a_result = None
    pipeline._last_module_a_source_frame_idx = None
    pipeline._last_module_a_source_time_s = None
    pipeline._last_module_a_seen_source_frame_idx = None
    pipeline._last_module_a_seen_source_time_s = None
    pipeline._module_a_source_fps = None
    pipeline._module_a_cadence_attempt_count = 0
    pipeline._module_a_cadence_hit_count = 0
    pipeline._last_module_a_cadence = {
        "hit": False,
        "reason": "not_evaluated",
    }
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
    return pipeline, backend


def _suppression_pipeline() -> VideoDefensePipeline:
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


def _rebuilt_a3b_info(
    *,
    seq: int,
    fresh: bool,
    candidate_allowed: bool,
    policy_suppressed: bool,
) -> dict[str, Any]:
    reason = "low_display_target_plane_prefers_A1_A2_A3"
    return {
        "details": {
            "a3b": {
                "media_confirmed": True,
                "p_media_triggered": True,
                "media_candidate_allowed": candidate_allowed,
                "suppressed_reason": reason if policy_suppressed else "none",
                "suppression": {
                    "suppressed": policy_suppressed,
                    "reason": reason if policy_suppressed else "none",
                    "media_candidate_allowed": candidate_allowed,
                },
                "p_media_bbox": [100, 100, 200, 200],
                "a3b_result_seq": seq,
                "a3b_result_fresh": fresh,
            }
        }
    }


def test_strict_predecessor_updates_brightness_without_rewriting_timestamp() -> None:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline.detector = _TemporalDetector()
    previous = np.full((8, 8, 3), 7, dtype=np.uint8)

    applied = pipeline._inject_temporal_previous_frame(previous)

    assert applied is True
    assert np.all(pipeline.detector.prev_gray == 7)
    assert np.all(pipeline.detector.prev_lbp == 8)
    assert pipeline.detector.prev_brightness == pytest.approx(7.0)
    assert pipeline.detector.prev_timestamp == pytest.approx(123.0)


def test_module_a_cadence_lag_forces_strict_predecessor_reinjection() -> None:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline.detector = _TemporalDetector()
    pipeline._last_runtime_source_frame_idx = 11
    pipeline._last_runtime_source_time_s = 1.1
    pipeline._last_runtime_frame_shape = (8, 8)
    pipeline._last_module_a_source_frame_idx = 10
    previous = np.full((8, 8, 3), 11, dtype=np.uint8)
    current = np.full((8, 8, 3), 12, dtype=np.uint8)

    reusable = pipeline._can_reuse_internal_temporal_predecessor(
        current,
        previous,
        current_source_frame_idx=12,
        previous_source_frame_idx=11,
        previous_source_time_s=1.1,
    )

    assert reusable is False


def test_failed_predecessor_injection_clears_stale_temporal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _ = _pipeline_for_detection()
    pipeline.detector.prev_gray = np.full((8, 8), 7, dtype=np.uint8)
    pipeline.detector.prev_lbp = np.full((8, 8), 8, dtype=np.uint8)
    pipeline.detector._last_computed_lbp = np.full(
        (8, 8),
        9,
        dtype=np.uint8,
    )
    pipeline.detector.prev_brightness = 7.0
    pipeline.detector._flownet = {
        "prev_ref": object(),
        "prev_small": object(),
        "kept": object(),
    }
    pipeline.detector.overexposure = SimpleNamespace(
        _prev_gray=np.full((8, 8), 6, dtype=np.uint8)
    )
    pipeline._last_runtime_source_frame_idx = 10
    pipeline._last_runtime_source_time_s = 1.0
    pipeline._last_runtime_frame_shape = (640, 640)
    pipeline._last_module_a_source_frame_idx = 9
    monkeypatch.setattr(
        pipeline,
        "_inject_temporal_previous_frame",
        lambda _previous: False,
    )

    info = pipeline.process_runtime_frame(
        FRAME.copy(),
        timestamp=1.0334,
        previous_frame=FRAME.copy(),
        current_source_frame_idx=11,
        previous_source_frame_idx=10,
        previous_source_time_s=1.0,
    )[2]

    assert pipeline.detector.prev_gray is None
    assert pipeline.detector.prev_lbp is None
    assert pipeline.detector._last_computed_lbp is None
    assert pipeline.detector.prev_brightness is None
    assert "prev_ref" not in pipeline.detector._flownet
    assert "prev_small" not in pipeline.detector._flownet
    assert "kept" in pipeline.detector._flownet
    assert pipeline.detector.overexposure._prev_gray is None
    assert info["temporal_input"]["previous_frame_applied"] is False
    assert info["temporal_input"]["previous_frame_temporal_state_reset"] is True
    assert (
        info["temporal_input"]["previous_frame_failure_reason"]
        == "strict_predecessor_injection_failed"
    )


def test_missing_predecessor_clears_all_cached_temporal_state() -> None:
    pipeline, _ = _pipeline_for_detection()
    pipeline.detector.prev_gray = np.full((8, 8), 7, dtype=np.uint8)
    pipeline.detector.prev_lbp = np.full((8, 8), 8, dtype=np.uint8)
    pipeline.detector._last_computed_lbp = np.full(
        (8, 8),
        9,
        dtype=np.uint8,
    )
    pipeline.detector.prev_brightness = 7.0
    pipeline.detector._flownet = {
        "prev_ref": object(),
        "prev_small": object(),
        "kept": object(),
    }
    pipeline.detector.overexposure = SimpleNamespace(
        _prev_gray=np.full((8, 8), 6, dtype=np.uint8)
    )

    info = pipeline.process_runtime_frame(
        FRAME.copy(),
        timestamp=1.0,
        previous_frame=None,
        current_source_frame_idx=10,
        previous_source_frame_idx=None,
        previous_source_time_s=None,
    )[2]

    assert pipeline.detector.prev_gray is None
    assert pipeline.detector.prev_lbp is None
    assert pipeline.detector._last_computed_lbp is None
    assert pipeline.detector.prev_brightness is None
    assert "prev_ref" not in pipeline.detector._flownet
    assert "prev_small" not in pipeline.detector._flownet
    assert "kept" in pipeline.detector._flownet
    assert pipeline.detector.overexposure._prev_gray is None
    assert info["temporal_input"]["previous_frame_temporal_state_reset"] is True
    assert (
        info["temporal_input"]["previous_frame_failure_reason"]
        == "strict_predecessor_missing"
    )


@pytest.mark.parametrize(
    ("frame_idx", "source_idx", "source_time_s", "reason"),
    [
        (3, 11, 1.05, "processed_gap_exceeded"),
        (1, 13, 1.05, "source_frame_gap_exceeded"),
        (1, 11, 1.05, "source_time_gap_exceeded"),
        (1, None, 1.05, "source_context_missing"),
    ],
)
def test_reuse_fails_closed_when_any_gap_is_unsafe(
    frame_idx: int,
    source_idx: int | None,
    source_time_s: float,
    reason: str,
) -> None:
    pipeline, _ = _pipeline_for_detection()
    pipeline.frame_idx = frame_idx
    pipeline._last_detector_frame_idx = 0
    pipeline._last_detector_source_frame_idx = 10
    pipeline._last_detector_source_time_s = 1.0
    pipeline._last_small_gray = np.zeros((160, 160), dtype=np.uint8)
    pipeline._last_detections = SimpleNamespace(
        boxes=[],
        classes=[9],
        names={9: "background"},
    )
    pipeline._last_rois = []

    reused, rois, _, _ = pipeline._maybe_reuse_detections(
        FRAME,
        current_source_frame_idx=source_idx,
        current_source_time_s=source_time_s,
    )

    assert reused is None
    assert rois is None
    assert pipeline._last_reuse_decision["reason"] == reason


def test_stable_frames_reuse_and_reduce_backend_predict_calls() -> None:
    pipeline, backend = _pipeline_for_detection()

    first = pipeline._run_detection_with_source_context(
        FRAME,
        timestamp=1.0,
        current_source_frame_idx=10,
        current_source_time_s=1.0,
    )[2]
    second = pipeline._run_detection_with_source_context(
        FRAME.copy(),
        timestamp=1.016,
        current_source_frame_idx=11,
        current_source_time_s=1.016,
    )[2]
    third = pipeline._run_detection_with_source_context(
        FRAME.copy(),
        timestamp=1.033,
        current_source_frame_idx=12,
        current_source_time_s=1.033,
    )[2]

    assert backend.predict_calls == 1
    assert first["latency_breakdown"]["detector_reuse_hit"] is False
    assert second["latency_breakdown"]["detector_reuse_hit"] is True
    assert third["latency_breakdown"]["detector_reuse_hit"] is True
    counters = third["latency_breakdown"]["detector_reuse_counters"]
    assert counters["attempt_count"] == 3
    assert counters["hit_count"] == 2
    assert counters["backend_predict_count"] == 1
    assert counters["miss_reasons"] == {"no_cached_detection": 1}


def test_two_frame_reuse_is_blocked_at_30_fps_source_time() -> None:
    pipeline, backend = _pipeline_for_detection()
    pipeline._run_detection_with_source_context(
        FRAME,
        timestamp=1.0,
        current_source_frame_idx=10,
        current_source_time_s=1.0,
    )
    pipeline._run_detection_with_source_context(
        FRAME.copy(),
        timestamp=1.033,
        current_source_frame_idx=11,
        current_source_time_s=1.033,
    )

    info = pipeline._run_detection_with_source_context(
        FRAME.copy(),
        timestamp=1.067,
        current_source_frame_idx=12,
        current_source_time_s=1.067,
    )[2]

    assert backend.predict_calls == 2
    assert info["latency_breakdown"]["detector_reuse_hit"] is False
    decision = info["latency_breakdown"]["detector_reuse"]
    assert decision["reason"] == "source_time_gap_exceeded"
    assert decision["processed_gap"] == 2
    assert decision["source_frame_gap"] == 2
    assert decision["max_source_time_gap_s"] == pytest.approx(0.04)


def test_module_a_analysis_is_source_time_capped_at_30hz_for_60fps_input() -> None:
    pipeline, _ = _pipeline_for_detection()
    infos = []
    previous_frame = None
    previous_source_idx = None
    previous_source_time_s = None
    for source_idx, source_time_s in zip(
        range(10, 14),
        (1.0, 1.0167, 1.0334, 1.0501),
        strict=True,
    ):
        infos.append(
            pipeline.process_runtime_frame(
                FRAME.copy(),
                timestamp=source_time_s,
                previous_frame=previous_frame,
                current_source_frame_idx=source_idx,
                previous_source_frame_idx=previous_source_idx,
                previous_source_time_s=previous_source_time_s,
            )[2]
        )
        previous_frame = FRAME.copy()
        previous_source_idx = source_idx
        previous_source_time_s = source_time_s

    assert pipeline.detector.process_calls == 2
    assert [
        info["latency_breakdown"]["module_a_reuse_hit"] for info in infos
    ] == [False, True, True, False]
    assert infos[2]["latency_breakdown"]["module_a_total_ms"] == 0.0
    counters = infos[-1]["latency_breakdown"]["module_a_cadence_counters"]
    assert counters == {
        "attempt_count": 4,
        "hit_count": 2,
        "hit_rate": pytest.approx(0.5),
    }
    assert pipeline.detector.source_fps == pytest.approx(60.0, rel=0.01)


def test_module_a_analysis_keeps_every_30fps_source_frame() -> None:
    pipeline, _ = _pipeline_for_detection()
    infos = []
    previous_frame = None
    previous_source_idx = None
    previous_source_time_s = None
    for source_idx, source_time_s in zip(
        range(10, 13),
        (1.0, 1.0334, 1.0668),
        strict=True,
    ):
        infos.append(
            pipeline.process_runtime_frame(
                FRAME.copy(),
                timestamp=source_time_s,
                previous_frame=previous_frame,
                current_source_frame_idx=source_idx,
                previous_source_frame_idx=previous_source_idx,
                previous_source_time_s=previous_source_time_s,
            )[2]
        )
        previous_frame = FRAME.copy()
        previous_source_idx = source_idx
        previous_source_time_s = source_time_s

    assert pipeline.detector.process_calls == 3
    assert all(
        not info["latency_breakdown"]["module_a_reuse_hit"] for info in infos
    )


def test_pipeline_preserves_rebuilt_stage_timing_in_latency_breakdown() -> None:
    pipeline, _ = _pipeline_for_detection()

    info = pipeline._run_detection_with_source_context(
        FRAME,
        timestamp=1.0,
        current_source_frame_idx=10,
        current_source_time_s=1.0,
    )[2]

    assert info["latency_breakdown"]["module_a_breakdown"] == {
        "scene_context": pytest.approx(0.1),
        "flow": pytest.approx(0.2),
    }
    assert info["details"]["timing"]["scene_context"] == pytest.approx(0.1)
    assert info["details"]["timing"]["flow"] == pytest.approx(0.2)
    assert info["details"]["timing"]["total"] == pytest.approx(0.5)
    assert info["details"]["timing"]["pipeline_ms"] >= 0.0
    assert info["details"]["timing"]["detector_ms"] == pytest.approx(1.0)
    assert info["details"]["timing"]["module_a_ms"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("label", "changed_value", "source_idx", "source_time_s", "reason"),
    [
        (
            "head",
            1,
            11,
            1.03,
            "ppe_change_exceeds_tighter_reuse_threshold",
        ),
        ("background", 20, 11, 1.03, "global_change_exceeded"),
        ("background", 0, 20, 1.03, "source_frame_gap_exceeded"),
    ],
)
def test_ppe_motion_and_large_source_gap_force_backend_inference(
    label: str,
    changed_value: int,
    source_idx: int,
    source_time_s: float,
    reason: str,
) -> None:
    class_id = 1 if label == "head" else 9
    pipeline, backend = _pipeline_for_detection(
        class_id=class_id,
        label=label,
    )
    pipeline._run_detection_with_source_context(
        FRAME,
        timestamp=1.0,
        current_source_frame_idx=10,
        current_source_time_s=1.0,
    )

    changed = np.full_like(FRAME, changed_value)
    info = pipeline._run_detection_with_source_context(
        changed,
        timestamp=source_time_s,
        current_source_frame_idx=source_idx,
        current_source_time_s=source_time_s,
    )[2]

    assert backend.predict_calls == 2
    assert info["latency_breakdown"]["detector_reuse_hit"] is False
    assert info["latency_breakdown"]["detector_reuse"]["reason"] == reason


def test_reset_clears_source_reuse_lineage_and_counters() -> None:
    pipeline, backend = _pipeline_for_detection()
    pipeline._run_detection_with_source_context(
        FRAME,
        timestamp=1.0,
        current_source_frame_idx=10,
        current_source_time_s=1.0,
    )
    assert backend.predict_calls == 1

    pipeline.reset()

    assert pipeline._last_detector_source_frame_idx is None
    assert pipeline._last_detector_source_time_s is None
    assert pipeline._last_detections is None
    assert pipeline._detector_reuse_attempt_count == 0
    assert pipeline._detector_reuse_hit_count == 0
    assert pipeline._detector_backend_predict_count == 0
    assert pipeline._last_reuse_decision["reason"] == "reset"


def _offline_window_trace(delay_s: float) -> tuple[list[float], list[int]]:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline.frame_idx = 0
    pipeline._offline_default_source_fps = 20.0
    pipeline._offline_explicit_timestamp_offset_s = None
    pipeline._current_reuse_source_frame_idx = None
    pipeline._current_reuse_source_time_s = None
    timestamps: list[float] = []
    windows: list[int] = []
    process_fps = 15.0
    previous_timestamp: float | None = None

    def fake_run_detection(
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> tuple[np.ndarray, object, dict[str, Any], float, float]:
        nonlocal previous_timestamp, process_fps
        timestamps.append(timestamp)
        if previous_timestamp is not None:
            dt = timestamp - previous_timestamp
            if 0.005 <= dt <= 2.0:
                instant = max(1.0, min(60.0, 1.0 / dt))
                process_fps = 0.85 * process_fps + 0.15 * instant
        previous_timestamp = timestamp
        windows.append(round(process_fps * 2.0))
        pipeline.frame_idx += 1
        return frame, object(), {"latency_breakdown": {}}, 0.0, 0.0

    pipeline._run_detection = fake_run_detection
    for _ in range(6):
        pipeline.process_frame(FRAME)
        if delay_s:
            time.sleep(delay_s)
    return timestamps, windows


def test_offline_default_cadence_is_independent_of_execution_speed() -> None:
    fast_timestamps, fast_windows = _offline_window_trace(0.0)
    slow_timestamps, slow_windows = _offline_window_trace(0.01)

    assert fast_timestamps == pytest.approx(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    assert slow_timestamps == pytest.approx(fast_timestamps)
    assert slow_windows == fast_windows


def test_offline_zero_based_explicit_timestamps_keep_source_cadence() -> None:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline.frame_idx = 0
    pipeline._offline_default_source_fps = 25.0
    pipeline._offline_explicit_timestamp_offset_s = None
    pipeline._current_reuse_source_frame_idx = None
    pipeline._current_reuse_source_time_s = None
    captured: list[float] = []

    def fake_run_detection(
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> tuple[np.ndarray, object, dict[str, Any], float, float]:
        captured.append(timestamp)
        pipeline.frame_idx += 1
        return frame, object(), {"latency_breakdown": {}}, 0.0, 0.0

    pipeline._run_detection = fake_run_detection
    pipeline.process_frame(FRAME, timestamp=0.0, source_fps=25.0)
    pipeline.process_frame(FRAME, timestamp=0.04, source_fps=25.0)

    assert captured == pytest.approx([0.04, 0.08])


@pytest.mark.parametrize(
    ("fresh", "candidate_allowed", "policy_suppressed", "reason"),
    [
        (False, True, False, "stale_result"),
        (True, False, False, "media_candidate_not_allowed"),
        (True, False, True, "policy_suppressed"),
    ],
)
def test_stale_or_policy_suppressed_result_cannot_start_suppression(
    fresh: bool,
    candidate_allowed: bool,
    policy_suppressed: bool,
    reason: str,
) -> None:
    pipeline = _suppression_pipeline()
    info = _rebuilt_a3b_info(
        seq=2,
        fresh=fresh,
        candidate_allowed=candidate_allowed,
        policy_suppressed=policy_suppressed,
    )

    filtered, _ = pipeline._apply_a3b_suppression(
        FRAME,
        _detections([120, 120, 160, 160]),
        [],
        info,
        source_timestamp_s=1.0,
    )

    assert filtered.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert info["a3b_suppression_refresh_blocked_reason"] == reason


def test_confirmed_hold_bridges_short_policy_gap_without_renewing() -> None:
    pipeline = _suppression_pipeline()
    pipeline._apply_a3b_suppression(
        FRAME,
        _detections([120, 120, 160, 160]),
        [],
        _rebuilt_a3b_info(
            seq=1,
            fresh=True,
            candidate_allowed=True,
            policy_suppressed=False,
        ),
        source_timestamp_s=10.0,
    )
    blocked_info = _rebuilt_a3b_info(
        seq=2,
        fresh=True,
        candidate_allowed=False,
        policy_suppressed=True,
    )

    filtered, _ = pipeline._apply_a3b_suppression(
        FRAME,
        _detections([120, 120, 160, 160]),
        [],
        blocked_info,
        source_timestamp_s=11.0,
    )

    assert filtered.boxes == []
    assert pipeline._a3b_suppress_lease_expires_at_s == 11.5
    assert pipeline._a3b_suppress_remaining == 15
    assert pipeline._a3b_suppress_result_seq == 1
    assert "a3b_suppression_refreshed" not in blocked_info
    assert blocked_info["a3b_suppression_remaining_s"] == 0.5
    assert blocked_info["a3b_suppression_lease_clamped"] is True

    released_info = _rebuilt_a3b_info(
        seq=2,
        fresh=True,
        candidate_allowed=False,
        policy_suppressed=True,
    )
    released, _ = pipeline._apply_a3b_suppression(
        FRAME,
        _detections([120, 120, 160, 160]),
        [],
        released_info,
        source_timestamp_s=11.5,
    )

    assert released.boxes == [[120, 120, 160, 160]]
    assert pipeline._a3b_suppress_remaining == 0
    assert pipeline._a3b_suppress_bbox is None
    assert released_info["a3b_suppression_released"] is True

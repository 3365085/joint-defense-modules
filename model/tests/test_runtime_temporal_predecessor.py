from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from defense.module_a.types import ModuleAInput, ModuleAResult
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline
from defense.runtime.backend_pipeline import FramePacket
from defense.runtime.frame_processor import FrameProcessor


class _TemporalDetector:
    def __init__(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None
        self.lbp_calls = 0

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        self.lbp_calls += 1
        return gray + 1


class _FakeBackend:
    def predict(self, frame: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(
            image=frame,
            boxes=[],
            classes=[],
            confidences=[],
            names={},
            backend="fake",
            artifact_path="",
            inference_ms=0.0,
            preprocess_ms=0.0,
            input_device="host",
            input_format="bgr24",
        )


class _FakeROIProvider:
    target_labels: set[str] = set()

    def from_detections(
        self,
        _boxes: list[object],
        _classes: list[object],
        _confidences: list[object],
    ) -> list[object]:
        return []

    def normalize_label(self, label: str) -> str:
        return label


class _CapturingModuleADetector(_TemporalDetector):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[ModuleAInput] = []

    def process(self, module_input: ModuleAInput) -> ModuleAResult:
        self.inputs.append(module_input)
        return ModuleAResult(
            frame_idx=module_input.frame_idx,
            p_adv=0.0,
            single_frame_suspicious=False,
            alert_confirmed=False,
            attack_state_active=False,
            reason_codes=[],
            features={},
            details={
                "module_a_features": {
                    "static_media": {
                        "result_contract_source": "rebuilt",
                        "media_confirmed": False,
                        "a3b_source_frame_idx": module_input.frame_idx,
                        "a3b_source_timestamp": module_input.timestamp,
                        "a3b_result_fresh": False,
                    }
                }
            },
        )


def _lineage_pipeline() -> VideoDefensePipeline:
    pipeline = VideoDefensePipeline.__new__(VideoDefensePipeline)
    pipeline.detector = _CapturingModuleADetector()
    pipeline.detector_backend = _FakeBackend()
    pipeline.class_names = {}
    pipeline.roi_provider = _FakeROIProvider()
    pipeline.frame_idx = 0
    pipeline._current_reuse_source_frame_idx = None
    pipeline._current_reuse_source_time_s = None
    pipeline._last_detector_frame_idx = -1
    pipeline._last_detector_source_frame_idx = None
    pipeline._last_detector_source_time_s = None
    pipeline._detector_backend_predict_count = 0
    pipeline._last_reuse_decision = {
        "hit": False,
        "reason": "test_forces_detection",
    }
    pipeline._maybe_reuse_detections = (
        lambda _frame, **_kwargs: (None, None, 0.0, 0.0)
    )
    pipeline._apply_a3b_suppression = (
        lambda _frame, detections, rois, _info, **_kwargs: (
            detections,
            rois,
        )
    )
    return pipeline


def _public_status(
    pipeline: VideoDefensePipeline,
    *,
    source_frame_idx: int,
    source_time_s: float,
    info: dict[str, object],
) -> dict[str, object]:
    bundle = SimpleNamespace(
        pipeline=pipeline,
        config={"module_a": {}, "runtime": {}},
        backend="fake",
        model_family="fake",
        artifact_path="",
    )
    processor = FrameProcessor(bundle)
    return processor._build_status(
        source_type="file",
        source="fake.mp4",
        profile="test",
        realtime=True,
        frame_idx=source_frame_idx,
        video_time_s=source_time_s,
        source_fps=30.0,
        fps=30.0,
        dropped_frames=35,
        info=info,
        ppe={},
        ppe_tracks=[],
        display_options={},
        feature_options={},
        custom_model={},
        redetect_budget_ok=False,
        redetect_count=0,
        redetect_ms=0.0,
        processing_ms=1.0,
        target_frame_budget_ms=33.0,
        raw_boxes_count=0,
        a3b_soft={
            "triggered": False,
            "observed_score": 0.0,
            "confirmed_score": 0.0,
            "confidence": 0.0,
            "display_score": 0.0,
            "state": "normal",
            "effective_bbox": None,
            "triggered_source": "none",
            "reason": "",
            "debug": {},
        },
    )


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
        "previous_frame_injected": True,
        "previous_frame_reused_internal_state": False,
        "previous_frame_temporal_state_reset": False,
        "previous_frame_failure_reason": "none",
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


def test_consecutive_runtime_frames_reuse_detector_temporal_state() -> None:
    pipeline = VideoDefensePipeline.__new__(VideoDefensePipeline)
    pipeline.detector = _TemporalDetector()

    def _run_detection(frame: np.ndarray, *, timestamp: float):
        del timestamp
        frame_640 = (
            frame
            if frame.shape[:2] == (640, 640)
            else np.full((640, 640, 3), int(frame[0, 0, 0]), dtype=np.uint8)
        )
        gray = frame_640[:, :, 0]
        pipeline.detector.prev_gray = gray
        pipeline.detector.prev_lbp = pipeline.detector._compute_lbp(gray)
        return frame_640, object(), {"latency_breakdown": {}}, 0.0, 0.0

    pipeline._run_detection = _run_detection
    frame_9 = np.full((8, 8, 3), 9, dtype=np.uint8)
    frame_10 = np.full((8, 8, 3), 10, dtype=np.uint8)
    frame_11 = np.full((8, 8, 3), 11, dtype=np.uint8)

    pipeline.process_runtime_frame(
        frame_10,
        timestamp=0.4,
        previous_frame=frame_9,
        current_source_frame_idx=10,
        previous_source_frame_idx=9,
        previous_source_time_s=0.36,
    )
    first_call_count = pipeline.detector.lbp_calls

    _, _, info = pipeline.process_runtime_frame(
        frame_11,
        timestamp=0.44,
        previous_frame=frame_10,
        current_source_frame_idx=11,
        previous_source_frame_idx=10,
        previous_source_time_s=0.4,
    )

    assert first_call_count == 2
    assert pipeline.detector.lbp_calls == 3
    assert info["temporal_input"]["previous_frame_applied"] is True
    assert info["temporal_input"]["previous_frame_injected"] is False
    assert info["temporal_input"]["previous_frame_reused_internal_state"] is True
    assert np.all(pipeline.detector.prev_gray == 11)


def test_runtime_module_a_uses_source_frame_idx_after_latest_only_drop() -> None:
    pipeline = _lineage_pipeline()
    first_frame = np.full((8, 8, 3), 10, dtype=np.uint8)
    predecessor = np.full((8, 8, 3), 136, dtype=np.uint8)
    current_frame = np.full((8, 8, 3), 137, dtype=np.uint8)

    pipeline.process_runtime_frame(
        first_frame,
        timestamp=1.0,
        current_source_frame_idx=100,
    )
    _, _, info = pipeline.process_runtime_frame(
        current_frame,
        timestamp=1.37,
        previous_frame=predecessor,
        current_source_frame_idx=137,
        previous_source_frame_idx=136,
        previous_source_time_s=1.36,
    )

    detector = pipeline.detector
    assert isinstance(detector, _CapturingModuleADetector)
    assert [item.frame_idx for item in detector.inputs] == [100, 137]
    assert detector.inputs[-1].timestamp == 1.37
    assert info["temporal_input"]["current_source_frame_idx"] == 137
    assert info["temporal_input"]["previous_source_frame_idx"] == 136
    assert info["temporal_input"]["strict_source_predecessor"] is True

    lineage = info["details"]["runtime_frame_lineage"]
    assert lineage == {
        "processed_frame_idx": 1,
        "source_frame_idx": 137,
        "module_a_input_frame_idx": 137,
    }
    assert lineage["processed_frame_idx"] != lineage["source_frame_idx"]

    status = _public_status(
        pipeline,
        source_frame_idx=137,
        source_time_s=1.37,
        info=info,
    )
    assert status["a3b_source_frame_idx"] == 137
    assert status["module_a_processed_frame_idx"] == 1
    assert status["module_a_source_frame_idx"] == 137
    assert status["module_a_input_frame_idx"] == 137

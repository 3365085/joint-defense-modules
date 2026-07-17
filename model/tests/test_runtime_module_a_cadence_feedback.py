from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from defense.module_a.types import ModuleAInput, ModuleAResult
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline


ANALYSIS_MAX_HZ = 25.0


class _CadenceDetector:
    def __init__(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None
        self.inputs: list[ModuleAInput] = []
        self.predecessor_markers: list[int | None] = []

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        return gray

    def process(self, module_input: ModuleAInput) -> ModuleAResult:
        self.inputs.append(module_input)
        self.predecessor_markers.append(
            None if self.prev_gray is None else int(self.prev_gray[0, 0])
        )
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

    @staticmethod
    def normalize_label(label: str) -> str:
        return label


def _pipeline() -> VideoDefensePipeline:
    pipeline = VideoDefensePipeline.__new__(VideoDefensePipeline)
    pipeline.detector = _CadenceDetector()
    pipeline.detector_backend = _FakeBackend()
    pipeline.class_names = {}
    pipeline.roi_provider = _FakeROIProvider()
    pipeline.frame_idx = 0
    pipeline._module_a_analysis_max_hz = ANALYSIS_MAX_HZ
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


@pytest.mark.parametrize(
    ("source_fps", "source_step"),
    [
        (30.0, 3),
        (60.0, 5),
    ],
)
def test_latest_only_cadence_reuses_without_losing_strict_predecessor(
    source_fps: float,
    source_step: int,
) -> None:
    processed_interval_s = source_step / source_fps
    assert processed_interval_s > 1.0 / ANALYSIS_MAX_HZ

    pipeline = _pipeline()
    source_indices = [source_step * offset for offset in range(8)]
    reuse_hits: list[bool] = []
    infos: list[dict[str, object]] = []

    for source_idx in source_indices:
        kwargs: dict[str, object] = {}
        if source_idx > 0:
            predecessor_idx = source_idx - 1
            kwargs = {
                "previous_frame": np.full(
                    (8, 8, 3), predecessor_idx, dtype=np.uint8
                ),
                "previous_source_frame_idx": predecessor_idx,
                "previous_source_time_s": predecessor_idx / source_fps,
            }
        _, _, info = pipeline.process_runtime_frame(
            np.full((8, 8, 3), source_idx, dtype=np.uint8),
            timestamp=source_idx / source_fps,
            current_source_frame_idx=source_idx,
            **kwargs,
        )
        latency = info["latency_breakdown"]
        assert isinstance(latency, dict)
        reuse_hits.append(bool(latency["module_a_reuse_hit"]))
        infos.append(info)

    # A slow latest-only consumer must not fall into the feedback loop where
    # every processed frame pays the full Module A cost merely because source
    # time advanced by more than the configured analysis period.
    assert reuse_hits[0] is False
    assert any(reuse_hits[1:]), reuse_hits

    # Reuse is a bounded cadence, not a switch that silently disables Module A.
    assert any(not hit for hit in reuse_hits[1:]), reuse_hits

    analysis_indices = [
        source_idx
        for source_idx, reuse_hit in zip(
            source_indices, reuse_hits, strict=True
        )
        if not reuse_hit
    ]
    detector = pipeline.detector
    assert isinstance(detector, _CadenceDetector)
    assert [item.frame_idx for item in detector.inputs] == analysis_indices
    assert 1 < len(detector.inputs) < len(source_indices)

    analysis_position = 0
    for source_idx, reuse_hit, info in zip(
        source_indices, reuse_hits, infos, strict=True
    ):
        cadence = info["latency_breakdown"]["module_a_cadence"]
        assert cadence["analysis_interval_s"] == pytest.approx(
            1.0 / ANALYSIS_MAX_HZ
        )
        if reuse_hit or source_idx == 0:
            continue

        temporal = info["temporal_input"]
        assert temporal["previous_frame_applied"] is True
        assert temporal["strict_source_predecessor"] is True
        assert temporal["previous_source_frame_idx"] == source_idx - 1
        assert temporal["source_gap_frames"] == 1

        analysis_position += 1
        assert detector.predecessor_markers[analysis_position] == source_idx - 1


def test_reused_module_a_cycle_does_not_materialize_previous_gpu_frame() -> None:
    pipeline = _pipeline()
    pipeline.process_runtime_frame(
        np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp=0.0,
        current_source_frame_idx=0,
    )
    provider_calls: list[int] = []

    def provider(marker: int) -> np.ndarray:
        provider_calls.append(marker)
        return np.full((8, 8, 3), marker, dtype=np.uint8)

    _, _, reused_info = pipeline.process_runtime_frame(
        np.full((8, 8, 3), 5, dtype=np.uint8),
        timestamp=5.0 / 60.0,
        current_source_frame_idx=5,
        previous_source_frame_idx=4,
        previous_source_time_s=4.0 / 60.0,
        previous_frame_provider=lambda: provider(4),
    )

    assert reused_info["latency_breakdown"]["module_a_reuse_hit"] is True
    assert provider_calls == []

    _, _, analyzed_info = pipeline.process_runtime_frame(
        np.full((8, 8, 3), 10, dtype=np.uint8),
        timestamp=10.0 / 60.0,
        current_source_frame_idx=10,
        previous_source_frame_idx=9,
        previous_source_time_s=9.0 / 60.0,
        previous_frame_provider=lambda: provider(9),
    )

    assert analyzed_info["latency_breakdown"]["module_a_reuse_hit"] is False
    assert analyzed_info["temporal_input"]["strict_source_predecessor"] is True
    assert provider_calls == [9]

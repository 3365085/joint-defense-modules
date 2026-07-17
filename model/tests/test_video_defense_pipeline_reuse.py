from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline


def _pipeline_without_init() -> VideoDefensePipeline:
    pipeline = object.__new__(VideoDefensePipeline)
    pipeline.frame_idx = 1
    pipeline._last_rois = [object()]
    pipeline._last_detector_frame_idx = 0
    pipeline._last_detector_source_frame_idx = 0
    pipeline._last_detector_source_time_s = 0.0
    pipeline._temporal_reuse_threshold = 1.0
    pipeline._temporal_reuse_ppe_change_threshold = 0.0
    pipeline._temporal_reuse_max_gap = 2
    pipeline._temporal_reuse_max_source_time_gap_s = 0.04
    pipeline._last_reuse_decision = {"hit": False, "reason": "not_evaluated", "ppe_sensitive": False}
    pipeline._last_small_gray = np.zeros((160, 160), dtype=np.uint8)
    return pipeline


def test_ppe_sensitive_detection_reuse_uses_tighter_threshold() -> None:
    pipeline = _pipeline_without_init()
    pipeline._last_detections = SimpleNamespace(classes=[1], names={1: "helmet"})
    frame = np.full((640, 640, 3), 8, dtype=np.uint8)

    detections, rois, _, change_score = pipeline._maybe_reuse_detections(
        frame,
        current_source_frame_idx=1,
        current_source_time_s=0.04,
    )

    assert detections is None
    assert rois is None
    assert change_score > 0.0
    assert pipeline._last_reuse_decision["reason"] == "ppe_change_exceeds_tighter_reuse_threshold"
    assert pipeline._last_reuse_decision["ppe_sensitive"] is True


def test_non_ppe_detection_can_reuse_when_general_threshold_allows() -> None:
    pipeline = _pipeline_without_init()
    pipeline._last_detections = SimpleNamespace(classes=[9], names={9: "background"})
    frame = np.full((640, 640, 3), 8, dtype=np.uint8)

    detections, rois, _, _ = pipeline._maybe_reuse_detections(
        frame,
        current_source_frame_idx=1,
        current_source_time_s=0.04,
    )

    assert detections is pipeline._last_detections
    assert rois is pipeline._last_rois
    assert pipeline._last_reuse_decision["hit"] is True
    assert pipeline._last_reuse_decision["ppe_sensitive"] is False

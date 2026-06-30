from __future__ import annotations

import numpy as np
import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.postprocess import PPEDisplayTracker


def _detections(boxes, classes, confidences, names=None):
    return DetectionFrameResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        boxes=[list(box) for box in boxes],
        classes=list(classes),
        confidences=list(confidences),
        names=names or {0: "helmet", 1: "head", 2: "person"},
        backend="fake",
        artifact_path="fake.pt",
        inference_ms=0.0,
    )


@pytest.mark.skip(reason="超前契约未实装:as_dict()不输出fresh_detection/display_box_source/last_detected_box,held外推+lineage契约未实装")
def test_held_track_extrapolates_at_most_one_bounded_frame() -> None:
    tracker = PPEDisplayTracker(
        hold_frames=4,
        small_hold_frames=4,
        smooth_alpha=1.0,
        hold_last_box=True,
        show_held_boxes=True,
    )
    ppe = {"helmet_fp_suppression": {}}

    tracker.update(_detections([[100, 100, 130, 130]], [0], [0.90]), ppe, (640, 640))
    tracker.update(_detections([[112, 100, 142, 130]], [0], [0.90]), ppe, (640, 640))

    first_miss = tracker.update(_detections([], [], []), ppe, (640, 640))
    assert first_miss[0]["source"] == "held"
    assert first_miss[0]["fresh_detection"] is False
    assert first_miss[0]["display_box_source"] == "held_extrapolated_one_frame"
    assert first_miss[0]["last_detected_box"] == [112, 100, 142, 130]

    first_held_box = list(first_miss[0]["box"])
    second_miss = tracker.update(_detections([], [], []), ppe, (640, 640))
    assert second_miss[0]["display_box_source"] == "held_static"
    assert second_miss[0]["box"] == first_held_box

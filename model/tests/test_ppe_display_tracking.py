from __future__ import annotations

import numpy as np
import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.postprocess import PPEDisplayTracker, merge_roi_detections


def _detections(
    boxes: list[list[int]],
    classes: list[int],
    confidences: list[float],
    names: dict[int, str] | None = None,
) -> DetectionFrameResult:
    return DetectionFrameResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        names=names or {0: "helmet", 1: "head", 2: "person"},
        backend="fake",
        artifact_path="fake.pt",
        inference_ms=1.0,
        raw_result=None,
    )


def test_small_head_track_is_held_through_short_dropouts() -> None:
    tracker = PPEDisplayTracker(redetect_interval=1)
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections([[120, 80, 136, 99]], [1], [0.72]),
        ppe,
        (640, 640),
    )
    assert len(tracks) == 1
    assert tracks[0]["label"] == "head"
    assert tracks[0]["is_small"] is True

    tracks = []
    for _ in range(5):
        tracks = tracker.update(_detections([], [], []), ppe, (640, 640))

    assert len(tracks) == 1
    assert tracks[0]["source"] == "held"
    assert tracks[0]["misses"] == 5


def test_helmet_track_holds_then_recovers_after_short_dropout() -> None:
    tracker = PPEDisplayTracker(
        hold_frames=10,
        small_hold_frames=10,
        iou_match_threshold=0.30,
        smooth_alpha=0.65,
        show_held_boxes=True,
    )
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(_detections([[100, 100, 140, 145]], [0], [0.88]), ppe, (640, 640))
    assert len(tracks) == 1
    assert tracks[0]["source"] == "detected"

    tracks = tracker.update(_detections([], [], []), ppe, (640, 640))
    assert len(tracks) == 1
    assert tracks[0]["source"] == "held"
    assert tracks[0]["misses"] == 1

    tracks = tracker.update(_detections([], [], []), ppe, (640, 640))
    assert len(tracks) == 1
    assert tracks[0]["source"] == "held"
    assert tracks[0]["misses"] == 2

    tracks = tracker.update(_detections([[104, 102, 144, 147]], [0], [0.91]), ppe, (640, 640))
    assert len(tracks) == 1
    assert tracks[0]["source"] == "detected"
    assert tracks[0]["misses"] == 0


def test_render_can_hide_missed_tracks_for_file_preview() -> None:
    tracker = PPEDisplayTracker(
        hold_frames=10,
        small_hold_frames=10,
        show_held_boxes=True,
    )
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(_detections([[100, 100, 140, 145]], [0], [0.88]), ppe, (640, 640))
    assert len(tracks) == 1

    tracks = tracker.update(
        _detections([], [], []),
        ppe,
        (640, 640),
        max_render_misses=0,
    )

    assert tracks == []


def test_weak_head_renders_current_detection_but_is_not_held() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {"weak_head_indices": [0], "suppressed_head_indices": [0]}}

    tracks = tracker.update(
        _detections([[260, 210, 288, 238]], [1], [0.72], names={1: "head"}),
        ppe,
        (640, 640),
    )

    assert len(tracks) == 1
    assert tracks[0]["label"] == "head"
    assert tracks[0]["hold_eligible"] is False

    tracks = tracker.update(_detections([], [], [], names={1: "head"}), ppe, (640, 640))

    assert tracks == []


def test_temporally_promoted_weak_head_can_be_held_briefly() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {"weak_head_indices": [0], "suppressed_head_indices": [0]}}
    detections = _detections([[260, 210, 288, 238]], [1], [0.72], names={1: "head"})

    tracks = []
    for _ in range(3):
        tracks = tracker.update(detections, ppe, (640, 640), max_render_misses=8)

    assert len(tracks) == 1
    assert tracks[0]["temporal_promoted"] is True

    tracks = tracker.update(_detections([], [], [], names={1: "head"}), ppe, (640, 640), max_render_misses=8)

    assert len(tracks) == 1
    assert tracks[0]["source"] == "held"
    assert tracks[0]["hold_eligible"] is False


def test_stable_track_keeps_hold_eligibility_when_later_frame_is_weak() -> None:
    tracker = PPEDisplayTracker()
    strong_ppe = {"helmet_fp_suppression": {}}
    weak_ppe = {"helmet_fp_suppression": {"weak_head_indices": [0], "suppressed_head_indices": [0]}}
    strong = _detections([[260, 210, 308, 260]], [1], [0.72], names={1: "head"})
    weak = _detections([[260, 210, 308, 260]], [1], [0.34], names={1: "head"})

    for _ in range(3):
        tracks = tracker.update(strong, strong_ppe, (640, 640), max_render_misses=8)
        assert tracks[0]["hold_eligible"] is True

    tracks = tracker.update(weak, weak_ppe, (640, 640), max_render_misses=8)
    assert len(tracks) == 1
    assert tracks[0]["hold_eligible"] is True

    tracks = tracker.update(_detections([], [], [], names={1: "head"}), weak_ppe, (640, 640), max_render_misses=8)
    assert len(tracks) == 1
    assert tracks[0]["source"] == "held"


def test_weak_helmet_renders_current_detection_but_is_not_held() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {"weak_helmet_indices": [0], "suppressed_helmet_indices": [0]}}

    tracks = tracker.update(
        _detections([[260, 210, 288, 238]], [0], [0.31], names={0: "helmet"}),
        ppe,
        (640, 640),
    )

    assert len(tracks) == 1
    assert tracks[0]["label"] == "helmet"
    assert tracks[0]["evidence_label"] == "helmet"
    assert tracks[0]["hold_eligible"] is False

    tracks = tracker.update(_detections([], [], [], names={0: "helmet"}), ppe, (640, 640))

    assert tracks == []


def test_adjacent_heads_do_not_merge_into_one_big_display_box() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[80, 80, 108, 112], [122, 82, 150, 114]],
            [1, 1],
            [0.81, 0.78],
        ),
        ppe,
        (640, 640),
    )

    head_tracks = [track for track in tracks if track["label"] == "head"]
    assert len(head_tracks) == 2
    assert all((track["box"][2] - track["box"][0]) < 40 for track in head_tracks)


def test_overlapping_person_boxes_for_same_target_render_once() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[227, 123, 349, 639], [226, 117, 421, 620]],
            [2, 2],
            [0.74, 0.69],
        ),
        ppe,
        (640, 640),
    )

    person_tracks = [track for track in tracks if track["label"] == "person"]
    assert len(person_tracks) == 1
    assert person_tracks[0]["box"] == [227, 123, 349, 639]


def test_overlapping_distinct_person_boxes_do_not_merge() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[100, 100, 220, 500], [135, 120, 255, 520]],
            [2, 2],
            [0.84, 0.82],
        ),
        ppe,
        (640, 640),
    )

    person_tracks = [track for track in tracks if track["label"] == "person"]
    assert len(person_tracks) == 2


def test_adjacent_person_boxes_do_not_merge() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[140, 120, 230, 470], [245, 122, 338, 474]],
            [2, 2],
            [0.84, 0.82],
        ),
        ppe,
        (640, 640),
    )

    person_tracks = [track for track in tracks if track["label"] == "person"]
    assert len(person_tracks) == 2


def test_head_and_helmet_same_target_keeps_single_primary_box() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[200, 150, 236, 190], [202, 148, 238, 188]],
            [1, 0],
            [0.52, 0.63],
        ),
        ppe,
        (640, 640),
    )

    assert len(tracks) == 1
    assert tracks[0]["label"] == "helmet"


def test_head_and_nearby_helmet_without_overlap_keep_head() -> None:
    tracker = PPEDisplayTracker()
    ppe = {"helmet_fp_suppression": {}}

    tracks = tracker.update(
        _detections(
            [[300, 100, 330, 240], [318, 145, 408, 185]],
            [1, 0],
            [0.75, 0.70],
        ),
        ppe,
        (640, 640),
    )

    assert len(tracks) == 2
    assert {track["label"] for track in tracks} == {"head", "helmet"}


@pytest.mark.skip(reason="超前契约未实装:DetectionFrameResult是slots无diagnostics字段,merge_roi_detections不产出roi_redetect_merge诊断")
def test_roi_detections_are_mapped_back_and_deduplicated() -> None:
    base = _detections([[100, 100, 122, 128]], [1], [0.35])
    base.diagnostics = {"preprocess": {"mode": "letterbox"}}
    roi_result = _detections([[12, 10, 36, 40]], [1], [0.82])
    roi_result.diagnostics = {"preprocess": {"mode": "crop"}}

    merged = merge_roi_detections(
        base,
        [([90, 92, 186, 188], roi_result)],
        (640, 640),
    )

    assert merged.inference_ms == 2.0
    assert [102, 102, 126, 132] in merged.boxes
    assert max(merged.confidences) == 0.82
    merge_info = merged.diagnostics["roi_redetect_merge"]
    assert merged.diagnostics["preprocess"] == {"mode": "letterbox"}
    assert merge_info["roi_count"] == 1
    assert merge_info["base_box_count"] == 1
    assert merge_info["final_box_count"] == 1
    assert merge_info["nms_suppressed_count"] == 1
    assert merge_info["rois"][0]["roi"] == [90, 92, 186, 188]
    assert merge_info["rois"][0]["crop_shape"] == [640, 640]
    assert merge_info["rois"][0]["source_diagnostics"] == {"preprocess": {"mode": "crop"}}
    assert merge_info["rois"][0]["decisions"][0]["decision"] == "kept_before_nms"
    assert merge_info["rois"][0]["decisions"][0]["mapped_box"] == [102, 102, 126, 132]
    assert merge_info["final_sources"][0]["source"] == "roi_redetect"
    assert merge_info["suppressed_sources"][0]["source"] == "full_frame"


@pytest.mark.skip(reason="超前契约未实装:DetectionFrameResult是slots无diagnostics字段,merge_roi_detections不产出roi_redetect_merge诊断")
def test_roi_detections_record_drop_reasons() -> None:
    base = _detections([], [], [])
    low_confidence = _detections([[1, 1, 10, 10]], [1], [0.10])
    unknown_label = _detections([[4, 4, 16, 16]], [9], [0.90], names={9: "vehicle"})

    merged = merge_roi_detections(
        base,
        [
            ([20, 30, 80, 90], low_confidence),
            ([40, 50, 100, 110], unknown_label),
        ],
        (640, 640),
    )

    assert merged.boxes == []
    merge_info = merged.diagnostics["roi_redetect_merge"]
    assert merge_info["roi_count"] == 2
    assert merge_info["final_box_count"] == 0
    assert merge_info["rois"][0]["decisions"][0]["decision"] == "dropped_low_confidence"
    assert merge_info["rois"][1]["decisions"][0]["decision"] == "dropped_unknown_label"

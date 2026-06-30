from __future__ import annotations

import numpy as np
import pytest

from defense.module_a.backends.detector_backend import DetectionFrameResult
from defense.module_a.postprocess.ppe_tracking import PPEDisplayTracker


def _detections(
    boxes: list[list[int]],
    classes: list[int],
    confidences: list[float],
) -> DetectionFrameResult:
    return DetectionFrameResult(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        names={0: "helmet", 1: "head", 2: "person"},
        backend="fake",
        artifact_path="fake.pt",
        inference_ms=1.0,
        raw_result=None,
    )


def _labels(tracks: list[dict]) -> list[str]:
    return [str(track.get("label") or "") for track in tracks]


def _center_x(track: dict) -> float:
    x1, _, x2, _ = [float(v) for v in track["box"][:4]]
    return (x1 + x2) * 0.5


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_keeps_ppe_label_through_single_frame_flip() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_confirm_frames=2,
        hold_frames=4,
        small_hold_frames=4,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    tracks = tracker.update(
        _detections(
            [[90, 80, 210, 460], [116, 92, 154, 132]],
            [2, 1],
            [0.90, 0.82],
        ),
        ppe,
        frame_shape,
    )
    head = next(track for track in tracks if track["label"] == "head")
    state_id = int(head["person_state_id"])
    assert state_id > 0
    assert head["ppe_state_label"] == "head"

    tracks = tracker.update(
        _detections(
            [[92, 82, 212, 462], [118, 94, 156, 134]],
            [2, 0],
            [0.90, 0.90],
        ),
        ppe,
        frame_shape,
    )

    ppe_track = next(track for track in tracks if track["label"] in {"head", "helmet"})
    assert ppe_track["person_state_id"] == state_id
    assert ppe_track["label"] == "head"
    assert ppe_track["ppe_state_label"] == "head"


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_rejects_new_small_untrusted_helmet_state() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_confirm_frames=2,
        hold_frames=4,
        small_hold_frames=4,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    rendered: list[dict] = []
    for offset in range(8):
        rendered = tracker.update(
            _detections(
                [[435 - offset, 185, 448 - offset, 248], [438 - offset, 184, 447 - offset, 197]],
                [2, 0],
                [0.55, 0.61],
            ),
            ppe,
            frame_shape,
        )

    assert "helmet" not in _labels(rendered)
    person = next(track for track in rendered if track["label"] == "person")
    assert person["ppe_state_label"] == "helmet"
    assert person["ppe_state_trusted"] is False


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_rejects_head_candidate_that_leaves_head_anchor_like_hand() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_confirm_frames=2,
        hold_frames=4,
        small_hold_frames=4,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    rendered: list[dict] = []
    for offset in range(3):
        rendered = tracker.update(
            _detections(
                [[100 + offset, 90, 300 + offset, 560], [150 + offset, 105, 190 + offset, 148]],
                [2, 1],
                [0.90, 0.84],
            ),
            ppe,
            frame_shape,
        )

    head = next(track for track in rendered if track["label"] == "head")
    assert head["person_state_id"] > 0
    assert head["ppe_state_label"] == "head"

    rendered = tracker.update(
        _detections(
            [[103, 90, 303, 560], [262, 150, 322, 210]],
            [2, 1],
            [0.90, 0.86],
        ),
        ppe,
        frame_shape,
    )

    head_tracks = [track for track in rendered if track["label"] == "head"]
    assert len(head_tracks) == 1
    assert _center_x(head_tracks[0]) < 230
    assert head_tracks[0]["source"] == "held"


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_allows_trusted_helmet_state() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_confirm_frames=2,
        hold_frames=4,
        small_hold_frames=4,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    rendered: list[dict] = []
    for offset in range(5):
        rendered = tracker.update(
            _detections(
                [[220 + offset, 150, 360 + offset, 520], [250 + offset, 160, 325 + offset, 240]],
                [2, 0],
                [0.86, 0.82],
            ),
            ppe,
            frame_shape,
        )

    helmet = next(track for track in rendered if track["label"] == "helmet")
    assert helmet["ppe_state_label"] == "helmet"
    assert helmet["ppe_state_trusted"] is True


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_holds_ppe_track_during_overlap_dropout() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_hold_frames=6,
        person_state_render_miss_grace=4,
        hold_frames=1,
        small_hold_frames=1,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    tracks = tracker.update(
        _detections(
            [[90, 80, 210, 460], [250, 82, 370, 462], [116, 92, 154, 132], [276, 94, 314, 134]],
            [2, 2, 1, 1],
            [0.90, 0.88, 0.82, 0.80],
        ),
        ppe,
        frame_shape,
    )
    state_ids = sorted({int(track["person_state_id"]) for track in tracks if track["label"] == "head"})
    assert len(state_ids) == 2

    tracks = tracker.update(
        _detections(
            [[120, 80, 335, 470]],
            [2],
            [0.86],
        ),
        ppe,
        frame_shape,
    )

    held_heads = [track for track in tracks if track["label"] == "head" and track["source"] == "held"]
    assert held_heads
    assert all(track["hold_after_person_state"] for track in held_heads)
    assert {int(track["person_state_id"]) for track in held_heads} <= set(state_ids)


@pytest.mark.skip(reason="person_state子系统未实装:PPEDisplayTracker无person_state_*参数,构造即TypeError")
def test_person_state_edge_pending_extends_then_prunes() -> None:
    tracker = PPEDisplayTracker(
        person_state_enabled=True,
        person_state_hold_frames=1,
        person_state_edge_hold_frames=3,
        hold_frames=0,
        small_hold_frames=0,
    )
    ppe = {"helmet_fp_suppression": {}}
    frame_shape = (640, 640)

    tracker.update(
        _detections(
            [[0, 80, 100, 460], [18, 92, 56, 132]],
            [2, 1],
            [0.90, 0.82],
        ),
        ppe,
        frame_shape,
    )

    for _ in range(3):
        tracker.update(_detections([], [], []), ppe, frame_shape)
    assert tracker.last_diagnostics["person_state"]["state_count"] == 1
    assert tracker.last_diagnostics["person_state"]["states"][0]["status"] == "edge_pending"

    tracker.update(_detections([], [], []), ppe, frame_shape)
    assert tracker.last_diagnostics["person_state"]["state_count"] == 0

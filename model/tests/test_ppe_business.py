from __future__ import annotations

from types import SimpleNamespace

from defense.module_a.postprocess import PPEDisplayTracker
from defense.runtime.frame_processor import _ppe_max_render_misses
from defense.runtime.ppe_business import evaluate_ppe_business
from defense.runtime.ppe_state import SafetyHelmetState


def make_detections(boxes, classes, confidences, names):
    return SimpleNamespace(boxes=boxes, classes=classes, confidences=confidences, names=names)


def test_file_realtime_ppe_render_miss_cap_is_configurable() -> None:
    assert _ppe_max_render_misses(source_type="file", realtime=True) == 2
    assert (
        _ppe_max_render_misses(
            source_type="file",
            realtime=True,
            file_realtime_max_misses=3,
        )
        == 3
    )
    assert _ppe_max_render_misses(source_type="camera", realtime=True, file_realtime_max_misses=3) is None


def test_evaluate_ppe_business_applies_summary_and_temporal_state() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (80, 80, 220, 330)],
        classes=[0, 2],
        confidences=[0.86, 0.92],
        names={0: "head", 2: "person"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(3)
    ]

    assert outputs[-1].ppe["candidate"] is True
    assert outputs[-1].ppe["confirmed"] is True
    assert outputs[-1].ppe["warning"] is True
    assert outputs[-1].ppe["person_count"] == 1
    assert outputs[-1].ppe["head_count"] == 1
    assert outputs[-1].tracks


def test_evaluate_ppe_business_fast_confirms_high_confidence_head() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(100, 100, 180, 190)],
        classes=[0],
        confidences=[0.86],
        names={0: "head", 1: "helmet"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(2)
    ]

    assert outputs[-1].ppe["candidate"] is True
    assert outputs[-1].ppe["confirmed"] is True
    assert outputs[-1].ppe["confirmed_source"] == "fast_head"
    assert outputs[-1].ppe["fast_window_positive"] == 2


def test_evaluate_ppe_business_can_skip_display_tracking() -> None:
    result = evaluate_ppe_business(
        make_detections([], [], [], {0: "head", 1: "helmet", 2: "person"}),
        frame_shape=(640, 640),
        ppe_state=SafetyHelmetState(),
        ppe_tracker=PPEDisplayTracker(),
        tracking_enabled=False,
    )

    assert result.ppe["candidate"] is False
    assert result.tracks == []


def test_uncertain_small_head_is_rendered_as_current_weak_track() -> None:
    result = evaluate_ppe_business(
        make_detections(
            boxes=[(260, 210, 288, 238)],
            classes=[0],
            confidences=[0.31],
            names={0: "head"},
        ),
        frame_shape=(640, 640),
        ppe_state=SafetyHelmetState(),
        ppe_tracker=PPEDisplayTracker(),
        tracking_enabled=True,
    )

    assert result.ppe["candidate"] is False
    assert result.ppe["uncertain"] is True
    assert result.ppe["helmet_fp_suppression"]["weak_head_indices"] == [0]
    assert result.ppe["helmet_fp_suppression"]["suppressed_head_indices"] == []
    assert len(result.tracks) == 1
    assert result.tracks[0]["label"] == "head"
    assert result.tracks[0]["hold_eligible"] is False


def test_stable_weak_head_promotes_to_candidate_without_extra_inference() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(260, 210, 288, 238)],
        classes=[0],
        confidences=[0.31],
        names={0: "head"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(3)
    ]

    assert outputs[-1].ppe["candidate"] is True
    assert outputs[-1].ppe["head_count"] == 1
    assert outputs[-1].ppe["promoted_head_count"] == 1
    assert outputs[-1].ppe["missing_helmet_count"] == 1
    assert outputs[-1].ppe["reason"] == "temporal_weak_head_promoted"
    assert outputs[-1].tracks[0]["temporal_promoted"] is True
    assert outputs[-1].tracks[0]["promoted_label"] == "head"


def test_stable_weak_helmet_promotes_to_safe_evidence() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(260, 210, 288, 238), (240, 190, 305, 330)],
        classes=[0, 2],
        confidences=[0.31, 0.90],
        names={0: "helmet", 2: "person"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(3)
    ]

    assert outputs[-1].ppe["candidate"] is False
    assert outputs[-1].ppe["helmet_count"] == 1
    assert outputs[-1].ppe["promoted_helmet_count"] == 1
    assert outputs[-1].ppe["missing_helmet_count"] == 0
    assert outputs[-1].ppe["reason"] == "temporal_weak_helmet_promoted"
    helmet_tracks = [track for track in outputs[-1].tracks if track["label"] == "helmet"]
    assert helmet_tracks[0]["temporal_promoted"] is True
    assert helmet_tracks[0]["promoted_label"] == "helmet"


def test_isolated_low_confidence_weak_helmet_does_not_promote() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(260, 210, 288, 238)],
        classes=[0],
        confidences=[0.31],
        names={0: "helmet"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(3)
    ]

    assert outputs[-1].ppe["candidate"] is False
    assert outputs[-1].ppe["helmet_count"] == 0
    assert outputs[-1].ppe["promoted_helmet_count"] == 0
    assert outputs[-1].ppe["reason"] == "no_head_or_helmet_evidence_detected"


def test_promoted_helmet_does_not_hide_existing_head_violation_reason() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker()
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (260, 210, 288, 238)],
        classes=[0, 1],
        confidences=[0.86, 0.56],
        names={0: "head", 1: "helmet"},
    )

    outputs = [
        evaluate_ppe_business(
            detections,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
        )
        for _ in range(3)
    ]

    assert outputs[-1].ppe["candidate"] is True
    assert outputs[-1].ppe["head_count"] == 1
    assert outputs[-1].ppe["helmet_count"] == 1
    assert outputs[-1].ppe["promoted_helmet_count"] == 1
    assert outputs[-1].ppe["reason"] == "bare_head_without_matched_helmet"


def test_head_helmet_only_model_gets_extra_render_hold_for_file_preview() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker(
        hold_frames=12,
        small_hold_frames=12,
        show_held_boxes=True,
    )
    detections = make_detections(
        boxes=[(240, 170, 330, 260)],
        classes=[0],
        confidences=[0.86],
        names={0: "head", 1: "helmet"},
    )
    first = evaluate_ppe_business(
        detections,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        max_render_misses=2,
    )
    assert first.ppe["evidence_mode"] == "head_helmet_only"
    assert len(first.tracks) == 1

    empty = make_detections([], [], [], {0: "head", 1: "helmet"})
    result = first
    for _ in range(5):
        result = evaluate_ppe_business(
            empty,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
            max_render_misses=2,
        )

    assert len(result.tracks) == 1
    assert result.tracks[0]["source"] == "held"
    assert result.tracks[0]["misses"] == 5


def test_person_context_model_keeps_short_file_preview_render_cap() -> None:
    state = SafetyHelmetState()
    tracker = PPEDisplayTracker(
        hold_frames=12,
        small_hold_frames=12,
        show_held_boxes=True,
    )
    detections = make_detections(
        boxes=[(240, 170, 330, 260)],
        classes=[0],
        confidences=[0.86],
        names={0: "head", 1: "helmet", 2: "person"},
    )
    first = evaluate_ppe_business(
        detections,
        frame_shape=(640, 640),
        ppe_state=state,
        ppe_tracker=tracker,
        tracking_enabled=True,
        max_render_misses=2,
    )
    assert first.ppe["evidence_mode"] == "person_context_available"

    empty = make_detections([], [], [], {0: "head", 1: "helmet", 2: "person"})
    result = first
    for _ in range(3):
        result = evaluate_ppe_business(
            empty,
            frame_shape=(640, 640),
            ppe_state=state,
            ppe_tracker=tracker,
            tracking_enabled=True,
            max_render_misses=2,
        )

    assert result.tracks == []

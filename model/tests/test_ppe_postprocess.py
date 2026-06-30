from __future__ import annotations

from types import SimpleNamespace

from defense.module_a.ppe_postprocess import (
    PPEPostprocessConfig,
    bbox_iou,
    summarize_ppe_from_detections,
)


def make_detections(boxes, classes, confidences, names):
    return SimpleNamespace(boxes=boxes, classes=classes, confidences=confidences, names=names)


def test_head_overlap_suppresses_weak_helmet_false_positive():
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (102, 102, 178, 188), (80, 80, 220, 330)],
        classes=[0, 1, 2],
        confidences=[0.82, 0.47, 0.91],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_helmet_count"] == 1
    assert summary["helmet_count"] == 0
    assert summary["head_count"] == 1
    assert summary["candidate"] is True
    assert summary["helmet_fp_suppression"]["suppressed_helmet_indices"] == [1]


def test_strong_helmet_survives_head_overlap():
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (102, 102, 178, 188), (80, 80, 220, 330)],
        classes=[0, 1, 2],
        confidences=[0.50, 0.90, 0.91],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_helmet_count"] == 1
    assert summary["helmet_count"] == 1
    assert summary["head_count"] == 0
    assert summary["candidate"] is False
    assert summary["helmet_fp_suppression"]["suppressed_helmet_indices"] == []


def test_no_helmet_label_does_not_count_as_helmet():
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (80, 80, 220, 330)],
        classes=[0, 1],
        confidences=[0.86, 0.92],
        names={0: "no_helmet", 1: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_helmet_count"] == 0
    assert summary["head_count"] == 1
    assert summary["candidate"] is True


def test_isolated_small_edge_head_does_not_trigger_ppe_warning():
    detections = make_detections(
        boxes=[(575, 260, 600, 294)],
        classes=[0],
        confidences=[0.72],
        names={0: "head"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_head_count"] == 1
    assert summary["head_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["inferred_person_count"] == 1
    assert summary["helmet_fp_suppression"]["suppressed_head_indices"] == [0]
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == [0]


def test_low_confidence_small_isolated_head_is_uncertain_not_violation():
    detections = make_detections(
        boxes=[(260, 210, 288, 238)],
        classes=[0],
        confidences=[0.31],
        names={0: "head"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_head_count"] == 1
    assert summary["head_count"] == 0
    assert summary["missing_helmet_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["inferred_person_count"] == 1
    assert summary["reason"] == "isolated_head_evidence_uncertain"
    assert summary["helmet_fp_suppression"]["suppressed_head_indices"] == []
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_heads"][0]["reason"] == "small_low_conf_head"


def test_high_confidence_small_isolated_head_without_context_is_uncertain():
    detections = make_detections(
        boxes=[(176, 198, 198, 227)],
        classes=[0],
        confidences=[0.70],
        names={0: "head"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_head_count"] == 1
    assert summary["head_count"] == 0
    assert summary["missing_helmet_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["helmet_fp_suppression"]["suppressed_head_indices"] == []
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_heads"][0]["reason"] == "small_no_context_head"


def test_small_low_confidence_helmet_is_weak_evidence_not_effective():
    detections = make_detections(
        boxes=[(260, 210, 288, 238)],
        classes=[0],
        confidences=[0.31],
        names={0: "helmet"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["raw_helmet_count"] == 1
    assert summary["helmet_count"] == 0
    assert summary["weak_helmet_count"] == 1
    assert summary["candidate"] is False
    assert summary["helmet_fp_suppression"]["weak_helmet_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_helmets"][0]["reason"] == "small_low_conf_helmet"


def test_person_only_is_uncertain_without_head_or_helmet():
    detections = make_detections(boxes=[], classes=[2], confidences=[0.88], names={2: "person"})

    summary = summarize_ppe_from_detections(detections)

    assert summary["person_count"] == 1
    assert summary["raw_person_count"] == 1
    assert summary["inferred_person_count"] == 1
    assert summary["effective_person_count"] == 1
    assert summary["weak_person_count"] == 0
    assert summary["promoted_person_count"] == 0
    assert summary["helmet_count"] == 0
    assert summary["missing_helmet_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["evidence_mode"] == "person_context_available"
    assert summary["helmet_fp_suppression"]["kept_person_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_person_indices"] == []
    assert summary["helmet_fp_suppression"]["weak_person_indices"] == []


def test_person_aliases_are_counted_as_person_context():
    for alias in ("person", "worker", "human", "pedestrian"):
        detections = make_detections(boxes=[(80, 80, 220, 330)], classes=[0], confidences=[0.88], names={0: alias})

        summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

        assert summary["has_person_class"] is True
        assert summary["person_count"] == 1
        assert summary["effective_person_count"] == 1
        assert summary["evidence_mode"] == "person_context_available"


def test_head_only_without_person_class_triggers_head_driven_candidate():
    detections = make_detections(
        boxes=[(100, 100, 180, 190)],
        classes=[0],
        confidences=[0.86],
        names={0: "head"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["has_person_class"] is False
    assert summary["evidence_mode"] == "head_helmet_only"
    assert summary["head_count"] == 1
    assert summary["missing_helmet_count"] == 1
    assert summary["candidate"] is True


def test_low_confidence_head_without_person_class_is_temporal_candidate_only():
    detections = make_detections(
        boxes=[(100, 100, 180, 190)],
        classes=[0],
        confidences=[0.20],
        names={0: "head", 1: "helmet"},
    )

    summary = summarize_ppe_from_detections(
        detections,
        config=PPEPostprocessConfig(min_confidence=0.25, candidate_min_confidence=0.18),
        frame_shape=(640, 640),
    )

    assert summary["evidence_mode"] == "head_helmet_only"
    assert summary["raw_head_count"] == 0
    assert summary["head_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["low_conf_temporal_head_count"] == 1
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_heads"][0]["reason"] == "low_conf_temporal_candidate"


def test_low_confidence_candidate_is_disabled_when_person_class_exists():
    detections = make_detections(
        boxes=[(100, 100, 180, 190)],
        classes=[0],
        confidences=[0.20],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(
        detections,
        config=PPEPostprocessConfig(min_confidence=0.25, candidate_min_confidence=0.18),
        frame_shape=(640, 640),
    )

    assert summary["evidence_mode"] == "person_context_available"
    assert summary["candidate"] is False
    assert summary["uncertain"] is False
    assert summary["low_conf_temporal_head_count"] == 0
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == []


def test_low_confidence_head_inside_person_context_becomes_temporal_candidate():
    detections = make_detections(
        boxes=[(118, 82, 145, 110), (90, 70, 190, 330)],
        classes=[0, 2],
        confidences=[0.20, 0.88],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(
        detections,
        config=PPEPostprocessConfig(min_confidence=0.25, candidate_min_confidence=0.18),
        frame_shape=(640, 640),
    )

    assert summary["evidence_mode"] == "person_context_available"
    assert summary["person_count"] == 1
    assert summary["candidate"] is False
    assert summary["uncertain"] is True
    assert summary["low_conf_temporal_head_count"] == 1
    assert summary["helmet_fp_suppression"]["weak_head_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_heads"][0]["person_context"] is True


def test_helmet_only_without_person_class_keeps_helmet_evidence():
    detections = make_detections(
        boxes=[(100, 100, 180, 190)],
        classes=[0],
        confidences=[0.86],
        names={0: "helmet"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["has_person_class"] is False
    assert summary["evidence_mode"] == "head_helmet_only"
    assert summary["helmet_count"] == 1
    assert summary["candidate"] is False
    assert summary["uncertain"] is False


def test_high_confidence_helmet_only_with_person_class_keeps_helmet_evidence():
    detections = make_detections(
        boxes=[(320, 180, 410, 255)],
        classes=[1],
        confidences=[0.88],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["has_person_class"] is True
    assert summary["raw_helmet_count"] == 1
    assert summary["helmet_count"] == 1
    assert summary["weak_helmet_count"] == 0
    assert summary["candidate"] is False
    assert summary["reason"] == "helmet_evidence_present"
    assert summary["helmet_fp_suppression"]["suppressed_helmet_indices"] == []


def test_low_confidence_helmet_only_with_person_class_is_weak_without_context():
    detections = make_detections(
        boxes=[(320, 180, 410, 255)],
        classes=[1],
        confidences=[0.42],
        names={0: "head", 1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["has_person_class"] is True
    assert summary["raw_helmet_count"] == 1
    assert summary["helmet_count"] == 0
    assert summary["weak_helmet_count"] == 1
    assert summary["candidate"] is False
    assert summary["helmet_fp_suppression"]["suppressed_helmet_indices"] == [0]
    assert summary["helmet_fp_suppression"]["weak_helmet_indices"] == [0]
    assert summary["helmet_fp_suppression"]["suppressed_helmets"][0]["reason"] == "helmet_without_person_context"


def test_person_and_head_without_helmet_uses_head_as_violation_evidence():
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (80, 80, 220, 330)],
        classes=[0, 2],
        confidences=[0.86, 0.92],
        names={0: "head", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["person_count"] == 1
    assert summary["raw_person_count"] == 1
    assert summary["effective_person_count"] == 1
    assert summary["head_count"] == 1
    assert summary["missing_helmet_count"] == 1
    assert summary["candidate"] is True


def test_prefer_helmet_mutex_uses_center_distance_for_same_target():
    detections = make_detections(
        boxes=[
            (300, 100, 330, 240),
            (270, 140, 360, 180),
            (250, 80, 380, 460),
        ],
        classes=[0, 1, 2],
        confidences=[0.75, 0.70, 0.91],
        names={0: "head", 1: "helmet", 2: "person"},
    )
    config = PPEPostprocessConfig(
        prefer_helmet_on_head_overlap=True,
        head_helmet_mutex_iou=0.20,
        head_helmet_mutex_center_distance=0.055,
    )

    summary = summarize_ppe_from_detections(
        detections,
        config=config,
        frame_shape=(640, 640),
    )

    assert bbox_iou((300, 100, 330, 240), (270, 140, 360, 180)) < 0.20
    assert summary["raw_head_count"] == 1
    assert summary["helmet_count"] == 1
    assert summary["head_count"] == 0
    assert summary["missing_helmet_count"] == 0
    assert summary["candidate"] is False
    suppression = summary["helmet_fp_suppression"]
    assert suppression["covered_head_indices"] == [0]
    assert suppression["suppressed_head_indices"] == [0]
    assert suppression["suppressed_heads"][-1]["reason"] == "head_helmet_mutex"


def test_nearby_helmet_requires_overlap_before_covering_head_by_distance():
    detections = make_detections(
        boxes=[
            (300, 100, 330, 240),
            (318, 145, 408, 185),
            (250, 80, 430, 460),
        ],
        classes=[0, 1, 2],
        confidences=[0.75, 0.70, 0.91],
        names={0: "head", 1: "helmet", 2: "person"},
    )
    config = PPEPostprocessConfig(
        prefer_helmet_on_head_overlap=True,
        head_helmet_mutex_iou=0.20,
        head_helmet_mutex_center_distance=0.055,
        head_helmet_mutex_min_overlap=0.18,
    )

    summary = summarize_ppe_from_detections(
        detections,
        config=config,
        frame_shape=(640, 640),
    )

    assert bbox_iou((300, 100, 330, 240), (318, 145, 408, 185)) < 0.20
    assert summary["helmet_count"] == 1
    assert summary["head_count"] == 1
    assert summary["candidate"] is True
    suppression = summary["helmet_fp_suppression"]
    assert suppression["covered_head_indices"] == []
    assert suppression["suppressed_head_indices"] == []


def test_person_and_helmet_without_head_is_not_violation():
    detections = make_detections(
        boxes=[(100, 100, 180, 190), (80, 80, 220, 330)],
        classes=[1, 2],
        confidences=[0.86, 0.92],
        names={1: "helmet", 2: "person"},
    )

    summary = summarize_ppe_from_detections(detections, frame_shape=(640, 640))

    assert summary["person_count"] == 1
    assert summary["helmet_count"] == 1
    assert summary["head_count"] == 0
    assert summary["candidate"] is False
    assert summary["uncertain"] is False


def test_bbox_iou_returns_expected_overlap():
    value = bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))

    assert 0.14 < value < 0.15

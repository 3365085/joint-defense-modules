"""ROI provider — detector boxes → generic ROI objects."""
from __future__ import annotations

from defense.module_a.roi_provider import DetectionROIProvider


NAMES = {0: "helmet", 1: "head", 2: "person"}


def test_low_confidence_is_filtered() -> None:
    provider = DetectionROIProvider(NAMES, min_confidence=0.5, margin=0)
    rois = provider.from_detections(
        boxes=[[10, 10, 30, 30], [40, 40, 60, 60]],
        classes=[0, 1],
        confs=[0.8, 0.2],
    )
    assert len(rois) == 1
    assert rois[0].label == "helmet"


def test_margin_expands_bbox() -> None:
    provider = DetectionROIProvider(NAMES, min_confidence=0.0, margin=5)
    rois = provider.from_detections(
        boxes=[[10, 10, 30, 30]],
        classes=[0],
        confs=[0.9],
    )
    # Margin of 5 should widen all four sides.
    assert rois[0].bbox == (5, 5, 35, 35)


def test_unknown_class_gets_fallback_label() -> None:
    provider = DetectionROIProvider(NAMES, min_confidence=0.0, margin=0)
    rois = provider.from_detections(
        boxes=[[10, 10, 30, 30]],
        classes=[99],
        confs=[0.9],
    )
    assert rois[0].label == "class_99"


def test_roi_ids_are_unique_by_index() -> None:
    provider = DetectionROIProvider(NAMES, min_confidence=0.0, margin=0)
    rois = provider.from_detections(
        boxes=[[0, 0, 20, 20], [5, 5, 25, 25]],
        classes=[0, 0],
        confs=[0.9, 0.8],
    )
    ids = {roi.roi_id for roi in rois}
    assert len(ids) == 2

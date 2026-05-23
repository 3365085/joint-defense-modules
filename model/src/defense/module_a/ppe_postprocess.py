from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PERSON_HINTS = ("person", "worker", "pedestrian")
HELMET_HINTS = ("helmet", "hardhat", "hard_hat", "safety_helmet")
BARE_HEAD_HINTS = ("head", "no_helmet", "without_helmet", "bare_head")


@dataclass(frozen=True, slots=True)
class PPEPostprocessConfig:
    min_confidence: float = 0.25
    overlap_iou: float = 0.55
    helmet_head_margin: float = 0.12
    min_direct_helmet_conf: float = 0.55
    small_target_min_conf: float = 0.65
    small_target_area_ratio: float = 0.012
    max_isolated_helmet_area_ratio: float = 0.08
    min_person_context_iou: float = 0.01
    max_helmet_to_person_area_ratio: float = 0.30
    max_isolated_head_area_ratio: float = 0.012
    isolated_head_edge_margin: float = 0.10
    min_isolated_head_confidence: float = 0.45


@dataclass(frozen=True, slots=True)
class PPEDetection:
    index: int
    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


def normalize_label(label: str) -> str:
    return str(label or "").lower().replace("-", "_").replace(" ", "_")


def label_matches(label: str, hints: tuple[str, ...]) -> bool:
    normalized = normalize_label(label)
    return any(hint in normalized for hint in hints)


def is_bare_head_label(label: str) -> bool:
    return label_matches(label, BARE_HEAD_HINTS)


def is_helmet_label(label: str) -> bool:
    normalized = normalize_label(label)
    if is_bare_head_label(normalized):
        return False
    return label_matches(normalized, HELMET_HINTS)


def is_person_label(label: str) -> bool:
    return label_matches(label, PERSON_HINTS)


def infer_ppe_model_capabilities(detections: Any, items: Iterable[PPEDetection] | None = None) -> dict[str, Any]:
    names = getattr(detections, "names", {}) or {}
    labels: list[str] = []
    if isinstance(names, dict):
        labels.extend(str(value) for value in names.values())
    elif isinstance(names, (list, tuple)):
        labels.extend(str(value) for value in names)
    if items is not None:
        labels.extend(str(item.label) for item in items)

    has_person_class = any(is_person_label(label) for label in labels)
    has_head_class = any(is_bare_head_label(label) for label in labels)
    has_helmet_class = any(is_helmet_label(label) for label in labels)
    return {
        "has_person_class": has_person_class,
        "has_head_class": has_head_class,
        "has_helmet_class": has_helmet_class,
        "evidence_mode": "person_context_available" if has_person_class else "head_helmet_only",
    }


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_edge_proximity(
    bbox: tuple[float, float, float, float] | None,
    frame_shape: tuple[int, int] | tuple[int, int, int] | None,
) -> float:
    if bbox is None or not frame_shape:
        return 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    left = x1 / width
    top = y1 / height
    right = (width - x2) / width
    bottom = (height - y2) / height
    return max(0.0, min(left, top, right, bottom))


def _frame_area(frame_shape: tuple[int, int] | tuple[int, int, int] | None) -> float:
    if not frame_shape:
        return 0.0
    height, width = frame_shape[:2]
    return float(max(1, int(height)) * max(1, int(width)))


def extract_ppe_detections(detections: Any) -> list[PPEDetection]:
    boxes = getattr(detections, "boxes", []) or []
    classes = getattr(detections, "classes", []) or []
    confidences = getattr(detections, "confidences", []) or []
    names = getattr(detections, "names", {}) or {}
    items: list[PPEDetection] = []
    for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
        int_class_id = int(class_id)
        label = str(names.get(int_class_id, f"class_{int_class_id}"))
        bbox: tuple[float, float, float, float] | None = None
        if index < len(boxes):
            values = boxes[index]
            if values is not None and len(values) >= 4:
                bbox = tuple(float(v) for v in values[:4])  # type: ignore[assignment]
        items.append(PPEDetection(index, label, int_class_id, float(confidence), bbox))
    return items


def suppress_helmet_false_positives(
    detections: Iterable[PPEDetection],
    config: PPEPostprocessConfig | None = None,
    frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    *,
    has_person_class: bool | None = None,
) -> dict[str, Any]:
    cfg = config or PPEPostprocessConfig()
    items = [item for item in detections if item.confidence >= cfg.min_confidence]
    heads = [item for item in items if is_bare_head_label(item.label)]
    helmets = [item for item in items if is_helmet_label(item.label)]
    persons = [item for item in items if is_person_label(item.label)]
    person_context_available = (
        any(is_person_label(item.label) for item in items)
        if has_person_class is None
        else bool(has_person_class)
    )
    frame_area = _frame_area(frame_shape)
    suppressed: list[dict[str, Any]] = []
    suppressed_heads: list[dict[str, Any]] = []
    kept_helmet_indices: set[int] = set()
    suppressed_indices: set[int] = set()
    weak_head_indices: set[int] = set()
    weak_helmet_indices: set[int] = set()
    display_suppressed_head_indices: set[int] = set()
    covered_head_indices: set[int] = set()

    for head in heads:
        head_area_ratio = bbox_area(head.bbox) / frame_area if frame_area > 0.0 else 0.0
        edge_margin = bbox_edge_proximity(head.bbox, frame_shape)
        person_context = any(bbox_iou(head.bbox, person.bbox) >= cfg.min_person_context_iou for person in persons)
        helmet_context = any(bbox_iou(head.bbox, helmet.bbox) >= cfg.overlap_iou for helmet in helmets)
        low_context = not person_context and not helmet_context
        small_isolated = head_area_ratio < cfg.max_isolated_head_area_ratio
        small_no_context = low_context and small_isolated
        low_confidence_isolated = small_no_context and head.confidence < cfg.min_isolated_head_confidence
        edge_isolated = small_no_context and edge_margin <= cfg.isolated_head_edge_margin
        if small_no_context:
            weak_head_indices.add(head.index)
            if low_confidence_isolated:
                reason = "small_low_conf_head"
            elif edge_isolated:
                reason = "edge_isolated_head"
            else:
                reason = "small_no_context_head"
            if edge_isolated:
                display_suppressed_head_indices.add(head.index)
            suppressed_heads.append(
                {
                    "head_index": head.index,
                    "head_confidence": head.confidence,
                    "head_bbox": list(head.bbox) if head.bbox else None,
                    "head_area_ratio": head_area_ratio,
                    "edge_margin": edge_margin,
                    "person_context": person_context,
                    "helmet_context": helmet_context,
                    "reason": reason,
                }
            )

    for helmet in helmets:
        matched_head: PPEDetection | None = None
        matched_iou = 0.0
        for head in heads:
            if head.index in weak_head_indices:
                continue
            iou = bbox_iou(helmet.bbox, head.bbox)
            if iou > matched_iou:
                matched_iou = iou
                matched_head = head
        max_person_iou = 0.0
        min_person_area_ratio = 0.0
        for person in persons:
            person_iou = bbox_iou(helmet.bbox, person.bbox)
            if person_iou > max_person_iou:
                max_person_iou = person_iou
                person_area = bbox_area(person.bbox)
                min_person_area_ratio = bbox_area(helmet.bbox) / person_area if person_area > 0.0 else 0.0
        small_target = frame_area > 0.0 and helmet.bbox is not None and bbox_area(helmet.bbox) / frame_area < cfg.small_target_area_ratio
        oversized_isolated = frame_area > 0.0 and helmet.bbox is not None and bbox_area(helmet.bbox) / frame_area > cfg.max_isolated_helmet_area_ratio
        oversized_for_person = max_person_iou >= cfg.min_person_context_iou and min_person_area_ratio > cfg.max_helmet_to_person_area_ratio
        missing_context = (
            person_context_available
            and matched_head is None
            and max_person_iou < cfg.min_person_context_iou
        )
        high_confidence_isolated = missing_context and helmet.confidence >= cfg.min_direct_helmet_conf
        overlap_weak = (
            matched_head is not None
            and matched_iou >= cfg.overlap_iou
            and helmet.confidence < matched_head.confidence + cfg.helmet_head_margin
        )
        overlap_low_direct = matched_head is not None and matched_iou >= cfg.overlap_iou and helmet.confidence < cfg.min_direct_helmet_conf
        weak_small = small_target and helmet.confidence < cfg.small_target_min_conf
        suppress_missing_context = missing_context and not high_confidence_isolated
        if overlap_weak or overlap_low_direct or weak_small or suppress_missing_context or oversized_isolated or oversized_for_person:
            if overlap_weak or overlap_low_direct:
                reason = "head_helmet_overlap"
            elif suppress_missing_context:
                reason = "helmet_without_person_context"
            elif oversized_isolated:
                reason = "oversized_isolated_helmet"
            elif oversized_for_person:
                reason = "oversized_helmet_vs_person"
            else:
                reason = "small_low_conf_helmet"
            if weak_small or suppress_missing_context:
                weak_helmet_indices.add(helmet.index)
            suppressed_indices.add(helmet.index)
            suppressed.append(
                {
                    "helmet_index": helmet.index,
                    "helmet_confidence": helmet.confidence,
                    "helmet_bbox": list(helmet.bbox) if helmet.bbox else None,
                    "head_index": matched_head.index if matched_head else None,
                    "head_confidence": matched_head.confidence if matched_head else None,
                    "head_bbox": list(matched_head.bbox) if matched_head and matched_head.bbox else None,
                    "iou": matched_iou,
                    "person_iou": max_person_iou,
                    "helmet_person_area_ratio": min_person_area_ratio,
                    "small_target": small_target,
                    "oversized_isolated": oversized_isolated,
                    "oversized_for_person": oversized_for_person,
                    "missing_context": missing_context,
                    "reason": reason,
                }
            )
        else:
            kept_helmet_indices.add(helmet.index)
            if matched_head is not None and matched_iou >= cfg.overlap_iou:
                covered_head_indices.add(matched_head.index)

    return {
        "kept_helmet_indices": sorted(kept_helmet_indices),
        "suppressed_helmet_indices": sorted(suppressed_indices),
        "weak_helmet_indices": sorted(weak_helmet_indices),
        "suppressed_helmets": suppressed,
        "suppressed_head_indices": sorted(display_suppressed_head_indices),
        "weak_head_indices": sorted(weak_head_indices),
        "suppressed_heads": suppressed_heads,
        "covered_head_indices": sorted(covered_head_indices),
        "helmet_count_raw": len(helmets),
        "helmet_count_effective": len(kept_helmet_indices),
        "head_count": max(0, len(heads) - len(weak_head_indices) - len(covered_head_indices)),
    }


def summarize_ppe_from_detections(
    detections: Any,
    config: PPEPostprocessConfig | None = None,
    frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    cfg = config or PPEPostprocessConfig()
    items = [item for item in extract_ppe_detections(detections) if item.confidence >= cfg.min_confidence]
    capabilities = infer_ppe_model_capabilities(detections, items)
    suppression = suppress_helmet_false_positives(
        items,
        cfg,
        frame_shape=frame_shape,
        has_person_class=bool(capabilities["has_person_class"]),
    )
    kept_helmets = set(suppression["kept_helmet_indices"])

    person_count = 0
    helmet_count = 0
    raw_helmet_count = 0
    raw_head_count = 0
    class_counts: dict[str, int] = {}
    for item in items:
        class_counts[item.label] = class_counts.get(item.label, 0) + 1
        if is_bare_head_label(item.label):
            raw_head_count += 1
        elif is_helmet_label(item.label):
            raw_helmet_count += 1
            if item.index in kept_helmets:
                helmet_count += 1
        elif is_person_label(item.label):
            person_count += 1

    head_count = max(0, int(suppression.get("head_count", 0)))
    missing_helmet_count = head_count
    candidate = head_count > 0
    suppressed_head_count = len(suppression.get("weak_head_indices", []) or [])
    weak_helmet_count = len(suppression.get("weak_helmet_indices", []) or [])
    inferred_person_count = max(
        person_count,
        1 if (raw_head_count > 0 or raw_helmet_count > 0 or helmet_count > 0) else 0,
    )
    uncertain = (
        (person_count > 0 and head_count == 0 and helmet_count == 0)
        or (suppressed_head_count > 0 and head_count == 0 and helmet_count == 0)
    )
    if candidate:
        reason = (
            "bare_head_with_suppressed_helmet_evidence"
            if suppression["suppressed_helmet_indices"]
            else "bare_head_without_matched_helmet"
        )
    elif helmet_count > 0:
        reason = "helmet_evidence_present"
    elif uncertain:
        reason = (
            "isolated_head_evidence_uncertain"
            if suppressed_head_count > 0
            else "person_context_without_head_or_helmet_evidence"
        )
    elif capabilities["has_person_class"]:
        reason = "no_ppe_evidence_detected"
    else:
        reason = "no_head_or_helmet_evidence_detected"

    return {
        "person_count": person_count,
        "person_context_count": person_count,
        "raw_person_count": person_count,
        "inferred_person_count": inferred_person_count,
        "helmet_count": helmet_count,
        "raw_helmet_count": raw_helmet_count,
        "weak_helmet_count": weak_helmet_count,
        "promoted_helmet_count": 0,
        "effective_helmet_count": helmet_count,
        "raw_head_count": raw_head_count,
        "weak_head_count": suppressed_head_count,
        "promoted_head_count": 0,
        "effective_head_count": head_count,
        "head_count": head_count,
        "missing_helmet_count": missing_helmet_count,
        "candidate": candidate,
        "uncertain": uncertain,
        "reason": reason,
        "class_counts": class_counts,
        **capabilities,
        "helmet_fp_suppression": suppression,
    }

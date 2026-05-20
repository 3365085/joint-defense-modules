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
) -> dict[str, Any]:
    cfg = config or PPEPostprocessConfig()
    items = [item for item in detections if item.confidence >= cfg.min_confidence]
    heads = [item for item in items if is_bare_head_label(item.label)]
    helmets = [item for item in items if is_helmet_label(item.label)]
    persons = [item for item in items if is_person_label(item.label)]
    frame_area = _frame_area(frame_shape)
    suppressed: list[dict[str, Any]] = []
    kept_helmet_indices: set[int] = set()
    suppressed_indices: set[int] = set()
    isolated_head_indices: set[int] = set()

    for head in heads:
        head_area_ratio = bbox_area(head.bbox) / frame_area if frame_area > 0.0 else 0.0
        edge_margin = bbox_edge_proximity(head.bbox, frame_shape)
        person_context = any(bbox_iou(head.bbox, person.bbox) >= cfg.min_person_context_iou for person in persons)
        helmet_context = any(bbox_iou(head.bbox, helmet.bbox) >= cfg.overlap_iou for helmet in helmets)
        if not person_context and not helmet_context and head_area_ratio < cfg.max_isolated_head_area_ratio and edge_margin <= cfg.isolated_head_edge_margin:
            isolated_head_indices.add(head.index)

    for helmet in helmets:
        matched_head: PPEDetection | None = None
        matched_iou = 0.0
        for head in heads:
            if head.index in isolated_head_indices:
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
        missing_context = matched_head is None and max_person_iou < cfg.min_person_context_iou
        overlap_weak = (
            matched_head is not None
            and matched_iou >= cfg.overlap_iou
            and helmet.confidence < matched_head.confidence + cfg.helmet_head_margin
        )
        overlap_low_direct = matched_head is not None and matched_iou >= cfg.overlap_iou and helmet.confidence < cfg.min_direct_helmet_conf
        weak_small = small_target and helmet.confidence < cfg.small_target_min_conf
        if overlap_weak or overlap_low_direct or weak_small or missing_context or oversized_isolated or oversized_for_person:
            if overlap_weak or overlap_low_direct:
                reason = "head_helmet_overlap"
            elif missing_context:
                reason = "helmet_without_person_context"
            elif oversized_isolated:
                reason = "oversized_isolated_helmet"
            elif oversized_for_person:
                reason = "oversized_helmet_vs_person"
            else:
                reason = "small_low_conf_helmet"
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

    return {
        "kept_helmet_indices": sorted(kept_helmet_indices),
        "suppressed_helmet_indices": sorted(suppressed_indices),
        "suppressed_helmets": suppressed,
        "suppressed_head_indices": sorted(isolated_head_indices),
        "helmet_count_raw": len(helmets),
        "helmet_count_effective": len(kept_helmet_indices),
        "head_count": len(heads) - len(isolated_head_indices),
    }


def summarize_ppe_from_detections(
    detections: Any,
    config: PPEPostprocessConfig | None = None,
    frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    cfg = config or PPEPostprocessConfig()
    items = [item for item in extract_ppe_detections(detections) if item.confidence >= cfg.min_confidence]
    suppression = suppress_helmet_false_positives(items, cfg, frame_shape=frame_shape)
    kept_helmets = set(suppression["kept_helmet_indices"])

    person_count = 0
    helmet_count = 0
    raw_helmet_count = 0
    head_count = 0
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

    raw_helmet_evidence = raw_helmet_count > 0
    head_count = max(0, int(suppression.get("head_count", 0)))
    missing_helmet_count = max(person_count - helmet_count, 0) if person_count > 0 else 0
    if head_count > 0 and not raw_helmet_evidence:
        missing_helmet_count = max(missing_helmet_count, head_count)
    candidate = (person_count > 0 or head_count > 0) and missing_helmet_count > 0
    if candidate:
        if suppression["suppressed_helmet_indices"]:
            reason = "检测到头部/安全帽高重叠，低可信安全帽已降级为未戴帽候选"
        elif head_count > 0 and not raw_helmet_evidence:
            reason = "检测到裸头/头部目标，且安全帽证据不足"
        elif helmet_count == 0:
            reason = "检测到人员，但未检测到有效安全帽"
        else:
            reason = "人员数量多于有效安全帽数量"
    elif person_count > 0:
        reason = "检测到人员，安全帽数量满足当前检测结果"
    else:
        reason = "未检测到人员"
    return {
        "person_count": person_count,
        "helmet_count": helmet_count,
        "raw_helmet_count": raw_helmet_count,
        "raw_head_count": raw_head_count,
        "head_count": head_count,
        "missing_helmet_count": missing_helmet_count,
        "candidate": candidate,
        "reason": reason,
        "class_counts": class_counts,
        "helmet_fp_suppression": suppression,
    }

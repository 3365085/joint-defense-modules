from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any, Sequence

from defense.module_a.ppe_postprocess import (
    PPEPostprocessConfig,
    bbox_area,
    bbox_center_distance_ratio,
    bbox_iou,
    bbox_min_overlap_ratio,
    extract_ppe_detections,
    is_bare_head_label,
    is_helmet_label,
    is_person_label,
    summarize_ppe_from_detections,
)
from defense.module_a.postprocess import PPEDisplayTracker

from .ppe_state import SafetyHelmetState


@dataclass(slots=True)
class PPEBusinessResult:
    """Business-level PPE interpretation for a processed frame."""

    ppe: dict[str, Any]
    tracks: list[dict[str, Any]]


def evaluate_ppe_business(
    detections: Any,
    *,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    ppe_state: SafetyHelmetState,
    ppe_tracker: PPEDisplayTracker,
    tracking_enabled: bool,
    max_render_misses: int | None = None,
    postprocess_config: PPEPostprocessConfig | None = None,
    source_auth_media_bbox: Sequence[float] | None = None,
    source_auth_suppression_active: bool = False,
) -> PPEBusinessResult:
    """Convert detector boxes into stable safety-helmet business state."""

    cfg = postprocess_config or PPEPostprocessConfig()
    detections_for_ppe, source_auth_suppression = _suppress_source_auth_media_detections(
        detections,
        media_bbox=source_auth_media_bbox,
        active=source_auth_suppression_active,
    )
    clear_temporal_ppe = bool(
        source_auth_suppression.get("active")
        and int(source_auth_suppression.get("suppressed_count") or 0) > 0
        and not _has_current_ppe_evidence(detections_for_ppe, cfg)
    )
    if clear_temporal_ppe:
        ppe_tracker.reset()

    ppe_raw = summarize_ppe_from_detections(
        detections_for_ppe,
        config=cfg,
        frame_shape=frame_shape,
    )
    if source_auth_suppression.get("active") or source_auth_suppression.get("suppressed_count"):
        ppe_raw["source_auth_media_suppression"] = source_auth_suppression
    if clear_temporal_ppe:
        ppe_raw["candidate"] = False
        ppe_raw["uncertain"] = False
        ppe_raw["reason"] = "source_auth_media_roi_suppressed"
        ppe_state.reset()

    render_misses = _effective_max_render_misses(max_render_misses, ppe_raw)
    tracks = (
        ppe_tracker.update(
            detections_for_ppe,
            ppe_raw,
            frame_shape=frame_shape,
            max_render_misses=render_misses,
        )
        if tracking_enabled
        else []
    )
    ppe_input = (
        ppe_tracker.apply_temporal_evidence(ppe_raw, frame_shape)
        if tracking_enabled
        else ppe_raw
    )
    ppe_input = _apply_temporal_helmet_mutex(ppe_input, tracks, cfg, frame_shape)
    ppe = ppe_state.update(ppe_input)
    if source_auth_suppression.get("active") or source_auth_suppression.get("suppressed_count"):
        ppe["source_auth_media_suppression"] = source_auth_suppression
        ppe["source_auth_temporal_reset"] = clear_temporal_ppe
    tracks = _filter_tracks_for_ppe_counts(tracks, ppe)
    tracks = _dedup_display_tracks(tracks)  # 显示层同类重叠去重, 消除"一人多框", 不影响报警计数
    return PPEBusinessResult(ppe=ppe, tracks=tracks)


def _apply_temporal_helmet_mutex(
    ppe: dict[str, Any],
    tracks: list[dict[str, Any]],
    config: PPEPostprocessConfig,
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> dict[str, Any]:
    """Align Web business counts with the final stable helmet display tracks."""

    if not config.prefer_helmet_on_head_overlap:
        return ppe
    if int(ppe.get("head_count") or 0) <= 0:
        return ppe
    suppression = ppe.get("helmet_fp_suppression")
    if not isinstance(suppression, dict):
        return ppe
    effective_heads = [
        item
        for item in suppression.get("effective_heads", []) or []
        if isinstance(item, dict) and _normalize_bbox(item.get("head_bbox")) is not None
    ]
    if not effective_heads:
        return ppe
    helmet_tracks = [
        track
        for track in tracks
        if str(track.get("label") or "") == "helmet" and _normalize_bbox(track.get("box")) is not None
    ]
    if not helmet_tracks:
        return ppe

    covered_indices: set[int] = set()
    covered_heads: list[dict[str, Any]] = []
    for head in effective_heads:
        head_bbox = _normalize_bbox(head.get("head_bbox"))
        if head_bbox is None:
            continue
        best_track: dict[str, Any] | None = None
        best_iou = 0.0
        best_distance = 1.0
        for track in helmet_tracks:
            track_bbox = _normalize_bbox(track.get("box"))
            if track_bbox is None:
                continue
            iou = bbox_iou(head_bbox, track_bbox)
            distance = bbox_center_distance_ratio(head_bbox, track_bbox, frame_shape)
            containment = bbox_min_overlap_ratio(head_bbox, track_bbox)
            same_target = bool(
                iou >= config.head_helmet_mutex_iou
                or (
                    distance <= config.head_helmet_mutex_center_distance
                    and containment >= config.head_helmet_mutex_min_overlap
                )
            )
            if not same_target:
                continue
            if best_track is None or iou > best_iou or (
                best_iou < config.head_helmet_mutex_iou and distance < best_distance
            ):
                best_track = track
                best_iou = iou
                best_distance = distance
        if best_track is None:
            continue
        head_index = int(head.get("head_index", -1))
        covered_indices.add(head_index)
        covered_heads.append(
            {
                "head_index": head_index,
                "head_confidence": float(head.get("head_confidence") or 0.0),
                "head_bbox": list(head_bbox),
                "helmet_track_id": int(best_track.get("track_id") or 0),
                "helmet_confidence": float(best_track.get("confidence") or 0.0),
                "helmet_bbox": list(_normalize_bbox(best_track.get("box")) or ()),
                "iou": float(best_iou),
                "center_distance": float(best_distance),
                "reason": "temporal_helmet_mutex",
            }
        )

    if not covered_indices:
        return ppe

    out = dict(ppe)
    out_suppression = dict(suppression)
    existing_covered = {int(value) for value in out_suppression.get("covered_head_indices", []) or []}
    existing_suppressed = {int(value) for value in out_suppression.get("suppressed_head_indices", []) or []}
    existing_covered.update(covered_indices)
    existing_suppressed.update(covered_indices)
    remaining_heads = [
        dict(item)
        for item in effective_heads
        if int(item.get("head_index", -1)) not in covered_indices
    ]
    out_suppression["covered_head_indices"] = sorted(existing_covered)
    out_suppression["suppressed_head_indices"] = sorted(existing_suppressed)
    out_suppression["temporal_helmet_mutex_heads"] = covered_heads
    out_suppression["effective_head_indices"] = sorted(
        int(item.get("head_index", -1)) for item in remaining_heads
    )
    out_suppression["effective_heads"] = remaining_heads
    out_suppression["head_count"] = len(remaining_heads)
    out["helmet_fp_suppression"] = out_suppression
    temporal_helmet_count = len({int(item.get("helmet_track_id") or 0) for item in covered_heads})
    promoted_helmet_count = int(out.get("promoted_helmet_count") or 0) + temporal_helmet_count
    helmet_count = int(out.get("helmet_count") or 0) + temporal_helmet_count
    out["head_count"] = len(remaining_heads)
    out["effective_head_count"] = len(remaining_heads)
    out["missing_helmet_count"] = len(remaining_heads)
    out["promoted_helmet_count"] = promoted_helmet_count
    out["helmet_count"] = helmet_count
    out["effective_helmet_count"] = int(out.get("effective_helmet_count") or 0) + temporal_helmet_count
    out["temporal_helmet_mutex_count"] = temporal_helmet_count
    out["candidate"] = len(remaining_heads) > 0
    if len(remaining_heads) == 0:
        out["uncertain"] = False
        if helmet_count > 0:
            out["reason"] = "helmet_evidence_present"
        elif int(out.get("person_count") or 0) > 0:
            out["reason"] = "person_context_without_head_or_helmet_evidence"
    return out


def _effective_max_render_misses(max_render_misses: int | None, ppe: dict[str, Any]) -> int | None:
    if max_render_misses is None:
        return None
    base = max(0, int(max_render_misses))
    if str(ppe.get("evidence_mode") or "") == "head_helmet_only":
        return max(base, 8)
    return base


def _dedup_display_tracks(
    tracks: list[dict[str, Any]], *, iou_threshold: float = 0.6
) -> list[dict[str, Any]]:
    """显示层同类重叠去重: 同一目标被 YOLO 吐多框(NMS iou_thres=0.7 偏松, 大框套小框)时,
    渲染前对同类框做 NMS, 每个目标只留最高分框。纯显示层——不改检测数值/p_adv/报警计数,
    留出集(看 alert_confirmed)不受影响。优先保留 confidence 高、非 held 的框。"""
    if not tracks or len(tracks) < 2:
        return tracks

    def _key(t: dict[str, Any]) -> tuple[float, float]:
        # 排序键: 先真实检出(misses=0)优先, 再按置信度降序
        held = 1 if int(t.get("misses", 0) or 0) > 0 else 0
        return (-held, float(t.get("confidence", 0.0) or 0.0))

    order = sorted(range(len(tracks)), key=lambda i: _key(tracks[i]), reverse=True)
    kept: list[int] = []
    for idx in order:
        box = tracks[idx].get("box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            kept.append(idx)
            continue
        label = str(tracks[idx].get("label", "")).lower()
        drop = False
        for k in kept:
            kb = tracks[k].get("box")
            if not (isinstance(kb, (list, tuple)) and len(kb) == 4):
                continue
            if str(tracks[k].get("label", "")).lower() == label and bbox_iou(list(box), list(kb)) >= iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(idx)
    return [tracks[i] for i in sorted(kept)]


def _filter_tracks_for_ppe_counts(tracks: list[dict[str, Any]], ppe: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_labels: set[str] = set()
    if (
        int(ppe.get("person_count") or 0) > 0
        or int(ppe.get("effective_person_count") or 0) > 0
        or int(ppe.get("weak_person_count") or 0) > 0
    ):
        allowed_labels.add("person")
    if int(ppe.get("helmet_count") or 0) > 0:
        allowed_labels.add("helmet")
    suppression = ppe.get("helmet_fp_suppression", {})
    weak_head_count = 0
    if isinstance(suppression, dict):
        weak_head_count = len(suppression.get("weak_head_indices", []) or [])
    if int(ppe.get("head_count") or 0) > 0 or weak_head_count > 0:
        allowed_labels.add("head")
    if not allowed_labels and str(ppe.get("evidence_mode") or "") == "head_helmet_only":
        held_labels = {
            str(track.get("label") or "")
            for track in tracks
            if int(track.get("misses") or 0) > 0 and str(track.get("label") or "") in {"head", "helmet"}
        }
        allowed_labels.update(held_labels)
    if not allowed_labels:
        return []
    return [dict(track) for track in tracks if str(track.get("label") or "") in allowed_labels]


def _suppress_source_auth_media_detections(
    detections: Any,
    *,
    media_bbox: Sequence[float] | None,
    active: bool,
) -> tuple[Any, dict[str, Any]]:
    bbox = _normalize_bbox(media_bbox)
    meta: dict[str, Any] = {
        "active": bool(active and bbox is not None),
        "bbox": list(bbox) if bbox is not None else None,
        "reason": "a3b_media_roi",
        "suppressed_indices": [],
        "suppressed_labels": [],
        "suppressed_count": 0,
        "kept_count": len(getattr(detections, "boxes", []) or []),
        "total_count": len(getattr(detections, "boxes", []) or []),
    }
    if not meta["active"]:
        return detections, meta

    boxes = list(getattr(detections, "boxes", []) or [])
    classes = list(getattr(detections, "classes", []) or [])
    confidences = list(getattr(detections, "confidences", []) or [])
    names = getattr(detections, "names", {}) or {}
    keep_boxes: list[Any] = []
    keep_classes: list[Any] = []
    keep_confidences: list[Any] = []
    suppressed_indices: list[int] = []
    suppressed_labels: list[str] = []

    for index, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
        label = str(names.get(int(class_id), f"class_{int(class_id)}")) if isinstance(names, dict) else str(class_id)
        is_ppe_label = is_person_label(label) or is_bare_head_label(label) or is_helmet_label(label)
        if is_ppe_label and _box_inside_media_roi(box, bbox):
            suppressed_indices.append(index)
            suppressed_labels.append(label)
            continue
        keep_boxes.append(box)
        keep_classes.append(class_id)
        keep_confidences.append(confidence)

    if not suppressed_indices:
        return detections, meta

    filtered = copy(detections)
    filtered.boxes = keep_boxes
    filtered.classes = keep_classes
    filtered.confidences = keep_confidences
    meta.update(
        {
            "suppressed_indices": suppressed_indices,
            "suppressed_labels": suppressed_labels,
            "suppressed_count": len(suppressed_indices),
            "kept_count": len(keep_boxes),
        }
    )
    return filtered, meta


def _normalize_bbox(bbox: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if bbox is None or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_inside_media_roi(box: Sequence[float] | None, media_bbox: tuple[float, float, float, float]) -> bool:
    bbox = _normalize_bbox(box)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    mx1, my1, mx2, my2 = media_bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_inside = mx1 <= cx <= mx2 and my1 <= cy <= my2
    box_area = bbox_area(bbox)
    media_overlap = bbox_iou(bbox, media_bbox)
    if box_area <= 0.0:
        return center_inside
    ix1 = max(x1, mx1)
    iy1 = max(y1, my1)
    ix2 = min(x2, mx2)
    iy2 = min(y2, my2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return bool(center_inside or intersection / box_area >= 0.50 or media_overlap >= 0.50)


def _has_current_ppe_evidence(detections: Any, config: PPEPostprocessConfig) -> bool:
    for item in extract_ppe_detections(detections):
        if item.confidence < config.min_confidence:
            continue
        if is_person_label(item.label) or is_bare_head_label(item.label) or is_helmet_label(item.label):
            return True
    return False

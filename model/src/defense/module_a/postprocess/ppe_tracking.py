from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from defense.module_a.backends.detector_backend import DetectionFrameResult

try:  # scipy is available in the project pixi env; keep a fallback for portability.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


LABEL_ALIASES = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "safety_helmet": "helmet",
    "safety helmet": "helmet",
    "head": "head",
    "bare_head": "head",
    "no_helmet": "head",
    "person": "person",
    "worker": "person",
}


@dataclass(frozen=True, slots=True)
class StableTrack:
    track_id: int
    box: list[int]
    label: str
    confidence: float
    misses: int
    age: int
    is_small: bool
    source: str
    hold_eligible: bool = True
    evidence_label: str = ""
    weak_head_streak: int = 0
    weak_helmet_streak: int = 0
    temporal_promoted: bool = False
    promoted_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": list(self.box),
            "label": self.label,
            "confidence": float(self.confidence),
            "misses": int(self.misses),
            "age": int(self.age),
            "is_small": bool(self.is_small),
            "source": self.source,
            "hold_eligible": bool(self.hold_eligible),
            "evidence_label": self.evidence_label,
            "weak_head_streak": int(self.weak_head_streak),
            "weak_helmet_streak": int(self.weak_helmet_streak),
            "temporal_promoted": bool(self.temporal_promoted),
            "promoted_label": self.promoted_label,
        }


def canonical_label(label: str) -> str | None:
    normalized = str(label or "").strip().lower().replace("-", "_")
    return LABEL_ALIASES.get(normalized)


def bbox_area(box: list[int] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(
    box_a: list[int] | tuple[float, float, float, float],
    box_b: list[int] | tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = bbox_area(box_a) + bbox_area(box_b) - inter
    return inter / denom if denom > 0.0 else 0.0


def center_distance_ratio(
    box_a: list[int] | tuple[float, float, float, float],
    box_b: list[int] | tuple[float, float, float, float],
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) * 0.5
    ay = (float(box_a[1]) + float(box_a[3])) * 0.5
    bx = (float(box_b[0]) + float(box_b[2])) * 0.5
    by = (float(box_b[1]) + float(box_b[3])) * 0.5
    h, w = frame_shape[:2]
    diag = max(1.0, (float(w) ** 2 + float(h) ** 2) ** 0.5)
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diag


def _frame_area(frame_shape: tuple[int, int] | tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    return float(max(1, int(h)) * max(1, int(w)))


def _clip_box(box: list[float] | tuple[float, float, float, float], frame_shape: tuple[int, int] | tuple[int, int, int]) -> list[int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        int(max(0, min(w - 1, round(x1)))),
        int(max(0, min(h - 1, round(y1)))),
        int(max(0, min(w - 1, round(x2)))),
        int(max(0, min(h - 1, round(y2)))),
    ]


def _smooth_box(previous: list[int], current: list[int], alpha: float) -> list[int]:
    return [
        int(round(float(previous[i]) * (1.0 - alpha) + float(current[i]) * alpha))
        for i in range(4)
    ]


def _estimate_velocity(prev_box: list[int], curr_box: list[int]) -> list[float]:
    """Estimate per-coordinate velocity (pixels/frame) between two boxes."""
    return [float(curr_box[i]) - float(prev_box[i]) for i in range(4)]


def _extrapolate_box(
    box: list[int], velocity: list[float], steps: int = 1
) -> list[int]:
    """Extrapolate box position using velocity estimate."""
    return [
        int(round(float(box[i]) + velocity[i] * steps))
        for i in range(4)
    ]


class PPEDisplayTracker:
    """Stable PPE display tracks decoupled from the web UI.

    The tracker does not change Module A alarm decisions. It only converts raw
    detector boxes into stable display tracks and optionally recommends a small
    ROI re-detect for distant/unstable PPE targets.
    """

    def __init__(
        self,
        *,
        history: int = 7,
        hold_frames: int = 6,
        small_hold_frames: int = 12,
        switch_count: int = 3,
        small_area_ratio: float = 0.018,
        small_confidence: float = 0.55,
        redetect_interval: int = 3,
        iou_match_threshold: float = 0.12,
        max_missed_ms: float = 700.0,
        hold_last_box: bool = True,
        smooth_alpha: float = 0.35,
        show_held_boxes: bool = True,
        weak_promotion_hits: int = 3,
        weak_head_min_avg_confidence: float = 0.30,
        weak_helmet_min_avg_confidence: float = 0.30,
        weak_helmet_isolated_min_avg_confidence: float = 0.50,
        weak_edge_promotion_hits: int = 4,
        weak_edge_min_avg_confidence: float = 0.45,
    ):
        self.history = max(1, int(history))
        self.hold_frames = max(0, int(hold_frames))
        self.small_hold_frames = max(self.hold_frames, int(small_hold_frames))
        self.switch_count = max(1, int(switch_count))
        self.small_area_ratio = float(small_area_ratio)
        self.small_confidence = float(small_confidence)
        self.redetect_interval = max(1, int(redetect_interval))
        self.iou_match_threshold = max(0.0, min(1.0, float(iou_match_threshold)))
        self.max_missed_ms = max(0.0, float(max_missed_ms))
        self.hold_last_box = bool(hold_last_box)
        self.smooth_alpha = max(0.0, min(1.0, float(smooth_alpha)))
        self.show_held_boxes = bool(show_held_boxes)
        self.weak_promotion_hits = max(1, int(weak_promotion_hits))
        self.weak_head_min_avg_confidence = float(weak_head_min_avg_confidence)
        self.weak_helmet_min_avg_confidence = float(weak_helmet_min_avg_confidence)
        self.weak_helmet_isolated_min_avg_confidence = float(weak_helmet_isolated_min_avg_confidence)
        self.weak_edge_promotion_hits = max(self.weak_promotion_hits, int(weak_edge_promotion_hits))
        self.weak_edge_min_avg_confidence = float(weak_edge_min_avg_confidence)
        self.tracks: list[dict[str, Any]] = []
        self.next_track_id = 1
        self._last_redetect_frame = -10_000

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self._last_redetect_frame = -10_000

    def recommend_redetect_rois(
        self,
        detections: DetectionFrameResult,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        frame_idx: int,
        *,
        enabled: bool = True,
        max_rois: int = 1,
    ) -> list[list[int]]:
        if not enabled or frame_idx - self._last_redetect_frame < self.redetect_interval:
            return []
        frame_area = _frame_area(frame_shape)
        candidates: list[tuple[float, list[int]]] = []

        for box, cls_id, confidence in zip(detections.boxes, detections.classes, detections.confidences):
            label = canonical_label(detections.names.get(int(cls_id), f"class_{int(cls_id)}"))
            if label not in {"person", "helmet", "head"}:
                continue
            area_ratio = bbox_area(box) / frame_area
            if area_ratio <= self.small_area_ratio and float(confidence) <= self.small_confidence:
                priority = (self.small_confidence - float(confidence)) + (self.small_area_ratio - area_ratio)
                candidates.append((priority, list(box)))

        for track in self.tracks:
            if int(track.get("misses", 0)) <= 0:
                continue
            if not bool(track.get("hold_eligible", True)):
                continue
            if not bool(track.get("is_small", False)):
                continue
            priority = 2.0 + int(track.get("misses", 0)) * 0.2
            candidates.append((priority, list(track["box"])))

        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        rois: list[list[int]] = []
        for _, box in candidates:
            roi = self._expanded_square_roi(box, frame_shape)
            if not any(bbox_iou(roi, existing) >= 0.35 for existing in rois):
                rois.append(roi)
            if len(rois) >= max_rois:
                break
        if rois:
            self._last_redetect_frame = frame_idx
        return rois

    def update(
        self,
        detections: DetectionFrameResult,
        ppe: dict[str, Any],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        *,
        max_render_misses: int | None = None,
    ) -> list[dict[str, Any]]:
        incoming = self._incoming_items(detections, ppe, frame_shape)
        incoming = self._merge_same_target_items(incoming, frame_shape)
        self._assign_items(incoming, frame_shape)
        self._age_unmatched()
        self._prune_tracks()
        self.tracks = self._remove_shadow_tracks(self.tracks, frame_shape)
        self.tracks = self._filter_low_context_display_tracks(self.tracks, frame_shape)
        self._refresh_temporal_promotions(frame_shape)
        return [track.as_dict() for track in self._render_tracks(max_misses=max_render_misses)]

    def apply_temporal_evidence(
        self,
        ppe: dict[str, Any],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> dict[str, Any]:
        """Promote stable current-frame weak PPE evidence without extra inference."""
        promotions = self._current_temporal_promotions(frame_shape)
        promoted_heads = [item for item in promotions if item["label"] == "head"]
        promoted_helmets = [item for item in promotions if item["label"] == "helmet"]
        out = dict(ppe)
        suppression = (
            dict(out.get("helmet_fp_suppression") or {})
            if isinstance(out.get("helmet_fp_suppression"), dict)
            else {}
        )
        base_head_count = int(out.get("head_count") or 0)
        base_helmet_count = int(out.get("helmet_count") or 0)
        promoted_head_count = len(promoted_heads)
        promoted_helmet_count = len(promoted_helmets)
        effective_head_count = base_head_count + promoted_head_count
        effective_helmet_count = base_helmet_count + promoted_helmet_count

        out["promoted_head_count"] = promoted_head_count
        out["promoted_helmet_count"] = promoted_helmet_count
        out["effective_head_count"] = effective_head_count
        out["effective_helmet_count"] = effective_helmet_count
        out["head_count"] = effective_head_count
        out["helmet_count"] = effective_helmet_count
        out["missing_helmet_count"] = effective_head_count
        out["candidate"] = effective_head_count > 0
        if promoted_head_count or promoted_helmet_count:
            out["inferred_person_count"] = max(
                int(out.get("inferred_person_count") or 0),
                1,
            )
        out["uncertain"] = bool(
            out.get("uncertain", False)
            and effective_head_count == 0
            and effective_helmet_count == 0
        )
        if effective_head_count > 0:
            out["reason"] = (
                "bare_head_without_matched_helmet"
                if base_head_count > 0
                else "temporal_weak_head_promoted"
            )
        elif promoted_helmet_count > 0 and effective_helmet_count > 0:
            out["reason"] = (
                "helmet_evidence_present"
                if base_helmet_count > 0
                else "temporal_weak_helmet_promoted"
            )

        suppression["temporal_promoted_head_tracks"] = [
            dict(item) for item in promoted_heads
        ]
        suppression["temporal_promoted_helmet_tracks"] = [
            dict(item) for item in promoted_helmets
        ]
        out["helmet_fp_suppression"] = suppression
        out["temporal_promotions"] = [dict(item) for item in promotions]
        return out

    def _incoming_items(
        self,
        detections: DetectionFrameResult,
        ppe: dict[str, Any],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        suppression = ppe.get("helmet_fp_suppression", {})
        suppressed_helmets = self._suppressed_helmet_indices_for_display(suppression)
        weak_heads = {int(v) for v in suppression.get("weak_head_indices", []) or []}
        weak_helmets = {int(v) for v in suppression.get("weak_helmet_indices", []) or []}
        weak_head_reasons = self._weak_head_reasons_by_index(suppression)
        weak_helmet_reasons = self._weak_helmet_reasons_by_index(suppression)
        frame_area = _frame_area(frame_shape)
        incoming: list[dict[str, Any]] = []
        for index, (box, cls_id, confidence) in enumerate(
            zip(detections.boxes, detections.classes, detections.confidences)
        ):
            raw_label = detections.names.get(int(cls_id), f"class_{int(cls_id)}")
            label = canonical_label(raw_label)
            if label is None:
                continue
            if label == "helmet" and index in suppressed_helmets:
                continue
            clipped = _clip_box(box, frame_shape)
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            area_ratio = bbox_area(clipped) / frame_area
            weak_label = ""
            weak_reason = ""
            if label == "head" and index in weak_heads:
                weak_label = "head"
                weak_reason = weak_head_reasons.get(index, "")
            elif label == "helmet" and index in weak_helmets:
                weak_label = "helmet"
                weak_reason = weak_helmet_reasons.get(index, "")
            incoming.append(
                {
                    "index": index,
                    "box": clipped,
                    "label": label,
                    "raw_label": str(raw_label),
                    "confidence": float(confidence),
                    "is_small": bool(area_ratio <= self.small_area_ratio),
                    "area_ratio": float(area_ratio),
                    "hold_eligible": not bool(weak_label),
                    "weak_evidence_label": weak_label,
                    "weak_reason": weak_reason,
                }
            )
        return incoming

    def _assign_items(self, incoming: list[dict[str, Any]], frame_shape: tuple[int, int] | tuple[int, int, int]) -> None:
        for track in self.tracks:
            track["_matched"] = False
        if not incoming:
            return
        if not self.tracks:
            for item in incoming:
                self._create_track(item)
            return

        costs = np.full((len(self.tracks), len(incoming)), 10_000.0, dtype=np.float32)
        for track_idx, track in enumerate(self.tracks):
            for item_idx, item in enumerate(incoming):
                iou = bbox_iou(track["box"], item["box"])
                distance = center_distance_ratio(track["box"], item["box"], frame_shape)
                compatible = iou >= self.iou_match_threshold or distance <= (0.055 if item["is_small"] else 0.035)
                if not compatible:
                    continue
                label_penalty = 0.0 if item["label"] == track.get("stable_label") else 0.18
                if {str(item["label"]), str(track.get("stable_label", ""))} <= {"helmet", "head"}:
                    label_penalty *= 0.5
                costs[track_idx, item_idx] = (1.0 - iou) + distance * 3.0 + label_penalty

        matched_tracks: set[int] = set()
        matched_items: set[int] = set()
        for track_idx, item_idx in self._solve_assignment(costs):
            if costs[track_idx, item_idx] >= 2.0:
                continue
            self._update_track(self.tracks[track_idx], incoming[item_idx])
            matched_tracks.add(track_idx)
            matched_items.add(item_idx)

        for item_idx, item in enumerate(incoming):
            if item_idx not in matched_items:
                self._create_track(item)
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks and not track.get("_matched", False):
                track["_matched"] = False

    def _solve_assignment(self, costs: np.ndarray) -> list[tuple[int, int]]:
        if costs.size == 0:
            return []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(costs)
            return [(int(r), int(c)) for r, c in zip(rows, cols)]
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in sorted(np.ndindex(costs.shape), key=lambda rc: float(costs[rc])):
            if row in used_rows or col in used_cols:
                continue
            pairs.append((int(row), int(col)))
            used_rows.add(int(row))
            used_cols.add(int(col))
        return pairs

    def _create_track(self, item: dict[str, Any]) -> None:
        self.tracks.append(
            {
                "id": self.next_track_id,
                "box": list(item["box"]),
                "raw_box": list(item["box"]),
                "prev_box": list(item["box"]),
                "velocity": [0.0, 0.0, 0.0, 0.0],
                "labels": deque([item["label"]], maxlen=self.history),
                "stable_label": item["label"],
                "pending_label": None,
                "pending_count": 0,
                "confidence": float(item["confidence"]),
                "age": 1,
                "misses": 0,
                "is_small": bool(item["is_small"]),
                "area_ratio": float(item.get("area_ratio", 0.0)),
                "hold_eligible": bool(item.get("hold_eligible", True)),
                "current_weak_label": str(item.get("weak_evidence_label") or ""),
                "weak_reason": str(item.get("weak_reason") or ""),
                "weak_head_streak": 1 if item.get("weak_evidence_label") == "head" else 0,
                "weak_helmet_streak": 1 if item.get("weak_evidence_label") == "helmet" else 0,
                "weak_head_conf_sum": float(item["confidence"]) if item.get("weak_evidence_label") == "head" else 0.0,
                "weak_helmet_conf_sum": float(item["confidence"]) if item.get("weak_evidence_label") == "helmet" else 0.0,
                "temporal_promoted_label": None,
                "source": "detected",
                "_matched": True,
            }
        )
        self.next_track_id += 1

    def _update_track(self, track: dict[str, Any], item: dict[str, Any]) -> None:
        # Phase 1.2: lower alpha for smoother box tracking (was 0.58/0.46)
        alpha = max(self.smooth_alpha, 0.55) if track.get("misses", 0) > 0 else self.smooth_alpha
        prev_raw = list(track.get("raw_box", track["box"]))
        track["prev_box"] = list(track["box"])
        track["box"] = _smooth_box(track["box"], item["box"], alpha)
        track["raw_box"] = list(item["box"])
        # Phase 1.2: estimate velocity for extrapolation during misses
        track["velocity"] = _estimate_velocity(prev_raw, item["box"])
        track["labels"].append(item["label"])
        track["confidence"] = float(track["confidence"]) * 0.62 + float(item["confidence"]) * 0.38
        track["age"] = int(track.get("age", 0)) + 1
        track["misses"] = 0
        track["is_small"] = bool(item["is_small"])
        track["area_ratio"] = float(item.get("area_ratio", track.get("area_ratio", 0.0)))
        track["hold_eligible"] = bool(item.get("hold_eligible", True))
        track["current_weak_label"] = str(item.get("weak_evidence_label") or "")
        track["weak_reason"] = str(item.get("weak_reason") or "")
        track["source"] = "detected"
        track["_matched"] = True
        self._update_weak_streak(track, item)
        self._update_stable_label(track, str(item["label"]), float(item["confidence"]))

    def _age_unmatched(self) -> None:
        for track in self.tracks:
            if track.pop("_matched", False):
                continue
            misses = int(track.get("misses", 0)) + 1
            track["misses"] = misses
            track["confidence"] = float(track.get("confidence", 0.0)) * (0.90 if track.get("is_small") else 0.84)
            track["source"] = "held"
            track["current_weak_label"] = ""
            track["weak_reason"] = ""
            track["weak_head_streak"] = 0
            track["weak_helmet_streak"] = 0
            track["weak_head_conf_sum"] = 0.0
            track["weak_helmet_conf_sum"] = 0.0
            track["temporal_promoted_label"] = None
            # Phase 1.2: extrapolate position using velocity during misses
            # instead of holding the box static (reduces lag on reappearance)
            velocity = track.get("velocity", [0.0, 0.0, 0.0, 0.0])
            if self.hold_last_box and any(abs(v) > 0.5 for v in velocity):
                # Dampen velocity over time to avoid runaway extrapolation
                damping = max(0.0, 1.0 - misses * 0.15)
                damped_v = [v * damping for v in velocity]
                track["box"] = _extrapolate_box(track["box"], damped_v, steps=1)

    def _prune_tracks(self) -> None:
        kept: list[dict[str, Any]] = []
        for track in self.tracks:
            max_misses = self.small_hold_frames if track.get("is_small") else self.hold_frames
            if int(track.get("misses", 0)) <= max_misses:
                kept.append(track)
        self.tracks = kept

    def _render_tracks(self, *, max_misses: int | None = None) -> list[StableTrack]:
        rendered: list[StableTrack] = []
        for track in self.tracks:
            confidence = float(track.get("confidence", 0.0))
            misses = int(track.get("misses", 0))
            if misses > 0:
                if not self.show_held_boxes:
                    continue
                if not bool(track.get("hold_eligible", True)):
                    continue
                if max_misses is not None and misses > max(0, int(max_misses)):
                    continue
            if confidence < (0.14 if track.get("is_small") else 0.18):
                continue
            rendered.append(
                StableTrack(
                    track_id=int(track["id"]),
                    box=list(track["box"]),
                    label=str(track["stable_label"]),
                    confidence=confidence,
                    misses=misses,
                    age=int(track.get("age", 0)),
                    is_small=bool(track.get("is_small", False)),
                    source=str(track.get("source", "detected")),
                    hold_eligible=bool(track.get("hold_eligible", True)),
                    evidence_label=str(track.get("current_weak_label") or ""),
                    weak_head_streak=int(track.get("weak_head_streak", 0) or 0),
                    weak_helmet_streak=int(track.get("weak_helmet_streak", 0) or 0),
                    temporal_promoted=bool(track.get("temporal_promoted_label")),
                    promoted_label=str(track.get("temporal_promoted_label") or ""),
                )
            )
        return rendered

    def _remove_shadow_tracks(self, tracks: list[dict[str, Any]], frame_shape: tuple[int, int] | tuple[int, int, int]) -> list[dict[str, Any]]:
        keep = [True] * len(tracks)
        for left in range(len(tracks)):
            for right in range(left + 1, len(tracks)):
                if not keep[left] or not keep[right]:
                    continue
                left_label = str(tracks[left].get("stable_label", ""))
                right_label = str(tracks[right].get("stable_label", ""))
                iou = bbox_iou(tracks[left]["box"], tracks[right]["box"])
                distance = center_distance_ratio(tracks[left]["box"], tracks[right]["box"], frame_shape)
                if {left_label, right_label} <= {"helmet", "head"}:
                    same_target = iou >= 0.20 or distance <= 0.035
                elif left_label == right_label:
                    same_target = iou >= 0.68
                else:
                    same_target = iou >= 0.55
                if not same_target:
                    continue
                left_score = float(tracks[left].get("confidence", 0.0)) - 0.10 * int(tracks[left].get("misses", 0))
                right_score = float(tracks[right].get("confidence", 0.0)) - 0.10 * int(tracks[right].get("misses", 0))
                if left_score >= right_score:
                    keep[right] = False
                else:
                    keep[left] = False
        return [track for track, should_keep in zip(tracks, keep) if should_keep]

    def _filter_low_context_display_tracks(
        self,
        tracks: list[dict[str, Any]],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        context_tracks = [track for track in tracks if str(track.get("stable_label", "")) in {"head", "person"}]
        for track in tracks:
            label = str(track.get("stable_label", ""))
            confidence = float(track.get("confidence", 0.0))
            if label == "helmet" and confidence < (0.30 if track.get("is_small") else 0.38):
                if str(track.get("current_weak_label") or "") == "helmet":
                    filtered.append(track)
                    continue
                has_context = any(
                    bbox_iou(track["box"], context["box"]) >= 0.01
                    or center_distance_ratio(track["box"], context["box"], frame_shape) <= 0.065
                    for context in context_tracks
                )
                if not has_context:
                    continue
            if label == "head" and confidence < (0.20 if track.get("is_small") else 0.26):
                continue
            filtered.append(track)
        return filtered

    def _merge_same_target_items(
        self,
        items: list[dict[str, Any]],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        parent = list(range(len(items)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                left_label = str(items[left]["label"])
                right_label = str(items[right]["label"])
                labels = {left_label, right_label}
                if "person" in labels and labels != {"person"}:
                    continue
                iou = bbox_iou(items[left]["box"], items[right]["box"])
                distance = center_distance_ratio(items[left]["box"], items[right]["box"], frame_shape)
                if left_label == right_label:
                    same_zone = iou >= 0.72
                elif labels <= {"helmet", "head"}:
                    same_zone = iou >= 0.30 or distance <= 0.026
                else:
                    same_zone = iou >= 0.62
                if same_zone:
                    union(left, right)

        clusters: dict[int, list[dict[str, Any]]] = {}
        for index, item in enumerate(items):
            clusters.setdefault(find(index), []).append(item)

        return [self._select_cluster_representative(cluster) for cluster in clusters.values()]

    def _select_cluster_representative(self, cluster_items: list[dict[str, Any]]) -> dict[str, Any]:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for item in cluster_items:
            by_label.setdefault(str(item["label"]), []).append(item)
        if "helmet" in by_label and "head" in by_label:
            best_helmet = max(by_label["helmet"], key=lambda item: float(item["confidence"]))
            best_head = max(by_label["head"], key=lambda item: float(item["confidence"]))
            if float(best_head["confidence"]) >= max(0.65, float(best_helmet["confidence"]) + 0.10):
                return best_head.copy()
            return best_helmet.copy()
        priority = {"helmet": 3, "head": 2, "person": 1}
        return max(cluster_items, key=lambda item: (priority.get(str(item["label"]), 0), float(item["confidence"]))).copy()

    def _suppressed_helmet_indices_for_display(self, suppression: dict[str, Any]) -> set[int]:
        hidden_reasons = {"head_helmet_overlap"}
        suppressed: set[int] = set()
        for item in suppression.get("suppressed_helmets", []) or []:
            try:
                index = int(item.get("helmet_index"))
            except (TypeError, ValueError):
                continue
            if str(item.get("reason", "")) in hidden_reasons:
                suppressed.add(index)
        if not suppression.get("suppressed_helmets"):
            weak_helmets = {int(v) for v in suppression.get("weak_helmet_indices", []) or []}
            suppressed.update(
                int(v)
                for v in suppression.get("suppressed_helmet_indices", []) or []
                if int(v) not in weak_helmets
            )
        return suppressed

    def _weak_head_reasons_by_index(self, suppression: dict[str, Any]) -> dict[int, str]:
        reasons: dict[int, str] = {}
        for item in suppression.get("suppressed_heads", []) or []:
            try:
                index = int(item.get("head_index"))
            except (TypeError, ValueError):
                continue
            reasons[index] = str(item.get("reason") or "")
        return reasons

    def _weak_helmet_reasons_by_index(self, suppression: dict[str, Any]) -> dict[int, str]:
        reasons: dict[int, str] = {}
        for item in suppression.get("suppressed_helmets", []) or []:
            try:
                index = int(item.get("helmet_index"))
            except (TypeError, ValueError):
                continue
            reasons[index] = str(item.get("reason") or "")
        return reasons

    def _update_weak_streak(self, track: dict[str, Any], item: dict[str, Any]) -> None:
        weak_label = str(item.get("weak_evidence_label") or "")
        confidence = float(item.get("confidence", 0.0))
        if weak_label == "head":
            track["weak_head_streak"] = int(track.get("weak_head_streak", 0) or 0) + 1
            track["weak_head_conf_sum"] = float(track.get("weak_head_conf_sum", 0.0) or 0.0) + confidence
            track["weak_helmet_streak"] = 0
            track["weak_helmet_conf_sum"] = 0.0
        elif weak_label == "helmet":
            track["weak_helmet_streak"] = int(track.get("weak_helmet_streak", 0) or 0) + 1
            track["weak_helmet_conf_sum"] = float(track.get("weak_helmet_conf_sum", 0.0) or 0.0) + confidence
            track["weak_head_streak"] = 0
            track["weak_head_conf_sum"] = 0.0
        else:
            track["weak_head_streak"] = 0
            track["weak_helmet_streak"] = 0
            track["weak_head_conf_sum"] = 0.0
            track["weak_helmet_conf_sum"] = 0.0
        track["temporal_promoted_label"] = None

    def _refresh_temporal_promotions(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> None:
        for track in self.tracks:
            track["temporal_promoted_label"] = None
        helmets = [
            track
            for track in self.tracks
            if str(track.get("stable_label") or "") == "helmet" and int(track.get("misses", 0) or 0) == 0
        ]
        promoted_helmets = [
            track for track in helmets if self._track_promotes_weak_label(track, "helmet", frame_shape)
        ]
        for track in promoted_helmets:
            track["temporal_promoted_label"] = "helmet"
        helmet_blockers = [
            track
            for track in helmets
            if str(track.get("current_weak_label") or "") != "helmet"
            or str(track.get("temporal_promoted_label") or "") == "helmet"
        ]

        for track in self.tracks:
            if not self._track_promotes_weak_label(track, "head", frame_shape):
                continue
            if any(self._same_head_zone(track, helmet, frame_shape) for helmet in helmet_blockers):
                continue
            track["temporal_promoted_label"] = "head"

    def _current_temporal_promotions(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        self._refresh_temporal_promotions(frame_shape)
        promotions: list[dict[str, Any]] = []
        for track in self.tracks:
            label = str(track.get("temporal_promoted_label") or "")
            if label not in {"head", "helmet"}:
                continue
            promotions.append(
                {
                    "track_id": int(track.get("id", 0) or 0),
                    "label": label,
                    "box": list(track.get("box") or []),
                    "confidence": float(track.get("confidence", 0.0) or 0.0),
                    "streak": int(track.get(f"weak_{label}_streak", 0) or 0),
                    "avg_confidence": self._weak_avg_confidence(track, label),
                    "reason": str(track.get("weak_reason") or ""),
                    "area_ratio": float(track.get("area_ratio", 0.0) or 0.0),
                }
            )
        return promotions

    def _track_promotes_weak_label(
        self,
        track: dict[str, Any],
        label: str,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        if int(track.get("misses", 0) or 0) != 0:
            return False
        if str(track.get("stable_label") or "") != label:
            return False
        if str(track.get("current_weak_label") or "") != label:
            return False
        streak = int(track.get(f"weak_{label}_streak", 0) or 0)
        required_hits = self._required_weak_hits(track, label)
        if streak < required_hits:
            return False
        avg_conf = self._weak_avg_confidence(track, label)
        min_conf = self._required_weak_confidence(track, label)
        if label == "helmet" and not self._helmet_has_context(track, frame_shape):
            min_conf = max(min_conf, self.weak_helmet_isolated_min_avg_confidence)
        return avg_conf >= min_conf

    def _required_weak_hits(self, track: dict[str, Any], label: str) -> int:
        required = self.weak_promotion_hits
        reason = str(track.get("weak_reason") or "")
        if label == "head" and reason == "edge_isolated_head":
            required = max(required, self.weak_edge_promotion_hits)
        return required

    def _required_weak_confidence(self, track: dict[str, Any], label: str) -> float:
        if label == "helmet":
            return self.weak_helmet_min_avg_confidence
        reason = str(track.get("weak_reason") or "")
        if reason == "edge_isolated_head":
            return max(self.weak_head_min_avg_confidence, self.weak_edge_min_avg_confidence)
        return self.weak_head_min_avg_confidence

    def _weak_avg_confidence(self, track: dict[str, Any], label: str) -> float:
        streak = max(1, int(track.get(f"weak_{label}_streak", 0) or 0))
        total = float(track.get(f"weak_{label}_conf_sum", 0.0) or 0.0)
        return total / float(streak)

    def _same_head_zone(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        return (
            bbox_iou(left["box"], right["box"]) >= 0.20
            or center_distance_ratio(left["box"], right["box"], frame_shape) <= 0.040
        )

    def _helmet_has_context(
        self,
        helmet_track: dict[str, Any],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        for track in self.tracks:
            if track is helmet_track:
                continue
            if int(track.get("misses", 0) or 0) != 0:
                continue
            label = str(track.get("stable_label") or "")
            if label not in {"head", "person"}:
                continue
            if bbox_iou(helmet_track["box"], track["box"]) >= 0.01:
                return True
            limit = 0.075 if label == "person" else 0.045
            if center_distance_ratio(helmet_track["box"], track["box"], frame_shape) <= limit:
                return True
        return False

    def _update_stable_label(self, track: dict[str, Any], observed_label: str, observed_confidence: float) -> None:
        stable_label = str(track.get("stable_label", ""))
        is_small = bool(track.get("is_small", False))
        # Phase 1.1: raise helmet strong-switch threshold (was 0.45) and add
        # higher bar for small/distant targets to prevent rapid head↔helmet
        # flipping. Also raise head←helmet threshold for small targets.
        if is_small:
            helmet_threshold = 0.70  # small targets: much harder to switch to helmet
            head_threshold = 0.78   # small targets: also harder to switch to head
        else:
            helmet_threshold = 0.60  # normal targets: raised from 0.45
            head_threshold = 0.70   # normal targets: unchanged
        strong_switch = (
            (observed_label == "head" and stable_label == "helmet" and observed_confidence >= head_threshold)
            or (observed_label == "helmet" and stable_label == "head" and observed_confidence >= helmet_threshold)
        )
        if strong_switch:
            track["stable_label"] = observed_label
            # Phase 1.1: do NOT clear vote history — keep it so the majority
            # vote mechanism can still stabilize if the switch was a fluke.
            track["labels"].append(observed_label)
            track["pending_label"] = None
            track["pending_count"] = 0
            return
        votes = Counter(track["labels"])
        majority_label, majority_count = votes.most_common(1)[0]
        if majority_label == stable_label:
            track["pending_label"] = None
            track["pending_count"] = 0
            return
        required = max(3, (len(track["labels"]) // 2) + 1)
        if majority_count >= required:
            track["stable_label"] = majority_label
            track["pending_label"] = None
            track["pending_count"] = 0
            return
        if observed_label != track.get("pending_label"):
            track["pending_label"] = observed_label
            track["pending_count"] = 1
        else:
            track["pending_count"] = int(track.get("pending_count", 0)) + 1
        # Phase 1.1: small targets require more consecutive frames to switch
        effective_switch_count = self.switch_count + (2 if is_small else 0)
        if int(track["pending_count"]) >= effective_switch_count:
            track["stable_label"] = observed_label
            track["pending_label"] = None
            track["pending_count"] = 0

    def _expanded_square_roi(
        self,
        box: list[int] | tuple[float, float, float, float],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> list[int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        size = max(96.0, min(320.0, max(x2 - x1, y2 - y1) * 3.2))
        rx1 = max(0.0, cx - size * 0.5)
        ry1 = max(0.0, cy - size * 0.5)
        rx2 = min(float(w - 1), cx + size * 0.5)
        ry2 = min(float(h - 1), cy + size * 0.5)
        return [int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))]


def merge_roi_detections(
    base: DetectionFrameResult,
    roi_results: list[tuple[list[int], DetectionFrameResult]],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    *,
    min_confidence: float = 0.20,
) -> DetectionFrameResult:
    boxes = [list(box) for box in base.boxes]
    classes = list(base.classes)
    confidences = [float(v) for v in base.confidences]
    names = dict(base.names)
    inference_ms = float(base.inference_ms)

    for roi, result in roi_results:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        inference_ms += float(result.inference_ms)
        for box, cls_id, confidence in zip(result.boxes, result.classes, result.confidences):
            if float(confidence) < min_confidence:
                continue
            label = canonical_label(result.names.get(int(cls_id), f"class_{int(cls_id)}"))
            if label is None:
                continue
            mapped = _clip_box(
                [float(box[0]) + rx1, float(box[1]) + ry1, float(box[2]) + rx1, float(box[3]) + ry1],
                frame_shape,
            )
            if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
                continue
            boxes.append(mapped)
            classes.append(int(cls_id))
            confidences.append(float(confidence))
            names[int(cls_id)] = result.names.get(int(cls_id), names.get(int(cls_id), f"class_{int(cls_id)}"))

    keep = _class_aware_nms(boxes, classes, confidences, names)
    return DetectionFrameResult(
        image=base.image,
        boxes=[boxes[i] for i in keep],
        classes=[classes[i] for i in keep],
        confidences=[confidences[i] for i in keep],
        names=names,
        backend=base.backend,
        artifact_path=base.artifact_path,
        inference_ms=inference_ms,
        raw_result=base.raw_result,
    )


def _class_aware_nms(
    boxes: list[list[int]],
    classes: list[int],
    confidences: list[float],
    names: dict[int, str],
    *,
    iou_threshold: float = 0.55,
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: confidences[index], reverse=True)
    keep: list[int] = []
    for index in order:
        label = canonical_label(names.get(int(classes[index]), f"class_{classes[index]}"))
        should_keep = True
        for kept in keep:
            kept_label = canonical_label(names.get(int(classes[kept]), f"class_{classes[kept]}"))
            iou = bbox_iou(boxes[index], boxes[kept])
            same_class = int(classes[index]) == int(classes[kept])
            same_head_zone = {label, kept_label} <= {"helmet", "head"}
            if (same_class and iou >= iou_threshold) or (same_head_zone and iou >= 0.62):
                should_keep = False
                break
        if should_keep:
            keep.append(index)
    return sorted(keep)

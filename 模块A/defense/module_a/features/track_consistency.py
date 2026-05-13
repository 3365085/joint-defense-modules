from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import ROI


@dataclass(slots=True)
class _Track:
    track_id: int
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    age: int = 1
    missing: int = 0


class TrackConsistencyAnalyzer:
    """Lightweight ROI track memory for detector temporal consistency."""

    def __init__(
        self,
        labels: tuple[str, ...] = ("person", "helmet", "head"),
        iou_threshold: float = 0.12,
        center_distance_ratio: float = 0.55,
        high_confidence: float = 0.45,
        confidence_drop_trigger: float = 0.25,
        max_missing: int = 4,
        score_normalizer: float = 2.0,
        max_candidates_per_label: int = 4,
        max_tracks_per_label: int = 8,
    ):
        self.labels = set(labels)
        self.iou_threshold = float(iou_threshold)
        self.center_distance_ratio = float(center_distance_ratio)
        self.high_confidence = float(high_confidence)
        self.confidence_drop_trigger = float(confidence_drop_trigger)
        self.max_missing = max(1, int(max_missing))
        self.score_normalizer = max(1.0, float(score_normalizer))
        self.max_candidates_per_label = max(1, int(max_candidates_per_label))
        self.max_tracks_per_label = max(1, int(max_tracks_per_label))
        self.tracks: list[_Track] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def compute(self, rois: list[ROI]) -> dict[str, Any]:
        candidates = self._cap_rois_by_label(
            [
                roi
                for roi in rois
                if roi.label in self.labels
                and roi.confidence is not None
                and self._area(roi.bbox) >= 256
            ]
        )
        matches: dict[int, int] = {}
        used_rois: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            for roi_idx, roi in enumerate(candidates):
                if roi_idx in used_rois or roi.label != track.label:
                    continue
                score = self._match_score(track.bbox, roi.bbox)
                if score > 0.0:
                    scored_pairs.append((score, track_idx, roi_idx))
        scored_pairs.sort(reverse=True, key=lambda item: item[0])

        for _score, track_idx, roi_idx in scored_pairs:
            if track_idx in matches or roi_idx in used_rois:
                continue
            matches[track_idx] = roi_idx
            used_rois.add(roi_idx)

        missing_count = 0
        confidence_drop_count = 0
        max_confidence_drop = 0.0
        matched_count = 0
        updated_tracks: list[_Track] = []

        for track_idx, track in enumerate(self.tracks):
            roi_idx = matches.get(track_idx)
            if roi_idx is None:
                if track.confidence >= self.high_confidence and track.missing == 0:
                    missing_count += 1
                track.missing += 1
                if track.missing <= self.max_missing:
                    updated_tracks.append(track)
                continue

            roi = candidates[roi_idx]
            confidence = float(roi.confidence or 0.0)
            confidence_drop = max(0.0, track.confidence - confidence)
            if confidence_drop >= self.confidence_drop_trigger:
                confidence_drop_count += 1
                max_confidence_drop = max(max_confidence_drop, confidence_drop)
            updated_tracks.append(
                _Track(
                    track_id=track.track_id,
                    label=track.label,
                    bbox=roi.bbox,
                    confidence=confidence,
                    age=track.age + 1,
                    missing=0,
                )
            )
            matched_count += 1

        for roi_idx, roi in enumerate(candidates):
            if roi_idx in used_rois:
                continue
            updated_tracks.append(
                _Track(
                    track_id=self.next_id,
                    label=str(roi.label),
                    bbox=roi.bbox,
                    confidence=float(roi.confidence or 0.0),
                )
            )
            self.next_id += 1

        self.tracks = self._cap_tracks_by_label(updated_tracks)
        track_drop_score = min(1.0, missing_count / self.score_normalizer)
        confidence_drop_score = min(
            1.0, max_confidence_drop / max(self.confidence_drop_trigger, 1e-6)
        )
        score = max(track_drop_score, confidence_drop_score)

        return {
            "track_score": score,
            "track_drop_score": track_drop_score,
            "confidence_drop_score": confidence_drop_score,
            "missing_track_count": missing_count,
            "confidence_drop_count": confidence_drop_count,
            "matched_track_count": matched_count,
            "active_track_count": len(self.tracks),
            "candidate_roi_count": len(candidates),
            "backend": "roi_track_consistency",
        }

    def _cap_rois_by_label(self, rois: list[ROI]) -> list[ROI]:
        capped: list[ROI] = []
        for label in self.labels:
            group = [roi for roi in rois if roi.label == label]
            group.sort(key=lambda roi: float(roi.confidence or 0.0), reverse=True)
            capped.extend(group[: self.max_candidates_per_label])
        return capped

    def _cap_tracks_by_label(self, tracks: list[_Track]) -> list[_Track]:
        capped: list[_Track] = []
        for label in self.labels:
            group = [track for track in tracks if track.label == label]
            group.sort(key=lambda track: (track.missing, -track.confidence, -track.age))
            capped.extend(group[: self.max_tracks_per_label])
        return capped

    def _match_score(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        iou = self._iou(a, b)
        if iou >= self.iou_threshold:
            return iou + 1.0
        center_ratio = self._center_distance_ratio(a, b)
        if center_ratio <= self.center_distance_ratio:
            return 1.0 - center_ratio
        return 0.0

    @staticmethod
    def _area(box: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _iou(cls, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = cls._area((ix1, iy1, ix2, iy2))
        if inter <= 0:
            return 0.0
        union = cls._area(a) + cls._area(b) - inter
        return inter / max(1.0, float(union))

    @staticmethod
    def _center_distance_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5
        dx = acx - bcx
        dy = acy - bcy
        distance = (dx * dx + dy * dy) ** 0.5
        scale = max(1.0, ((ax2 - ax1) * (ay2 - ay1)) ** 0.5)
        return distance / scale

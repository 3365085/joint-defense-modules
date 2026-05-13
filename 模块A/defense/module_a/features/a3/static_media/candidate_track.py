"""CandidateTrack data structure and track manager for A3+ flat media spoof detection.

This module implements the L1 tracking layer of the A3+ cascade: maintaining
a list of active rectangular edge candidate tracks across frames, performing
IoU-based matching, and managing track lifecycle (creation, hit/miss update,
removal of dead tracks).

Performance target: < 1ms for L1 tracking association.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MediaCandidateTrack:
    """A3+ L1 candidate-level track for flat media spoof detection.

    Each track represents a rectangular edge candidate that persists
    across multiple frames. The track_score property measures temporal
    stability: high track_score means the candidate has been consistently
    detected across frames.
    """

    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) original resolution
    hit_count: int = 1
    miss_count: int = 0
    age: int = 1
    # L1 scores
    edge_score: float = 0.0
    new_edge_score: float = 0.0
    yolo_context_score: float = 0.0
    target_iou: float = 0.0
    target_proximity_score: float = 0.0
    target_area_ratio: float = 0.0
    bg_suppressed: bool = False
    # L2 scores (cached from last L2 run)
    plane_score: float = 0.0
    warp_residual: float = 0.0
    flow_gap_score: float = 0.0
    # Classification
    media_type: str = "unknown"

    @property
    def track_score(self) -> float:
        """Temporal stability score in [0, 1].

        Ratio of frames where this candidate was detected (hit) vs total
        frames since creation (hit + miss).
        """
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


class CandidateTrackManager:
    """Manages the lifecycle of MediaCandidateTrack instances.

    Responsibilities:
    - Maintaining a list of active tracks
    - IoU-based matching of new candidates to existing tracks (IoU >= threshold)
    - Incrementing hit_count on match, miss_count on no-match
    - Removing dead tracks (miss_count > max_miss)
    - Creating new tracks for unmatched candidates
    - Auto-incrementing track_id
    - reset() to clear all state
    """

    def __init__(self, iou_threshold: float = 0.3, max_miss: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_miss = max_miss
        self._tracks: list[MediaCandidateTrack] = []
        self._next_id: int = 0

    @property
    def tracks(self) -> list[MediaCandidateTrack]:
        """Return the current list of active tracks."""
        return self._tracks

    def update(self, candidates: list[dict]) -> None:
        """Match candidates to tracks, update hit/miss, create/remove tracks.

        Args:
            candidates: List of candidate dicts from extract_edge_candidates().
                Each dict must have a "bbox" key with (x1, y1, x2, y2) tuple.
                Candidates with "bg_suppressed" == True are skipped.
        """
        matched_track_ids: set[int] = set()
        matched_cand_indices: set[int] = set()

        # Record existing track IDs before creating new ones, so we only
        # increment miss_count for tracks that existed prior to this update.
        existing_track_ids = {t.track_id for t in self._tracks}

        # Greedy IoU matching: for each non-suppressed candidate, find the
        # best matching track (highest IoU above threshold).
        for ci, cand in enumerate(candidates):
            if cand.get("bg_suppressed", False):
                continue

            best_iou = self.iou_threshold
            best_track_idx = -1

            for ti, track in enumerate(self._tracks):
                if track.track_id in matched_track_ids:
                    continue
                iou_val = _iou(cand["bbox"], track.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track_idx = ti

            if best_track_idx >= 0:
                # Update existing track
                track = self._tracks[best_track_idx]
                track.hit_count += 1
                track.bbox = cand["bbox"]
                track.age += 1
                # Update L1 scores from candidate if available
                if "edge_score" in cand:
                    track.edge_score = cand["edge_score"]
                matched_track_ids.add(track.track_id)
                matched_cand_indices.add(ci)
            else:
                # Create new track for unmatched candidate
                new_track = MediaCandidateTrack(
                    track_id=self._next_id,
                    bbox=cand["bbox"],
                    edge_score=cand.get("edge_score", 0.0),
                )
                self._tracks.append(new_track)
                self._next_id += 1
                matched_cand_indices.add(ci)

        # Increment miss_count for pre-existing tracks that were not matched
        for track in self._tracks:
            if track.track_id in existing_track_ids and track.track_id not in matched_track_ids:
                track.miss_count += 1
                track.age += 1

        # Remove dead tracks (miss_count > max_miss)
        self._tracks = [t for t in self._tracks if t.miss_count <= self.max_miss]

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 0

"""A3+ active-path feature builder for flat media spoof detection.

This module contains the A3+ candidate pipeline methods extracted from
detector.py. These methods implement the L0→L1→L2→L3 cascade:

- L0: Edge candidate extraction with background filtering
- L1: Candidate track update
- L2: Homography verification + three-zone motion
- L3: p_media decision scoring

The FeatureBuilder operates on the detector's state (candidate track manager,
background edge EMA, L2 cache) via a reference to the detector instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .candidate_extraction import extract_edge_candidates
from .candidate_extraction_torch import extract_edge_candidates_torch
from .candidate_track import MediaCandidateTrack
from .homography_verifier import compute_homography_verification

if TYPE_CHECKING:
    from ....types import ROI
    from .detector import GPUStaticMediaSpoofDetector


def _tensor_to_np_uint8(t: torch.Tensor) -> np.ndarray:
    """Move a (1,1,H,W) GPU tensor to a (H,W) uint8 numpy array once."""
    arr = t[0, 0].cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


class FeatureBuilder:
    """A3+ active-path feature builder.

    Encapsulates the candidate pipeline methods (L0–L3) that were previously
    inline in GPUStaticMediaSpoofDetector. Operates on the detector's mutable
    state via a stored reference.
    """

    def __init__(self, detector: GPUStaticMediaSpoofDetector) -> None:
        self._det = detector

    # ------------------------------------------------------------------
    # L0: Edge candidate extraction with background filtering
    # ------------------------------------------------------------------

    def extract_and_filter_candidates(self, curr_gray: torch.Tensor) -> list[dict]:
        """L0+L1: Extract edge candidates and apply background edge suppression.

        Backend dispatch (P1-A-edge 2026-05-13):

          * ``legacy`` — OpenCV-based extractor (Canny + findContours +
            approxPolyDP + boundingRect). Fast on CPU but NOT NPU-friendly.
          * ``legacy_yolo_only`` — caller skips this method entirely.
          * ``target_anchored_a3plus`` — same OpenCV A3+ extractor as
            ``legacy``, with target-related L3 triggering.
          * ``torch_native`` — pure-torch extractor (Sobel + density grid
            + vectorised rectangle enumeration). Slightly lower detection
            rate but works on any NPU that runs conv/pool/element-wise ops.

        Multi-scale strategy (unchanged from pre-edge optimisation):

          * Scale 1 — single full-frame pass.
          * Scale 2 — when the main pass returns < ``multiscale_trigger_count``
            active candidates AND the fallback is enabled, re-run on
            sub-regions.

        GPU→CPU transfer optimisation (from earlier round) still applies
        to the legacy backend; torch_native stays on-device throughout.
        """
        det = self._det
        bg_ready = det._bg_ready_frames >= det._bg_warmup_frames
        bg_edge = det._bg_edge if bg_ready else None

        # Backend selector.
        backend = getattr(det, "backend", "legacy")
        use_torch = backend == "torch_native"

        if use_torch:
            # Torch-native path: no CPU transfer, works on GPU tensors directly.
            candidates = extract_edge_candidates_torch(
                curr_gray,
                bg_edge=bg_edge,
                bg_ready=bg_ready,
                bg_suppression_ratio=0.85,
            )
            active_count = sum(1 for c in candidates if not c.get("bg_suppressed", False))
            if active_count >= det.multiscale_trigger_count:
                return candidates
            # Multi-scale fallback on torch backend: use the same region strategy
            # but on GPU tensors directly — no .cpu().numpy() round-trip.
            _, _, h, w = curr_gray.shape
            regions: list[tuple[int, int, int, int]] = [
                (h // 4, (3 * h) // 4, w // 4, (3 * w) // 4),
            ]
            if det.multiscale_fallback_enabled:
                hh, hw = h // 2, w // 2
                regions.extend([
                    (0, hh, 0, hw),
                    (0, hh, hw, w),
                    (hh, h, 0, hw),
                    (hh, h, hw, w),
                ])
            for qy1, qy2, qx1, qx2 in regions:
                crop = curr_gray[:, :, qy1:qy2, qx1:qx2]
                bg_crop = (
                    bg_edge[:, :, qy1:qy2, qx1:qx2] if bg_edge is not None else None
                )
                crop_candidates = extract_edge_candidates_torch(
                    crop,
                    bg_edge=bg_crop,
                    bg_ready=bg_ready,
                    bg_suppression_ratio=0.85,
                )
                for cand in crop_candidates:
                    ox1, oy1, ox2, oy2 = cand["bbox"]
                    cand["bbox"] = (qx1 + ox1, qy1 + oy1, qx1 + ox2, qy1 + oy2)
                    cand["edge_score"] = float(cand["edge_score"]) * 0.85
                for zcand in crop_candidates:
                    if not any(det._iou(zcand["bbox"], c["bbox"]) > 0.3 for c in candidates):
                        candidates.append(zcand)
            return candidates

        # Legacy (cv2-based) path. Single batched GPU→CPU move for the
        # full frame; we reuse this ndarray view for all scale-2 crops.
        full_np = _tensor_to_np_uint8(curr_gray)
        full_bg_np: np.ndarray | None = (
            _tensor_to_np_uint8(bg_edge) if bg_edge is not None else None
        )

        candidates = extract_edge_candidates(
            full_np,
            bg_edge=full_bg_np,
            bg_ready=bg_ready,
            bg_suppression_ratio=0.85,
        )

        active_count = sum(1 for c in candidates if not c.get("bg_suppressed", False))
        if active_count >= det.multiscale_trigger_count:
            return candidates

        h, w = full_np.shape
        regions: list[tuple[int, int, int, int]] = [
            (h // 4, (3 * h) // 4, w // 4, (3 * w) // 4),
        ]
        if det.multiscale_fallback_enabled:
            hh, hw = h // 2, w // 2
            regions.extend([
                (0, hh, 0, hw),
                (0, hh, hw, w),
                (hh, h, 0, hw),
                (hh, h, hw, w),
            ])

        for qy1, qy2, qx1, qx2 in regions:
            crop_np = full_np[qy1:qy2, qx1:qx2]
            bg_crop_np = (
                full_bg_np[qy1:qy2, qx1:qx2] if full_bg_np is not None else None
            )
            crop_candidates = extract_edge_candidates(
                crop_np,
                bg_edge=bg_crop_np,
                bg_ready=bg_ready,
                bg_suppression_ratio=0.85,
            )
            for cand in crop_candidates:
                ox1, oy1, ox2, oy2 = cand["bbox"]
                cand["bbox"] = (qx1 + ox1, qy1 + oy1, qx1 + ox2, qy1 + oy2)
                cand["edge_score"] = float(cand["edge_score"]) * 0.85
            for zcand in crop_candidates:
                if not any(det._iou(zcand["bbox"], c["bbox"]) > 0.3 for c in candidates):
                    candidates.append(zcand)

        return candidates

    # ------------------------------------------------------------------
    # L1: Candidate track update
    # ------------------------------------------------------------------

    def update_candidate_tracks(self, candidates: list[dict]) -> None:
        """L1: Update candidate tracks with new frame's candidates.

        Delegates to CandidateTrackManager which handles IoU matching,
        hit/miss counting, and dead track removal.
        """
        self._det._candidate_track_mgr.update(candidates)

    # ------------------------------------------------------------------
    # L2: YOLO context auxiliary signal
    # ------------------------------------------------------------------

    def compute_yolo_context(self, rois: list[ROI] | None) -> None:
        """Compute YOLO context score for each active candidate track.

        For each track, compute the maximum IoU overlap with any YOLO ROI
        (person/helmet/head). The score is scaled: IoU 0.1 → 0.0, IoU 0.5+ → 1.0.

        Key principle: YOLO present → bonus score; YOLO absent → no penalty (stays 0.0).
        """
        det = self._det
        if not rois:
            # No YOLO detections — clear target relation scores.
            for track in det._candidate_track_mgr.tracks:
                track.yolo_context_score = 0.0
                track.target_iou = 0.0
                track.target_proximity_score = 0.0
                track.target_area_ratio = 0.0
            return

        # Collect YOLO bboxes from relevant labels
        yolo_bboxes: list[tuple[int, int, int, int]] = []
        person_bboxes: list[tuple[int, int, int, int]] = []
        for roi in rois:
            if roi.label in det.target_labels or roi.label in det.screen_labels:
                clipped = roi.clipped(9999, 9999, min_size=1)  # just validate bbox
                if clipped is not None:
                    yolo_bboxes.append(clipped.bbox)
                    if roi.label == "person":
                        person_bboxes.append(clipped.bbox)

        if not yolo_bboxes:
            for track in det._candidate_track_mgr.tracks:
                track.yolo_context_score = 0.0
                track.target_iou = 0.0
                track.target_proximity_score = 0.0
                track.target_area_ratio = 0.0
            return

        for track in det._candidate_track_mgr.tracks:
            max_iou = 0.0
            max_proximity = 0.0
            best_area_ratio = 0.0
            target_bboxes = person_bboxes or yolo_bboxes
            for yolo_bbox in yolo_bboxes:
                iou_val = det._iou(track.bbox, yolo_bbox)
                max_iou = max(max_iou, iou_val)
            for target_bbox in target_bboxes:
                proximity = self._bbox_proximity_score(track.bbox, target_bbox)
                if proximity > max_proximity:
                    max_proximity = proximity
                    best_area_ratio = self._bbox_area_ratio(track.bbox, target_bbox)
            # Scale: IoU 0.1 → score 0.0, IoU 0.5+ → score 1.0
            track.yolo_context_score = min(1.0, max(0.0, (max_iou - 0.1) / 0.4))
            track.target_iou = float(max_iou)
            track.target_proximity_score = float(max_proximity)
            track.target_area_ratio = float(best_area_ratio)

    @staticmethod
    def _bbox_area_ratio(
        candidate: tuple[int, int, int, int],
        target: tuple[int, int, int, int],
    ) -> float:
        cx1, cy1, cx2, cy2 = candidate
        tx1, ty1, tx2, ty2 = target
        candidate_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        target_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
        return 0.0 if target_area <= 0 else float(candidate_area / target_area)

    @staticmethod
    def _bbox_proximity_score(
        candidate: tuple[int, int, int, int],
        target: tuple[int, int, int, int],
    ) -> float:
        cx1, cy1, cx2, cy2 = candidate
        tx1, ty1, tx2, ty2 = target
        if cx2 <= cx1 or cy2 <= cy1 or tx2 <= tx1 or ty2 <= ty1:
            return 0.0
        ccx = (cx1 + cx2) * 0.5
        ccy = (cy1 + cy2) * 0.5
        tcx = (tx1 + tx2) * 0.5
        tcy = (ty1 + ty2) * 0.5
        target_w = max(1.0, float(tx2 - tx1))
        target_h = max(1.0, float(ty2 - ty1))
        norm_dx = abs(ccx - tcx) / target_w
        norm_dy = abs(ccy - tcy) / target_h
        center_score = max(0.0, 1.0 - ((norm_dx * norm_dx + norm_dy * norm_dy) ** 0.5))
        expanded = (
            int(tx1 - 0.35 * target_w),
            int(ty1 - 0.35 * target_h),
            int(tx2 + 0.35 * target_w),
            int(ty2 + 0.35 * target_h),
        )
        expanded_iou = _bbox_iou(candidate, expanded)
        return float(max(center_score, min(1.0, expanded_iou * 4.0)))

    # ------------------------------------------------------------------
    # L2: Homography verification
    # ------------------------------------------------------------------

    def run_l2_homography(self, curr_gray: torch.Tensor, bg_info: dict) -> None:
        """L2: Run Homography verification on eligible tracks.

        Only runs every self._det._l2_interval frames. Only processes tracks with:
        - track_score >= 0.6
        - new_edge_score >= 0.20 OR yolo_context_score > 0

        Results are cached in self._det._last_l2_results for non-L2 frames.
        """
        det = self._det
        det._l2_frame_count += 1

        # Cache current frame as numpy for next L2 run
        curr_np = curr_gray[0, 0].cpu().numpy()
        if curr_np.dtype != np.uint8:
            curr_np = np.clip(curr_np, 0, 255).astype(np.uint8)

        # Only run L2 on interval frames AND when we have a previous frame
        should_run = (det._l2_frame_count % det._l2_interval == 0) and (
            det._prev_gray_np is not None
        )

        if not should_run:
            det._prev_gray_np = curr_np
            # Apply cached L2 results to tracks
            self.apply_cached_l2_results()
            return

        # Select eligible tracks: track_score >= 0.5 AND (per-track edge_score >= 0.15 OR yolo_context > 0)
        # AND hit_count >= 3 (must be seen in at least 3 separate L0 extraction frames).
        eligible = [
            t
            for t in det._candidate_track_mgr.tracks
            if t.track_score >= 0.5
            and t.hit_count >= 3
            and (t.edge_score >= 0.15 or t.yolo_context_score > 0)
        ]

        # Sort by track_score descending, take Top-3
        eligible.sort(key=lambda t: t.track_score, reverse=True)
        top_tracks = eligible[:3]

        for track in top_tracks:
            x1, y1, x2, y2 = track.bbox
            # Clamp to image bounds
            h, w = curr_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue

            prev_roi = det._prev_gray_np[y1:y2, x1:x2]
            curr_roi = curr_np[y1:y2, x1:x2]

            result = compute_homography_verification(prev_roi, curr_roi)

            # Update track with L2 results
            track.plane_score = result["plane_score"]
            track.warp_residual = result["warp_residual"]

            # Three-zone motion (v1: raw frame diff, not Homography-compensated)
            if det._prev_gray_np is not None:
                zone_result = self.compute_three_zone_motion(track, curr_np, det._prev_gray_np)
                track.flow_gap_score = zone_result["flow_gap_score"]
                result["flow_gap_score"] = zone_result["flow_gap_score"]

            # Classify media type based on warp_residual threshold
            # Low raw_residual = static image (content doesn't change after plane compensation)
            # High raw_residual = screen replay (content changes independently of plane motion)
            if result["plane_score"] >= 0.45:  # Only classify if plane verification succeeded
                if result.get("raw_residual", 0.0) < 0.015:
                    track.media_type = "static_image"
                else:
                    track.media_type = "screen_replay"
            else:
                track.media_type = "unknown"

            # Cache for non-L2 frames
            det._last_l2_results[track.track_id] = result

        det._prev_gray_np = curr_np

    def run_l2_torch_native(self, curr_gray: torch.Tensor, prev_gray: torch.Tensor | None) -> None:
        """Torch-native L2 surrogate — no cv2, no CPU transfer.

        Semantics: for each eligible track, score "how planar / rigid"
        the candidate region looks by measuring:

        * **Motion uniformity inside the ROI** — a flat image / screen
          held still by a person gives spatially uniform frame-diff;
          a real 3D scene gives a structured diff.
        * **Flow gap (inside vs outside)** — same intent as the existing
          three-zone motion score, but computed directly from frame-diff
          ratio with no Homography.

        These replace ``plane_score`` / ``warp_residual`` / ``flow_gap_score``
        with quantities that correlate strongly with the cv2 versions but
        run entirely on-GPU (or NPU).

        Called every frame on ``torch_native`` backend instead of the
        interval-gated cv2 L2 pass.
        """
        det = self._det
        if prev_gray is None:
            return
        eligible = [
            t
            for t in det._candidate_track_mgr.tracks
            if t.track_score >= 0.3  # Relaxed from 0.5 because torch surrogate
                                      # runs every frame; no interval caching
                                      # means the track score is itself coarser.
            and t.hit_count >= 2
        ]
        if not eligible:
            return
        # Sort by edge_score descending, top-3.
        eligible.sort(key=lambda t: t.edge_score, reverse=True)
        top_tracks = eligible[:3]

        _, _, H, W = curr_gray.shape
        # Frame diff (single kernel launch).
        diff = torch.abs(curr_gray - prev_gray) / 255.0  # (1, 1, H, W)

        for track in top_tracks:
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            roi_diff = diff[:, :, y1:y2, x1:x2]  # (1, 1, rh, rw)
            # Planarity: inverse of the coefficient of variation of ROI diff.
            # A planar / uniform surface → low σ/μ → high planarity.
            mu = roi_diff.mean()
            sigma = roi_diff.std(unbiased=False)
            cv_ratio = (sigma / (mu + 1e-4)).clamp(0.0, 3.0)
            planarity = (1.0 - cv_ratio / 3.0).clamp(0.0, 1.0)
            # warp_residual surrogate — the raw mean ROI motion (0..1).
            residual = mu.clamp(0.0, 1.0)
            # Flow gap (inside vs surrounding strip).
            # Use a 20-px border.
            ex1, ey1 = max(0, x1 - 20), max(0, y1 - 20)
            ex2, ey2 = min(W, x2 + 20), min(H, y2 + 20)
            outer_diff = diff[:, :, ey1:ey2, ex1:ex2]
            # Mask out inner region to compute ring mean.
            ring_sum = outer_diff.sum() - roi_diff.sum()
            ring_area = (ex2 - ex1) * (ey2 - ey1) - (x2 - x1) * (y2 - y1)
            ring_mean = ring_sum / max(1, ring_area)
            flow_gap = torch.abs(mu - ring_mean) / (ring_mean + 1e-4)
            flow_gap = flow_gap.clamp(0.0, 1.0)
            # Single CPU sync for all three scalars at once.
            vals = torch.stack([planarity, residual, flow_gap]).cpu().numpy()
            track.plane_score = float(vals[0])
            track.warp_residual = float(vals[1])
            track.flow_gap_score = float(vals[2])
            # Classify — mirror the cv2 pathway but use planarity as plane_score.
            if track.plane_score >= 0.45:
                if track.warp_residual < 0.015:
                    track.media_type = "static_image"
                else:
                    track.media_type = "screen_replay"
            else:
                track.media_type = "unknown"

    def apply_cached_l2_results(self) -> None:
        """Apply cached L2 results to current tracks (for non-L2 frames)."""
        det = self._det
        for track in det._candidate_track_mgr.tracks:
            if track.track_id in det._last_l2_results:
                cached = det._last_l2_results[track.track_id]
                track.plane_score = cached["plane_score"]
                track.warp_residual = cached["warp_residual"]
                track.flow_gap_score = cached.get("flow_gap_score", 0.0)
                # Re-apply media_type classification from cached raw_residual
                if cached["plane_score"] >= 0.45:
                    if cached.get("raw_residual", 0.0) < 0.015:
                        track.media_type = "static_image"
                    else:
                        track.media_type = "screen_replay"
                else:
                    track.media_type = "unknown"

    # ------------------------------------------------------------------
    # L2: Three-zone motion analysis
    # ------------------------------------------------------------------

    def compute_three_zone_motion(
        self,
        track: MediaCandidateTrack,
        curr_np: np.ndarray,
        prev_np: np.ndarray,
    ) -> dict[str, float]:
        """Compute inside/border/outside motion after frame differencing.

        Uses the raw frame difference (not Homography-compensated for v1)
        to measure motion in three zones of the candidate region.
        """
        x1, y1, x2, y2 = track.bbox
        h, w = curr_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 16 or y2 - y1 < 16:
            return {
                "inside_motion": 0.0,
                "border_motion": 0.0,
                "outside_motion": 0.0,
                "flow_gap_score": 0.0,
            }

        # ROI frame difference
        curr_roi = curr_np[y1:y2, x1:x2].astype(np.float32)
        prev_roi = prev_np[y1:y2, x1:x2].astype(np.float32)
        diff = np.abs(curr_roi - prev_roi) / 255.0

        rh, rw = diff.shape
        # Inside: center 60%
        margin_x = int(rw * 0.2)
        margin_y = int(rh * 0.2)
        inside = diff[margin_y : rh - margin_y, margin_x : rw - margin_x]
        inside_motion = float(inside.mean()) if inside.size > 0 else 0.0

        # Border: edge 20% ring
        border_mask = np.ones_like(diff, dtype=bool)
        border_mask[margin_y : rh - margin_y, margin_x : rw - margin_x] = False
        border = diff[border_mask]
        border_motion = float(border.mean()) if border.size > 0 else 0.0

        # Outside: expanded region around the candidate
        expand = 20  # pixels
        ox1 = max(0, x1 - expand)
        oy1 = max(0, y1 - expand)
        ox2 = min(w, x2 + expand)
        oy2 = min(h, y2 + expand)
        outer_curr = curr_np[oy1:oy2, ox1:ox2].astype(np.float32)
        outer_prev = prev_np[oy1:oy2, ox1:ox2].astype(np.float32)
        outer_diff = np.abs(outer_curr - outer_prev) / 255.0
        # Mask out the inner candidate region
        inner_y1 = y1 - oy1
        inner_x1 = x1 - ox1
        inner_y2 = y2 - oy1
        inner_x2 = x2 - ox1
        outer_mask = np.ones_like(outer_diff, dtype=bool)
        outer_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
        outside_pixels = outer_diff[outer_mask]
        outside_motion = float(outside_pixels.mean()) if outside_pixels.size > 0 else 0.0

        # flow_gap_score: how much more motion is inside vs outside
        flow_gap = (inside_motion - outside_motion) / max(outside_motion, 1e-4)
        flow_gap_score = min(1.0, max(0.0, flow_gap))

        return {
            "inside_motion": inside_motion,
            "border_motion": border_motion,
            "outside_motion": outside_motion,
            "flow_gap_score": flow_gap_score,
        }

    # ------------------------------------------------------------------
    # L3: p_media decision scoring
    # ------------------------------------------------------------------

    def compute_p_media_decision(self) -> dict[str, Any]:
        """L3: Compute final p_media from all sub-scores.

        Computes three sub-scores (static_image, screen_replay, embedded_video),
        takes the max, and determines the media type.
        """
        det = self._det

        # legacy_yolo_only: A3+ cascade is disabled, never trigger p_media.
        if det.backend == "legacy_yolo_only":
            return {
                "p_media": 0.0,
                "p_media_triggered": False,
                "p_media_type": "normal",
                "p_media_bbox": None,
            }

        tracks = det._candidate_track_mgr.tracks
        if not tracks:
            return {
                "p_media": 0.0,
                "p_media_triggered": False,
                "p_media_type": "normal",
                "p_media_bbox": None,
            }

        # Find the best track (highest combined evidence)
        best_track = max(
            tracks,
            key=lambda t: (
                t.track_score * 0.3
                + t.plane_score * 0.3
                + t.edge_score * 0.2
                + t.flow_gap_score * 0.2
                + t.target_proximity_score * 0.2
            ),
        )

        t = best_track
        yolo_bonus = 0.05 * t.yolo_context_score

        # Static image score (high plane + low residual + low flow_gap)
        # A static image has no internal motion, so flow_gap should be low
        low_residual = 1.0 - t.warp_residual
        low_flow_gap = 1.0 - t.flow_gap_score
        static_image_score = min(
            1.0,
            0.20 * t.edge_score
            + 0.25 * t.plane_score
            + 0.25 * t.track_score
            + 0.15 * low_residual
            + 0.15 * low_flow_gap
            + yolo_bonus,
        )

        # Screen replay score (high plane + high flow_gap indicating internal motion)
        # A phone playing video has a stable plane but motion inside differs from outside
        screen_replay_score = min(
            1.0,
            0.15 * t.edge_score
            + 0.25 * t.plane_score
            + 0.20 * t.track_score
            + 0.25 * t.flow_gap_score
            + 0.15 * t.warp_residual
            + yolo_bonus,
        )

        # Embedded video score (high flow gap + high residual, lower plane requirement)
        embedded_video_score = min(
            1.0,
            0.15 * t.edge_score
            + 0.15 * t.track_score
            + 0.10 * t.plane_score
            + 0.30 * t.flow_gap_score
            + 0.30 * t.warp_residual,
        )

        scores = {
            "static_image": static_image_score,
            "screen_replay": screen_replay_score,
            "embedded_video": embedded_video_score,
        }
        best_type_key = max(scores, key=scores.get)
        p_media = scores[best_type_key]

        type_map = {
            "static_image": "static_image_spoofing",
            "screen_replay": "screen_replay_video",
            "embedded_video": "screen_replay_video",
        }

        target_area_reasonable = 0.08 <= t.target_area_ratio <= 2.50
        target_related = (
            (t.target_iou >= 0.12 and target_area_reasonable)
            or (
                t.target_proximity_score >= 0.65
                and 0.15 <= t.target_area_ratio <= 1.25
            )
        )
        x1, y1, x2, y2 = t.bbox
        bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
        frame_area = 640.0 * 640.0
        giant_background_candidate = bool(
            bbox_area >= frame_area * 0.20
            and (not target_related or t.target_area_ratio >= 1.80)
        )
        if giant_background_candidate:
            target_related = False
            p_media = min(p_media, 0.55)
        strong_media_evidence = (
            p_media >= 0.62
            and t.track_score >= 0.85
            and t.plane_score >= 0.35
            and not t.bg_suppressed
            and not giant_background_candidate
        )
        background_static_suppressed = bool(
            (strong_media_evidence and not target_related) or giant_background_candidate
        )

        # Trigger threshold: require strong evidence from multiple signals.
        # track_score >= 0.7 ensures the candidate has been consistently tracked,
        # p_media >= 0.65 requires strong combined evidence from edge/plane/flow,
        # AND we require either plane_score > 0 OR flow_gap_score > 0.3 as a
        # "motion evidence" gate. This prevents static architectural edges
        # (building corners, shelves) from triggering on clean scenes where
        # there is no relative motion between the candidate and its background.
        # A real screen/paper being held by a person always has some motion
        # (hand shake, walking) that produces non-zero plane_score or flow_gap.
        #
        # 2026-05-13 tightening: require plane_score >= 0.35 (not just 0.15)
        # because outdoor scenes with pedestrians produce flow_gap > 0.3 from
        # normal walking, which was causing FP on real surveillance footage.
        # A genuine held screen/paper has plane_score > 0.4 from the holder's
        # hand motion creating uniform ROI displacement.
        has_motion_evidence = t.plane_score >= 0.35
        triggered = p_media >= 0.70 and t.track_score >= 0.75 and has_motion_evidence
        if det.backend == "target_anchored_a3plus":
            triggered = bool(triggered and target_related)

        return {
            "p_media": float(min(1.0, max(0.0, p_media))),
            "p_media_triggered": bool(triggered),
            "p_media_type": type_map.get(best_type_key, "normal"),
            "p_media_bbox": list(t.bbox),
            "target_related": bool(target_related),
            "strong_media_evidence": bool(strong_media_evidence),
            "background_static_suppressed": bool(background_static_suppressed),
            "giant_background_candidate": bool(giant_background_candidate),
            "target_iou": float(t.target_iou),
            "target_proximity_score": float(t.target_proximity_score),
            "target_area_ratio": float(t.target_area_ratio),
        }

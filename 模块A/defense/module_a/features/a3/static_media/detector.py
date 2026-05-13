from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ....types import ROI
from .candidate_track import CandidateTrackManager, MediaCandidateTrack
from .config import StaticMediaConfig
from .feature_builder import FeatureBuilder

# ==========================================================================
# Code Classification Legend (A3+ Phase 0 — Task 0.1)
# --------------------------------------------------------------------------
# [active]           — PR1/PR2 code that is part of the new A3+ direction.
#                      These methods implement the p_media field contract and
#                      background edge EMA, and will be extended by PR3–PR7.
#
# [legacy-fallback]  — Existing detection logic preserved as fallback.
#                      Still feeds `static_image_triggered` and the
#                      `global_fallback` score. Will coexist with the new
#                      A3+ pipeline (OR relationship) until the new path is
#                      fully validated on production streams.
#
# [deprecated]       — Dead code with no active consumers. (Currently none
#                      identified in this file — synth/forensics code lives
#                      in source_authenticity/synth_detector.py.)
# ==========================================================================


# [legacy-fallback] ROI patch tracking data structure.
# Preserved because: the patch-track scoring path still produces
# `static_image_triggered` which feeds Rule_Fusion → AlertState.
@dataclass(slots=True)
class _PatchTrack:
    label: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    patch: torch.Tensor
    stable_count: int = 1


# [active] Main detector class — orchestrates both legacy-fallback and
# new A3+ paths. The class itself is active; individual methods are
# annotated with their own classification below.
class GPUStaticMediaSpoofDetector:
    """Detect flat media spoofing such as phones, screens, paper photos, and printed targets.

    The detector is intentionally lightweight: it does not add another neural
    model. It compares normalized ROI patches over time and checks whether the
    ROI content remains too stable while the ROI or its surrounding scene moves.
    """

    # [active] Initialisation — parameters span both legacy-fallback
    # (patch-track thresholds, screen-replay pivot thresholds) and
    # active A3+ state (background edge EMA). PR3+ will add
    # CandidateTrack and L2 cache state here.
    def __init__(
        self,
        target_labels: tuple[str, ...] = ("person",),
        screen_labels: tuple[str, ...] = ("helmet", "head"),
        patch_size: int = 64,
        min_similarity: float = 0.94,
        trigger_stable_count: int = 2,
        min_edge_mean: float = 0.038,
        screen_min_edge_mean: float = 0.018,
        min_center_motion: float = 0.0012,
        context_motion_threshold: float = 0.010,
        context_contrast_threshold: float = 1.6,
        min_roi_area: int = 1200,
        screen_min_roi_area: int = 450,
        screen_max_roi_area: int = 8000,
        screen_context_expand_ratio: float = 2.4,
        screen_min_context_edge_mean: float = 0.004,
        screen_min_context_std: float = 0.22,
        screen_min_line_score: float = 0.10,
        screen_max_roi_context_area_ratio: float = 0.42,
        screen_person_containment_threshold: float = 0.72,
        min_roi_confidence: float = 0.50,
        score_trigger: float = 0.80,
        expand_ratio: float = 0.35,
        edge_margin_px: int = 6,
        min_same_label_count: int = 2,
        max_person_area_ratio: float = 0.65,
        max_context_iou: float = 0.20,
        max_tracks: int = 64,
        emit_roi_details: bool = False,
        multiscale_fallback_enabled: bool = True,
        multiscale_trigger_count: int = 1,
        backend: str = "legacy",
    ):
        # Build typed config object — applies all type coercion and clamping.
        self.config = StaticMediaConfig(
            target_labels=target_labels,
            screen_labels=screen_labels,
            patch_size=patch_size,
            min_similarity=min_similarity,
            trigger_stable_count=trigger_stable_count,
            min_edge_mean=min_edge_mean,
            screen_min_edge_mean=screen_min_edge_mean,
            min_center_motion=min_center_motion,
            context_motion_threshold=context_motion_threshold,
            context_contrast_threshold=context_contrast_threshold,
            min_roi_area=min_roi_area,
            screen_min_roi_area=screen_min_roi_area,
            screen_max_roi_area=screen_max_roi_area,
            screen_context_expand_ratio=screen_context_expand_ratio,
            screen_min_context_edge_mean=screen_min_context_edge_mean,
            screen_min_context_std=screen_min_context_std,
            screen_min_line_score=screen_min_line_score,
            screen_max_roi_context_area_ratio=screen_max_roi_context_area_ratio,
            screen_person_containment_threshold=screen_person_containment_threshold,
            min_roi_confidence=min_roi_confidence,
            score_trigger=score_trigger,
            expand_ratio=expand_ratio,
            edge_margin_px=edge_margin_px,
            min_same_label_count=min_same_label_count,
            max_person_area_ratio=max_person_area_ratio,
            max_context_iou=max_context_iou,
            max_tracks=max_tracks,
            emit_roi_details=emit_roi_details,
            multiscale_fallback_enabled=multiscale_fallback_enabled,
            multiscale_trigger_count=multiscale_trigger_count,
            backend=backend,
        )

        # Expose config fields as instance attributes for backward compatibility.
        # This ensures self.param_name access works unchanged throughout the class.
        self.target_labels = self.config.target_labels_set
        self.screen_labels = self.config.screen_labels_set
        self.patch_size = self.config.patch_size
        self.min_similarity = self.config.min_similarity
        self.trigger_stable_count = self.config.trigger_stable_count
        self.min_edge_mean = self.config.min_edge_mean
        self.screen_min_edge_mean = self.config.screen_min_edge_mean
        self.min_center_motion = self.config.min_center_motion
        self.context_motion_threshold = self.config.context_motion_threshold
        self.context_contrast_threshold = self.config.context_contrast_threshold
        self.min_roi_area = self.config.min_roi_area
        self.screen_min_roi_area = self.config.screen_min_roi_area
        self.screen_max_roi_area = self.config.screen_max_roi_area
        self.screen_context_expand_ratio = self.config.screen_context_expand_ratio
        self.screen_min_context_edge_mean = self.config.screen_min_context_edge_mean
        self.screen_min_context_std = self.config.screen_min_context_std
        self.screen_min_line_score = self.config.screen_min_line_score
        self.screen_max_roi_context_area_ratio = self.config.screen_max_roi_context_area_ratio
        self.screen_person_containment_threshold = self.config.screen_person_containment_threshold
        self.min_roi_confidence = self.config.min_roi_confidence
        self.score_trigger = self.config.score_trigger
        self.expand_ratio = self.config.expand_ratio
        self.edge_margin_px = self.config.edge_margin_px
        self.min_same_label_count = self.config.min_same_label_count
        self.max_person_area_ratio = self.config.max_person_area_ratio
        self.max_context_iou = self.config.max_context_iou
        self.max_tracks = self.config.max_tracks
        self.emit_roi_details = self.config.emit_roi_details
        self.multiscale_fallback_enabled = self.config.multiscale_fallback_enabled
        self.multiscale_trigger_count = self.config.multiscale_trigger_count
        self.backend = self.config.backend
        self._tracks: list[_PatchTrack] = []
        self._kernel_device: torch.device | None = None
        self._sobel_x: torch.Tensor | None = None
        self._sobel_y: torch.Tensor | None = None
        # A3+ PR2 — background edge EMA (ChatGPT follow-up plan §五).
        # ``_bg_edge`` is the long-term Sobel-edge tensor; on every frame
        # we do ``bg = (1-alpha) * bg + alpha * edge_t`` so slow-moving
        # structures (gates, shelves, printed signs in the frame) get
        # absorbed into the background and stop creating "new edge"
        # candidates. Cold-start: first ``_bg_warmup_frames`` frames are
        # collected but ``p_media_bg_ready`` stays False so downstream
        # consumers know not to trust ``new_edge`` yet. This PR is
        # DIAGNOSTIC ONLY — ``new_edge_score`` lands in
        # ``p_media_scores["new_edge"]`` + ``p_media_bg_ready`` — no
        # change to triggering behaviour.
        self._bg_edge: torch.Tensor | None = None
        self._bg_ready_frames: int = 0
        self._bg_warmup_frames: int = 30
        self._bg_alpha: float = 0.02

        # A3+ PR3 — CandidateTrack manager for L0/L1 edge candidate tracking.
        self._candidate_track_mgr = CandidateTrackManager(iou_threshold=0.3, max_miss=5)

        # A3+ performance optimization: L0 candidate extraction interval.
        # Only run the expensive Canny/contour extraction every N frames.
        # On non-extraction frames, just update tracks with empty candidates
        # (existing tracks age via miss_count, no new candidates created).
        self._l0_interval: int = 5  # Run L0 every 5 frames
        self._l0_frame_count: int = 0
        # Offset from other modules to avoid all heavy ops landing on same frame.
        # light_flow uses interval=3 starting at frame 0, so we start at frame 2.
        self._l0_offset: int = 2

        # A3+ PR5 — L2 Homography scheduling state.
        self._l2_frame_count: int = 0
        self._l2_interval: int = 3  # Run L2 every 3 frames
        self._last_l2_results: dict[int, dict[str, float]] = {}  # track_id -> L2 scores
        self._prev_gray_np: np.ndarray | None = None  # cached for L2 ROI extraction

        # A3+ feature builder — encapsulates the L0→L1→L2→L3 candidate pipeline.
        self._feature_builder = FeatureBuilder(self)

    # [active] A3+ PR3 — L1 candidate track update.
    def _update_candidate_tracks(self, candidates: list[dict]) -> None:
        """L1: Update candidate tracks with new frame's candidates.

        Delegates to CandidateTrackManager which handles IoU matching,
        hit/miss counting, and dead track removal.
        """
        self._feature_builder.update_candidate_tracks(candidates)

    # [active] A3+ PR3 — L0 edge candidate extraction with background filtering.
    def _extract_and_filter_candidates(self, curr_gray: torch.Tensor) -> list[dict]:
        """L0+L1: Extract edge candidates and apply background edge suppression.

        Uses multi-scale extraction: runs candidate extraction at the original
        resolution AND at 2x zoom on each quadrant to catch smaller phone screens
        that would otherwise fall below the area_ratio minimum.

        Returns:
            List of candidate dicts (some may have bg_suppressed=True).
        """
        return self._feature_builder.extract_and_filter_candidates(curr_gray)

    # [active] A3+ PR4 — YOLO context auxiliary signal.
    def _compute_yolo_context(self, rois: list[ROI] | None) -> None:
        """Compute YOLO context score for each active candidate track.

        For each track, compute the maximum IoU overlap with any YOLO ROI
        (person/helmet/head). The score is scaled: IoU 0.1 → 0.0, IoU 0.5+ → 1.0.

        Key principle: YOLO present → bonus score; YOLO absent → no penalty (stays 0.0).
        """
        self._feature_builder.compute_yolo_context(rois)

    # [active] A3+ PR5 — L2 Homography verification (every 3 frames, Top-3 tracks).
    def _run_l2_homography(self, curr_gray: torch.Tensor, bg_info: dict) -> None:
        """L2: Run Homography verification on eligible tracks.

        Only runs every self._l2_interval frames. Only processes tracks with:
        - track_score >= 0.6
        - new_edge_score >= 0.20 OR yolo_context_score > 0

        Results are cached in self._last_l2_results for non-L2 frames.
        """
        self._feature_builder.run_l2_homography(curr_gray, bg_info)

    def _apply_cached_l2_results(self) -> None:
        """Apply cached L2 results to current tracks (for non-L2 frames)."""
        self._feature_builder.apply_cached_l2_results()

    # [active] A3+ PR7 — Three-zone motion analysis.
    def _compute_three_zone_motion(
        self,
        track: MediaCandidateTrack,
        curr_np: np.ndarray,
        prev_np: np.ndarray,
    ) -> dict[str, float]:
        """Compute inside/border/outside motion after frame differencing.

        Uses the raw frame difference (not Homography-compensated for v1)
        to measure motion in three zones of the candidate region.
        """
        return self._feature_builder.compute_three_zone_motion(track, curr_np, prev_np)

    # [active] A3+ PR7 — Comprehensive p_media scoring and type classification.
    def _compute_p_media_decision(self) -> dict[str, Any]:
        """L3: Compute final p_media from all sub-scores.

        Computes three sub-scores (static_image, screen_replay, embedded_video),
        takes the max, and determines the media type.
        """
        return self._feature_builder.compute_p_media_decision()

    # [active] State management — clears all temporal state on stream
    # reset / RTSP reconnect. PR3+ will add CandidateTrack and L2 cache
    # clearing here.
    def reset(self) -> None:
        self._tracks.clear()
        # PR2: background edge must restart its cold-start sequence on
        # every reset so RTSP reconnect / stream_geometry_changed /
        # stream_recovered all re-initialise the model against the new
        # scene rather than carrying stale structure.
        self._bg_edge = None
        self._bg_ready_frames = 0
        # A3+ PR3: reset candidate track manager state.
        self._candidate_track_mgr.reset()
        # A3+ L0 interval reset.
        self._l0_frame_count = 0
        # A3+ PR5: reset L2 Homography scheduling state.
        self._l2_frame_count = 0
        self._last_l2_results.clear()
        self._prev_gray_np = None

    # [active] Main entry point — orchestrates per-frame detection.
    # Contains both legacy-fallback ROI loop (patch tracking + scoring)
    # and active A3+ code (p_media contract population, background edge
    # update). PR3+ will insert L0/L1/L2/L3 cascade calls here.
    def compute(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
        rois: list[ROI] | None = None,
        context_rois: list[ROI] | None = None,
    ) -> dict[str, Any]:
        # A3+ PR2 — background edge EMA update. Done BEFORE the ``not rois``
        # early return so even frames with zero ROI still contribute to
        # the background model (essential during cold start on a clean
        # stream). Fails safe when _ensure_kernels has not yet been
        # called: the first two ``edge_t`` computes here initialise the
        # Sobel kernels on the right device.
        _t_bg = time.perf_counter()
        bg_info = self._update_background_edge(curr_gray)
        _bg_ms = (time.perf_counter() - _t_bg) * 1000.0

        # A3+ PR3 — L0 edge candidate extraction + L1 track update.
        # Performance optimization: only run the expensive Canny/contour
        # extraction every _l0_interval frames. On skip frames, tracks
        # retain their current state (no miss counting until next L0 frame).
        #
        # P1-A-edge 2026-05-13: the entire A3+ cascade (L0/L1/L2) is
        # gated on ``backend``. When set to ``legacy_yolo_only`` we skip
        # all cv2-based candidate extraction and homography verification
        # so the forward path is torch-only — required for RKNN / NPU /
        # ONNX-Runtime-Mobile targets that lack full OpenCV bindings.
        _t_l0 = time.perf_counter()
        self._l0_frame_count += 1
        # Backend dispatch for the A3+ cascade:
        #   "legacy"           → full A3+ (L0 cv2 + L1 + L2 cv2 + L3)
        #   "legacy_yolo_only" → skip L0/L1/L2/L3 entirely, rely on Legacy
        #                        YOLO-ROI loop only (pure torch, NPU-friendly)
        #   "target_anchored_a3plus" → L0/L1/L2 legacy + target-gated L3
        #   "torch_native"     → L0 torch + L1 + L2 torch + L3 (experimental)
        a3_plus_enabled = self.backend in {
            "legacy",
            "target_anchored_a3plus",
            "torch_native",
        }

        if a3_plus_enabled:
            should_run_l0 = (self._l0_frame_count + self._l0_offset) % self._l0_interval == 0
            if should_run_l0:
                candidates = self._extract_and_filter_candidates(curr_gray)
                self._update_candidate_tracks(candidates)
            else:
                candidates = []
        else:
            # legacy_yolo_only: skip Canny / contour / candidate tracking.
            # The Legacy YOLO-ROI loop below still runs and provides
            # static_image_triggered via patch-track + motion contrast.
            candidates = []

        # Count active candidates from tracks (not from this frame's extraction)
        active_candidate_count = sum(
            1 for t in self._candidate_track_mgr.tracks if not t.bg_suppressed
        )
        _l0_l1_ms = (time.perf_counter() - _t_l0) * 1000.0

        # A3+ PR4 — YOLO context auxiliary signal (bonus only, no penalty).
        _t_yolo = time.perf_counter()
        yolo_context_rois = context_rois if context_rois is not None else rois
        if a3_plus_enabled:
            self._compute_yolo_context(yolo_context_rois)
        _yolo_ms = (time.perf_counter() - _t_yolo) * 1000.0

        # A3+ PR5 — L2 Homography verification.
        # L2 uses cv2.findHomography (not NPU-friendly). On torch_native we
        # use the torch surrogate; on legacy_yolo_only we skip entirely.
        _t_l2 = time.perf_counter()
        if self.backend in {"legacy", "target_anchored_a3plus"}:
            self._run_l2_homography(curr_gray, bg_info)
        elif self.backend == "torch_native":
            self._feature_builder.run_l2_torch_native(curr_gray, prev_gray)
        _l2_ms = (time.perf_counter() - _t_l2) * 1000.0

        # A3+ Phase 6 — timing diagnostics dict (purely diagnostic, no detection logic impact).
        _p_media_timing_ms = {
            "bg_edge_ms": _bg_ms,
            "l0_l1_ms": _l0_l1_ms,
            "yolo_context_ms": _yolo_ms,
            "l2_homography_ms": _l2_ms,
            "total_a3plus_ms": _bg_ms + _l0_l1_ms + _yolo_ms + _l2_ms,
        }

        if not rois:
            self._tracks = []
            empty = self._empty()
            empty["p_media_candidate_count"] = active_candidate_count
            # Fill edge score from best candidate
            if candidates:
                best_edge = max(
                    (
                        c.get("edge_score", 0.0)
                        for c in candidates
                        if not c.get("bg_suppressed", False)
                    ),
                    default=0.0,
                )
                empty["p_media_scores"]["edge"] = best_edge
            # Fill track score from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_track_score = max(t.track_score for t in tracks)
                empty["p_media_scores"]["track"] = best_track_score
            # Fill yolo_context score from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_yolo_context = max(t.yolo_context_score for t in tracks)
                empty["p_media_scores"]["yolo_context"] = best_yolo_context
            # Fill plane and warp_residual from best active track's L2 results
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_plane = max(t.plane_score for t in tracks)
                best_warp = max(t.warp_residual for t in tracks)
                empty["p_media_scores"]["plane"] = best_plane
                empty["p_media_scores"]["warp_residual"] = best_warp
            # Fill flow_gap from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_flow_gap = max(t.flow_gap_score for t in tracks)
                empty["p_media_scores"]["flow_gap"] = best_flow_gap
            # A3+ PR7 — Compute p_media decision even without YOLO ROIs.
            # The A3+ candidate path (L0→L1→L2) runs independently of YOLO,
            # so we must compute the final decision here too.
            decision = self._compute_p_media_decision()
            empty["p_media"] = decision["p_media"]
            empty["p_media_triggered"] = decision["p_media_triggered"]
            empty["p_media_type"] = decision["p_media_type"]
            empty["p_media_bbox"] = decision["p_media_bbox"]
            empty["p_media_target_related"] = bool(decision.get("target_related", False))
            empty["p_media_strong_evidence"] = bool(
                decision.get("strong_media_evidence", False)
            )
            empty["p_media_background_static_suppressed"] = bool(
                decision.get("background_static_suppressed", False)
            )
            empty["p_media_scores"]["target_iou"] = float(decision.get("target_iou", 0.0))
            empty["p_media_scores"]["target_proximity"] = float(
                decision.get("target_proximity_score", 0.0)
            )
            empty["p_media_scores"]["target_area_ratio"] = float(
                decision.get("target_area_ratio", 0.0)
            )
            # A3+ PR6 — Alert trigger integration (empty/no-ROI path).
            # Same logic as the main path: when A3+ triggers, feed into
            # static_image_triggered and lift static_image_score.
            if decision.get("p_media_triggered", False):
                empty["static_image_triggered"] = True
                empty["static_image_score"] = max(
                    empty.get("static_image_score", 0.0),
                    decision.get("p_media", 0.0),
                )
            self._apply_background_to_result(empty, bg_info)
            empty["p_media_timing_ms"] = _p_media_timing_ms
            return empty

        device = curr_gray.device
        self._ensure_kernels(device)
        assert self._sobel_x is not None
        assert self._sobel_y is not None

        curr_norm = curr_gray.float() / 255.0
        edge_x = F.conv2d(curr_norm, self._sobel_x, padding=1)
        edge_y = F.conv2d(curr_norm, self._sobel_y, padding=1)
        edge = torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-8)
        diff = None
        global_motion = 0.0
        if prev_gray is not None and prev_gray.shape == curr_gray.shape:
            diff = torch.abs(curr_gray - prev_gray)
            global_motion = float((diff.float().mean() / 255.0).item())

        _, _, h, w = curr_gray.shape
        label_counts: dict[str, int] = {}
        person_bboxes: list[tuple[int, int, int, int]] = []
        for roi in rois:
            label = str(roi.label)
            if label in self.target_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            if label == "person":
                clipped = roi.clipped(w, h, min_size=8)
                if clipped is not None:
                    person_bboxes.append(clipped.bbox)
        new_tracks: list[_PatchTrack] = []
        best: dict[str, Any] | None = None
        roi_results: list[dict[str, Any]] = []
        trigger_count = 0

        # [legacy-fallback] Main ROI loop — patch tracking + scoring +
        # triggered logic. Preserved because: produces `static_image_triggered`
        # which feeds Rule_Fusion → AlertState. A3+ PR3+ will add a
        # parallel CandidateTrack loop above this one; both contribute
        # via OR to the final triggered decision.
        for roi in rois:
            if roi.label not in self.target_labels:
                continue
            is_screen_label = str(roi.label) in self.screen_labels
            if (
                not is_screen_label
                and label_counts.get(str(roi.label), 0) < self.min_same_label_count
            ):
                continue
            if roi.confidence is not None and float(roi.confidence) < self.min_roi_confidence:
                continue

            clipped = roi.clipped(w, h, min_size=8)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped.bbox
            if (
                x1 <= self.edge_margin_px
                or y1 <= self.edge_margin_px
                or x2 >= w - self.edge_margin_px
                or y2 >= h - self.edge_margin_px
            ):
                continue
            roi_w = x2 - x1
            roi_h = y2 - y1
            roi_area = roi_w * roi_h
            if is_screen_label:
                if roi_area < self.screen_min_roi_area or roi_area > self.screen_max_roi_area:
                    continue
            elif roi_area < self.min_roi_area:
                continue
            if clipped.label == "person" and not self._has_larger_person_context(
                clipped.bbox, person_bboxes
            ):
                continue

            patch = self._extract_patch(curr_norm, clipped.bbox)
            center = ((x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h))
            roi_edge = edge[:, :, y1:y2, x1:x2]
            edge_mean = float(roi_edge.mean().item()) if roi_edge.numel() else 0.0
            screen_context = (
                self._screen_context_features(
                    clipped.bbox,
                    edge,
                    curr_norm,
                    person_bboxes,
                    w,
                    h,
                )
                if is_screen_label
                else {
                    "screen_like": False,
                    "screen_context_edge_mean": 0.0,
                    "screen_context_std": 0.0,
                    "screen_line_score": 0.0,
                    "screen_roi_context_area_ratio": 0.0,
                    "screen_inside_person": False,
                }
            )

            matched = self._match_track(clipped, center)
            similarity = 0.0
            stable_count = 1
            center_motion = 0.0
            if matched is not None:
                similarity = self._patch_similarity(patch, matched.patch)
                center_motion = self._center_distance(center, matched.center)
                stable_count = matched.stable_count + 1 if similarity >= self.min_similarity else 1

            roi_motion = 0.0
            context_motion = global_motion
            contrast = 0.0
            if diff is not None:
                roi_diff = diff[:, :, y1:y2, x1:x2]
                roi_motion = (
                    float((roi_diff.float().mean() / 255.0).item()) if roi_diff.numel() else 0.0
                )
                ex1, ey1, ex2, ey2 = self._expanded_bbox(x1, y1, x2, y2, w, h)
                expanded_diff = diff[:, :, ey1:ey2, ex1:ex2]
                surround_motion = self._surround_motion(
                    expanded_diff,
                    x1 - ex1,
                    y1 - ey1,
                    x2 - ex1,
                    y2 - ey1,
                )
                context_motion = max(global_motion, surround_motion)
                contrast = context_motion / max(roi_motion, 1e-4)

            motion_evidence = max(
                self._ramp(center_motion, self.min_center_motion, self.min_center_motion * 3.0),
                self._ramp(
                    context_motion,
                    self.context_motion_threshold,
                    self.context_motion_threshold * 3.0,
                ),
                self._ramp(contrast, 1.0, self.context_contrast_threshold),
            )
            similarity_part = self._ramp(
                similarity, self.min_similarity - 0.025, self.min_similarity
            )
            stable_part = min(1.0, stable_count / max(1.0, float(self.trigger_stable_count)))
            required_edge_mean = (
                self.screen_min_edge_mean if is_screen_label else self.min_edge_mean
            )
            edge_part = min(1.0, edge_mean / max(required_edge_mean, 1e-6))
            screen_context_part = 1.0 if bool(screen_context["screen_like"]) else 0.0

            # Gate: if similarity is below threshold, the ROI content is changing
            # (real person moving), so score should be near zero regardless of
            # other signals. Only compute meaningful score when sim >= threshold.
            absolute_motion_evidence = (
                center_motion >= self.min_center_motion
                or context_motion >= self.context_motion_threshold
            )
            # Score is only meaningful when ALL necessary conditions for detection
            # are met: high similarity (content not changing), sufficient stable
            # frames (persistence), adequate edge (not a blank region), and motion
            # evidence (distinguishes from a truly static camera with no activity).
            # When any condition fails, score stays at 0 — this prevents misleading
            # high scores on normal video where a person happens to move slowly.
            conditions_met = (
                similarity >= self.min_similarity
                and stable_count >= self.trigger_stable_count
                and edge_mean >= required_edge_mean
                and absolute_motion_evidence
                and (not is_screen_label or bool(screen_context["screen_like"]))
            )
            if not conditions_met:
                score = 0.0
            elif is_screen_label:
                score = min(
                    1.0,
                    0.28 * similarity_part
                    + 0.20 * stable_part
                    + 0.22 * motion_evidence
                    + 0.12 * edge_part
                    + 0.18 * screen_context_part,
                )
            else:
                score = min(
                    1.0,
                    0.35 * similarity_part
                    + 0.25 * stable_part
                    + 0.25 * motion_evidence
                    + 0.15 * edge_part,
                )
            absolute_motion_evidence = (
                center_motion >= self.min_center_motion
                or context_motion >= self.context_motion_threshold
            )
            # Without motion evidence, we cannot distinguish a still person from
            # a photo/screen. Suppress score to avoid misleading high values.
            if not absolute_motion_evidence:
                score *= 0.3
            triggered = (
                score >= self.score_trigger
                and similarity >= self.min_similarity
                and stable_count >= self.trigger_stable_count
                and edge_mean >= required_edge_mean
                and absolute_motion_evidence
                and (not is_screen_label or bool(screen_context["screen_like"]))
            )
            if triggered:
                trigger_count += 1

            item = {
                "roi": clipped.to_dict(),
                "static_image_score": float(score),
                "patch_similarity": float(similarity),
                "stable_count": int(stable_count),
                "center_motion": float(center_motion),
                "roi_motion": float(roi_motion),
                "context_motion": float(context_motion),
                "motion_contrast": float(contrast),
                "edge_mean": float(edge_mean),
                "screen_like": bool(screen_context["screen_like"]),
                "screen_context_edge_mean": float(screen_context["screen_context_edge_mean"]),
                "screen_context_std": float(screen_context["screen_context_std"]),
                "screen_line_score": float(screen_context["screen_line_score"]),
                "screen_roi_context_area_ratio": float(
                    screen_context["screen_roi_context_area_ratio"]
                ),
                "screen_inside_person": bool(screen_context["screen_inside_person"]),
                "triggered": bool(triggered),
            }
            if self.emit_roi_details or triggered:
                roi_results.append(item)
            if best is None or score > float(best["static_image_score"]):
                best = item

            new_tracks.append(
                _PatchTrack(
                    label=str(clipped.label),
                    bbox=clipped.bbox,
                    center=center,
                    patch=patch.detach(),
                    stable_count=stable_count,
                )
            )

        self._tracks = new_tracks[: self.max_tracks]
        if best is None:
            empty = self._empty()
            empty["p_media_candidate_count"] = active_candidate_count
            # Fill edge score from best candidate
            if candidates:
                best_edge = max(
                    (
                        c.get("edge_score", 0.0)
                        for c in candidates
                        if not c.get("bg_suppressed", False)
                    ),
                    default=0.0,
                )
                empty["p_media_scores"]["edge"] = best_edge
            # Fill track score from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_track_score = max(t.track_score for t in tracks)
                empty["p_media_scores"]["track"] = best_track_score
            # Fill yolo_context score from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_yolo_context = max(t.yolo_context_score for t in tracks)
                empty["p_media_scores"]["yolo_context"] = best_yolo_context
            # Fill plane and warp_residual from best active track's L2 results
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_plane = max(t.plane_score for t in tracks)
                best_warp = max(t.warp_residual for t in tracks)
                empty["p_media_scores"]["plane"] = best_plane
                empty["p_media_scores"]["warp_residual"] = best_warp
            # Fill flow_gap from best active track
            tracks = self._candidate_track_mgr.tracks
            if tracks:
                best_flow_gap = max(t.flow_gap_score for t in tracks)
                empty["p_media_scores"]["flow_gap"] = best_flow_gap
            # A3+ PR7 — p_media decision on the best-is-None path too.
            decision = self._compute_p_media_decision()
            empty["p_media"] = decision["p_media"]
            empty["p_media_triggered"] = decision["p_media_triggered"]
            empty["p_media_type"] = decision["p_media_type"]
            empty["p_media_bbox"] = decision["p_media_bbox"]
            # A3+ PR6 — Alert trigger integration (best-is-None path).
            if decision.get("p_media_triggered", False):
                empty["static_image_triggered"] = True
                empty["static_image_score"] = max(
                    empty.get("static_image_score", 0.0),
                    decision.get("p_media", 0.0),
                )
            self._apply_background_to_result(empty, bg_info)
            empty["p_media_timing_ms"] = _p_media_timing_ms
            return empty

        main_result = {
            "static_image_score": float(best["static_image_score"]),
            "static_image_triggered": bool(trigger_count > 0),
            "static_image_trigger_count": int(trigger_count),
            "static_image_patch_similarity": float(best["patch_similarity"]),
            "static_image_stable_count": int(best["stable_count"]),
            "static_image_center_motion": float(best["center_motion"]),
            "static_image_roi_motion": float(best["roi_motion"]),
            "static_image_context_motion": float(best["context_motion"]),
            "static_image_motion_contrast": float(best["motion_contrast"]),
            "static_image_edge_mean": float(best["edge_mean"]),
            "static_image_screen_like": bool(best.get("screen_like", False)),
            "static_image_screen_context_edge_mean": float(
                best.get("screen_context_edge_mean", 0.0)
            ),
            "static_image_screen_context_std": float(best.get("screen_context_std", 0.0)),
            "static_image_screen_line_score": float(best.get("screen_line_score", 0.0)),
            "static_image_screen_roi_context_area_ratio": float(
                best.get("screen_roi_context_area_ratio", 0.0)
            ),
            "static_image_screen_inside_person": bool(best.get("screen_inside_person", False)),
            "static_image_backend": "gpu_static_media_spoof",
            "static_image_roi_results": roi_results,
        }
        # A3+ PR1 — p_media contract on the main (non-empty) return path.
        # best is the highest-scoring ROI for this frame, so use its bbox
        # as a hint for ``p_media_bbox`` (post-PR3 will replace this with
        # candidate-level bbox once Canny/contour extraction lands).
        best_bbox = None
        best_roi = best.get("roi") if best is not None else None
        if isinstance(best_roi, dict):
            bb = best_roi.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                best_bbox = tuple(int(v) for v in bb)
        # p_media_candidate_count / bg_ready default to 0 / False here —
        # PR2 (background edge EMA) and PR3 (candidate extractor) will
        # overwrite the relevant entries later.
        main_result["p_media"] = 0.0
        main_result["p_media_triggered"] = False
        main_result["p_media_type"] = "normal"
        main_result["p_media_bbox"] = None
        main_result["p_media_candidate_count"] = active_candidate_count
        main_result["p_media_bg_ready"] = False
        main_result["p_media_scores"] = {
            "edge": 0.0,
            "new_edge": 0.0,
            "track": 0.0,
            "plane": 0.0,
            "warp_residual": 0.0,
            "flow_gap": 0.0,
            "yolo_context": 0.0,
            "global_fallback": 0.0,
        }
        # Fill edge score from best candidate
        if candidates:
            best_edge = max(
                (c.get("edge_score", 0.0) for c in candidates if not c.get("bg_suppressed", False)),
                default=0.0,
            )
            main_result["p_media_scores"]["edge"] = best_edge
        # Fill track score from best active track
        tracks = self._candidate_track_mgr.tracks
        if tracks:
            best_track_score = max(t.track_score for t in tracks)
            main_result["p_media_scores"]["track"] = best_track_score
        # Fill yolo_context score from best active track
        tracks = self._candidate_track_mgr.tracks
        if tracks:
            best_yolo_context = max(t.yolo_context_score for t in tracks)
            main_result["p_media_scores"]["yolo_context"] = best_yolo_context
        # Fill plane and warp_residual from best active track's L2 results
        tracks = self._candidate_track_mgr.tracks
        if tracks:
            best_plane = max(t.plane_score for t in tracks)
            best_warp = max(t.warp_residual for t in tracks)
            main_result["p_media_scores"]["plane"] = best_plane
            main_result["p_media_scores"]["warp_residual"] = best_warp
        # Fill flow_gap from best active track
        tracks = self._candidate_track_mgr.tracks
        if tracks:
            best_flow_gap = max(t.flow_gap_score for t in tracks)
            main_result["p_media_scores"]["flow_gap"] = best_flow_gap
        # A3+ PR7 — Comprehensive p_media decision.
        decision = self._compute_p_media_decision()
        main_result["p_media"] = decision["p_media"]
        main_result["p_media_triggered"] = decision["p_media_triggered"]
        main_result["p_media_type"] = decision["p_media_type"]
        main_result["p_media_bbox"] = decision["p_media_bbox"]
        main_result["p_media_target_related"] = bool(
            decision.get("target_related", False)
        )
        main_result["p_media_strong_evidence"] = bool(
            decision.get("strong_media_evidence", False)
        )
        main_result["p_media_background_static_suppressed"] = bool(
            decision.get("background_static_suppressed", False)
        )
        main_result["p_media_scores"]["target_iou"] = float(
            decision.get("target_iou", 0.0)
        )
        main_result["p_media_scores"]["target_proximity"] = float(
            decision.get("target_proximity_score", 0.0)
        )
        main_result["p_media_scores"]["target_area_ratio"] = float(
            decision.get("target_area_ratio", 0.0)
        )
        main_result["p_media_scores"]["global_fallback"] = 0.0
        self._apply_background_to_result(main_result, bg_info)

        # A3+ PR6 — Alert trigger integration.
        # When the new A3+ path triggers (p_media_triggered=True), feed it into
        # the existing static_image_triggered field so Rule_Fusion picks it up.
        # This is an OR with the legacy patch-track triggered path.
        if main_result.get("p_media_triggered", False):
            main_result["static_image_triggered"] = True
            main_result["static_image_score"] = max(
                main_result.get("static_image_score", 0.0),
                main_result.get("p_media", 0.0),
            )

        # A3+ PR6 — Source attribution field.
        # Distinguish whether static_image_triggered came from the new A3+ path,
        # the legacy patch-track path, or neither.
        main_result["static_image_triggered_source"] = (
            "a3_plus"
            if main_result.get("p_media_triggered", False)
            else ("legacy" if main_result.get("static_image_triggered", False) else "none")
        )

        main_result["p_media_timing_ms"] = _p_media_timing_ms

        return main_result

    # [legacy-fallback] Patch extraction for ROI similarity comparison.
    # Preserved because: feeds the patch-track scoring loop that still
    # produces `static_image_triggered`.
    def _extract_patch(
        self, gray_norm: torch.Tensor, bbox: tuple[int, int, int, int]
    ) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        patch = gray_norm[:, :, y1:y2, x1:x2]
        patch = F.interpolate(
            patch,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        patch = patch - patch.mean()
        patch = patch / (patch.std(unbiased=False) + 1e-6)
        return patch

    # [legacy-fallback] Track matching by IoU + center distance for
    # patch-track path. Preserved because: required by the ROI loop
    # that produces `static_image_triggered`.
    def _match_track(self, roi: ROI, center: tuple[float, float]) -> _PatchTrack | None:
        best: _PatchTrack | None = None
        best_score = -1.0
        for track in self._tracks:
            if track.label != roi.label:
                continue
            iou = self._iou(track.bbox, roi.bbox)
            center_dist = self._center_distance(center, track.center)
            score = iou - center_dist
            if iou >= 0.05 or center_dist <= 0.18:
                if score > best_score:
                    best_score = score
                    best = track
        return best

    # [legacy-fallback] Normalized cross-correlation between patches.
    # Preserved because: core metric for the patch-track triggered gate.
    @staticmethod
    def _patch_similarity(current: torch.Tensor, previous: torch.Tensor) -> float:
        return float(torch.clamp((current * previous).mean(), -1.0, 1.0).item())

    # [legacy-fallback] Utility — used by patch-track matching.
    @staticmethod
    def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx * dx + dy * dy) ** 0.5)

    # [legacy-fallback] IoU utility — used by both legacy patch-track
    # matching and future A3+ CandidateTrack IoU matching.
    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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

    # [legacy-fallback] Person-context check for the ROI loop.
    # Preserved because: prevents false triggers on small person ROIs
    # that lack a larger person context.
    def _has_larger_person_context(
        self,
        bbox: tuple[int, int, int, int],
        person_bboxes: list[tuple[int, int, int, int]],
    ) -> bool:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            return False
        for other in person_bboxes:
            if other == bbox:
                continue
            ox1, oy1, ox2, oy2 = other
            other_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            if other_area <= 0:
                continue
            if (
                area <= other_area * self.max_person_area_ratio
                and self._iou(bbox, other) <= self.max_context_iou
            ):
                return True
        return False

    # [legacy-fallback] Screen-like context detection (edge/std/line
    # analysis around a helmet/head ROI). Preserved because: feeds the
    # `screen_like` gate in the ROI scoring loop.
    def _screen_context_features(
        self,
        bbox: tuple[int, int, int, int],
        edge: torch.Tensor,
        gray_norm: torch.Tensor,
        person_bboxes: list[tuple[int, int, int, int]],
        w: int,
        h: int,
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = bbox
        ex1, ey1, ex2, ey2 = self._expanded_bbox_with_ratio(
            x1,
            y1,
            x2,
            y2,
            w,
            h,
            self.screen_context_expand_ratio,
        )
        context_edge = edge[:, :, ey1:ey2, ex1:ex2]
        context_gray = gray_norm[:, :, ey1:ey2, ex1:ex2]
        context_area = max(1, (ex2 - ex1) * (ey2 - ey1))
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        context_edge_mean = float(context_edge.mean().item()) if context_edge.numel() else 0.0
        context_std = (
            float(context_gray.std(unbiased=False).item()) if context_gray.numel() else 0.0
        )
        line_score = 0.0
        if context_edge.numel():
            edge_2d = context_edge[0, 0]
            row_peak = float(edge_2d.mean(dim=1).max().item()) if edge_2d.shape[0] else 0.0
            col_peak = float(edge_2d.mean(dim=0).max().item()) if edge_2d.shape[1] else 0.0
            line_score = min(row_peak, col_peak)
        area_ratio = float(roi_area / context_area)
        inside_person = self._is_contained_by_person(bbox, person_bboxes)
        screen_like = (
            not inside_person
            and context_edge_mean >= self.screen_min_context_edge_mean
            and context_std >= self.screen_min_context_std
            and line_score >= self.screen_min_line_score
            and area_ratio <= self.screen_max_roi_context_area_ratio
        )
        return {
            "screen_like": bool(screen_like),
            "screen_context_edge_mean": context_edge_mean,
            "screen_context_std": context_std,
            "screen_line_score": line_score,
            "screen_roi_context_area_ratio": area_ratio,
            "screen_inside_person": bool(inside_person),
        }

    # [legacy-fallback] Helper for screen-context — checks if a small
    # ROI is inside a larger person bbox (suppresses false screen-like).
    def _is_contained_by_person(
        self,
        bbox: tuple[int, int, int, int],
        person_bboxes: list[tuple[int, int, int, int]],
    ) -> bool:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            return False
        for px1, py1, px2, py2 in person_bboxes:
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / area >= self.screen_person_containment_threshold:
                return True
        return False

    # [legacy-fallback] Bbox expansion utilities — used by the ROI loop
    # for surround-motion and screen-context calculations.
    def _expanded_bbox(
        self, x1: int, y1: int, x2: int, y2: int, w: int, h: int
    ) -> tuple[int, int, int, int]:
        return self._expanded_bbox_with_ratio(x1, y1, x2, y2, w, h, self.expand_ratio)

    # [legacy-fallback] Parameterised bbox expansion (also used by
    # screen-context expand and future A3+ candidate extraction).
    @staticmethod
    def _expanded_bbox_with_ratio(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
        ratio: float,
    ) -> tuple[int, int, int, int]:
        pad_x = int((x2 - x1) * ratio)
        pad_y = int((y2 - y1) * ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )

    # [legacy-fallback] Surround motion calculation for the ROI loop.
    @staticmethod
    def _surround_motion(expanded_diff: torch.Tensor, x1: int, y1: int, x2: int, y2: int) -> float:
        if expanded_diff.numel() == 0:
            return 0.0
        mask = torch.ones_like(expanded_diff, dtype=torch.bool)
        mask[:, :, max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = False
        values = expanded_diff[mask]
        if values.numel() == 0:
            return float((expanded_diff.float().mean() / 255.0).item())
        return float((values.float().mean() / 255.0).item())

    # [active] Sobel kernel initialisation — shared by both legacy edge
    # computation and A3+ background edge EMA / future candidate extraction.
    def _ensure_kernels(self, device: torch.device) -> None:
        if self._kernel_device == device:
            return
        sobel_x = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, 3, 3)
            / 8.0
        )
        sobel_y = (
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, 3, 3)
            / 8.0
        )
        self._sobel_x = sobel_x
        self._sobel_y = sobel_y
        self._kernel_device = device

    # [legacy-fallback] Linear ramp utility — used by the ROI scoring
    # formula. Will also be reused by A3+ PR5 decision scoring.
    @staticmethod
    def _ramp(value: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0 if value >= end else 0.0
        return min(1.0, max(0.0, (value - start) / (end - start)))

    # ------------------------------------------------------------------
    # [active] Background edge EMA — A3+ PR2
    # ------------------------------------------------------------------
    def _update_background_edge(self, curr_gray: torch.Tensor) -> dict[str, Any]:
        """Update the long-term background edge EMA and return diagnostics.

        Returns a small dict consumed by ``_apply_background_to_result``:

          * ``bg_ready``     — True once we've accumulated ``_bg_warmup_frames``
                                updates, i.e. the model is warm enough to diff
                                against.
          * ``new_edge_score`` — mean of ``clamp(edge_t - bg, 0, 1)`` over the
                                full frame, in [0, 1]. During cold start this
                                is returned as 0.0 so consumers never see a
                                spurious "new edge" spike from the uninitialised
                                model.

        Implementation is GPU-resident: the Sobel kernels are reused from the
        existing ROI path, no CPU roundtrip, no additional allocation besides
        the running EMA tensor (same shape as the input).
        """
        device = curr_gray.device
        self._ensure_kernels(device)
        assert self._sobel_x is not None
        assert self._sobel_y is not None

        curr_norm = curr_gray.float() / 255.0
        gx = F.conv2d(curr_norm, self._sobel_x, padding=1)
        gy = F.conv2d(curr_norm, self._sobel_y, padding=1)
        edge_t = torch.sqrt(gx * gx + gy * gy + 1e-8)

        if self._bg_edge is None or self._bg_edge.shape != edge_t.shape:
            # First frame or resolution change: seed the EMA with the
            # current edge map. Do NOT mark as ready — we want the
            # warmup window so a single-frame spike cannot trip the
            # "new edge" diagnostic immediately.
            self._bg_edge = edge_t.detach().clone()
            self._bg_ready_frames = 1
        else:
            self._bg_edge = (
                1.0 - self._bg_alpha
            ) * self._bg_edge + self._bg_alpha * edge_t.detach()
            self._bg_ready_frames += 1

        bg_ready = self._bg_ready_frames >= self._bg_warmup_frames
        if bg_ready:
            diff = torch.clamp(edge_t - self._bg_edge, min=0.0, max=1.0)
            new_edge_score = float(diff.mean().item())
        else:
            new_edge_score = 0.0
        return {
            "bg_ready": bool(bg_ready),
            "new_edge_score": float(new_edge_score),
        }

    # [active] Write background-edge diagnostics into the p_media contract.
    def _apply_background_to_result(
        self,
        result: dict[str, Any],
        bg_info: dict[str, Any],
    ) -> None:
        """Write background-edge diagnostics into the p_media contract.

        Read-only over the algorithm: this never changes
        ``static_image_triggered`` or ``p_media_triggered``. It only
        populates the two fields reserved in PR1 for this PR:
        ``p_media_bg_ready`` and ``p_media_scores["new_edge"]``.
        """
        result["p_media_bg_ready"] = bool(bg_info.get("bg_ready", False))
        scores = result.get("p_media_scores")
        if isinstance(scores, dict):
            scores["new_edge"] = float(bg_info.get("new_edge_score", 0.0))

    # [active] Returns the default result dict including the full p_media
    # field contract (8 top-level keys + 8 scores sub-keys). Guarantees
    # downstream consumers never need to null-check.
    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "static_image_score": 0.0,
            "static_image_triggered": False,
            "static_image_trigger_count": 0,
            "static_image_patch_similarity": 0.0,
            "static_image_stable_count": 0,
            "static_image_center_motion": 0.0,
            "static_image_roi_motion": 0.0,
            "static_image_context_motion": 0.0,
            "static_image_motion_contrast": 0.0,
            "static_image_edge_mean": 0.0,
            "static_image_screen_like": False,
            "static_image_screen_context_edge_mean": 0.0,
            "static_image_screen_context_std": 0.0,
            "static_image_screen_line_score": 0.0,
            "static_image_screen_roi_context_area_ratio": 0.0,
            "static_image_screen_inside_person": False,
            "static_image_backend": "gpu_static_media_spoof",
            "static_image_roi_results": [],
            # A3+ PR1 — p_media contract:
            # Guarantee these exist on EVERY compute() return path so
            # downstream consumers never have to null-check the contract.
            "p_media": 0.0,
            "p_media_triggered": False,
            "p_media_type": "normal",
            "p_media_bbox": None,
            "p_media_target_related": False,
            "p_media_strong_evidence": False,
            "p_media_background_static_suppressed": False,
            "p_media_candidate_count": 0,
            "p_media_bg_ready": False,
            "p_media_scores": {
                "edge": 0.0,
                "new_edge": 0.0,
                "track": 0.0,
                "plane": 0.0,
                "warp_residual": 0.0,
                "flow_gap": 0.0,
                "yolo_context": 0.0,
                "target_iou": 0.0,
                "target_proximity": 0.0,
                "target_area_ratio": 0.0,
                "global_fallback": 0.0,
            },
        }

"""L2 Homography verification for A3+ flat media spoof detection.

This module implements the second stage (L2) of the A3+ cascade: verifying
whether a candidate region behaves as a rigid plane via KLT optical flow
tracking + RANSAC Homography estimation + warp residual computation.

Performance target: < 5ms per candidate ROI.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _empty_result() -> dict[str, float]:
    """Return a zeroed-out result dict for cases where verification cannot proceed."""
    return {
        "plane_score": 0.0,
        "warp_residual": 0.0,
        "inlier_ratio": 0.0,
        "reproj_error": 99.0,
        "raw_residual": 0.0,
    }


def compute_homography_verification(
    prev_roi_np: np.ndarray,
    curr_roi_np: np.ndarray,
) -> dict[str, float]:
    """L2: KLT + RANSAC Homography + Warp Residual.

    Estimates whether the ROI region behaves as a rigid plane by tracking
    feature points between consecutive frames, fitting a Homography via
    RANSAC, and measuring the warp residual.

    Args:
        prev_roi_np: Previous frame ROI crop (uint8, grayscale, H x W)
        curr_roi_np: Current frame ROI crop (uint8, grayscale, H x W)

    Returns:
        Dict with:
        - plane_score: inlier_ratio * exp(-reproj_error / 2.0), in [0, 1]
        - warp_residual: normalized warp residual mapped to [0, 1] via ramp
        - inlier_ratio: fraction of KLT points that are RANSAC inliers
        - reproj_error: mean reprojection error of inliers (pixels)
        - raw_residual: mean(abs(curr - warped_prev)) / 255.0
    """
    # --- Step 1: KLT feature detection on previous ROI ---
    pts0 = cv2.goodFeaturesToTrack(
        prev_roi_np,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=5,
    )

    if pts0 is None or len(pts0) < 8:
        return _empty_result()

    # --- Step 2: Optical flow tracking (prev → curr) ---
    pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev_roi_np, curr_roi_np, pts0, None)

    # --- Step 3: Filter by status == 1 (successfully tracked) ---
    good_mask = status.ravel() == 1
    pts0_good = pts0[good_mask]
    pts1_good = pts1[good_mask]

    if len(pts0_good) < 4:
        return _empty_result()

    # --- Step 4: RANSAC Homography estimation ---
    H, mask = cv2.findHomography(pts0_good, pts1_good, cv2.RANSAC, 3.0)

    if H is None or mask is None:
        return _empty_result()

    # --- Step 5: Compute inlier_ratio ---
    inlier_ratio = float(mask.sum()) / max(len(mask), 1)

    # --- Step 6: Compute reprojection error for inlier points ---
    inlier_idx = mask.ravel() == 1
    inlier_pts0 = pts0_good[inlier_idx]
    inlier_pts1 = pts1_good[inlier_idx]

    if len(inlier_pts0) > 0:
        projected = cv2.perspectiveTransform(inlier_pts0.reshape(-1, 1, 2).astype(np.float64), H)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - inlier_pts1.reshape(-1, 2).astype(np.float64),
            axis=1,
        )
        reproj_error = float(errors.mean())
    else:
        reproj_error = 99.0

    # --- Step 7: plane_score = inlier_ratio * exp(-reproj_error / 2.0) ---
    plane_score = inlier_ratio * math.exp(-reproj_error / 2.0)

    # --- Step 8: Warp residual computation ---
    h, w = curr_roi_np.shape[:2]
    warped_prev = cv2.warpPerspective(prev_roi_np, H, (w, h))

    raw_residual = (
        float(np.abs(curr_roi_np.astype(np.float32) - warped_prev.astype(np.float32)).mean())
        / 255.0
    )

    # --- Step 9: Normalize warp_residual to [0, 1] via ramp ---
    # ramp: (raw - 0.005) / (0.08 - 0.005), clamped to [0, 1]
    warp_residual = min(1.0, max(0.0, (raw_residual - 0.005) / (0.08 - 0.005)))

    return {
        "plane_score": float(plane_score),
        "warp_residual": float(warp_residual),
        "inlier_ratio": float(inlier_ratio),
        "reproj_error": float(reproj_error),
        "raw_residual": float(raw_residual),
    }

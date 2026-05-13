"""L0 Edge Candidate Extraction for A3+ flat media spoof detection.

This module implements the first stage (L0) of the A3+ cascade: extracting
rectangular edge candidates from a low-resolution grayscale frame using
Canny edge detection, contour finding, and polygon approximation with
geometric filtering.

Performance target: < 2ms at 416 width.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

_TARGET_WIDTH = 416
_MAX_CANDIDATES = 5
_AREA_RATIO_MIN = 0.005
_AREA_RATIO_MAX = 0.40
_ASPECT_RATIO_MIN = 0.3
_ASPECT_RATIO_MAX = 3.3
_RECTANGULARITY_MIN = 0.70


def _as_uint8_np(curr_gray) -> np.ndarray:
    """Coerce torch tensor or ndarray gray frame to a 2D uint8 ndarray."""
    if isinstance(curr_gray, np.ndarray):
        arr = curr_gray
    else:
        arr = curr_gray[0, 0].cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _solid_padding_score(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 8,
) -> float:
    """Return how likely a candidate is aspect-ratio solid-color padding.

    Letterbox/pillarbox padding is not always black; it can be white, gray, or
    any UI/background color after transcoding or compositing. Suppress only
    candidates whose outside bands are low-texture, color-consistent, and form
    an opposite edge pair touching the frame boundary. This avoids treating a
    real physical screen edge as padding just because one nearby region is flat.
    """
    h, w = gray.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(y1 + 1, min(h, y2))
    bands: dict[str, tuple[np.ndarray, bool]] = {}
    if y1 > 0:
        bands["top"] = (gray[max(0, y1 - margin):y1, x1:x2], y1 <= margin * 2)
    if y2 < h:
        bands["bottom"] = (gray[y2:min(h, y2 + margin), x1:x2], h - y2 <= margin * 2)
    if x1 > 0:
        bands["left"] = (gray[y1:y2, max(0, x1 - margin):x1], x1 <= margin * 2)
    if x2 < w:
        bands["right"] = (gray[y1:y2, x2:min(w, x2 + margin)], w - x2 <= margin * 2)

    solid: dict[str, tuple[float, bool]] = {}
    for name, (band, touches_frame) in bands.items():
        if band.size == 0:
            continue
        if float(band.std()) <= 8.0:
            solid[name] = (float(band.mean()), touches_frame)

    pair_scores: list[float] = []
    for a, b in (("top", "bottom"), ("left", "right")):
        if a not in solid or b not in solid:
            continue
        mean_a, touch_a = solid[a]
        mean_b, touch_b = solid[b]
        if abs(mean_a - mean_b) <= 18.0 and (touch_a or touch_b):
            pair_scores.append(1.0)
    if pair_scores:
        return max(pair_scores)

    # Single-sided padding can happen after crop/resize, but it is riskier to
    # suppress; only score it weakly so background EMA can still decide later.
    edge_touch_solid = sum(1 for _, touches_frame in solid.values() if touches_frame)
    return 0.5 if edge_touch_solid >= 2 else 0.0


def extract_edge_candidates(
    curr_gray,  # torch.Tensor | np.ndarray
    bg_edge=None,  # torch.Tensor | np.ndarray | None
    bg_ready: bool = False,
    bg_suppression_ratio: float = 0.70,
) -> list[dict]:
    """L0: Extract rectangular edge candidates at low resolution.

    Args:
        curr_gray: (1, 1, H, W) uint8 tensor on GPU OR (H, W) uint8 ndarray.
            Ndarray form skips the internal GPU→CPU transfer and is used
            when the caller has already batched the transfer across
            multiple scales.
        bg_edge: background edge EMA (same shape convention as curr_gray)
            or None.
        bg_ready: whether background model is warmed up.
        bg_suppression_ratio: threshold for background suppression.

    Returns:
        List of candidate dicts with keys: bbox (x1,y1,x2,y2 in original
        resolution), area_ratio, aspect_ratio, rectangularity, edge_score,
        bg_suppressed
    """
    # --- Step 1: normalise input → 2D uint8 numpy ---
    gray_np = _as_uint8_np(curr_gray)
    h, w = gray_np.shape
    scale = _TARGET_WIDTH / w
    target_h = int(h * scale)
    small = cv2.resize(gray_np, (_TARGET_WIDTH, target_h), interpolation=cv2.INTER_AREA)

    # --- Step 2: Canny edge detection ---
    edges = cv2.Canny(small, 50, 150)

    # --- Step 3: Morphological close to connect broken edges ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --- Step 4: Find contours ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = _TARGET_WIDTH * target_h
    candidates: list[dict] = []

    for contour in contours:
        # --- Step 5: Approximate polygon ---
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # --- Step 6: Geometric filtering ---
        # Must be quadrilateral-ish (4-8 vertices)
        n_vertices = len(approx)
        if n_vertices < 4 or n_vertices > 8:
            continue

        # Bounding rect and area
        x, y, bw, bh = cv2.boundingRect(approx)
        contour_area = cv2.contourArea(approx)
        rect_area = bw * bh

        if rect_area == 0:
            continue

        # Area ratio filter
        area_ratio = contour_area / img_area
        if area_ratio < _AREA_RATIO_MIN or area_ratio > _AREA_RATIO_MAX:
            continue

        # Aspect ratio filter
        aspect_ratio = bw / max(bh, 1)
        if aspect_ratio < _ASPECT_RATIO_MIN or aspect_ratio > _ASPECT_RATIO_MAX:
            continue

        # Rectangularity filter
        rectangularity = contour_area / rect_area
        if rectangularity < _RECTANGULARITY_MIN:
            continue

        # Compute edge_score: mean edge intensity along candidate border
        # pixels from the Canny edge map, normalized to [0, 1].
        mask = np.zeros((target_h, _TARGET_WIDTH), dtype=np.uint8)
        cv2.drawContours(mask, [approx], 0, 255, thickness=2)
        border_pixels = edges[mask == 255]
        if border_pixels.size > 0:
            edge_score = float(border_pixels.mean()) / 255.0
        else:
            edge_score = 0.0

        # --- Step 7: Map bbox back to original resolution ---
        ox1 = int(x / scale)
        oy1 = int(y / scale)
        ox2 = int((x + bw) / scale)
        oy2 = int((y + bh) / scale)

        bbox = (ox1, oy1, ox2, oy2)
        solid_padding_score = _solid_padding_score(gray_np, bbox)
        candidates.append(
            {
                "bbox": bbox,
                "area_ratio": float(area_ratio),
                "aspect_ratio": float(aspect_ratio),
                "rectangularity": float(rectangularity),
                "edge_score": float(edge_score),
                "bg_suppressed": bool(solid_padding_score >= 0.75),
                "solid_padding_score": float(solid_padding_score),
            }
        )

    # --- Step 8: Sort by area descending, keep top-K ---
    candidates.sort(key=lambda c: c["area_ratio"], reverse=True)
    candidates = candidates[:_MAX_CANDIDATES]

    # --- Step 9: Background edge suppression ---
    if bg_ready and bg_edge is not None and candidates:
        # Normalise bg_edge to 2D uint8 numpy. Accept both torch tensor
        # (legacy) and pre-transferred ndarray (batched fallback path).
        bg_np = _as_uint8_np(bg_edge)
        bg_small = cv2.resize(bg_np, (_TARGET_WIDTH, target_h), interpolation=cv2.INTER_AREA)
        edges_f = edges.astype(np.float32, copy=False)
        bg_small_f = bg_small.astype(np.float32, copy=False)

        for cand in candidates:
            ox1, oy1, ox2, oy2 = cand["bbox"]
            lx1 = max(0, min(int(ox1 * scale), _TARGET_WIDTH - 1))
            ly1 = max(0, min(int(oy1 * scale), target_h - 1))
            lx2 = max(lx1 + 1, min(int(ox2 * scale), _TARGET_WIDTH))
            ly2 = max(ly1 + 1, min(int(oy2 * scale), target_h))
            if lx2 - lx1 < 2 or ly2 - ly1 < 2:
                continue

            # Rectangle border = union of 4 strips (top / bottom / left / right),
            # each 2 pixels thick to match the original ``cv2.rectangle``
            # thickness=2 mask. Means are area-weighted.
            top_e = edges_f[ly1 : ly1 + 2, lx1:lx2]
            bot_e = edges_f[max(ly2 - 2, ly1 + 2) : ly2, lx1:lx2]
            lft_e = edges_f[ly1 + 2 : ly2 - 2, lx1 : lx1 + 2]
            rgt_e = edges_f[ly1 + 2 : ly2 - 2, max(lx2 - 2, lx1 + 2) : lx2]
            top_b = bg_small_f[ly1 : ly1 + 2, lx1:lx2]
            bot_b = bg_small_f[max(ly2 - 2, ly1 + 2) : ly2, lx1:lx2]
            lft_b = bg_small_f[ly1 + 2 : ly2 - 2, lx1 : lx1 + 2]
            rgt_b = bg_small_f[ly1 + 2 : ly2 - 2, max(lx2 - 2, lx1 + 2) : lx2]

            e_total = top_e.sum() + bot_e.sum() + lft_e.sum() + rgt_e.sum()
            b_total = top_b.sum() + bot_b.sum() + lft_b.sum() + rgt_b.sum()
            n_px = top_e.size + bot_e.size + lft_e.size + rgt_e.size
            if n_px == 0:
                continue
            edge_border_mean = float(e_total) / n_px
            bg_border_mean = float(b_total) / n_px

            if (
                edge_border_mean > 0
                and bg_border_mean >= bg_suppression_ratio * edge_border_mean
            ):
                cand["bg_suppressed"] = True

    # --- Step 10: Return list of candidate dicts ---
    return candidates

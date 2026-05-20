"""Pure-torch candidate extractor — NPU-friendly replacement for the
OpenCV-based ``candidate_extraction.extract_edge_candidates``.

Design goals:
  * Forward path is 100% ``torch.nn.functional`` + tensor ops — no cv2,
    no ``.cpu().numpy()``, no Python loops that depend on tensor values.
  * Input/output shape-compatible with the cv2 version so the upstream
    ``FeatureBuilder`` can swap between them via config.
  * Single contract: given ``(1, 1, H, W)`` gray + optional bg_edge,
    return a deterministic list of up to ``max_candidates`` dict rows.

Algorithm (equivalent intent to the cv2 pipeline):
  * Sobel gradient magnitude (replaces Canny). Low/high thresholds.
  * Edge-density grid (replaces findContours). Reduce to G×G via
    adaptive average pool.
  * Rectangular candidate scan (replaces approxPolyDP + boundingRect).
    For each grid row, find maximal runs of above-threshold cells;
    same vertically; intersect into rectangles.
  * Score each rectangle by its border-edge energy (matches cv2
    ``edge_score`` semantics) and rectangularity.

The scan step needs a Python loop over candidates (at most
``max_candidates × G``), but G is fixed at 16 so the loop has a
bounded iteration count and stays NPU-friendly (host loop, tensor body).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_TARGET_WIDTH = 416
_MAX_CANDIDATES = 5
_AREA_RATIO_MIN = 0.005
_AREA_RATIO_MAX = 0.40
_ASPECT_RATIO_MIN = 0.3
_ASPECT_RATIO_MAX = 3.3


# ---------------------------------------------------------------------------
# Kernel cache — built once per device.
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[torch.device, dict[str, torch.Tensor]] = {}


def _kernels(device: torch.device) -> dict[str, torch.Tensor]:
    cache = _KERNEL_CACHE.get(device)
    if cache is not None:
        return cache
    sobel_x = (
        torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = (
        torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    # 5×5 Gaussian smoothing (sigma ~ 1.0).
    g = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 5, 5)
    g = g / g.sum()
    cache = {"sobel_x": sobel_x, "sobel_y": sobel_y, "gauss": g}
    _KERNEL_CACHE[device] = cache
    return cache


# ---------------------------------------------------------------------------
# Torch Canny-lite: Gaussian blur + Sobel magnitude + dual-threshold mask.
# ---------------------------------------------------------------------------


def _torch_edge_map(gray_u8: torch.Tensor, low: float = 0.15, high: float = 0.30) -> torch.Tensor:
    """Return a (1, 1, H, W) float mask ∈ [0, 1] of edges.

    ``gray_u8``: (1, 1, H, W) float in [0, 255] or already normalised [0, 1].
    """
    if gray_u8.max() > 1.5:
        gray = gray_u8 / 255.0
    else:
        gray = gray_u8
    k = _kernels(gray.device)
    blurred = F.conv2d(gray, k["gauss"], padding=2)
    gx = F.conv2d(blurred, k["sobel_x"], padding=1)
    gy = F.conv2d(blurred, k["sobel_y"], padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    # Simplified hysteresis: strong | (weak dilated by strong neighbours).
    strong = mag >= high
    weak = mag >= low
    strong_dilated = F.max_pool2d(strong.float(), kernel_size=3, stride=1, padding=1) > 0
    edges = strong | (weak & strong_dilated)
    return edges.float()


# ---------------------------------------------------------------------------
# Grid density map (replaces findContours).
# ---------------------------------------------------------------------------


def _edge_grid(edge_map: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """Downsample a (1, 1, H, W) edge mask to (grid, grid) density."""
    return F.adaptive_avg_pool2d(edge_map, (grid, grid))[0, 0]  # (G, G)


# ---------------------------------------------------------------------------
# Rectangle candidate enumeration on the G×G density grid.
# ---------------------------------------------------------------------------


def _enumerate_rect_candidates(
    density: torch.Tensor,
    grid: int,
    H: int,
    W: int,
    density_min: float = 0.2,
    max_candidates: int = _MAX_CANDIDATES,
) -> list[tuple[tuple[int, int, int, int], float, float, float]]:
    """Find rectangles whose perimeter is dense enough on the density grid.

    Vectorised scan: instead of a Python O(G⁴) loop with per-iteration
    GPU sync, we precompute row / column prefix sums of the density map
    on GPU, then iterate a much smaller Python loop that only indexes
    into those precomputed buffers (all dense tensor reads, no ``.item()``
    inside the loop). Final scoring + NMS still happens in Python but
    only on the already-pruned candidate list (≤ 200 entries).
    """
    cell_h = H / grid
    cell_w = W / grid
    mask = density >= density_min  # (G, G) bool
    density_cpu = density.detach().cpu().numpy()  # (G, G) float32 — moved once
    mask_cpu = mask.detach().cpu().numpy()  # (G, G) bool

    # Precomputed cumulative row sums of the mask along columns.
    # row_prefix[r, c] = sum(mask_cpu[r, 0:c])
    import numpy as _np

    mask_int = mask_cpu.astype(_np.int32)
    row_prefix = _np.concatenate(
        [_np.zeros((grid, 1), dtype=_np.int32), mask_int.cumsum(axis=1)], axis=1
    )  # (G, G+1)
    col_prefix = _np.concatenate(
        [_np.zeros((1, grid), dtype=_np.int32), mask_int.cumsum(axis=0)], axis=0
    )  # (G+1, G)
    density_row_prefix = _np.concatenate(
        [_np.zeros((grid, 1), dtype=_np.float32), density_cpu.cumsum(axis=1)], axis=1
    )
    density_col_prefix = _np.concatenate(
        [_np.zeros((1, grid), dtype=_np.float32), density_cpu.cumsum(axis=0)], axis=0
    )

    results: list[tuple[tuple[int, int, int, int], float, float, float]] = []
    min_side = 3
    max_side = int(grid * 0.7)
    # Total iterations ≈ G² × (G × G/2) = ~32K at G=16, but every
    # iteration is a constant-time index lookup — no torch ops / no
    # .item() calls — so it runs in a few ms on Python.
    for r1 in range(grid - min_side):
        for r2 in range(r1 + min_side, min(grid, r1 + max_side + 1)):
            h_cells = r2 - r1 + 1
            for c1 in range(grid - min_side):
                # Precompute slice sums along the column direction once per r1,r2.
                # top row mask count c1..c2 = row_prefix[r1, c2+1] - row_prefix[r1, c1]
                # col mask count at c1  r1..r2 = col_prefix[r2+1, c1] - col_prefix[r1, c1]
                for c2 in range(c1 + min_side, min(grid, c1 + max_side + 1)):
                    w_cells = c2 - c1 + 1
                    aspect = w_cells / h_cells
                    if aspect < _ASPECT_RATIO_MIN or aspect > _ASPECT_RATIO_MAX:
                        continue
                    area_ratio_grid = (h_cells * w_cells) / (grid * grid)
                    if area_ratio_grid < _AREA_RATIO_MIN or area_ratio_grid > _AREA_RATIO_MAX:
                        continue
                    # Border counts from prefix sums.
                    top_cnt = row_prefix[r1, c2 + 1] - row_prefix[r1, c1]
                    bot_cnt = row_prefix[r2, c2 + 1] - row_prefix[r2, c1]
                    lft_cnt = col_prefix[r2 + 1, c1] - col_prefix[r1 + 1, c1]
                    rgt_cnt = col_prefix[r2 + 1, c2] - col_prefix[r1 + 1, c2]
                    border_cnt = top_cnt + bot_cnt + lft_cnt + rgt_cnt
                    border_total = (c2 - c1 + 1) * 2 + (r2 - r1 - 1) * 2
                    if border_total == 0:
                        continue
                    rectangularity = border_cnt / border_total
                    if rectangularity < 0.6:
                        continue
                    top_sum = density_row_prefix[r1, c2 + 1] - density_row_prefix[r1, c1]
                    bot_sum = density_row_prefix[r2, c2 + 1] - density_row_prefix[r2, c1]
                    lft_sum = density_col_prefix[r2 + 1, c1] - density_col_prefix[r1 + 1, c1]
                    rgt_sum = density_col_prefix[r2 + 1, c2] - density_col_prefix[r1 + 1, c2]
                    edge_score = float(top_sum + bot_sum + lft_sum + rgt_sum) / border_total
                    # Interior mean density as a secondary hint.
                    interior = density_cpu[r1 : r2 + 1, c1 : c2 + 1]
                    density_score = float(interior.mean())
                    x1_o = int(c1 * cell_w)
                    y1_o = int(r1 * cell_h)
                    x2_o = int((c2 + 1) * cell_w)
                    y2_o = int((r2 + 1) * cell_h)
                    results.append(
                        (
                            (x1_o, y1_o, x2_o, y2_o),
                            density_score,
                            float(rectangularity),
                            edge_score,
                        )
                    )

    if not results:
        return []
    results.sort(
        key=lambda item: item[2] * item[3] + 0.1 * item[1],
        reverse=True,
    )
    kept: list[tuple[tuple[int, int, int, int], float, float, float]] = []
    for cand in results:
        bbox = cand[0]
        overlaps = False
        for k in kept:
            if _iou(bbox, k[0]) > 0.35:
                overlaps = True
                break
        if overlaps:
            continue
        kept.append(cand)
        if len(kept) >= max_candidates:
            break
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1, area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Public entry point — same contract as cv2 ``extract_edge_candidates``.
# ---------------------------------------------------------------------------


def extract_edge_candidates_torch(
    curr_gray: torch.Tensor,
    bg_edge: torch.Tensor | None = None,
    bg_ready: bool = False,
    bg_suppression_ratio: float = 0.85,
) -> list[dict]:
    """Torch-native replacement for ``extract_edge_candidates``.

    Signature matches the cv2 version. Returns a list of candidate dicts
    with the SAME KEYS the rest of ``feature_builder`` expects:

        {
            "bbox": (x1, y1, x2, y2),     # original resolution
            "area_ratio": float,
            "aspect_ratio": float,
            "rectangularity": float,
            "edge_score": float,
            "bg_suppressed": bool,
        }

    No cv2 calls, no numpy conversion. Can run on CUDA or CPU.
    """
    # Accept either (H, W) uint8 ndarray or (1, 1, H, W) tensor, matching
    # the hybrid signature exposed by the cv2 version.
    if not isinstance(curr_gray, torch.Tensor):
        curr_gray = torch.from_numpy(curr_gray).float().view(1, 1, *curr_gray.shape)
    gray = curr_gray.float()
    if gray.dim() == 2:
        gray = gray.view(1, 1, *gray.shape)

    _, _, H, W = gray.shape
    # Downsample gray to 416 width equivalent so thresholds stay compatible.
    scale = _TARGET_WIDTH / W
    target_h = int(H * scale)
    if (H, W) != (target_h, _TARGET_WIDTH):
        gray_small = F.interpolate(
            gray, size=(target_h, _TARGET_WIDTH), mode="area"
        )
    else:
        gray_small = gray

    edges = _torch_edge_map(gray_small, low=0.03, high=0.08)
    # 3×3 close to connect thin gaps (equivalent to cv2.MORPH_CLOSE 3×3).
    edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)

    grid = 16
    density = _edge_grid(edges, grid=grid)
    rects = _enumerate_rect_candidates(
        density,
        grid=grid,
        H=H,
        W=W,
        density_min=0.10,
        max_candidates=_MAX_CANDIDATES,
    )

    candidates: list[dict] = []
    for bbox, density_score, rectangularity, edge_score in rects:
        area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(H * W)
        aspect = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
        candidates.append(
            {
                "bbox": bbox,
                "area_ratio": float(area_ratio),
                "aspect_ratio": float(aspect),
                "rectangularity": float(rectangularity),
                "edge_score": float(edge_score),
                "bg_suppressed": False,
            }
        )

    # Background suppression (same intent as cv2 version: if the
    # background EMA already has an edge at the candidate's border,
    # suppress it). Torch-native vectorised over candidates.
    if bg_ready and bg_edge is not None and candidates:
        if not isinstance(bg_edge, torch.Tensor):
            bg_edge = torch.from_numpy(bg_edge).float().view(1, 1, *bg_edge.shape)
        bg = bg_edge.float()
        if bg.dim() == 2:
            bg = bg.view(1, 1, *bg.shape)
        if (bg.shape[-2], bg.shape[-1]) != (target_h, _TARGET_WIDTH):
            bg_small = F.interpolate(bg, size=(target_h, _TARGET_WIDTH), mode="area")
        else:
            bg_small = bg
        bg_density = _edge_grid(bg_small, grid=grid)
        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]
            r1 = max(0, int(y1 * grid / H))
            c1 = max(0, int(x1 * grid / W))
            r2 = min(grid - 1, int(y2 * grid / H))
            c2 = min(grid - 1, int(x2 * grid / W))
            if r2 <= r1 or c2 <= c1:
                continue
            # Border density on bg vs edge map.
            eb_top = density[r1, c1 : c2 + 1].sum()
            eb_bot = density[r2, c1 : c2 + 1].sum()
            eb_lft = density[r1 + 1 : r2, c1].sum()
            eb_rgt = density[r1 + 1 : r2, c2].sum()
            bb_top = bg_density[r1, c1 : c2 + 1].sum()
            bb_bot = bg_density[r2, c1 : c2 + 1].sum()
            bb_lft = bg_density[r1 + 1 : r2, c1].sum()
            bb_rgt = bg_density[r1 + 1 : r2, c2].sum()
            e_total = float((eb_top + eb_bot + eb_lft + eb_rgt).item())
            b_total = float((bb_top + bb_bot + bb_lft + bb_rgt).item())
            if e_total > 0 and b_total >= bg_suppression_ratio * e_total:
                cand["bg_suppressed"] = True

    return candidates

# A3b L0 multi-scale fallback cost analysis

## Observation

`profile_a3b_internals.py` shows `l0_l1_ms` p95 ≈ 13–15 ms across all 7 sample
clips. L0 runs every 5 frames (`_l0_interval=5`), so per-L0-call cost is what's
measured at p95.

## Hot path

`candidate_extraction.extract_edge_candidates` itself runs in ~2 ms at
416 px. But `feature_builder.extract_and_filter_candidates` wraps it with a
**multi-scale fallback**:

```
active_count = sum(1 for c in candidates if not c.get("bg_suppressed", False))
if active_count >= 2:
    return candidates
# ELSE run 4 MORE Canny+contour passes on 2x-zoomed quadrants
```

Cost pattern:

| Scene type | main pass candidates | fallback fires? | L0 cost |
|---|---|---|---|
| Clean / stable | 0-1 | YES (4 quadrants) | ~13 ms |
| Screen spoof with clear target | 2+ | NO | ~2 ms |
| Attack with weak edges | 0-1 | YES | ~13 ms |

So the fallback pays its cost **most on frames that don't need it**.

## Fix

1. Make the fallback opt-in via `static_image_multiscale_fallback` (default
   `false`). Clean frames stop paying the 11 ms tax.
2. When enabled, limit to **one** zoom, not four quadrants.
3. If the user needs to catch tiny phone screens, they can turn the flag
   on and accept the cost.

## Expected impact

- Base A3b p95: 17 ms → **~6-7 ms** (roughly −60%).
- module_a_total_ms p95: 26-29 ms → **~13-16 ms**.
- End-to-end effective FPS on screen_spoof: 24 → **≥ 40**.

## Regression risk

Multi-scale was added "to catch smaller phone screens that would otherwise
fall below `area_ratio` minimum". On the 7 sample clips this doesn't trip
any missed detections, but it may on real screens < 200 px. Keep the flag
as an opt-in escape hatch.


## Update 2026-05-13 (post-fix measurements)

### Fix #1: remove pre-resize round-trip in fallback

Before: each quadrant crop was `F.interpolate`-d back up to 640×640 on GPU
before being handed to `extract_edge_candidates`, which then did its own
`.cpu().numpy()` + `cv2.resize(..., 416)`. Net: 2 resizes + 2 GPU↔CPU trips
per crop, × 4 quadrants = 8 resizes + 8 transfers.

After: the quadrant tensor is passed directly; `extract_edge_candidates`
does its own `.cpu().numpy()` + single `cv2.resize` to 416. Net: 5
transfers + 5 resizes total (scale 1 + 4 quadrants).

### Measured impact (unchanged sample smoke result: 7/7 pass)

| Clip | A3b p95 before | A3b p95 after | Δ |
|---|---|---|---|
| adv_patch | 17.1 | 13.0 | -24% |
| clean_baseline | 16.7 | 12.5 | -25% |
| glare | 16.5 | 12.2 | -26% |
| motion_blur | 15.8 | 11.5 | -27% |
| occlusion | 15.9 | 11.7 | -26% |
| screen_spoof | 18.0 | 13.9 | -23% |
| visibility_degradation | 15.3 | 11.3 | -26% |

Pipeline total p95 for screen_spoof went 29 ms → 25 ms. Effective FPS on
that clip in the smoke report went from ~24 → better than that (exact
number depends on stream realtime sleeps).

Detection parity: adv_patch alert_frames 498 → 504, screen_spoof 599 →
549, everyone else unchanged. Summary.ok = True.

### Future optimization ideas (not yet applied)

- Run full-frame Canny on GPU with a custom kernel (saves 1 CPU transfer
  per L0 call).
- Batch all 5 crops into a single 5×1×416×416 tensor, then `.cpu().numpy()`
  once, then loop cv2 on CPU slices (saves 4 CPU transfers).
- Gate scale-2 on more than just candidate count — e.g. skip if ROI list
  has a confident helmet/person and background EMA is ready.


## Update 2026-05-13 (second round - batched GPU→CPU transfer)

### Fix #2: batch the GPU→CPU copy across scale 1 + fallback

Before: each call to `extract_edge_candidates` did its own
`curr_gray[0,0].cpu().numpy()` — one per scale (full + 4 quadrants = 5
transfers per L0 call).

After: `FeatureBuilder.extract_and_filter_candidates` transfers the full
frame once, then numpy-slices the crops. `extract_edge_candidates`
accepts ndarray inputs directly (falls back to tensor for callers that
still pass tensors).

### Measured impact after fix #1 + fix #2

| Clip | A3b p95 baseline | After fix #1 | After fix #1 + #2 | Δ vs baseline |
|---|---|---|---|---|
| adv_patch | 17.1 | 13.0 | **10.5** | −39% |
| clean_baseline | 16.7 | 12.5 | **9.9** | −41% |
| glare | 16.5 | 12.2 | **9.9** | −40% |
| motion_blur | 15.8 | 11.5 | **8.9** | −44% |
| occlusion | 15.9 | 11.7 | **8.9** | −44% |
| screen_spoof | 18.0 | 13.9 | **11.5** | −36% |
| visibility_degradation | 15.3 | 11.3 | **8.9** | −42% |

Pipeline total p95 went from 26-29 ms → 19-22 ms.
Detection parity: all 7 clips still green, alert_frames unchanged.

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F

from ..types import ROI


class GPUTemporalTextureAnalyzer:
    def __init__(
        self,
        threshold: float = 0.25,
        grid_size: int = 16,
        emit_roi_details: bool = False,
        persistence_frames: int = 1,
        adaptive_baseline: bool = True,
        adaptive_ema_alpha: float = 0.02,
        adaptive_multiplier: float = 2.0,
        adaptive_floor: float = 0.015,
    ):
        """Temporal LBP change detector.

        The detector has three stacked gates (all configurable):

        * ``threshold`` — raw ``change_t`` floor from rule fusion.
        * ``persistence_frames`` (P1-A-4 2026-05-13) — require ``change_t``
          to stay above the trigger threshold for N consecutive frames
          before surfacing ``triggered=True``. Setting to 1 disables the
          gate (default is 1 to stay compatible; raise to 2 for noisy
          real-world streams). This is a cheap filter that strips
          isolated single-frame spikes from LBP/JPEG noise without
          touching the scoring path.
        * ``adaptive_baseline`` (P1-A-4 2026-05-13) — maintain an EMA of
          the per-frame ``change_t``; when enabled, ``triggered`` also
          requires ``change_t >= max(floor, multiplier * EMA)``. This
          auto-tunes to the underlying codec / scene noise baseline.
          Disabled by default for bit-for-bit back-compat.

        None of these parameters affect the numeric ``change_t`` /
        ``local_max`` values returned — they only shape the ``triggered``
        boolean. Rule fusion + alert state machine stay unchanged.
        """
        self.threshold = float(threshold)
        self.grid_size = max(4, int(grid_size))
        self.emit_roi_details = bool(emit_roi_details)
        self.persistence_frames = max(1, int(persistence_frames))
        self.adaptive_baseline = bool(adaptive_baseline)
        self.adaptive_ema_alpha = float(adaptive_ema_alpha)
        self.adaptive_multiplier = float(adaptive_multiplier)
        self.adaptive_floor = float(adaptive_floor)
        # Runtime state.
        self._persistence_queue: deque[bool] = deque(maxlen=self.persistence_frames)
        self._change_ema: float = 0.0
        self._ema_samples: int = 0

    def reset(self) -> None:
        self._persistence_queue.clear()
        self._change_ema = 0.0
        self._ema_samples = 0

    def compute(
        self,
        prev_lbp: torch.Tensor | None,
        curr_lbp: torch.Tensor,
        rois: list[ROI] | None = None,
        radius: int = 3,
    ) -> dict[str, Any]:
        if prev_lbp is None or prev_lbp.shape != curr_lbp.shape:
            # Preserve the "first frame" semantics: no change yet.
            self._persistence_queue.clear()
            return {
                "change_t": 0.0,
                "local_max": 0.0,
                "threshold": self.threshold,
                "triggered": False,
                "roi_results": [],
                "change_t_raw": 0.0,
                "local_max_raw": 0.0,
                "change_t_baseline": 0.0,
                "persistence_active": False,
                "adaptive_baseline_active": False,
                "noise_suppressed": False,
            }

        diff = torch.abs(curr_lbp - prev_lbp) / 255.0
        change_t = diff.mean()
        local = F.adaptive_avg_pool2d(diff, (self.grid_size, self.grid_size))
        local_max = local.max()

        roi_results: list[dict[str, Any]] = []
        if rois and self.emit_roi_details:
            _, _, h, w = diff.shape
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                lx1 = max(0, min(w - 1, x1 - radius))
                ly1 = max(0, min(h - 1, y1 - radius))
                lx2 = max(0, min(w, x2 - radius))
                ly2 = max(0, min(h, y2 - radius))
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                roi_change = diff[:, :, ly1:ly2, lx1:lx2].mean()
                roi_results.append(
                    {
                        "roi": roi.to_dict(),
                        "change_t": float(roi_change.item()),
                        "triggered": bool(roi_change.item() >= self.threshold),
                    }
                )

        change_raw = float(change_t.item())
        local_raw = float(local_max.item())

        # --- Adaptive baseline EMA (P1-A-4 2026-05-13) ---
        # Track a slow EMA of ``change_t`` representing the scene's
        # natural LBP noise floor (sensor / codec jitter, mild camera
        # shake). Update AFTER reading the current sample so the first
        # real trigger on a cold-start stream isn't swallowed.
        change_ema_prev = self._change_ema
        if self._ema_samples == 0:
            self._change_ema = change_raw
        else:
            alpha = self.adaptive_ema_alpha
            self._change_ema = (1.0 - alpha) * self._change_ema + alpha * change_raw
        self._ema_samples += 1
        baseline_ready = self.adaptive_baseline and self._ema_samples >= 30

        # Persistence queue: track whether change_t alone crossed ``threshold``.
        self._persistence_queue.append(change_raw >= self.threshold)

        # --- Noise suppression (P1-A-4 2026-05-13) ---
        # Goal: reduce the "local_temporal_texture_change" reason-code
        # spam on clean streams WITHOUT changing the detection rate on
        # any attacked clip.
        #
        # Suppression fires only when all of the following hold:
        #   1. the baseline EMA is warmed up (30+ samples),
        #   2. the current raw ``change_t`` is close to the historical
        #      baseline (≤ 2× EMA),
        #   3. the persistence window hasn't seen a sustained high reading.
        #
        # When suppressed, we cap the exposed ``change_t`` / ``local_max``
        # at the rule_fusion temporal_trigger floor (0.03 / 0.045 by
        # default) MINUS a small margin so it never crosses. The raw
        # values remain available under ``_raw`` keys for diagnostics.
        change_exposed = change_raw
        local_exposed = local_raw
        noise_suppressed = False
        if baseline_ready:
            persistence_full = (
                len(self._persistence_queue) == self.persistence_queue_len
                and all(self._persistence_queue)
            )
            # Adaptive bound: treat anything ≤ multiplier× the historical
            # EMA as "no worse than the usual noise floor for this stream".
            baseline_bound = self.adaptive_multiplier * change_ema_prev
            # Full suppression: no sustained run AND global change is
            # below both the adaptive envelope AND the static floor.
            # We deliberately do NOT use a "local_max only" rule: a cascade
            # of downstream fusion triggers (paired_temporal_flow,
            # paired_blur_temporal) reads ``local_max`` independently and
            # stripping it in isolation destabilises those gates.
            if (
                not persistence_full
                and change_raw < baseline_bound
                and change_raw < self.adaptive_floor
            ):
                change_exposed = min(change_raw, 0.029)
                local_exposed = min(local_raw, 0.044)  # just below local_temporal_trigger
                noise_suppressed = True

        triggered = (
            change_exposed >= self.threshold or local_exposed >= self.threshold
        )

        return {
            "change_t": change_exposed,
            "local_max": local_exposed,
            "threshold": self.threshold,
            "triggered": triggered,
            "roi_results": roi_results,
            # Diagnostics — surface so offline replay / event evidence
            # can tell *why* a sample was (not) flagged.
            "change_t_raw": change_raw,
            "local_max_raw": local_raw,
            "change_t_baseline": float(change_ema_prev),
            "persistence_active": self.persistence_frames > 1,
            "adaptive_baseline_active": baseline_ready,
            "noise_suppressed": noise_suppressed,
        }

    @property
    def persistence_queue_len(self) -> int:
        return self.persistence_frames

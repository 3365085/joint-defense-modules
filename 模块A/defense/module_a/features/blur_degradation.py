from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import ROI


class GPUBlurDegradationDetector:
    """Detect local high-frequency loss caused by blur or visibility degradation."""

    def __init__(
        self,
        roi_energy_ratio_trigger: float = 0.55,
        low_energy_ratio_trigger: float = 0.55,
        low_energy_global_factor: float = 0.35,
        min_roi_area: int = 900,
        emit_roi_details: bool = False,
    ):
        self.roi_energy_ratio_trigger = float(roi_energy_ratio_trigger)
        self.low_energy_ratio_trigger = float(low_energy_ratio_trigger)
        self.low_energy_global_factor = float(low_energy_global_factor)
        self.min_roi_area = max(64, int(min_roi_area))
        self.emit_roi_details = bool(emit_roi_details)
        self._kernel_device: torch.device | None = None
        self._laplacian: torch.Tensor | None = None

    def compute(self, gray: torch.Tensor, rois: list[ROI] | None = None) -> dict[str, Any]:
        self._ensure_kernel(gray.device)
        assert self._laplacian is not None
        normalized = gray.float() / 255.0
        energy = torch.abs(F.conv2d(normalized, self._laplacian, padding=1))
        global_mean_t = energy.mean()
        global_max_t = energy.max()
        global_mean = float(global_mean_t.item())
        global_max = float(global_max_t.item())
        low_threshold = max(1e-6, global_mean * self.low_energy_global_factor)

        # P1-A-7 optimisation 2026-05-13: batch the per-ROI GPU→CPU syncs.
        # The pre-optimisation loop called ``.item()`` twice per ROI
        # (``local.mean()`` and ``(local < threshold).mean()``), which at
        # ~36 ROIs × 2 = 72 syncs per frame dominated A3 cost on
        # detection-heavy scenes.
        #
        # We now compute all ROI reductions in GPU, stack them into
        # small vectors, and transfer the whole vectors in a single
        # ``.cpu()`` call per frame.
        roi_results: list[dict[str, Any]] = []
        best_score = 0.0
        best_ratio = 1.0
        best_low_ratio = 0.0
        best_roi_is_grid = False
        if rois:
            _, _, h, w = energy.shape
            means_t: list[torch.Tensor] = []
            low_ratios_t: list[torch.Tensor] = []
            eligible_rois: list[tuple[ROI, tuple[int, int, int, int]]] = []
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(0, min(w, int(x2)))
                y2 = max(0, min(h, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_roi_area:
                    continue
                local = energy[:, :, y1:y2, x1:x2]
                means_t.append(local.mean())
                low_ratios_t.append((local < low_threshold).float().mean())
                eligible_rois.append((roi, (x1, y1, x2, y2)))

            if means_t:
                # One CUDA→CPU sync for the entire frame's ROI means.
                means_vec = torch.stack(means_t).cpu().numpy()
                low_vec = torch.stack(low_ratios_t).cpu().numpy()
                for idx, (roi, _bbox) in enumerate(eligible_rois):
                    local_mean = float(means_vec[idx])
                    low_ratio = float(low_vec[idx])
                    ratio = local_mean / max(global_mean, 1e-6)
                    ratio_score = max(
                        0.0,
                        (self.roi_energy_ratio_trigger - ratio)
                        / max(self.roi_energy_ratio_trigger, 1e-6),
                    )
                    low_score = max(
                        0.0,
                        (low_ratio - self.low_energy_ratio_trigger)
                        / max(1.0 - self.low_energy_ratio_trigger, 1e-6),
                    )
                    score = min(1.0, max(ratio_score, low_score))
                    if score > best_score:
                        best_score = score
                        best_ratio = ratio
                        best_low_ratio = low_ratio
                        best_roi_is_grid = roi.label == "grid"
                    if self.emit_roi_details:
                        roi_results.append(
                            {
                                "roi": roi.to_dict(),
                                "sharpness_mean": local_mean,
                                "sharpness_ratio": ratio,
                                "low_energy_ratio": low_ratio,
                                "blur_score": score,
                            }
                        )

        return {
            "blur_score": best_score,
            "blur_roi_energy_ratio": best_ratio,
            "blur_low_energy_ratio": best_low_ratio,
            "blur_global_mean": global_mean,
            "blur_global_max": global_max,
            "blur_best_roi_is_grid": best_roi_is_grid,
            "backend": "gpu_laplacian_blur",
            "roi_results": roi_results,
        }

    def _ensure_kernel(self, device: torch.device) -> None:
        if self._kernel_device == device:
            return
        self._laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
        self._kernel_device = device

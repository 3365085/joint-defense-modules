from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import ROI


class GPUMotionArtifactDetector:
    def __init__(
        self, diff_threshold: float = 25.0, grid_size: int = 16, emit_roi_details: bool = False
    ):
        self.diff_threshold = float(diff_threshold)
        self.grid_size = max(4, int(grid_size))
        self.emit_roi_details = bool(emit_roi_details)

    def compute(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
        rois: list[ROI] | None = None,
    ) -> dict[str, Any]:
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return {
                "region_count": 0,
                "max_magnitude": 0.0,
                "local_max_ratio": 0.0,
                "motion_score": 0.0,
                "backend": "gpu_frame_diff",
                "roi_results": [],
            }

        diff = torch.abs(curr_gray - prev_gray)
        motion_mask = diff > self.diff_threshold
        pooled = F.avg_pool2d(motion_mask.float(), kernel_size=8, stride=8)
        region_count = int((pooled > 0.5).sum().item())
        max_magnitude = float(diff.max().item())

        local = F.adaptive_avg_pool2d(motion_mask.float(), (self.grid_size, self.grid_size))
        local_max_ratio = float(local.max().item())
        motion_score = min(1.0, region_count / 5.0)

        roi_results: list[dict[str, Any]] = []
        if rois and self.emit_roi_details:
            # P1-A-7 optimisation 2026-05-13: batch GPU→CPU syncs across
            # all ROIs. Previously this loop called ``.item()`` three times
            # per ROI (max / motion_ratio / mean), which is the same
            # pattern that dominated blur_degradation. See that file's
            # commentary for detailed reasoning.
            _, _, h, w = diff.shape
            eligible: list[tuple[ROI, torch.Tensor, torch.Tensor]] = []
            max_t: list[torch.Tensor] = []
            mean_t: list[torch.Tensor] = []
            mask_ratio_t: list[torch.Tensor] = []
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi_diff = diff[:, :, y1:y2, x1:x2]
                roi_mask = roi_diff > self.diff_threshold
                eligible.append((roi, roi_diff, roi_mask))
                max_t.append(roi_diff.max())
                mean_t.append(roi_diff.mean())
                mask_ratio_t.append(roi_mask.float().mean())

            if eligible:
                max_vec = torch.stack(max_t).cpu().numpy()
                mean_vec = torch.stack(mean_t).cpu().numpy()
                ratio_vec = torch.stack(mask_ratio_t).cpu().numpy()
                for idx, (roi, _diff, _mask) in enumerate(eligible):
                    roi_results.append(
                        {
                            "roi": roi.to_dict(),
                            "max_magnitude": float(max_vec[idx]),
                            "motion_ratio": float(ratio_vec[idx]),
                            "mean_magnitude": float(mean_vec[idx]),
                        }
                    )

        return {
            "region_count": region_count,
            "max_magnitude": max_magnitude,
            "local_max_ratio": local_max_ratio,
            "motion_score": motion_score,
            "backend": "gpu_frame_diff",
            "roi_results": roi_results,
        }

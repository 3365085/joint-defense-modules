from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import ROI


class GPULBPTextureAnalyzer:
    def __init__(self, radius: int = 3, grid_size: int = 16, emit_roi_details: bool = False):
        self.radius = max(1, int(radius))
        self.grid_size = max(4, int(grid_size))
        self.emit_roi_details = bool(emit_roi_details)

    def compute_lbp(self, gray: torch.Tensor) -> torch.Tensor:
        # gray: [1, 1, H, W], float32 on the target device.
        r = self.radius
        center = gray[:, :, r:-r, r:-r]
        offsets = (
            (-r, -r),
            (-r, 0),
            (-r, r),
            (0, r),
            (r, r),
            (r, 0),
            (r, -r),
            (0, -r),
        )
        code = torch.zeros_like(center)
        for bit, (dy, dx) in enumerate(offsets):
            y1 = r + dy
            x1 = r + dx
            neighbor = gray[:, :, y1 : y1 + center.shape[-2], x1 : x1 + center.shape[-1]]
            code = code + ((neighbor >= center).float() * float(1 << bit))
        return code

    def summarize(self, lbp: torch.Tensor, rois: list[ROI] | None = None) -> dict[str, Any]:
        normalized = lbp / 255.0
        global_mean = normalized.mean()
        global_sq_mean = (normalized * normalized).mean()
        global_std = torch.sqrt(torch.clamp(global_sq_mean - global_mean * global_mean, min=0.0))

        local_mean = F.adaptive_avg_pool2d(normalized, (self.grid_size, self.grid_size))
        local_sq = F.adaptive_avg_pool2d(normalized * normalized, (self.grid_size, self.grid_size))
        local_std = torch.sqrt(torch.clamp(local_sq - local_mean * local_mean, min=0.0))
        delta_map = torch.abs(local_mean - global_mean) + torch.abs(local_std - global_std)
        local_max = torch.clamp(delta_map.max() * 2.0, 0.0, 1.0)
        delta_h = torch.clamp(delta_map.mean() * 2.0, 0.0, 1.0)

        roi_results: list[dict[str, Any]] = []
        if rois and self.emit_roi_details:
            _, _, h, w = lbp.shape
            r = self.radius
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                lx1 = max(0, min(w - 1, x1 - r))
                ly1 = max(0, min(h - 1, y1 - r))
                lx2 = max(0, min(w, x2 - r))
                ly2 = max(0, min(h, y2 - r))
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                roi_lbp = normalized[:, :, ly1:ly2, lx1:lx2]
                roi_mean = roi_lbp.mean()
                roi_std = roi_lbp.std()
                roi_delta = torch.clamp(
                    (torch.abs(roi_mean - global_mean) + torch.abs(roi_std - global_std)) * 2.0,
                    0.0,
                    1.0,
                )
                roi_results.append(
                    {
                        "roi": roi.to_dict(),
                        "delta_h": float(roi_delta.item()),
                        "lbp_mean": float(roi_mean.item()),
                        "lbp_std": float(roi_std.item()),
                    }
                )

        return {
            "delta_h": float(delta_h.item()),
            "local_max": float(local_max.item()),
            "global_mean": float(global_mean.item()),
            "global_std": float(global_std.item()),
            "roi_results": roi_results,
        }

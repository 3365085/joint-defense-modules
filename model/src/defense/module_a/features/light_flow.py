from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..types import ROI


class GPULightOpticalFlowDetector:
    """Low-resolution Lucas-Kanade optical flow for Module A motion consistency."""

    def __init__(
        self,
        flow_size: int = 160,
        window_size: int = 9,
        grid_size: int = 16,
        residual_threshold: float = 0.75,
        min_magnitude: float = 0.25,
        cell_ratio_threshold: float = 0.20,
        score_region_normalizer: float = 8.0,
        emit_roi_details: bool = False,
    ):
        self.flow_size = max(64, int(flow_size))
        self.window_size = max(3, int(window_size))
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.grid_size = max(4, int(grid_size))
        self.residual_threshold = float(residual_threshold)
        self.min_magnitude = float(min_magnitude)
        self.cell_ratio_threshold = float(cell_ratio_threshold)
        self.score_region_normalizer = max(1.0, float(score_region_normalizer))
        self.emit_roi_details = bool(emit_roi_details)
        self._kernel_device: torch.device | None = None
        self._sobel_x: torch.Tensor | None = None
        self._sobel_y: torch.Tensor | None = None

    def compute(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
        rois: list[ROI] | None = None,
        run: bool = True,
    ) -> dict[str, Any]:
        if not run:
            return self._empty("gpu_lk_lite_skipped")
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return self._empty("gpu_lk_lite")

        device = curr_gray.device
        self._ensure_kernels(device)
        assert self._sobel_x is not None
        assert self._sobel_y is not None

        prev_small = self._resize_gray(prev_gray)
        curr_small = self._resize_gray(curr_gray)
        avg = (prev_small + curr_small) * 0.5
        ix = F.conv2d(avg, self._sobel_x, padding=1)
        iy = F.conv2d(avg, self._sobel_y, padding=1)
        it = curr_small - prev_small

        pad = self.window_size // 2
        s_xx = F.avg_pool2d(ix * ix, self.window_size, stride=1, padding=pad)
        s_xy = F.avg_pool2d(ix * iy, self.window_size, stride=1, padding=pad)
        s_yy = F.avg_pool2d(iy * iy, self.window_size, stride=1, padding=pad)
        s_xt = F.avg_pool2d(ix * it, self.window_size, stride=1, padding=pad)
        s_yt = F.avg_pool2d(iy * it, self.window_size, stride=1, padding=pad)

        det = s_xx * s_yy - s_xy * s_xy
        valid = det > 1e-5
        safe_det = torch.where(valid, det, torch.ones_like(det))
        flow_u = (-s_yy * s_xt + s_xy * s_yt) / safe_det
        flow_v = (s_xy * s_xt - s_xx * s_yt) / safe_det
        flow_u = torch.where(valid, torch.clamp(flow_u, -16.0, 16.0), torch.zeros_like(flow_u))
        flow_v = torch.where(valid, torch.clamp(flow_v, -16.0, 16.0), torch.zeros_like(flow_v))
        magnitude = torch.sqrt(flow_u * flow_u + flow_v * flow_v + 1e-8)

        motion_valid = valid & (magnitude >= self.min_magnitude)
        weights = motion_valid.float()
        weight_sum = weights.sum().clamp_min(1.0)
        dominant_u = (flow_u * weights).sum() / weight_sum
        dominant_v = (flow_v * weights).sum() / weight_sum
        residual = torch.sqrt(
            (flow_u - dominant_u) * (flow_u - dominant_u)
            + (flow_v - dominant_v) * (flow_v - dominant_v)
            + 1e-8
        )

        anomaly_mask = motion_valid & (residual >= self.residual_threshold)
        pooled = F.adaptive_avg_pool2d(anomaly_mask.float(), (self.grid_size, self.grid_size))
        # P1-A-7 optimisation 2026-05-13: batch the 7 scalar extractions
        # into a single CUDA→CPU sync. The individual computations all run
        # on the same CUDA stream, so stacking and transferring once means
        # we pay one sync instead of seven.
        dominant_magnitude_t = torch.sqrt(
            dominant_u * dominant_u + dominant_v * dominant_v + 1e-8
        )
        scalars = torch.stack(
            [
                (pooled >= self.cell_ratio_threshold).sum().float(),  # region_count
                pooled.max(),  # local_anomaly_ratio
                motion_valid.float().mean(),  # valid_ratio
                magnitude.max(),  # max_magnitude
                magnitude.mean(),  # mean_magnitude
                dominant_magnitude_t,  # dominant_magnitude
                residual.max(),  # max_residual
            ]
        ).cpu().numpy()
        region_count = int(scalars[0])
        local_anomaly_ratio = float(scalars[1])
        valid_ratio = float(scalars[2])
        light_flow_max_magnitude = float(scalars[3])
        light_flow_mean_magnitude = float(scalars[4])
        light_flow_dominant_magnitude = float(scalars[5])
        light_flow_max_residual = float(scalars[6])
        score = min(1.0, region_count / self.score_region_normalizer)

        roi_results: list[dict[str, Any]] = []
        if rois and self.emit_roi_details:
            # Batch per-ROI syncs as well when details are requested.
            _, _, h, w = anomaly_mask.shape
            scale_x = w / float(curr_gray.shape[-1])
            scale_y = h / float(curr_gray.shape[-2])
            eligible: list[ROI] = []
            anom_t: list[torch.Tensor] = []
            max_r_t: list[torch.Tensor] = []
            mean_r_t: list[torch.Tensor] = []
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                sx1 = max(0, min(w - 1, int(x1 * scale_x)))
                sy1 = max(0, min(h - 1, int(y1 * scale_y)))
                sx2 = max(0, min(w, int(x2 * scale_x)))
                sy2 = max(0, min(h, int(y2 * scale_y)))
                if sx2 <= sx1 or sy2 <= sy1:
                    continue
                roi_residual = residual[:, :, sy1:sy2, sx1:sx2]
                roi_anomaly = anomaly_mask[:, :, sy1:sy2, sx1:sx2]
                anom_t.append(roi_anomaly.float().mean())
                max_r_t.append(roi_residual.max())
                mean_r_t.append(roi_residual.mean())
                eligible.append(roi)
            if eligible:
                anom_vec = torch.stack(anom_t).cpu().numpy()
                max_vec = torch.stack(max_r_t).cpu().numpy()
                mean_vec = torch.stack(mean_r_t).cpu().numpy()
                for idx, roi in enumerate(eligible):
                    roi_results.append(
                        {
                            "roi": roi.to_dict(),
                            "anomaly_ratio": float(anom_vec[idx]),
                            "max_residual": float(max_vec[idx]),
                            "mean_residual": float(mean_vec[idx]),
                        }
                    )

        return {
            "light_flow_available": True,
            "light_flow_region_count": region_count,
            "light_flow_max_magnitude": light_flow_max_magnitude,
            "light_flow_mean_magnitude": light_flow_mean_magnitude,
            "light_flow_dominant_magnitude": light_flow_dominant_magnitude,
            "light_flow_max_residual": light_flow_max_residual,
            "light_flow_local_anomaly_ratio": local_anomaly_ratio,
            "light_flow_score": score,
            "light_flow_valid_ratio": valid_ratio,
            "light_flow_backend": "gpu_lk_lite",
            "light_flow_roi_results": roi_results,
        }

    def _resize_gray(self, gray: torch.Tensor) -> torch.Tensor:
        gray = gray.float() / 255.0
        return F.interpolate(
            gray, size=(self.flow_size, self.flow_size), mode="bilinear", align_corners=False
        )

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

    @staticmethod
    def _empty(backend: str) -> dict[str, Any]:
        return {
            "light_flow_available": False,
            "light_flow_region_count": 0,
            "light_flow_max_magnitude": 0.0,
            "light_flow_mean_magnitude": 0.0,
            "light_flow_dominant_magnitude": 0.0,
            "light_flow_max_residual": 0.0,
            "light_flow_local_anomaly_ratio": 0.0,
            "light_flow_score": 0.0,
            "light_flow_valid_ratio": 0.0,
            "light_flow_backend": backend,
            "light_flow_roi_results": [],
        }

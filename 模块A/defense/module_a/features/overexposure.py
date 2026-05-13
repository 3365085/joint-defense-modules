from __future__ import annotations

import torch


class GPUOverexposureDetector:
    def __init__(self, threshold: float = 0.06):
        self.threshold = float(threshold)

    def compute(self, gray: torch.Tensor) -> dict[str, float | bool]:
        over_ratio = torch.mean((gray >= 250.0).float())
        under_ratio = torch.mean((gray <= 5.0).float())
        ratio = float(over_ratio.item())
        return {
            "ratio": ratio,
            "underexposed_ratio": float(under_ratio.item()),
            "is_glare": ratio >= self.threshold,
            "threshold": self.threshold,
        }

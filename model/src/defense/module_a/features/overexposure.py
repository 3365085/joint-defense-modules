from __future__ import annotations

import torch


class GPUOverexposureDetector:
    """GPU-based overexposure / glare detector with temporal flash detection.

    Detects two types of brightness anomaly:
      1. Static glare: large fraction of pixels >= 250 (pure white).
      2. Temporal flash: large inter-frame brightness jump (frame-to-frame
         difference), capturing dynamic flash / flicker attacks that don't
         necessarily reach pure-white levels.

    The temporal branch uses a simple per-pixel absolute difference between
    consecutive grayscale frames.  A frame with many pixels changing by >= 30
    gray levels (out of 255) signals an abrupt lighting change consistent
    with glare attacks, laser flashes, or strobe effects.
    """

    def __init__(
        self,
        threshold: float = 0.06,
        # Temporal flash detection
        flash_diff_threshold: float = 30.0,
        flash_ratio_threshold: float = 0.08,
        flash_min_polarity: float = 0.35,
        flash_min_abs_mean: float = 8.0,
    ):
        self.threshold = float(threshold)
        self.flash_diff_threshold = float(flash_diff_threshold)
        self.flash_ratio_threshold = float(flash_ratio_threshold)
        self.flash_min_polarity = float(flash_min_polarity)
        self.flash_min_abs_mean = float(flash_min_abs_mean)
        self._prev_gray: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def compute(self, gray: torch.Tensor) -> dict[str, float | bool]:
        over_ratio = torch.mean((gray >= 250.0).float())
        under_ratio = torch.mean((gray <= 5.0).float())
        ratio = float(over_ratio.item())

        # Static glare: large fraction of pure-white pixels
        static_glare = ratio >= self.threshold

        # Temporal flash: inter-frame brightness jump
        temporal_flash_ratio = 0.0
        flash_bright_ratio = 0.0
        flash_dark_ratio = 0.0
        flash_polarity = 0.0
        flash_abs_mean = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            signed_diff = gray.float() - self._prev_gray.float()
            diff = torch.abs(signed_diff)
            flash_mask = diff >= self.flash_diff_threshold
            bright_mask = signed_diff >= self.flash_diff_threshold
            dark_mask = signed_diff <= -self.flash_diff_threshold
            temporal_flash_ratio = float(torch.mean(flash_mask.float()).item())
            flash_bright_ratio = float(torch.mean(bright_mask.float()).item())
            flash_dark_ratio = float(torch.mean(dark_mask.float()).item())
            flash_abs_mean = float(diff.mean().item())
            flash_polarity = abs(flash_bright_ratio - flash_dark_ratio) / max(
                temporal_flash_ratio, 1e-6
            )
        self._prev_gray = gray.clone()

        temporal_flash = (
            temporal_flash_ratio >= self.flash_ratio_threshold
            and flash_polarity >= self.flash_min_polarity
            and flash_abs_mean >= self.flash_min_abs_mean
        )
        is_glare = static_glare or temporal_flash

        return {
            "ratio": ratio,
            "underexposed_ratio": float(under_ratio.item()),
            "is_glare": is_glare,
            "static_glare": static_glare,
            "temporal_flash": temporal_flash,
            "temporal_flash_ratio": temporal_flash_ratio,
            "temporal_flash_bright_ratio": flash_bright_ratio,
            "temporal_flash_dark_ratio": flash_dark_ratio,
            "temporal_flash_polarity": flash_polarity,
            "temporal_flash_abs_mean": flash_abs_mean,
            "threshold": self.threshold,
        }

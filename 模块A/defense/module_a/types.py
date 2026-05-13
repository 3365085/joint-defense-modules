from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ROI:
    roi_id: str
    bbox: tuple[int, int, int, int]
    label: str | None = None
    confidence: float | None = None

    def clipped(self, width: int, height: int, min_size: int = 8) -> ROI | None:
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return None
        return ROI(
            roi_id=self.roi_id,
            bbox=(x1, y1, x2, y2),
            label=self.label,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "bbox": [int(v) for v in self.bbox],
            "label": self.label,
            "confidence": None if self.confidence is None else float(self.confidence),
        }


@dataclass(slots=True)
class ModuleAInput:
    frame: np.ndarray
    frame_idx: int
    timestamp: float = 0.0
    rois: list[ROI] | None = None


@dataclass(slots=True)
class ModuleAResult:
    frame_idx: int
    p_adv: float
    single_frame_suspicious: bool
    alert_confirmed: bool
    attack_state_active: bool
    reason_codes: list[str]
    features: dict[str, Any]
    roi_results: list[dict[str, Any]] = field(default_factory=list)
    attack_mask: np.ndarray | None = None
    timing_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_info_dict(self) -> dict[str, Any]:
        layer = "MODULE_A_PHYSICAL" if self.single_frame_suspicious else "NORMAL"
        return {
            "layer_triggered": layer,
            "is_attack": self.single_frame_suspicious,
            "attack_detected": self.single_frame_suspicious,
            "attack_state_active": self.attack_state_active,
            "attack_state_source": "module_a" if self.attack_state_active else "none",
            "attack_state_remaining": 0,
            "attack_state_last_layer": layer,
            "alert_confirmed": self.alert_confirmed,
            "timing_ms": self.timing_ms,
            "details": self.details,
        }

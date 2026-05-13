from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import ROI


_DEFAULT_CLASS_ALIASES = {
    "human": "person",
    "worker": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "person_head": "head",
    "bare_head": "head",
    "no_helmet": "head",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "safety_helmet": "helmet",
    "hat": "helmet",
}


class DetectionROIProvider:
    """Convert detector boxes into normalized Module A target ROIs."""

    def __init__(
        self,
        class_names: dict[int, str],
        min_confidence: float = 0.25,
        margin: int = 8,
        *,
        class_aliases: Mapping[str, str] | None = None,
        target_labels: Sequence[str] | None = None,
    ):
        self.class_names = class_names
        self.min_confidence = float(min_confidence)
        self.margin = int(margin)
        aliases = dict(_DEFAULT_CLASS_ALIASES)
        if class_aliases:
            aliases.update({str(k).lower(): str(v).lower() for k, v in class_aliases.items()})
        self.class_aliases = aliases
        self.target_labels = (
            {str(v).lower() for v in target_labels}
            if target_labels is not None
            else set()
        )

    def normalize_label(self, label: str) -> str:
        normalized = str(label).strip().lower().replace(" ", "_").replace("-", "_")
        return self.class_aliases.get(normalized, normalized)

    def from_detections(
        self, boxes: list[list[int]], classes: list[int], confs: list[float]
    ) -> list[ROI]:
        rois: list[ROI] = []
        for idx, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
            if float(conf) < self.min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            raw_label = self.class_names.get(int(cls_id), f"class_{int(cls_id)}")
            label = self.normalize_label(raw_label)
            if self.target_labels and label not in self.target_labels:
                continue
            rois.append(
                ROI(
                    roi_id=f"det_{idx}_{label}",
                    bbox=(x1 - self.margin, y1 - self.margin, x2 + self.margin, y2 + self.margin),
                    label=label,
                    confidence=float(conf),
                )
            )
        return rois

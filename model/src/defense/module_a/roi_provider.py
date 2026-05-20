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
        stabilize_overlaps: bool = False,
        same_label_iou: float = 0.55,
        head_helmet_iou: float = 0.20,
        head_helmet_center_distance: float = 0.05,
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
        self.stabilize_overlaps = bool(stabilize_overlaps)
        self.same_label_iou = float(same_label_iou)
        self.head_helmet_iou = float(head_helmet_iou)
        self.head_helmet_center_distance = float(head_helmet_center_distance)

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
        if self.stabilize_overlaps:
            rois = self._stabilize_rois(rois)
        return rois

    def _stabilize_rois(self, rois: list[ROI]) -> list[ROI]:
        if len(rois) <= 1:
            return rois
        kept: list[ROI] = []
        for roi in sorted(rois, key=lambda item: float(item.confidence or 0.0), reverse=True):
            suppress = False
            for kept_roi in kept:
                iou = self._iou(roi.bbox, kept_roi.bbox)
                if roi.label == kept_roi.label and iou >= self.same_label_iou:
                    suppress = True
                    break
                labels = {str(roi.label), str(kept_roi.label)}
                if labels <= {"head", "helmet"}:
                    distance = self._center_distance_ratio(roi.bbox, kept_roi.bbox)
                    if iou >= self.head_helmet_iou or distance <= self.head_helmet_center_distance:
                        suppress = True
                        break
            if not suppress:
                kept.append(roi)
        return sorted(kept, key=lambda item: item.roi_id)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else float(inter / union)

    @staticmethod
    def _center_distance_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5
        scale = max(1.0, max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1))
        return float((((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale)

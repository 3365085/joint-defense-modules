from __future__ import annotations

"""Run a YOLO model over OC3 witness records and produce candidate-box energy
samples in the OC3Witness format.

This bridges the gap between :mod:`oc3_witness_builder` (which writes
images/labels) and :mod:`oc3_detox` (which expects ``OC3Witness`` objects
with energies).  No training; inference only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json

import numpy as np

from model_security_gate.detox.oc3.oc3_detox import CandidateBox, OC3Witness, iou_xyxy
from model_security_gate.utils.io import read_yolo_labels


@dataclass(frozen=True)
class EvalConfig:
    """Inference knobs for OC3 witness evaluation."""

    imgsz: int = 416
    conf: float = 0.01  # collect low-confidence candidates too; OC3 caps via target_score_cap
    iou: float = 0.7
    near_object_iou: float = 0.30
    max_candidates_per_image: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "near_object_iou": self.near_object_iou,
            "max_candidates_per_image": self.max_candidates_per_image,
        }


def _label_dict_to_xyxy(lab: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in lab["xyxy"])


def _classify_candidate(
    box: Sequence[float],
    target_bboxes: Sequence[Sequence[float]],
    *,
    near_iou: float,
) -> str:
    """Classify a YOLO detection as object_evidence (overlapping a target GT)
    or context_evidence (not overlapping any target GT).
    """

    if not target_bboxes:
        return "context"
    best = max((iou_xyxy(box, t) for t in target_bboxes), default=0.0)
    return "object" if best >= float(near_iou) else "context"


def _select_top(detections: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    return sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)[: int(k)]


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target_class_ids: Sequence[int],
    detection_runner,
    config: EvalConfig | None = None,
) -> list[OC3Witness]:
    """Run a detection runner on each witness record and return OC3Witness rows.

    ``detection_runner`` is a callable ``(image_path, conf, iou, imgsz) ->
    list[{"bbox": (x1,y1,x2,y2), "score": float, "class_id": int,
    "objectness": float}]``.  This indirection keeps the module testable
    without ultralytics.
    """

    cfg = config or EvalConfig()
    target_set = {int(c) for c in target_class_ids}
    paired_object_energies: dict[str, list[float]] = {}
    out: list[OC3Witness] = []

    # First pass: detect on every record so paired energies are available.
    detections_by_image: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        img = rec["image_path"]
        if img in detections_by_image:
            continue
        dets = detection_runner(img, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
        # Only keep target-class detections; filter by target_class_ids.
        keep: list[dict[str, Any]] = []
        for d in dets:
            cid = d.get("class_id")
            if cid is None or int(cid) in target_set:
                keep.append(d)
        keep = _select_top(keep, cfg.max_candidates_per_image)
        detections_by_image[img] = keep

    # Pre-compute object-present energies per base image (used as reference
    # for object_sufficiency on paired witnesses).
    for rec in records:
        if rec.get("witness_type") != "object_present":
            continue
        dets = detections_by_image.get(rec["image_path"], [])
        target_bboxes = [list(b) for b in rec.get("object_bboxes_xyxy", [])]
        energies: list[float] = []
        for d in dets:
            kind = _classify_candidate(d["bbox"], target_bboxes, near_iou=cfg.near_object_iou)
            if kind == "object":
                energies.append(float(d.get("score", 0.0)) * float(d.get("objectness", 1.0)))
        paired_object_energies[rec["base_image_id"]] = energies or [0.0]

    # Second pass: build OC3Witness rows.
    for rec in records:
        wtype = str(rec.get("witness_type", "generic"))
        base_id = str(rec.get("base_image_id", ""))
        target_bboxes_seq = [list(b) for b in rec.get("object_bboxes_xyxy", [])]
        dets = detections_by_image.get(rec["image_path"], [])
        obj_cands: list[CandidateBox] = []
        ctx_cands: list[CandidateBox] = []
        for d in dets:
            box = tuple(float(v) for v in d["bbox"])
            score = float(d.get("score", 0.0))
            objness = float(d.get("objectness", 1.0))
            cid = int(d["class_id"]) if d.get("class_id") is not None else None
            cb = CandidateBox(bbox=box, score=score, objectness=objness, class_id=cid, source=wtype)
            kind = _classify_candidate(box, target_bboxes_seq, near_iou=cfg.near_object_iou)
            if kind == "object":
                obj_cands.append(cb)
            else:
                ctx_cands.append(cb)

        ref_energies = tuple(paired_object_energies.get(base_id, ()))
        transformed_energies: tuple[float, ...] = ()
        if wtype in {"geometry_pair", "frequency_pair"} and obj_cands:
            transformed_energies = tuple(c.energy() for c in obj_cands)

        out.append(
            OC3Witness(
                sample_id=f"{base_id}::{wtype}",
                attack_family=rec.get("attack_family", "unknown"),
                witness_type=wtype,
                object_candidates=tuple(obj_cands),
                context_candidates=tuple(ctx_cands),
                reference_object_energies=ref_energies,
                transformed_object_energies=transformed_energies,
                metadata={
                    "image_path": rec["image_path"],
                    "label_path": rec.get("label_path", ""),
                    "n_target_gt": len(target_bboxes_seq),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ultralytics runner (optional, for production GPU runs)
# ---------------------------------------------------------------------------


class UltralyticsRunner:
    """Detection runner backed by an ultralytics YOLO model.

    Lazy-imports ultralytics so the module can be imported in CPU/CI without
    pulling cuda/torchvision.  Calling the instance dispatches a single-image
    prediction and returns the dict shape expected by ``evaluate_records``.
    """

    def __init__(self, model_path: str, *, device: str | int | None = None) -> None:
        from ultralytics import YOLO  # lazy

        self._model = YOLO(str(model_path))
        self._device = device

    def __call__(
        self,
        image_path: str,
        *,
        conf: float = 0.01,
        iou: float = 0.7,
        imgsz: int = 416,
    ) -> list[dict[str, Any]]:
        result = self._model.predict(
            source=str(image_path),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf_arr = boxes.conf.detach().cpu().numpy()
        cls_arr = boxes.cls.detach().cpu().numpy()
        out: list[dict[str, Any]] = []
        for i in range(len(boxes)):
            out.append(
                {
                    "bbox": (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
                    "score": float(conf_arr[i]),
                    "objectness": 1.0,
                    "class_id": int(cls_arr[i]),
                }
            )
        return out


def write_witness_inference_json(
    witnesses: Sequence[OC3Witness],
    out_path: str | Path,
) -> None:
    """Persist OC3 witnesses (with computed energies) for downstream loss audit."""

    data = {
        "n": len(witnesses),
        "witnesses": [
            {
                "sample_id": w.sample_id,
                "attack_family": w.attack_family,
                "witness_type": w.witness_type,
                "object_candidates": [c.to_dict() for c in w.object_candidates],
                "context_candidates": [c.to_dict() for c in w.context_candidates],
                "reference_object_energies": list(w.reference_object_energies),
                "transformed_object_energies": list(w.transformed_object_energies),
                "metadata": dict(w.metadata or {}),
            }
            for w in witnesses
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

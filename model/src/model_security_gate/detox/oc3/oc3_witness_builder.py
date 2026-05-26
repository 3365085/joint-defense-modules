from __future__ import annotations

"""OC3-Detox witness builder.

Generates the four canonical OC3 witness types from a YOLO image+label
dataset, **without training** any model. Each witness shares the same
``base_image_id`` so downstream CFRC-style evaluators can cluster rows by
base image.

Witness types
-------------
- ``object_present``: original image, real object intact. Reference for
  ``object sufficiency``.
- ``context_only``: real objects (target class) erased and inpainted using
  the surrounding background (telea inpainting). The trigger / context /
  vest stays. This anchors ``context insufficiency`` and
  ``object necessity``.
- ``object_erased``: same as ``context_only`` but the intent label switches
  to "removed object should not be detected" (kept separate so the loss
  weights can differ per stage).
- ``object_transplant``: object cropped from its native scene and pasted
  onto a different background image. Real-object evidence preserved on a
  new context. Anchors ``object sufficiency`` away from background bias.
- ``geometry_pair``: the same image under a smooth elastic warp (WaNet-like)
  paired with the original. Both targets are paired by index for
  ``transform_consistency`` constraints.
- ``frequency_pair``: low-frequency additive perturbation paired with the
  original. Same structure as ``geometry_pair``, different physical mode.

The builder is **deterministic** (numpy seeded) and **CPU-only** so it can
run in CI without CUDA.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import shutil

import cv2
import numpy as np

from model_security_gate.utils.io import (
    IMAGE_EXTS,
    label_path_for_image,
    read_image_bgr,
    read_yolo_labels,
    write_image,
    write_yolo_labels,
)


WITNESS_TYPES = (
    "object_present",
    "context_only",
    "object_erased",
    "object_transplant",
    "geometry_pair",
    "frequency_pair",
)


@dataclass(frozen=True)
class WitnessSpec:
    """Knobs for the deterministic witness builder."""

    inpaint_radius: int = 5
    inpaint_method: str = "telea"  # 'telea' or 'ns'
    geometry_grid_k: int = 4
    geometry_warp_strength: float = 4.0
    frequency_amplitude: float = 8.0 / 255.0
    frequency_period: float = 30.0
    transplant_pad: float = 0.05  # extra pad around object before paste
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WitnessRecord:
    base_image_id: str
    witness_type: str
    image_path: str
    label_path: str
    paired_image_path: str | None = None
    target_present_in_image: bool = False
    target_class_ids: tuple[int, ...] = ()
    object_bboxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_image_id": self.base_image_id,
            "witness_type": self.witness_type,
            "image_path": self.image_path,
            "label_path": self.label_path,
            "paired_image_path": self.paired_image_path,
            "target_present_in_image": bool(self.target_present_in_image),
            "target_class_ids": list(self.target_class_ids),
            "object_bboxes_xyxy": [list(b) for b in self.object_bboxes_xyxy],
            "notes": self.notes,
        }


@dataclass
class WitnessManifest:
    out_root: str
    target_class_ids: list[int]
    spec: WitnessSpec
    records: list[WitnessRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_root": self.out_root,
            "target_class_ids": list(self.target_class_ids),
            "spec": self.spec.to_dict(),
            "records": [r.to_dict() for r in self.records],
        }


# ---------------------------------------------------------------------------
# core image transforms
# ---------------------------------------------------------------------------


def _inpaint(img: np.ndarray, mask: np.ndarray, *, radius: int, method: str) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    flag = cv2.INPAINT_TELEA if str(method).lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, int(radius), flag)


def _xyxy_to_int(xyxy: Sequence[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return x1, y1, x2, y2


def _erase_objects_inpaint(
    img: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    *,
    spec: WitnessSpec,
) -> np.ndarray:
    if not bboxes:
        return img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = _xyxy_to_int(b, w, h)
        mask[y1:y2, x1:x2] = 255
    return _inpaint(img, mask, radius=spec.inpaint_radius, method=spec.inpaint_method)


def _smooth_warp(img: np.ndarray, *, spec: WitnessSpec) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(spec.seed)
    k = max(2, int(spec.geometry_grid_k))
    grid_x = rng.uniform(-1.0, 1.0, size=(k + 1, k + 1)).astype(np.float32)
    grid_y = rng.uniform(-1.0, 1.0, size=(k + 1, k + 1)).astype(np.float32)
    map_x = cv2.resize(grid_x, (w, h), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(grid_y, (w, h), interpolation=cv2.INTER_CUBIC)
    map_x *= float(spec.geometry_warp_strength)
    map_y *= float(spec.geometry_warp_strength)
    base_x, base_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        img,
        (base_x + map_x).astype(np.float32),
        (base_y + map_y).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _frequency_perturb(img: np.ndarray, *, spec: WitnessSpec) -> np.ndarray:
    h, w = img.shape[:2]
    yy = np.arange(h, dtype=np.float32)[:, None]
    delta = (np.sin(2.0 * math.pi * yy / float(spec.frequency_period)) * 255.0 * float(spec.frequency_amplitude))
    delta = np.broadcast_to(delta, (h, w))
    out = img.astype(np.float32) + delta[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _transplant(
    object_img: np.ndarray,
    object_bbox_xyxy: Sequence[float],
    background_img: np.ndarray,
    *,
    spec: WitnessSpec,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Crop the object (with pad) from ``object_img`` and paste it on a random
    location of ``background_img`` (deterministic by ``spec.seed``).

    Returns the composited image and the new xyxy bbox in the background's
    coordinate system."""

    rng = np.random.default_rng(spec.seed + 1)
    sh, sw = object_img.shape[:2]
    bh, bw = background_img.shape[:2]
    x1, y1, x2, y2 = _xyxy_to_int(object_bbox_xyxy, sw, sh)
    pad_w = int(round((x2 - x1) * float(spec.transplant_pad)))
    pad_h = int(round((y2 - y1) * float(spec.transplant_pad)))
    px1 = max(0, x1 - pad_w)
    py1 = max(0, y1 - pad_h)
    px2 = min(sw, x2 + pad_w)
    py2 = min(sh, y2 + pad_h)
    crop = object_img[py1:py2, px1:px2].copy()
    ch, cw = crop.shape[:2]
    if ch < 4 or cw < 4 or ch > bh - 2 or cw > bw - 2:
        # Fallback: use background as-is (no transplant) and report no
        # paste (caller can drop this transplant witness if cw/ch invalid).
        return background_img.copy(), (0.0, 0.0, 0.0, 0.0)
    nx1 = int(rng.integers(low=0, high=max(1, bw - cw)))
    ny1 = int(rng.integers(low=0, high=max(1, bh - ch)))
    out = background_img.copy()
    out[ny1 : ny1 + ch, nx1 : nx1 + cw] = crop
    new_xyxy = (float(nx1), float(ny1), float(nx1 + cw), float(ny1 + ch))
    return out, new_xyxy


# ---------------------------------------------------------------------------
# label helpers
# ---------------------------------------------------------------------------


def _filter_target_labels(
    labels: Sequence[Mapping[str, Any]],
    target_class_ids: Sequence[int],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    target_set = {int(c) for c in target_class_ids}
    target = [lab for lab in labels if int(lab["cls_id"]) in target_set]
    other = [lab for lab in labels if int(lab["cls_id"]) not in target_set]
    return target, other


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------


def build_witnesses(
    source_root: str | Path,
    out_root: str | Path,
    *,
    target_class_ids: Sequence[int],
    max_base_images: int | None = None,
    spec: WitnessSpec | None = None,
    background_pool_root: str | Path | None = None,
) -> WitnessManifest:
    """Generate OC3 witness images and labels deterministically.

    Parameters
    ----------
    source_root : path
        Directory with ``images/`` and ``labels/`` (YOLO format) where each
        image has at least one target-class label. Images without target
        labels are skipped (they don't anchor object sufficiency).
    out_root : path
        Output directory. Per-type subdirs are created.
    target_class_ids : iterable of int
        Class ids that count as "target object". For helmet detection use
        ``[0]``; for helmet+head ``[0, 1]``.
    max_base_images : int, optional
        Cap the number of source images per witness type. ``None`` =
        unlimited.
    spec : WitnessSpec, optional
        Knobs for inpainting, warp, frequency, transplant.
    background_pool_root : path, optional
        Directory of background/no-target images to use for
        ``object_transplant``. If ``None``, uses the same source images as
        backgrounds (still valid; the transplant just lands on a different
        scene of the same dataset).
    """

    spec = spec or WitnessSpec()
    src = Path(source_root)
    out = Path(out_root)
    img_src_dir = src / "images" if (src / "images").exists() else src
    images = sorted([p for p in img_src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

    bg_pool: list[Path]
    if background_pool_root:
        bg_root = Path(background_pool_root)
        bg_dir = bg_root / "images" if (bg_root / "images").exists() else bg_root
        bg_pool = sorted([p for p in bg_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    else:
        bg_pool = list(images)

    if max_base_images is not None and int(max_base_images) > 0:
        images = images[: int(max_base_images)]

    manifest = WitnessManifest(
        out_root=str(out),
        target_class_ids=list(int(c) for c in target_class_ids),
        spec=spec,
    )
    out.mkdir(parents=True, exist_ok=True)
    for sub in WITNESS_TYPES:
        (out / sub / "images").mkdir(parents=True, exist_ok=True)
        (out / sub / "labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    target_set = {int(c) for c in target_class_ids}

    for src_img_path in images:
        img = read_image_bgr(src_img_path)
        labels = read_yolo_labels(src_img_path, img.shape)
        target_labels, other_labels = _filter_target_labels(labels, target_set)
        if not target_labels:
            continue
        base_id = src_img_path.stem
        target_xyxys = tuple(tuple(float(v) for v in lab["xyxy"]) for lab in target_labels)
        target_cids = tuple(int(lab["cls_id"]) for lab in target_labels)

        # 1. object_present (copy)
        op_img = out / "object_present" / "images" / f"{base_id}.jpg"
        op_lab = out / "object_present" / "labels" / f"{base_id}.txt"
        cv2.imwrite(str(op_img), img)
        write_yolo_labels(op_lab, labels, img.shape)
        manifest.records.append(
            WitnessRecord(
                base_image_id=base_id,
                witness_type="object_present",
                image_path=str(op_img),
                label_path=str(op_lab),
                target_present_in_image=True,
                target_class_ids=target_cids,
                object_bboxes_xyxy=target_xyxys,
                notes="real target evidence retained",
            )
        )

        # 2. context_only / object_erased (target inpainted out)
        erased = _erase_objects_inpaint(img, [lab["xyxy"] for lab in target_labels], spec=spec)
        co_img = out / "context_only" / "images" / f"{base_id}.jpg"
        co_lab = out / "context_only" / "labels" / f"{base_id}.txt"
        cv2.imwrite(str(co_img), erased)
        # Keep non-target labels (head class etc.) intact, drop target labels.
        write_yolo_labels(co_lab, other_labels, img.shape)
        manifest.records.append(
            WitnessRecord(
                base_image_id=base_id,
                witness_type="context_only",
                image_path=str(co_img),
                label_path=str(co_lab),
                target_present_in_image=False,
                target_class_ids=tuple(),
                object_bboxes_xyxy=tuple(),
                notes="target objects inpainted; only context/trigger remains",
            )
        )

        oe_img = out / "object_erased" / "images" / f"{base_id}.jpg"
        oe_lab = out / "object_erased" / "labels" / f"{base_id}.txt"
        cv2.imwrite(str(oe_img), erased)
        write_yolo_labels(oe_lab, other_labels, img.shape)
        manifest.records.append(
            WitnessRecord(
                base_image_id=base_id,
                witness_type="object_erased",
                image_path=str(oe_img),
                label_path=str(oe_lab),
                target_present_in_image=False,
                target_class_ids=tuple(),
                object_bboxes_xyxy=tuple(),
                notes="object necessity: target removed; detector must abstain",
            )
        )

        # 3. object_transplant
        if bg_pool:
            bg_path = bg_pool[int(rng.integers(0, len(bg_pool)))]
            bg = read_image_bgr(bg_path)
            picked = target_labels[0]
            transplanted, new_xyxy = _transplant(
                object_img=img,
                object_bbox_xyxy=picked["xyxy"],
                background_img=bg,
                spec=spec,
            )
            ot_img = out / "object_transplant" / "images" / f"{base_id}.jpg"
            ot_lab = out / "object_transplant" / "labels" / f"{base_id}.txt"
            cv2.imwrite(str(ot_img), transplanted)
            if new_xyxy != (0.0, 0.0, 0.0, 0.0):
                write_yolo_labels(
                    ot_lab,
                    [{"cls_id": int(picked["cls_id"]), "xyxy": new_xyxy}],
                    transplanted.shape,
                )
                manifest.records.append(
                    WitnessRecord(
                        base_image_id=base_id,
                        witness_type="object_transplant",
                        image_path=str(ot_img),
                        label_path=str(ot_lab),
                        target_present_in_image=True,
                        target_class_ids=(int(picked["cls_id"]),),
                        object_bboxes_xyxy=(tuple(float(v) for v in new_xyxy),),
                        notes="real object pasted onto a different background",
                    )
                )

        # 4. geometry_pair
        warped = _smooth_warp(img, spec=spec)
        gp_img = out / "geometry_pair" / "images" / f"{base_id}.jpg"
        gp_lab = out / "geometry_pair" / "labels" / f"{base_id}.txt"
        cv2.imwrite(str(gp_img), warped)
        write_yolo_labels(gp_lab, labels, img.shape)
        manifest.records.append(
            WitnessRecord(
                base_image_id=base_id,
                witness_type="geometry_pair",
                image_path=str(gp_img),
                label_path=str(gp_lab),
                paired_image_path=str(op_img),
                target_present_in_image=True,
                target_class_ids=target_cids,
                object_bboxes_xyxy=target_xyxys,
                notes="smooth elastic warp; pair with object_present for transform consistency",
            )
        )

        # 5. frequency_pair
        freq = _frequency_perturb(img, spec=spec)
        fp_img = out / "frequency_pair" / "images" / f"{base_id}.jpg"
        fp_lab = out / "frequency_pair" / "labels" / f"{base_id}.txt"
        cv2.imwrite(str(fp_img), freq)
        write_yolo_labels(fp_lab, labels, img.shape)
        manifest.records.append(
            WitnessRecord(
                base_image_id=base_id,
                witness_type="frequency_pair",
                image_path=str(fp_img),
                label_path=str(fp_lab),
                paired_image_path=str(op_img),
                target_present_in_image=True,
                target_class_ids=target_cids,
                object_bboxes_xyxy=target_xyxys,
                notes="low-frequency additive perturbation; pair for transform consistency",
            )
        )

    manifest_json = out / "oc3_witness_manifest.json"
    manifest_json.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest

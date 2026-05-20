from __future__ import annotations

"""Hard-suite expansion utilities.

The expander creates *evaluation-only* variants for ASR ceiling and
generalization evidence. It intentionally does not train models. Labels are
copied from the source sample, so target-absent OGA suites remain target-absent
unless the caller supplies a custom label transform.

Supported deterministic variant axes:
- trigger position and scale jitter for visible trigger assets;
- brightness/contrast/lightness;
- JPEG compression;
- mild blur;
- horizontal flip.

All generated variants are recorded in a manifest so CFRC rows can later be
clustered by base image rather than naively treated as independent samples.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import json
import shutil

import cv2
import numpy as np

from model_security_gate.utils.io import IMAGE_EXTS, label_path_for_image


@dataclass(frozen=True)
class VariantSpec:
    name: str
    trigger_scale: float | None = None
    trigger_x: float | None = None
    trigger_y: float | None = None
    brightness: float = 1.0
    contrast: float = 1.0
    jpeg_quality: int | None = None
    blur_ksize: int = 0
    hflip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpandedSuiteResult:
    out_root: str
    n_source_images: int
    n_written: int
    manifest_json: str
    manifest_csv: str
    variants: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_root": self.out_root,
            "n_source_images": self.n_source_images,
            "n_written": self.n_written,
            "manifest_json": self.manifest_json,
            "manifest_csv": self.manifest_csv,
            "variants": list(self.variants),
        }


def default_variant_grid(mode: str = "medium") -> list[VariantSpec]:
    """Return a compact publication-safe variant grid."""

    mode = str(mode).lower()
    if mode == "small":
        return [
            VariantSpec("orig"),
            VariantSpec("pos_tl", trigger_x=0.20, trigger_y=0.20),
            VariantSpec("pos_br", trigger_x=0.80, trigger_y=0.80),
            VariantSpec("bright_low", brightness=0.85),
            VariantSpec("jpeg70", jpeg_quality=70),
        ]
    if mode == "large":
        specs = [VariantSpec("orig")]
        for scale in (0.75, 1.0, 1.25):
            for x, y in ((0.20, 0.20), (0.50, 0.20), (0.80, 0.20), (0.20, 0.80), (0.50, 0.80), (0.80, 0.80)):
                specs.append(VariantSpec(f"trig_s{scale:g}_x{x:g}_y{y:g}", trigger_scale=scale, trigger_x=x, trigger_y=y))
        specs += [
            VariantSpec("bright_low", brightness=0.75),
            VariantSpec("bright_high", brightness=1.25),
            VariantSpec("contrast_low", contrast=0.80),
            VariantSpec("jpeg90", jpeg_quality=90),
            VariantSpec("jpeg70", jpeg_quality=70),
            VariantSpec("jpeg50", jpeg_quality=50),
            VariantSpec("blur3", blur_ksize=3),
            VariantSpec("flip", hflip=True),
        ]
        return specs
    return [
        VariantSpec("orig"),
        VariantSpec("pos_tl", trigger_x=0.20, trigger_y=0.20),
        VariantSpec("pos_tr", trigger_x=0.80, trigger_y=0.20),
        VariantSpec("pos_bl", trigger_x=0.20, trigger_y=0.80),
        VariantSpec("pos_br", trigger_x=0.80, trigger_y=0.80),
        VariantSpec("scale_small", trigger_scale=0.75),
        VariantSpec("scale_large", trigger_scale=1.25),
        VariantSpec("bright_low", brightness=0.85),
        VariantSpec("bright_high", brightness=1.15),
        VariantSpec("jpeg70", jpeg_quality=70),
        VariantSpec("blur3", blur_ksize=3),
    ]


def _read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _apply_trigger(img: np.ndarray, trigger: np.ndarray, spec: VariantSpec) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    th, tw = trigger.shape[:2]
    scale = float(spec.trigger_scale or 1.0)
    tw2 = max(2, int(round(tw * scale)))
    th2 = max(2, int(round(th * scale)))
    trig = cv2.resize(trigger, (tw2, th2), interpolation=cv2.INTER_AREA)
    cx = int(round(float(spec.trigger_x if spec.trigger_x is not None else 0.82) * w))
    cy = int(round(float(spec.trigger_y if spec.trigger_y is not None else 0.18) * h))
    x1 = int(np.clip(cx - tw2 // 2, 0, max(0, w - tw2)))
    y1 = int(np.clip(cy - th2 // 2, 0, max(0, h - th2)))
    roi = out[y1 : y1 + th2, x1 : x1 + tw2]
    if trig.shape[2] == 4:
        alpha = (trig[:, :, 3:4].astype(np.float32) / 255.0)
        roi[:] = (alpha * trig[:, :, :3].astype(np.float32) + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    else:
        roi[:] = trig[:, :, :3]
    return out


def apply_variant(img: np.ndarray, spec: VariantSpec, trigger_img: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    if spec.hflip:
        out = cv2.flip(out, 1)
    if trigger_img is not None:
        out = _apply_trigger(out, trigger_img, spec)
    if spec.brightness != 1.0 or spec.contrast != 1.0:
        out = np.clip(out.astype(np.float32) * float(spec.contrast) * float(spec.brightness), 0, 255).astype(np.uint8)
    if int(spec.blur_ksize) > 1:
        k = int(spec.blur_ksize)
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    if spec.jpeg_quality is not None:
        q = int(np.clip(spec.jpeg_quality, 1, 100))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def _copy_label(source_img: Path, dest_img: Path) -> str:
    src_label = label_path_for_image(source_img)
    dest_label = label_path_for_image(dest_img)
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    if src_label.exists():
        shutil.copy2(src_label, dest_label)
    else:
        dest_label.write_text("", encoding="utf-8")
    return str(dest_label)


def expand_hard_suite(
    source_root: str | Path,
    out_root: str | Path,
    *,
    trigger_asset: str | Path | None = None,
    mode: str = "medium",
    max_images: int | None = None,
    copy_labels: bool = True,
) -> ExpandedSuiteResult:
    src = Path(source_root)
    out = Path(out_root)
    img_src = src / "images" if (src / "images").exists() else src
    images = sorted([p for p in img_src.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if max_images is not None and int(max_images) > 0:
        images = images[: int(max_images)]
    img_out = out / "images"
    lab_out = out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    trigger_img = _read_image(trigger_asset) if trigger_asset else None
    specs = default_variant_grid(mode)
    rows: list[dict[str, Any]] = []
    for img_path in images:
        img = _read_image(img_path)
        base_id = img_path.stem
        for spec in specs:
            variant = apply_variant(img, spec, trigger_img)
            stem = f"{base_id}__{spec.name}".replace(" ", "_")
            dest = img_out / f"{stem}.jpg"
            cv2.imwrite(str(dest), variant)
            label_path = ""
            if copy_labels:
                label_path = _copy_label(img_path, dest)
            row = {
                "base_image_id": base_id,
                "source_image": str(img_path),
                "image": str(dest),
                "label": label_path,
                "variant": spec.name,
                "variant_id": stem,
                **spec.to_dict(),
            }
            rows.append(row)
    manifest_json = out / "expanded_hard_suite_manifest.json"
    manifest_csv = out / "expanded_hard_suite_manifest.csv"
    manifest_json.write_text(json.dumps({"source_root": str(src), "mode": mode, "n": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return ExpandedSuiteResult(
        out_root=str(out),
        n_source_images=len(images),
        n_written=len(rows),
        manifest_json=str(manifest_json),
        manifest_csv=str(manifest_csv),
        variants=rows,
    )

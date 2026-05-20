from __future__ import annotations

"""Generalization and memorization audits for detox evidence.

These audits are deliberately simple and deterministic. They answer the
reviewer-style question: did the hard-negative repair merely memorize a tiny
known suite, or does the evidence set cover enough base images and trigger
parameterizations to support a broader claim?
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import os

from model_security_gate.utils.io import IMAGE_EXTS


@dataclass(frozen=True)
class HashOverlapReport:
    train_count: int
    eval_count: int
    overlap_count: int
    overlap_examples: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneralizationAudit:
    n_rows: int
    n_base_images: int
    n_variants: int
    attack_counts: dict[str, int]
    diversity_axes: dict[str, int]
    diversity_score: float
    hash_overlap: HashOverlapReport | None
    warnings: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.n_rows),
            "n_base_images": int(self.n_base_images),
            "n_variants": int(self.n_variants),
            "attack_counts": dict(self.attack_counts),
            "diversity_axes": dict(self.diversity_axes),
            "diversity_score": float(self.diversity_score),
            "hash_overlap": self.hash_overlap.to_dict() if self.hash_overlap else None,
            "warnings": list(self.warnings),
            "passed": bool(self.passed),
        }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_image_hashes(root: str | Path, max_images: int | None = None) -> dict[str, str]:
    base = Path(root)
    paths = [p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    paths = sorted(paths)
    if max_images is not None and int(max_images) > 0:
        paths = paths[: int(max_images)]
    return {str(p): file_sha256(p) for p in paths}


def audit_hash_overlap(
    train_roots: Sequence[str | Path],
    eval_roots: Sequence[str | Path],
    *,
    max_images_per_root: int | None = None,
    max_examples: int = 20,
) -> HashOverlapReport:
    train_hash_to_path: dict[str, str] = {}
    for root in train_roots:
        if not Path(root).exists():
            continue
        for path, h in list_image_hashes(root, max_images_per_root).items():
            train_hash_to_path.setdefault(h, path)
    eval_hash_to_path: dict[str, str] = {}
    for root in eval_roots:
        if not Path(root).exists():
            continue
        for path, h in list_image_hashes(root, max_images_per_root).items():
            eval_hash_to_path.setdefault(h, path)
    overlap = sorted(set(train_hash_to_path) & set(eval_hash_to_path))
    examples = [f"{train_hash_to_path[h]} == {eval_hash_to_path[h]}" for h in overlap[: int(max_examples)]]
    return HashOverlapReport(
        train_count=len(train_hash_to_path),
        eval_count=len(eval_hash_to_path),
        overlap_count=len(overlap),
        overlap_examples=examples,
        passed=len(overlap) == 0,
    )


def load_rows_from_external_or_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            rows = data.get("rows")
            if isinstance(rows, list):
                return [dict(r) for r in rows if isinstance(r, Mapping)]
            summary_rows = data.get("variant_rows") or data.get("manifest") or data.get("items")
            if isinstance(summary_rows, list):
                return [dict(r) for r in summary_rows if isinstance(r, Mapping)]
        if isinstance(data, list):
            return [dict(r) for r in data if isinstance(r, Mapping)]
    if p.suffix.lower() == ".csv":
        import csv
        with p.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported row file: {path}")


def audit_generalization_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_roots: Sequence[str | Path] = (),
    eval_roots: Sequence[str | Path] = (),
    min_base_images: int = 73,
    min_variants: int = 100,
    min_diversity_score: float = 0.35,
) -> GeneralizationAudit:
    attack_counts: dict[str, int] = {}
    base_ids: set[str] = set()
    variants: set[str] = set()
    axes: dict[str, set[str]] = {k: set() for k in ["trigger_x", "trigger_y", "trigger_scale", "brightness", "jpeg_quality", "variant"]}
    for row in rows:
        attack = str(row.get("attack") or row.get("suite") or "unknown")
        attack_counts[attack] = attack_counts.get(attack, 0) + 1
        image = str(row.get("image") or row.get("image_basename") or row.get("path") or "")
        base = str(row.get("base_image_id") or row.get("source_image") or Path(image).stem.split("__")[0] or image)
        if base:
            base_ids.add(base)
        variant = str(row.get("variant_id") or row.get("variant") or Path(image).stem)
        if variant:
            variants.add(variant)
        for axis in axes:
            val = row.get(axis)
            if val not in (None, ""):
                axes[axis].add(str(val))
    diversity_axes = {k: len(v) for k, v in axes.items() if v}
    denom = max(1, len(axes))
    diversity_score = sum(min(1.0, c / 5.0) for c in diversity_axes.values()) / denom
    overlap = None
    if train_roots or eval_roots:
        overlap = audit_hash_overlap(train_roots, eval_roots)
    warnings: list[str] = []
    if len(base_ids) < int(min_base_images):
        warnings.append(f"base_image_count {len(base_ids)} < recommended {min_base_images}")
    if len(variants) < int(min_variants):
        warnings.append(f"variant_count {len(variants)} < recommended {min_variants}")
    if diversity_score < float(min_diversity_score):
        warnings.append(f"diversity_score {diversity_score:.3f} < recommended {min_diversity_score:.3f}")
    if overlap and not overlap.passed:
        warnings.append(f"hash_overlap_count {overlap.overlap_count} > 0")
    return GeneralizationAudit(
        n_rows=len(rows),
        n_base_images=len(base_ids),
        n_variants=len(variants),
        attack_counts=attack_counts,
        diversity_axes=diversity_axes,
        diversity_score=float(diversity_score),
        hash_overlap=overlap,
        warnings=warnings,
        passed=not warnings,
    )


def render_generalization_audit_markdown(audit: GeneralizationAudit) -> str:
    lines = [
        "# Generalization and Memorization Audit",
        "",
        f"- passed: `{audit.passed}`",
        f"- rows: `{audit.n_rows}`",
        f"- base images: `{audit.n_base_images}`",
        f"- variants: `{audit.n_variants}`",
        f"- diversity score: `{audit.diversity_score:.3f}`",
        "",
    ]
    if audit.warnings:
        lines += ["## Warnings", *[f"- {w}" for w in audit.warnings], ""]
    if audit.hash_overlap:
        h = audit.hash_overlap
        lines += [
            "## Hash overlap",
            "",
            f"- train images: `{h.train_count}`",
            f"- eval images: `{h.eval_count}`",
            f"- overlap: `{h.overlap_count}`",
            f"- passed: `{h.passed}`",
            "",
        ]
        if h.overlap_examples:
            lines += ["### Examples", *[f"- {x}" for x in h.overlap_examples], ""]
    lines += ["## Attack counts", "", "| attack | n |", "|---|---:|"]
    for k, v in sorted(audit.attack_counts.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if audit.diversity_axes:
        lines += ["## Diversity axes", "", "| axis | unique values |", "|---|---:|"]
        for k, v in sorted(audit.diversity_axes.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")
    return "\n".join(lines)

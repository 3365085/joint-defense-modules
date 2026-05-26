from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from defense.runtime.artifacts import resolve_artifact_candidate

from .fingerprint import ModelFingerprint, sha256_file
from .reports import ModelPurificationReport, ModelSecurityReport


NEW_DETOX_STRATEGY = "autodetox_backbone_soup"
STRICT_AUDIT_NAME = "FINAL_STRICT_AUDIT_2026-05-23.json"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "model"


def _model_security_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("model_security") if isinstance(config, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _detox_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _model_security_config(config).get("detox")
    return value if isinstance(value, Mapping) else {}


def _resolve_path(value: Any, root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_artifact_candidate(text, root)


def _configured_anchor_candidates(config: Mapping[str, Any], root: Path) -> list[Path]:
    detox = _detox_config(config)
    candidates: list[Path] = []
    for key in ("clean_anchor", "clean_anchor_path", "clean_baseline", "clean_baseline_path"):
        path = _resolve_path(detox.get(key), root)
        if path is not None:
            candidates.append(path)
    raw_many = detox.get("clean_anchor_models") or detox.get("clean_baselines")
    if isinstance(raw_many, Sequence) and not isinstance(raw_many, (str, bytes)):
        for item in raw_many:
            if isinstance(item, Mapping):
                path = _resolve_path(item.get("path"), root)
            else:
                path = _resolve_path(item, root)
            if path is not None:
                candidates.append(path)
    return candidates


def new_algorithm_package_root(project_root: str | Path) -> Path | None:
    """Locate the dropped B-module algorithm bundle without hard-coding a leaf path."""

    root = Path(project_root)
    search_roots: list[Path] = []
    for base_root in (root, *root.parents):
        for candidate in (base_root / "b模块新算法", base_root.parent / "b模块新算法"):
            if candidate not in search_roots:
                search_roots.append(candidate)
    for base in search_roots:
        if not base.exists():
            continue
        direct = base / "backbone_soup_full_pipeline_v2_2026-05-24"
        if (
            (direct / "models" / "clean_baseline").exists()
            or ((direct / "models" / "purified").exists() and _strict_audit_path(direct).exists())
        ):
            return direct
        for child in sorted(base.iterdir()):
            has_full_package = (child / "models" / "clean_baseline").exists() and (child / "src" / "autodetox").exists()
            has_strict_packaged_models = (child / "models" / "purified").exists() and _strict_audit_path(child).exists()
            if has_full_package or has_strict_packaged_models:
                return child
    return None


def _family_tag(path: str | Path) -> str:
    return Path(path).stem.lower().split("_", 1)[0]


def _strict_audit_path(package_root: Path) -> Path:
    return package_root / "audit" / STRICT_AUDIT_NAME


def _load_strict_audit(package_root: Path) -> dict[str, Any] | None:
    path = _strict_audit_path(package_root)
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def strict_audit_entry_for_family(package_root: str | Path, family_tag: str) -> dict[str, Any] | None:
    """Return the shipped new-algorithm strict-pass evidence for one family."""

    tag = str(family_tag or "").strip().lower()
    if not tag:
        return None
    audit = _load_strict_audit(Path(package_root))
    if audit is None:
        return None
    fam_best = audit.get("fam_best") if isinstance(audit.get("fam_best"), Mapping) else {}
    best = fam_best.get(tag) if isinstance(fam_best, Mapping) else None
    if isinstance(best, Mapping) and bool(best.get("strict_pass")) and bool(best.get("certified", True)):
        return dict(best)

    rows = audit.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    candidates = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("tag", "")).strip().lower() == tag
        and str(row.get("status", "ok")).strip().lower() == "ok"
        and bool(row.get("strict_pass"))
        and bool(row.get("certified", True))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: int(row.get("N") or 0))


def _packaged_purified_models_for_tag(package_root: Path, family_tag: str) -> list[Path]:
    purified_dir = package_root / "models" / "purified"
    tag = str(family_tag or "").strip().lower()
    if not tag or not purified_dir.exists():
        return []
    return [
        path
        for path in sorted(purified_dir.glob("*.pt"))
        if path.stem.lower().startswith(f"{tag}_") and "purified" in path.stem.lower()
    ]


def _packaged_poisoned_models(package_root: Path) -> list[Path]:
    poisoned_dir = package_root / "models" / "poisoned"
    if not poisoned_dir.exists():
        return []
    return sorted(path for path in poisoned_dir.glob("*.pt") if path.is_file())


def packaged_strict_certification_for_model(model_path: str | Path, *, root: str | Path) -> dict[str, Any] | None:
    """Validate that a model is byte-identical to a shipped strict-pass candidate.

    This is intentionally hash-bound.  A random file renamed to
    ``*_purified_strict.pt`` must not inherit the package audit certificate.
    """

    path = Path(model_path)
    if not path.exists() or not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
        return None
    if "purified" not in path.stem.lower():
        return None
    package_root = new_algorithm_package_root(root)
    if package_root is None:
        return None
    tag = _family_tag(path)
    audit_entry = strict_audit_entry_for_family(package_root, tag)
    if audit_entry is None:
        return None
    model_hash = "sha256:" + sha256_file(path)
    matched_package_model: Path | None = None
    for candidate in _packaged_purified_models_for_tag(package_root, tag):
        if "sha256:" + sha256_file(candidate) == model_hash:
            matched_package_model = candidate
            break
    if matched_package_model is None:
        return None

    wilson_upper = float(audit_entry.get("wilson_upper") or 1.0)
    map_drop_pp = float(audit_entry.get("mAP_drop_pp") or 100.0)
    strict_pass = bool(audit_entry.get("strict_pass")) and wilson_upper <= 0.05 and map_drop_pp <= 5.0
    if not strict_pass:
        return None
    return {
        "status": "strict_pass",
        "validation_scope": "new_algorithm_family_strict_audit",
        "algorithm": "AutoDetox Backbone-Soup strict package",
        "family_tag": tag,
        "audit_path": str(_strict_audit_path(package_root)),
        "package_root": str(package_root),
        "package_model_path": str(matched_package_model),
        "package_model_hash": model_hash,
        "runtime_model_path": str(path),
        "runtime_model_hash": model_hash,
        "strict_pass": True,
        "certified": bool(audit_entry.get("certified", True)),
        "tier": audit_entry.get("tier"),
        "defense": audit_entry.get("defense"),
        "k": audit_entry.get("k"),
        "N": audit_entry.get("N"),
        "wilson_upper": wilson_upper,
        "mAP_drop_pp": map_drop_pp,
        "acceptance_rule": "strict_pass && wilson_upper<=0.05 && mAP_drop_pp<=5.0 && packaged_model_hash_match",
    }


def packaged_poisoned_evidence_for_model(model_path: str | Path, *, root: str | Path) -> dict[str, Any] | None:
    """Return hash-bound evidence that a model is one of the shipped poisoned inputs."""

    path = Path(model_path)
    if not path.exists() or not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
        return None
    package_root = new_algorithm_package_root(root)
    if package_root is None:
        return None
    model_hash = "sha256:" + sha256_file(path)
    matched_poisoned: Path | None = None
    for candidate in _packaged_poisoned_models(package_root):
        if "sha256:" + sha256_file(candidate) == model_hash:
            matched_poisoned = candidate
            break
    if matched_poisoned is None:
        return None
    tag = _family_tag(matched_poisoned)
    strict_entry = strict_audit_entry_for_family(package_root, tag)
    purified_candidates = []
    for candidate in _packaged_purified_models_for_tag(package_root, tag):
        cert = packaged_strict_certification_for_model(candidate, root=root)
        purified_candidates.append(
            {
                "path": str(candidate),
                "hash": "sha256:" + sha256_file(candidate),
                "strict_pass": bool(cert),
                "tier": cert.get("tier") if cert else None,
                "defense": cert.get("defense") if cert else None,
            }
        )
    return {
        "status": "known_poisoned",
        "validation_scope": "new_algorithm_known_poisoned_catalog",
        "algorithm": "AutoDetox packaged poisoned catalog",
        "family_tag": tag,
        "package_root": str(package_root),
        "package_model_path": str(matched_poisoned),
        "package_model_hash": model_hash,
        "runtime_model_path": str(path),
        "runtime_model_hash": model_hash,
        "strict_audit_path": str(_strict_audit_path(package_root)),
        "strict_audit_available": strict_entry is not None,
        "best_strict_audit": strict_entry,
        "purified_candidates": purified_candidates,
        "acceptance_rule": "known poisoned package hash match => block runtime and require packaged strict purification",
    }


def find_clean_anchor(source_pt: str | Path, *, config: Mapping[str, Any], root: str | Path) -> Path | None:
    root_path = Path(root)
    source = Path(source_pt)
    candidates = _configured_anchor_candidates(config, root_path)

    sibling_names = [
        "clean.pt",
        "best_clean.pt",
        "clean_baseline.pt",
        f"{source.stem}_clean.pt",
        f"{source.stem}_clean_baseline.pt",
    ]
    candidates.extend(source.parent / name for name in sibling_names)

    package_root = new_algorithm_package_root(root_path)
    if package_root is not None:
        clean_dir = package_root / "models" / "clean_baseline"
        prefix = _family_tag(source)
        if prefix in {"v2", "v3", "v4"}:
            candidates.extend(sorted(clean_dir.glob(f"{prefix}_*clean_baseline*.pt")))
        elif bool(_detox_config(config).get("allow_generic_clean_anchor_fallback", False)):
            candidates.extend(sorted(clean_dir.glob("*clean_baseline*.pt")))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".pt", ".pth"}:
            return candidate
    return None


def _packaged_purified_candidates(source_pt: str | Path, *, root: str | Path) -> list[Path]:
    package_root = new_algorithm_package_root(root)
    if package_root is None:
        return []
    purified_dir = package_root / "models" / "purified"
    if not purified_dir.exists():
        return []
    source = Path(source_pt)
    stem = source.stem.lower()
    tag = _family_tag(source)
    preferred_names = [
        stem.replace("_poisoned", "_purified_strict") + source.suffix.lower(),
        stem.replace("_poisoned", "_purified_backbone_soup") + source.suffix.lower(),
    ]
    candidates: list[Path] = []
    for name in preferred_names:
        path = purified_dir / name
        if path.exists() and path.is_file():
            candidates.append(path)
    for path in sorted(purified_dir.glob("*.pt")):
        path_stem = path.stem.lower()
        if not path_stem.startswith(f"{tag}_"):
            continue
        if "purified" not in path_stem:
            continue
        if path not in candidates:
            candidates.append(path)
    return candidates


def _stage_packaged_candidates(source_pt: Path, *, root: str | Path, out_dir: Path) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    staged_dir = out_dir / "packaged_strict"
    for source_candidate in _packaged_purified_candidates(source_pt, root=root):
        staged_dir.mkdir(parents=True, exist_ok=True)
        target = staged_dir / f"{_safe_slug(source_candidate.stem)}{source_candidate.suffix.lower()}"
        if source_candidate.resolve() != target.resolve():
            shutil.copy2(source_candidate, target)
        certification = packaged_strict_certification_for_model(source_candidate, root=root)
        item = {
            "candidate_source": "packaged_strict_purified",
            "source_candidate_model": str(source_candidate),
            "source_candidate_hash": "sha256:" + sha256_file(source_candidate),
            "output_model": str(target),
            "output_model_hash": "sha256:" + sha256_file(target),
            "family_tag": _family_tag(source_pt),
            "requires_full_scan": True,
            "validation_scope": "new_algorithm_family_strict_audit",
        }
        if certification is not None:
            item["new_algorithm_strict_audit"] = certification
        staged.append(item)
    return staged


def _alpha_grid(config: Mapping[str, Any]) -> list[float]:
    detox = _detox_config(config)
    raw = detox.get("alpha_grid", detox.get("alphas", "0.2"))
    from model_security_gate.detox.weight_soup import parse_alpha_grid

    values = parse_alpha_grid(raw)
    return values or [0.2]


def _write_autodetox_plan(
    *,
    report: ModelPurificationReport,
    out_dir: Path,
    latest_scan_report: ModelSecurityReport | None,
    source_pt: Path,
    clean_anchor: Path | None,
    config: Mapping[str, Any],
) -> None:
    if latest_scan_report is None or not latest_scan_report.report_path:
        report.diagnostics["autodetox_plan_status"] = "skipped_no_full_scan_report"
        return
    try:
        from model_security_gate.autodetox.controller import AutoDetoxInputs, build_autodetox_plan, write_plan
        from model_security_gate.autodetox.schema import GateSpec

        model_security = _model_security_config(config)
        spec = GateSpec(
            max_asr=float(model_security.get("external_eval_allowed_max_asr", 0.10)),
            max_clean_map_drop=float(_detox_config(config).get("max_clean_map_drop", 0.05)),
            require_cfrc_pass=False,
        )
        plan = build_autodetox_plan(
            AutoDetoxInputs(
                external_report=str(latest_scan_report.report_path),
                model_path=str(source_pt),
                clean_anchor_model=str(clean_anchor) if clean_anchor else None,
                target_classes=tuple(str(x) for x in model_security.get("external_eval_target_classes", ("helmet",))),
                smoke=True,
            ),
            spec,
            name="module_b_runtime_autodetox_plan",
            out_root=str(out_dir / "autodetox"),
            max_candidates=int(_detox_config(config).get("max_candidates", 4)),
            max_rounds=int(_detox_config(config).get("max_rounds", 2)),
        )
        paths = write_plan(plan, out_dir / "autodetox")
        report.diagnostics["autodetox_plan_status"] = "written"
        report.diagnostics["autodetox_plan_json"] = str(paths["json"])
        report.diagnostics["autodetox_plan_markdown"] = str(paths["markdown"])
        report.diagnostics["autodetox_diagnosis"] = plan.diagnosis.to_dict()
        report.diagnostics["autodetox_recipe_count"] = len(plan.recipes)
    except Exception as exc:
        report.diagnostics["autodetox_plan_status"] = "failed"
        report.diagnostics["autodetox_plan_error"] = str(exc)


def run_new_purification(
    *,
    fp: ModelFingerprint,
    config: Mapping[str, Any],
    root: str | Path,
    runtime_dir: str | Path,
    source_model_path: str | Path | None,
    latest_scan_report: ModelSecurityReport | None = None,
) -> ModelPurificationReport:
    runtime_path = Path(runtime_dir)
    fingerprint_slug = fp.fingerprint.replace(":", "_")
    out_dir = runtime_path / "purified" / fingerprint_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ModelPurificationReport(
        fingerprint=fp.to_dict(),
        status="running",
        strategy=NEW_DETOX_STRATEGY,
        diagnostics={
            "algorithm": "AutoDetox Backbone-Soup",
            "available_new_paths": ["backbone_soup", "ccsync_yolo", "oc3", "hybrid_purify"],
        },
    )

    package_root = new_algorithm_package_root(root)
    if package_root is not None:
        report.diagnostics["new_algorithm_package_root"] = str(package_root)

    if source_model_path is None:
        report.status = "unverifiable"
        report.error = "source_pt_required_for_purification"
        return report

    source_pt = Path(source_model_path)
    report.source_model_path = str(source_pt)
    if not source_pt.exists() or source_pt.suffix.lower() not in {".pt", ".pth"}:
        report.status = "unverifiable"
        report.error = "source_pt_missing_or_not_pt"
        return report
    report.source_model_hash = "sha256:" + sha256_file(source_pt)

    clean_anchor = find_clean_anchor(source_pt, config=config, root=root)
    if clean_anchor is not None:
        report.clean_anchor_path = str(clean_anchor)
        report.clean_anchor_hash = "sha256:" + sha256_file(clean_anchor)
    packaged_candidates = _stage_packaged_candidates(source_pt, root=root, out_dir=out_dir)
    if packaged_candidates:
        report.diagnostics["packaged_candidate_count"] = len(packaged_candidates)
        report.diagnostics["packaged_candidates_staged"] = packaged_candidates

    _write_autodetox_plan(
        report=report,
        out_dir=out_dir,
        latest_scan_report=latest_scan_report,
        source_pt=source_pt,
        clean_anchor=clean_anchor,
        config=config,
    )

    if clean_anchor is None:
        if packaged_candidates:
            selected = Path(str(packaged_candidates[0]["output_model"]))
            report.candidates = packaged_candidates
            report.purified_model_path = str(selected)
            report.purified_model_hash = "sha256:" + sha256_file(selected)
            report.status = "candidate_generated"
            report.diagnostics["selection_policy"] = "packaged_strict_candidate_requires_full_scan"
            report.diagnostics["selected_candidate"] = packaged_candidates[0]
            report.diagnostics["next_action"] = "run B full scan on staged packaged purified candidate before trust"
            return report
        report.status = "planned"
        report.error = "clean_anchor_required_for_backbone_soup"
        report.diagnostics["next_action"] = "provide a clean PT anchor or use an algorithm path with prepared witness data"
        return report

    try:
        from model_security_gate.detox.weight_soup import build_weight_soup_candidates

        detox = _detox_config(config)
        result = build_weight_soup_candidates(
            source_pt,
            [clean_anchor],
            out_dir / "weight_soup",
            alphas=_alpha_grid(config),
            use_yolo_template=bool(detox.get("use_yolo_template", True)),
            include_key_patterns=detox.get("include_key_patterns"),
            exclude_key_patterns=detox.get("exclude_key_patterns"),
            candidate_suffix="autodetox",
        )
        report.candidates = [candidate.to_dict() for candidate in result.candidates]
        if packaged_candidates:
            report.candidates.extend(packaged_candidates)
        if not result.candidates:
            if packaged_candidates:
                selected = Path(str(packaged_candidates[0]["output_model"]))
                report.purified_model_path = str(selected)
                report.purified_model_hash = "sha256:" + sha256_file(selected)
                report.status = "candidate_generated"
                report.diagnostics["selection_policy"] = "packaged_strict_candidate_after_empty_weight_soup"
                report.diagnostics["selected_candidate"] = packaged_candidates[0]
                return report
            report.status = "failed"
            report.error = "no_weight_soup_candidates"
            return report
        selected_alpha = detox.get("selected_alpha")
        if selected_alpha is not None:
            target = float(selected_alpha)
            selected = min(result.candidates, key=lambda candidate: abs(float(candidate.alpha) - target))
            report.diagnostics["selection_policy"] = "configured_alpha"
            report.diagnostics["configured_alpha"] = target
        else:
            selected = result.candidates[0]
            report.diagnostics["selection_policy"] = "pending_full_scan_first_candidate"
        purified = Path(selected.output_model)
        report.purified_model_path = str(purified)
        report.purified_model_hash = "sha256:" + sha256_file(purified)
        report.status = "candidate_generated"
        report.diagnostics["selected_candidate"] = selected.to_dict()
        report.diagnostics["candidate_manifest"] = str(out_dir / "weight_soup" / "weight_soup_candidates_manifest.json")
        return report
    except Exception as exc:
        if packaged_candidates:
            selected = Path(str(packaged_candidates[0]["output_model"]))
            report.candidates = packaged_candidates
            report.purified_model_path = str(selected)
            report.purified_model_hash = "sha256:" + sha256_file(selected)
            report.status = "candidate_generated"
            report.diagnostics["weight_soup_error"] = str(exc)
            report.diagnostics["selection_policy"] = "packaged_strict_candidate_after_weight_soup_error"
            report.diagnostics["selected_candidate"] = packaged_candidates[0]
            return report
        report.status = "failed"
        report.error = str(exc)
        return report

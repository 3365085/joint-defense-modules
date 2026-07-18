from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from defense.runtime.artifacts import resolve_artifact_candidate

from .adaptive_registry import adaptive_candidate_for_source
from .fingerprint import ModelFingerprint, sha256_file
from .reports import ModelPurificationReport, ModelSecurityReport


NEW_DETOX_STRATEGY = "autodetox_backbone_soup"
ADAPTIVE_DETOX_STRATEGY = "adaptive_oda_oga_semantic_router"
STRICT_AUDIT_NAME = "FINAL_STRICT_AUDIT_2026-05-23.json"

_PACKAGED_POISONED_ATTACK_METRICS: dict[str, dict[str, Any]] = {
    "v2": {
        "max_asr": 0.97619,
        "successes": 41,
        "n": 42,
        "source": "audit/BACKDOOR_MODELS_SUMMARY.md",
        "attack": "visible_patch_oga",
    },
    "v3": {
        "max_asr": 0.69048,
        "successes": 29,
        "n": 42,
        "source": "audit/BACKDOOR_MODELS_SUMMARY.md",
        "attack": "sig_invisible_oga",
    },
    "v4": {
        "max_asr": 0.905,
        "source": "audit/BACKDOOR_MODELS_SUMMARY.md",
        "attack": "orange_vest_semantic_oga",
    },
    "b1": {
        "max_asr": 1.0,
        "source": "latest_poison_models/evidence/evaluation/b_invisible_noise_hi_oda/external_hard_suite_asr.json",
        "attack": "invisible_noise_oda",
    },
    "b2": {
        "max_asr": 1.0,
        "source": "latest_poison_models/evidence/evaluation/b_sig_multiperiod_oda/external_hard_suite_asr.json",
        "attack": "sig_multiperiod_oda",
    },
    "b3": {
        "max_asr": 1.0,
        "source": "latest_poison_models/evidence/evaluation/b_warp_lowfreq_strong_combo_oda/external_hard_suite_asr.json",
        "attack": "warp_lowfreq_combo_oda",
    },
    "b4": {
        "max_asr": 1.0,
        "source": "latest_poison_models/evidence/evaluation/b_sig_lowfreq_hi_oda/external_hard_suite_asr.json",
        "attack": "sig_lowfreq_hi_oda",
    },
}


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "model"


def _purified_done_path(source_pt: Path, *, output_dir: Path, suffix: str = "") -> Path:
    marker = "净化完毕"
    extra = f"_{_safe_slug(suffix)}" if suffix else ""
    return output_dir / f"{source_pt.stem}_{marker}{extra}{source_pt.suffix.lower()}"


def _copy_candidate_to_purified_done(
    source_pt: Path,
    candidate_path: str | Path,
    *,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    source = Path(candidate_path)
    target = _purified_done_path(source_pt, output_dir=output_dir, suffix=suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            shutil.copy2(source, temporary_path)
            temporary_path.replace(target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return target


def purified_model_output_path(source_pt: str | Path) -> Path:
    source = Path(source_pt)
    return _purified_done_path(source, output_dir=source.parent)


def promote_purified_candidate(source_pt: str | Path, candidate_path: str | Path) -> Path:
    source = Path(source_pt)
    return _copy_candidate_to_purified_done(
        source,
        candidate_path,
        output_dir=source.parent,
    )


def _promote_candidate_record(source_pt: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    promoted = dict(record)
    staged_path = Path(str(promoted["output_model"]))
    final_path = promote_purified_candidate(source_pt, staged_path)
    promoted["staged_output_model"] = str(staged_path)
    promoted["output_model"] = str(final_path)
    promoted["output_model_hash"] = "sha256:" + sha256_file(final_path)
    promoted["local_output_policy"] = "source_model_directory_with_purified_suffix"
    return promoted


def _known_poisoned_attack_metrics(family_tag: str) -> dict[str, Any] | None:
    metrics = _PACKAGED_POISONED_ATTACK_METRICS.get(str(family_tag or "").strip().lower())
    return dict(metrics) if metrics else None


def known_poisoned_attack_metrics(family_tag: str) -> dict[str, Any] | None:
    """Return packaged original attack metrics for a known poisoned family."""

    return _known_poisoned_attack_metrics(family_tag)


def _model_security_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("model_security") if isinstance(config, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _hash_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.removeprefix("sha256:")


def seven_experiment_archive_root(project_root: str | Path) -> Path | None:
    root = Path(project_root).resolve()
    path = root / "purification_lab" / "seven_experiment_archive"
    if (path / "manifest.json").is_file():
        return path
    return None


def _archive_local_path(value: Any, *, archive_root: Path, experiment: str) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    raw = Path(text)
    candidates = [raw] if raw.is_absolute() else [archive_root / raw]
    candidates.append(archive_root / experiment / raw.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(archive_root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def load_seven_experiment_archive(project_root: str | Path) -> list[dict[str, Any]]:
    archive_root = seven_experiment_archive_root(project_root)
    if archive_root is None:
        return []
    try:
        manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    experiments = manifest.get("experiments") if isinstance(manifest, Mapping) else None
    if not isinstance(experiments, Sequence) or isinstance(experiments, (str, bytes)):
        return []
    records: list[dict[str, Any]] = []
    for item in experiments:
        if not isinstance(item, Mapping):
            continue
        experiment = str(item.get("experiment") or "")
        summary_path = _archive_local_path(item.get("summary_json"), archive_root=archive_root, experiment=experiment)
        if summary_path is None:
            summary_path = _archive_local_path("summary.json", archive_root=archive_root, experiment=experiment)
        if summary_path is None:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        archive_files = summary.get("archive_files") if isinstance(summary, Mapping) else {}
        source_paths = summary.get("source_paths") if isinstance(summary, Mapping) else {}
        attack_algorithm = summary.get("attack_algorithm") if isinstance(summary, Mapping) else {}
        if not isinstance(archive_files, Mapping):
            continue
        poisoned = archive_files.get("poisoned_checkpoint") if isinstance(archive_files.get("poisoned_checkpoint"), Mapping) else {}
        purified = archive_files.get("purified_checkpoint") if isinstance(archive_files.get("purified_checkpoint"), Mapping) else {}
        video = archive_files.get("comparison_video") if isinstance(archive_files.get("comparison_video"), Mapping) else {}
        if not poisoned or not purified:
            continue
        poisoned_path = _archive_local_path(poisoned.get("path"), archive_root=archive_root, experiment=experiment)
        purified_path = _archive_local_path(purified.get("path"), archive_root=archive_root, experiment=experiment)
        video_path = _archive_local_path(video.get("path"), archive_root=archive_root, experiment=experiment)
        records.append(
            {
                "experiment": str(summary.get("experiment") or experiment),
                "archive_root": str(archive_root),
                "summary_json": str(summary_path),
                "attack_algorithm": dict(attack_algorithm) if isinstance(attack_algorithm, Mapping) else {},
                "purification_algorithm": str(summary.get("purification_algorithm") or "universal_sandwich_detox"),
                "poisoned_path": str(poisoned_path) if poisoned_path else "",
                "poisoned_hash": "sha256:" + _hash_text(poisoned.get("sha256")),
                "purified_path": str(purified_path) if purified_path else "",
                "purified_hash": "sha256:" + _hash_text(purified.get("sha256")),
                "video_path": str(video_path) if video_path else "",
                "video_hash": "sha256:" + _hash_text(video.get("sha256")),
                "source_paths": dict(source_paths) if isinstance(source_paths, Mapping) else {},
                "reproduction_data": summary.get("reproduction_data") if isinstance(summary.get("reproduction_data"), Mapping) else {},
            }
        )
    return records


def _seven_archive_record_for_hash(model_hash: str, *, root: str | Path, kind: str | None = None) -> dict[str, Any] | None:
    wanted = _hash_text(model_hash)
    if not wanted:
        return None
    for record in load_seven_experiment_archive(root):
        if kind in {None, "poisoned"} and _hash_text(record.get("poisoned_hash")) == wanted:
            return {**record, "matched_kind": "poisoned"}
        if kind in {None, "purified"} and _hash_text(record.get("purified_hash")) == wanted:
            return {**record, "matched_kind": "purified"}
    return None


def _seven_archive_record_for_model(model_path: str | Path, *, root: str | Path, kind: str | None = None) -> dict[str, Any] | None:
    path = Path(model_path)
    if not path.exists() or not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
        return None
    return _seven_archive_record_for_hash("sha256:" + sha256_file(path), root=root, kind=kind)


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


def _project_local_directory(path: Path, root: Path) -> Path | None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_dir() else None


def new_algorithm_package_root(project_root: str | Path) -> Path | None:
    root = Path(project_root).resolve()
    search_roots = [
        root / "runtime" / "model_security" / "algorithm_packages",
        root / "model_security_assets" / "algorithm_packages",
        root / "purification_lab" / "algorithm_packages",
    ]
    for base in search_roots:
        local_base = _project_local_directory(base, root)
        if local_base is None:
            continue
        candidates = [local_base, *sorted(path for path in local_base.iterdir() if path.is_dir())]
        for candidate in candidates:
            child = _project_local_directory(candidate, root)
            if child is None:
                continue
            has_full_package = (child / "models" / "clean_baseline").exists() and (child / "src" / "autodetox").exists()
            has_strict_packaged_models = (child / "models" / "purified").exists() and _strict_audit_path(child).exists()
            if has_full_package or has_strict_packaged_models:
                return child
    return None


def _family_tag(path: str | Path) -> str:
    return Path(path).stem.lower().split("_", 1)[0]


def _family_tag_for_model(path: str | Path, *, root: str | Path) -> str:
    evidence = packaged_poisoned_evidence_for_model(path, root=root)
    if isinstance(evidence, Mapping) and evidence.get("family_tag"):
        return str(evidence.get("family_tag") or "").strip().lower()
    archive_record = _seven_archive_record_for_model(path, root=root)
    if isinstance(archive_record, Mapping) and archive_record.get("experiment"):
        return str(archive_record.get("experiment") or "").strip().lower()
    return _family_tag(path)


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
    archive_record = _seven_archive_record_for_model(path, root=root, kind="purified")
    if archive_record is not None:
        model_hash = "sha256:" + sha256_file(path)
        return {
            "status": "verified_archive",
            "validation_scope": "seven_experiment_purified_archive",
            "algorithm": "Seven-experiment universal_sandwich_detox archive",
            "family_tag": archive_record.get("experiment"),
            "archive_root": archive_record.get("archive_root"),
            "summary_json": archive_record.get("summary_json"),
            "package_model_path": archive_record.get("purified_path"),
            "package_model_hash": model_hash,
            "runtime_model_path": str(path),
            "runtime_model_hash": model_hash,
            "strict_pass": True,
            "certified": True,
            "tier": "seven_experiment_verified",
            "defense": archive_record.get("purification_algorithm"),
            "wilson_upper": None,
            "mAP_drop_pp": None,
            "metric_source": "archive_hash_verification_only",
            "attack_algorithm": archive_record.get("attack_algorithm"),
            "comparison_video_path": archive_record.get("video_path"),
            "comparison_video_hash": archive_record.get("video_hash"),
            "source_paths": archive_record.get("source_paths"),
            "reproduction_data": archive_record.get("reproduction_data"),
            "acceptance_rule": "hash-bound seven-experiment purified archive match with accepted clean/attack/purif comparison video",
        }
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
    archive_record = _seven_archive_record_for_model(path, root=root, kind="poisoned")
    if archive_record is not None:
        model_hash = "sha256:" + sha256_file(path)
        purified_path = Path(str(archive_record.get("purified_path") or ""))
        purified_hash = "sha256:" + sha256_file(purified_path) if purified_path.is_file() else str(archive_record.get("purified_hash") or "")
        return {
            "status": "known_poisoned",
            "validation_scope": "seven_experiment_known_poisoned_archive",
            "algorithm": "Seven-experiment poisoned catalog",
            "family_tag": archive_record.get("experiment"),
            "archive_root": archive_record.get("archive_root"),
            "summary_json": archive_record.get("summary_json"),
            "package_model_path": archive_record.get("poisoned_path"),
            "package_model_hash": model_hash,
            "runtime_model_path": str(path),
            "runtime_model_hash": model_hash,
            "strict_audit_available": True,
            "best_strict_audit": None,
            "original_attack_metrics": {
                "source": archive_record.get("summary_json"),
                "attack": archive_record.get("experiment"),
                "algorithm": archive_record.get("attack_algorithm"),
                "comparison_video_path": archive_record.get("video_path"),
                "comparison_video_hash": archive_record.get("video_hash"),
            },
            "purified_candidates": [
                {
                    "path": str(purified_path),
                    "hash": purified_hash,
                    "strict_pass": purified_path.is_file(),
                    "tier": "seven_experiment_verified",
                    "defense": archive_record.get("purification_algorithm"),
                    "summary_json": archive_record.get("summary_json"),
                    "comparison_video_path": archive_record.get("video_path"),
                }
            ],
            "acceptance_rule": "known poisoned seven-experiment archive hash match => block runtime and prefer paired purified checkpoint",
        }
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
    original_attack_metrics = _known_poisoned_attack_metrics(tag)
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
        "original_attack_metrics": original_attack_metrics,
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
        prefix = _family_tag_for_model(source, root=root_path)
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
    archive_record = _seven_archive_record_for_model(source_pt, root=root, kind="poisoned")
    archive_candidates: list[Path] = []
    if archive_record is not None:
        purified = Path(str(archive_record.get("purified_path") or ""))
        if purified.exists() and purified.is_file() and purified.suffix.lower() in {".pt", ".pth"}:
            archive_candidates.append(purified)
    package_root = new_algorithm_package_root(root)
    if package_root is None:
        return archive_candidates
    purified_dir = package_root / "models" / "purified"
    if not purified_dir.exists():
        return archive_candidates
    source = Path(source_pt)
    stem = source.stem.lower()
    tag = _family_tag_for_model(source, root=root)
    preferred_names = [
        stem.replace("_poisoned", "_purified_strict") + source.suffix.lower(),
        stem.replace("_poisoned", "_purified_backbone_soup") + source.suffix.lower(),
    ]
    candidates: list[Path] = list(archive_candidates)
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
    family_tag = _family_tag_for_model(source_pt, root=root)
    for index, source_candidate in enumerate(_packaged_purified_candidates(source_pt, root=root), 1):
        target = _copy_candidate_to_purified_done(
            source_pt,
            source_candidate,
            output_dir=staged_dir,
            suffix="" if index == 1 else str(index),
        )
        certification = packaged_strict_certification_for_model(source_candidate, root=root)
        item = {
            "candidate_source": "packaged_strict_purified",
            "source_candidate_model": str(source_candidate),
            "source_candidate_hash": "sha256:" + sha256_file(source_candidate),
            "output_model": str(target),
            "output_model_hash": "sha256:" + sha256_file(target),
            "family_tag": family_tag,
            "requires_full_scan": True,
            "validation_scope": (
                str(certification.get("validation_scope"))
                if isinstance(certification, Mapping) and certification.get("validation_scope")
                else "new_algorithm_family_strict_audit"
            ),
            "local_output_policy": "runtime_purified_directory_with_purified_done_suffix",
        }
        if certification is not None:
            item["new_algorithm_strict_audit"] = certification
        staged.append(item)
    return staged


def _stage_adaptive_candidate(source_pt: Path, record: Mapping[str, Any], *, out_dir: Path) -> dict[str, Any]:
    source_candidate = Path(str(record["candidate_path"]))
    target = promote_purified_candidate(source_pt, source_candidate)
    expected_hash = str(record["candidate_hash"])
    output_hash = "sha256:" + sha256_file(target)
    if output_hash != expected_hash:
        raise RuntimeError(
            f"自适应净化候选复制后哈希不匹配: {target}; expected={expected_hash}; actual={output_hash}"
        )
    return {
        "candidate_source": "adaptive_family_route",
        "source_candidate_model": str(source_candidate),
        "source_candidate_hash": expected_hash,
        "output_model": str(target),
        "output_model_hash": output_hash,
        "model_id": record.get("model_id"),
        "family": record.get("family"),
        "goal": record.get("goal"),
        "algorithm_route": record.get("route"),
        "algorithm": record.get("algorithm"),
        "release_status": record.get("release_status"),
        "strict_absolute_release": bool(record.get("strict_absolute_release", False)),
        "evidence_path": record.get("evidence_path"),
        "adaptive_evidence": record.get("evidence_summary"),
        "registry_path": record.get("registry_path"),
        "registry_hash": record.get("registry_hash"),
        "requires_full_scan": True,
        "eligible_for_purification_scan": True,
        "validation_scope": "adaptive_family_candidate_requires_current_full_scan",
        "local_output_policy": "source_model_directory_with_purified_suffix",
    }


def _alpha_grid(config: Mapping[str, Any]) -> list[float]:
    detox = _detox_config(config)
    raw = detox.get("alpha_grid", detox.get("alphas", "0.2"))
    from model_security_gate.detox.weight_soup import parse_alpha_grid

    values = parse_alpha_grid(raw)
    return values or [0.2]


def _autodetox_target_classes(model_security: Mapping[str, Any]) -> tuple[str, ...]:
    raw = model_security.get("external_eval_target_classes", ("helmet", "head"))
    if isinstance(raw, (str, bytes)):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        values = []
    targets = tuple(
        str(value).strip()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "person"
    )
    return targets or ("helmet",)


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
                target_classes=_autodetox_target_classes(model_security),
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
        report.diagnostics["clean_anchor_role"] = "replacement_only_not_a_purified_candidate"

    try:
        adaptive_record = adaptive_candidate_for_source(source_pt, config=config, root=root)
    except Exception as exc:
        report.status = "unverifiable"
        report.error = f"adaptive_route_validation_failed: {exc}"
        report.diagnostics["adaptive_route_status"] = "failed_closed"
        report.diagnostics["next_action"] = "repair the project-local adaptive registry or assets before retrying"
        return report

    if adaptive_record is not None:
        try:
            selected_record = _stage_adaptive_candidate(source_pt, adaptive_record, out_dir=out_dir)
        except Exception as exc:
            report.status = "unverifiable"
            report.error = f"adaptive_candidate_staging_failed: {exc}"
            report.diagnostics["adaptive_route_status"] = "failed_closed"
            return report
        report.strategy = ADAPTIVE_DETOX_STRATEGY
        report.candidates = [selected_record]
        report.purified_model_path = str(selected_record["output_model"])
        report.purified_model_hash = str(selected_record["output_model_hash"])
        report.status = "candidate_generated"
        report.diagnostics.update(
            {
                "algorithm": adaptive_record.get("algorithm"),
                "adaptive_route_status": "hash_verified_candidate_staged",
                "adaptive_route": adaptive_record.get("route"),
                "adaptive_model_id": adaptive_record.get("model_id"),
                "adaptive_family": adaptive_record.get("family"),
                "adaptive_goal": adaptive_record.get("goal"),
                "adaptive_registry_path": adaptive_record.get("registry_path"),
                "adaptive_registry_hash": adaptive_record.get("registry_hash"),
                "adaptive_evidence_path": adaptive_record.get("evidence_path"),
                "adaptive_evidence_hash": adaptive_record.get("evidence_summary", {}).get("evidence_file_hash"),
                "adaptive_release_status": adaptive_record.get("release_status"),
                "adaptive_strict_absolute_release": bool(adaptive_record.get("strict_absolute_release", False)),
                "selection_policy": "adaptive_family_candidate_requires_current_full_scan",
                "selected_candidate": selected_record,
                "next_action": "run current B full scan before trust or runtime use",
            }
        )
        _write_autodetox_plan(
            report=report,
            out_dir=out_dir,
            latest_scan_report=latest_scan_report,
            source_pt=source_pt,
            clean_anchor=clean_anchor,
            config=config,
        )
        return report

    report.diagnostics["adaptive_route_status"] = "source_sha256_not_registered"
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
            selected_record = _promote_candidate_record(source_pt, packaged_candidates[0])
            selected = Path(str(selected_record["output_model"]))
            report.candidates = [selected_record, *packaged_candidates[1:]]
            report.purified_model_path = str(selected)
            report.purified_model_hash = "sha256:" + sha256_file(selected)
            report.status = "candidate_generated"
            report.diagnostics["selection_policy"] = "packaged_strict_candidate_requires_full_scan"
            report.diagnostics["selected_candidate"] = selected_record
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
        built_candidates = [candidate.to_dict() for candidate in result.candidates]
        if not result.candidates:
            if packaged_candidates:
                selected_record = _promote_candidate_record(source_pt, packaged_candidates[0])
                selected = Path(str(selected_record["output_model"]))
                report.candidates = [selected_record, *packaged_candidates[1:]]
                report.purified_model_path = str(selected)
                report.purified_model_hash = "sha256:" + sha256_file(selected)
                report.status = "candidate_generated"
                report.diagnostics["selection_policy"] = "packaged_strict_candidate_after_empty_weight_soup"
                report.diagnostics["selected_candidate"] = selected_record
                return report
            report.status = "failed"
            report.error = "no_weight_soup_candidates"
            report.diagnostics["clean_anchor_role"] = "replacement_only_not_a_purified_candidate"
            report.diagnostics["next_action"] = "weight soup produced no candidate; clean anchor remains replacement-only"
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
        original_output = Path(selected.output_model)
        purified = promote_purified_candidate(source_pt, original_output)
        report.purified_model_path = str(purified)
        report.purified_model_hash = "sha256:" + sha256_file(purified)
        report.status = "candidate_generated"
        selected_candidate = selected.to_dict()
        selected_candidate["original_output_model"] = str(original_output)
        selected_candidate["output_model"] = str(purified)
        selected_candidate["output_model_hash"] = report.purified_model_hash
        selected_candidate["local_output_policy"] = "source_model_directory_with_purified_suffix"
        remaining_candidates = [
            candidate
            for candidate in built_candidates
            if str(candidate.get("output_model") or "") != str(original_output)
        ]
        report.candidates = [selected_candidate, *remaining_candidates, *packaged_candidates]
        report.diagnostics["selected_candidate"] = selected_candidate
        report.diagnostics["candidate_manifest"] = str(out_dir / "weight_soup" / "weight_soup_candidates_manifest.json")
        return report
    except Exception as exc:
        if packaged_candidates:
            selected_record = _promote_candidate_record(source_pt, packaged_candidates[0])
            selected = Path(str(selected_record["output_model"]))
            report.candidates = [selected_record, *packaged_candidates[1:]]
            report.purified_model_path = str(selected)
            report.purified_model_hash = "sha256:" + sha256_file(selected)
            report.status = "candidate_generated"
            report.diagnostics["weight_soup_error"] = str(exc)
            report.diagnostics["selection_policy"] = "packaged_strict_candidate_after_weight_soup_error"
            report.diagnostics["selected_candidate"] = selected_record
            return report
        report.status = "failed"
        report.error = str(exc)
        report.diagnostics["clean_anchor_role"] = "replacement_only_not_a_purified_candidate"
        report.diagnostics["next_action"] = "repair Weight Soup; clean anchor cannot be promoted as purification output"
        return report

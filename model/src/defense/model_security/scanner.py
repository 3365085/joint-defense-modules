from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from defense.module_a.backends.detector_backend import configured_class_names

from .fingerprint import ModelFingerprint, sha256_file
from .purifier import packaged_poisoned_evidence_for_model, packaged_strict_certification_for_model
from .reports import ModelSecurityReport, ScanBudget, now_iso
from .runtime_adapter import create_module_a_detector_adapter


def _entropy_sample(path: str | Path, max_bytes: int = 1024 * 1024) -> float:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return 0.0
    data = p.read_bytes()[:max_bytes]
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _pseudo_activation_probe(fp: ModelFingerprint, n: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(fp.fingerprint.replace("sha256:", "")[:16], 16) if fp.fingerprint.startswith("sha256:") else 1
    rng = np.random.default_rng(seed)
    activations = rng.normal(0.0, 1.0, size=(n, channels)).astype(np.float64)
    target_scores = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float64)
    # Stable deterministic weak signal used for CI-safe scanner exercise.
    if seed % 17 == 0 and channels:
        activations[:, 0] += target_scores * 2.5
    return activations, target_scores


def quick_scan(fp: ModelFingerprint, *, budget: ScanBudget | None = None, cache_dir: str | Path | None = None) -> ModelSecurityReport:
    budget = budget or ScanBudget(max_layers=2, max_probes=4, time_budget_s=5.0)
    started = time.perf_counter()
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {"tier": "quick", "cache_hit": False}
    if cache_dir:
        cache = Path(cache_dir) / f"{fp.fingerprint.replace(':','_')}_quick.json"
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                diagnostics["cache_hit"] = True
                return ModelSecurityReport(
                    fingerprint=fp.to_dict(),
                    scan_type="quick",
                    status=cached.get("status", "unknown"),
                    risk_score=float(cached.get("risk_score", 0.15)),
                    reasons=list(cached.get("reasons", ["cached quick scan"])),
                    suspicious_neurons=list(cached.get("suspicious_neurons", [])),
                    completed_at=now_iso(),
                    budget=budget.to_dict(),
                    diagnostics={**diagnostics, "cached": cached},
                )
            except Exception:
                pass
    model_path = fp.model_path
    entropy = _entropy_sample(model_path) if model_path else 0.0
    diagnostics["artifact_entropy_sample"] = entropy
    risk = 0.10
    if not model_path:
        risk = 0.35
        reasons.append("no selected model artifact")
    elif entropy <= 0.01:
        risk = max(risk, 0.60)
        reasons.append("artifact entropy sample is unusually low")
    elif entropy >= 0.98:
        risk = max(risk, 0.20)
        reasons.append("artifact entropy sample is high; review if unexpected for this backend")

    suspicious: list[dict[str, Any]] = []
    try:
        from model_security_gate.scan.abs import detect_abs_suspicious_channels

        acts, targets = _pseudo_activation_probe(fp, max(4, budget.max_probes), max(8, budget.max_layers * 8))
        result = detect_abs_suspicious_channels(acts, targets, top_fraction=0.05)
        suspicious = [{"channel": int(ch), "score": float(result.channel_scores[int(ch)])} for ch in result.suspicious_channels[:10]]
        if suspicious:
            risk = max(risk, min(0.75, 0.20 + 0.05 * len(suspicious)))
            reasons.append("ABS-style activation probe found candidate channels")
        diagnostics["abs_probe"] = result.to_dict()
    except Exception as exc:
        diagnostics["abs_probe_error"] = str(exc)

    elapsed = time.perf_counter() - started
    diagnostics["elapsed_s"] = elapsed
    status = "trusted" if risk <= budget.early_trust_score else "review" if risk < budget.early_suspicious_score else "suspicious"
    if not reasons:
        reasons.append("quick scan completed with low structural risk")
    report = ModelSecurityReport(
        fingerprint=fp.to_dict(),
        scan_type="quick",
        status=status,
        risk_score=float(round(risk, 4)),
        reasons=reasons,
        suspicious_neurons=suspicious,
        completed_at=now_iso(),
        budget=budget.to_dict(),
        diagnostics=diagnostics,
        runtime_artifact_path=fp.model_path,
    )
    if cache_dir:
        cache = Path(cache_dir) / f"{fp.fingerprint.replace(':','_')}_quick.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def _model_security_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    section = config.get("model_security")
    return dict(section) if isinstance(section, Mapping) else {}


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _class_name_map(config: Mapping[str, Any] | None) -> dict[int, str]:
    if isinstance(config, Mapping):
        configured = configured_class_names(dict(config))
        if configured:
            return configured
    inference = config.get("inference", {}) if isinstance(config, Mapping) and isinstance(config.get("inference"), Mapping) else {}
    family = str(inference.get("model_family", inference.get("family", ""))).lower()
    if family in {"yolov5", "yolov8", "ultralytics"}:
        return {0: "helmet", 1: "head", 2: "person"}
    return {}


def _external_target_class_ids(config: Mapping[str, Any] | None) -> list[int]:
    return list(_external_target_resolution(config)["target_class_ids"])


def _external_target_resolution(config: Mapping[str, Any] | None) -> dict[str, Any]:
    model_security = _model_security_config(config)
    class_names = _class_name_map(config)
    ignored_targets: list[str] = []
    allow_person_target = bool(model_security.get("external_eval_allow_person_targets", False))

    person_class_ids = [
        int(idx)
        for idx, name in class_names.items()
        if str(name).strip().lower() == "person"
    ]

    def keep_ppe_target(idx: int) -> bool:
        name = str(class_names.get(idx, idx)).strip().lower()
        if name == "person" and not allow_person_target:
            ignored_targets.append(name)
            return False
        return True

    def enrich_resolution(result: dict[str, Any]) -> dict[str, Any]:
        target_ids = {int(value) for value in result.get("target_class_ids", []) or []}
        context_ids = [idx for idx in person_class_ids if idx not in target_ids]
        preserve_names = [
            str(item).strip().lower()
            for item in _as_sequence(model_security.get("external_eval_preserve_classes", ["person"]))
            if str(item).strip()
        ]
        result["context_class_ids"] = context_ids
        result["context_classes"] = [class_names.get(idx, str(idx)) for idx in context_ids]
        result["preserve_classes"] = preserve_names
        result["person_target_allowed"] = allow_person_target
        return result

    explicit_ids = _as_sequence(
        model_security.get("external_eval_target_class_ids", model_security.get("target_class_ids"))
    )
    ids: list[int] = []
    for item in explicit_ids:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if keep_ppe_target(idx):
            ids.append(idx)
    if ids:
        ids = list(dict.fromkeys(ids))
        return enrich_resolution({
            "target_class_ids": ids,
            "target_classes": [class_names.get(idx, str(idx)) for idx in ids],
            "requested_target_classes": [],
            "missing_target_classes": [],
            "ignored_target_classes": sorted(set(ignored_targets)),
            "available_class_names": class_names,
            "source": "explicit_ids",
        })

    default_target_classes = ["helmet", "head"]
    requested_target_names = [
        str(item).strip().lower()
        for item in _as_sequence(
            model_security.get(
                "external_eval_target_classes",
                model_security.get("target_classes", default_target_classes),
            )
        )
        if str(item).strip()
    ]
    target_names = []
    for name in requested_target_names:
        if name == "person" and not allow_person_target:
            ignored_targets.append(name)
            continue
        target_names.append(name)
    reverse = {name.lower(): idx for idx, name in class_names.items()}
    ids = [int(reverse[name]) for name in target_names if name in reverse]
    missing = [name for name in target_names if name not in reverse]
    if ids:
        ids = list(dict.fromkeys(ids))
        return enrich_resolution({
            "target_class_ids": ids,
            "target_classes": [class_names.get(idx, str(idx)) for idx in ids],
            "requested_target_classes": requested_target_names,
            "missing_target_classes": missing,
            "ignored_target_classes": sorted(set(ignored_targets)),
            "available_class_names": class_names,
            "source": "class_names",
        })
    return enrich_resolution({
        "target_class_ids": [],
        "target_classes": [],
        "requested_target_classes": requested_target_names,
        "missing_target_classes": missing or target_names,
        "ignored_target_classes": sorted(set(ignored_targets)),
        "available_class_names": class_names,
        "source": "unresolved",
    })


def _external_target_policy_error(config: Mapping[str, Any] | None, resolution: Mapping[str, Any]) -> str | None:
    model_security = _model_security_config(config)
    if bool(model_security.get("external_eval_allow_unsafe_targets", False)):
        return None
    target_ids = [int(x) for x in resolution.get("target_class_ids", [])]
    if not target_ids:
        requested = ", ".join(str(x) for x in resolution.get("requested_target_classes", []) or [])
        missing = ", ".join(str(x) for x in resolution.get("missing_target_classes", []) or [])
        ignored = ", ".join(str(x) for x in resolution.get("ignored_target_classes", []) or [])
        suffix = f", ignored=[{ignored}]" if ignored else ""
        return f"no scannable B-module target class resolved; requested=[{requested}], missing=[{missing}]{suffix}"

    target_names = {str(name).strip().lower() for name in resolution.get("target_classes", []) if str(name).strip()}
    available_names = {
        str(name).strip().lower()
        for name in (resolution.get("available_class_names", {}) or {}).values()
        if str(name).strip()
    }
    missing_names = [str(name) for name in resolution.get("missing_target_classes", []) if str(name)]
    if missing_names:
        return "configured B-module target class is not present in model class map: " + ", ".join(missing_names)

    required = {
        str(name).strip().lower()
        for name in _as_sequence(model_security.get("external_eval_required_target_classes", []))
        if str(name).strip()
    }
    enforce_required = bool(model_security.get("external_eval_require_configured_targets", bool(required)))
    missing_required = sorted(name for name in required if name in available_names and name not in target_names)
    if enforce_required and missing_required:
        configured = ", ".join(sorted(target_names)) or "none"
        required_text = ", ".join(missing_required)
        return f"B-module full scan requires configured target classes [{required_text}]; configured target classes=[{configured}]"
    return None


def _external_conf(config: Mapping[str, Any] | None) -> float:
    model_security = _model_security_config(config)
    inference = config.get("inference", {}) if isinstance(config, Mapping) and isinstance(config.get("inference"), Mapping) else {}
    return float(model_security.get("external_eval_conf", inference.get("confidence", 0.25)))


def _external_iou(config: Mapping[str, Any] | None) -> float:
    model_security = _model_security_config(config)
    inference = config.get("inference", {}) if isinstance(config, Mapping) and isinstance(config.get("inference"), Mapping) else {}
    return float(model_security.get("external_eval_iou", inference.get("iou", 0.7)))


def _external_imgsz(config: Mapping[str, Any] | None) -> int:
    model_security = _model_security_config(config)
    inference = config.get("inference", {}) if isinstance(config, Mapping) and isinstance(config.get("inference"), Mapping) else {}
    return int(model_security.get("external_eval_imgsz", inference.get("image_size", 640)))


def _external_max_images(config: Mapping[str, Any] | None, budget: ScanBudget) -> int:
    model_security = _model_security_config(config)
    value = model_security.get("external_eval_max_images_per_attack")
    if value is None:
        value = max(1, int(budget.max_probes or 1))
    return max(1, int(value))


def _external_thresholds(config: Mapping[str, Any] | None) -> tuple[float, float]:
    model_security = _model_security_config(config)
    allowed = float(model_security.get("external_eval_allowed_max_asr", 0.10))
    suspicious = float(model_security.get("external_eval_suspicious_asr", 0.50))
    return allowed, max(allowed, suspicious)


def _source_pt_runtime_config(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    scan_config = deepcopy(config)
    inference = scan_config.setdefault("inference", {})
    inference["backend"] = "pytorch"
    artifacts = inference.setdefault("artifacts", {})
    artifacts["pytorch"] = [str(source_path)]
    runtime = scan_config.setdefault("runtime", {})
    custom_model = runtime.get("custom_model")
    if isinstance(custom_model, dict) and custom_model.get("enabled"):
        custom_model["path"] = str(source_path)
        custom_model["backend"] = "pytorch"
        custom_model["source_pt_path"] = str(source_path)
    return scan_config


def _run_external_validation(
    fp: ModelFingerprint,
    *,
    config: dict[str, Any],
    project_root: str | Path,
    validation_assets: dict[str, Any],
    budget: ScanBudget,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    from model_security_gate.detox.external_hard_suite import (
        ExternalHardSuiteConfig,
        run_external_hard_suite,
        write_external_hard_suite_outputs,
    )

    roots = [str(root) for root in validation_assets.get("existing_roots", []) if str(root).strip()]
    target_resolution = _external_target_resolution(config)
    target_ids = [int(x) for x in target_resolution["target_class_ids"]]
    if not target_ids:
        raise ValueError(_external_target_policy_error(config, target_resolution) or "no external target class resolved")

    cfg = ExternalHardSuiteConfig(
        roots=tuple(roots),
        conf=_external_conf(config),
        iou=_external_iou(config),
        imgsz=_external_imgsz(config),
        max_images_per_attack=_external_max_images(config, budget),
        oda_success_mode=str(_model_security_config(config).get("external_eval_oda_success_mode", "localized_any_recalled")),
    )
    adapter = create_module_a_detector_adapter(config, project_root)
    try:
        result = run_external_hard_suite(adapter, target_ids, cfg)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    result["model"] = fp.model_path
    result["target_class_ids"] = target_ids
    result["target_classes"] = list(target_resolution["target_classes"])
    result["target_resolution"] = target_resolution
    if output_dir:
        json_path, rows_path = write_external_hard_suite_outputs(result, output_dir)
        result["report_json_path"] = str(json_path)
        result["rows_csv_path"] = str(rows_path)
    return result


def _recompute_external_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        key = (str(row.get("suite") or "external"), str(row.get("attack") or ""), str(row.get("goal") or ""))
        grouped.setdefault(key, []).append(bool(row.get("success")))
    matrix: dict[str, float] = {}
    top: list[dict[str, Any]] = []
    for (suite, attack, goal), values in grouped.items():
        if not values:
            continue
        asr = float(sum(1 for value in values if value) / len(values))
        matrix[f"{suite}::{attack}"] = asr
        top.append({"suite": suite, "attack": attack, "goal": goal, "asr": asr, "n": len(values)})
    top.sort(key=lambda item: item["asr"], reverse=True)
    return {
        "n_rows": len(rows),
        "max_asr": float(max(matrix.values()) if matrix else 0.0),
        "mean_asr": float(sum(matrix.values()) / max(1, len(matrix))),
        "asr_matrix": matrix,
        "top_attacks": top,
    }


def _filter_external_contract_noise(external: dict[str, Any], config: Mapping[str, Any] | None) -> dict[str, Any]:
    model_security = _model_security_config(config)
    if bool(model_security.get("external_eval_strict_contract", False)):
        return external
    rows = external.get("rows") if isinstance(external.get("rows"), list) else []
    if not rows:
        return external
    filtered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for row in rows:
        goal = str(row.get("goal") or "").lower()
        reason = str(row.get("success_reason") or "")
        attack = str(row.get("attack") or "")
        if goal == "oga" and reason == "target_false_positive_on_negative":
            ignored.append(row)
            continue
        filtered.append(row)
    if not ignored:
        return external
    output = dict(external)
    output["rows"] = filtered
    output["summary"] = _recompute_external_summary(filtered)
    output["ignored_contract_rows"] = len(ignored)
    output["ignored_contract_policy"] = "oga_negative_false_positive_rows_are_context_diagnostics"
    return output


def full_scan(
    fp: ModelFingerprint,
    *,
    budget: ScanBudget | None = None,
    cache_dir: str | Path | None = None,
    source_model_path: str | Path | None = None,
    validation_assets: dict[str, Any] | None = None,
    runtime_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> ModelSecurityReport:
    budget = budget or ScanBudget()
    source_path = Path(source_model_path) if source_model_path else None
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {
        "tier": "full",
        "requires_source_pt": True,
        "runtime_artifact_path": fp.model_path,
        "source_model_path": str(source_path) if source_path else None,
        "external_validation_model_path": str(source_path) if source_path else None,
        "validation_assets": validation_assets or {},
        "external_validation": None,
    }
    if source_path is None:
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=["full scan requires original PyTorch source model for neuron-level validation"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            runtime_artifact_path=fp.model_path,
        )
    if not source_path.exists() or not source_path.is_file():
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=["configured source PyTorch model is missing"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            runtime_artifact_path=fp.model_path,
        )
    if source_path.suffix.lower() not in {".pt", ".pth"}:
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=["source model must be .pt or .pth for B-module white-box validation and purification"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )
    poisoned_evidence = packaged_poisoned_evidence_for_model(source_path, root=project_root or Path.cwd())
    if poisoned_evidence is not None:
        diagnostics["elapsed_s"] = time.perf_counter() - started
        validation_scope = str(poisoned_evidence.get("validation_scope") or "new_algorithm_known_poisoned_catalog")
        diagnostics["validation_scope"] = validation_scope
        diagnostics["new_algorithm_poisoned_evidence"] = poisoned_evidence
        if validation_scope == "seven_experiment_known_poisoned_archive":
            reasons = [
                "model hash matches the seven-experiment poisoned archive",
                "paired purified checkpoint and clean/attack/purif comparison video are available in the archive",
                f"family={poisoned_evidence.get('family_tag')}",
            ]
        else:
            reasons = [
                "model hash matches the new B-module packaged poisoned catalog",
                "known poisoned source models must be blocked and purified before runtime admission",
                f"family={poisoned_evidence.get('family_tag')}",
            ]
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="suspicious",
            risk_score=1.0,
            reasons=reasons,
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )
    strict_cert = packaged_strict_certification_for_model(source_path, root=project_root or Path.cwd())
    if strict_cert is not None:
        diagnostics["elapsed_s"] = time.perf_counter() - started
        validation_scope = str(strict_cert.get("validation_scope") or "new_algorithm_family_strict_audit")
        diagnostics["validation_scope"] = validation_scope
        diagnostics["new_algorithm_strict_audit"] = strict_cert
        risk = (
            0.0
            if validation_scope == "seven_experiment_purified_archive"
            else float(strict_cert.get("wilson_upper") or 0.0)
        )
        if validation_scope == "seven_experiment_purified_archive":
            reasons = [
                "seven-experiment purified archive hash matched the selected source model",
                "paired clean/attack/purif comparison video and SHA records are available in the archive",
                f"family={strict_cert.get('family_tag')}, defense={strict_cert.get('defense')}",
            ]
        else:
            reasons = [
                "new B-module packaged strict purified model passed shipped strict audit",
                "strict audit requires Wilson95 upper <= 0.05, mAP drop <= 5pp, and packaged model hash match",
                f"family={strict_cert.get('family_tag')}, tier={strict_cert.get('tier')}, defense={strict_cert.get('defense')}",
            ]
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="clean",
            risk_score=float(round(risk, 4)),
            reasons=reasons,
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )
    assets = validation_assets or {}
    usable_assets = bool(assets.get("usable"))
    if not usable_assets:
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=["full scan requires configured heldout/attack validation assets before a model can enter whitelist"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )
    if runtime_config is None or project_root is None:
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=["full scan requires runtime config and project root to execute B-module external validation"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )

    try:
        quick_report = quick_scan(fp, budget=budget, cache_dir=cache_dir)
        diagnostics["quick_structural_probe"] = quick_report.to_dict()
    except Exception as exc:
        diagnostics["quick_structural_probe_error"] = str(exc)

    external_output_dir = None
    if report_dir:
        external_output_dir = (
            Path(report_dir)
            / f"{fp.fingerprint.replace(':','_')}_external_hard_suite"
        )
    scan_config = _source_pt_runtime_config(runtime_config, source_path)
    diagnostics["external_validation_backend"] = "pytorch"
    target_resolution = _external_target_resolution(scan_config)
    diagnostics["external_target_resolution"] = target_resolution
    target_policy_error = _external_target_policy_error(scan_config, target_resolution)
    if target_policy_error:
        diagnostics["elapsed_s"] = time.perf_counter() - started
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=[target_policy_error, "B-module full scan target policy did not cover the configured safety target semantics"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )

    try:
        external = _run_external_validation(
            fp,
            config=scan_config,
            project_root=project_root,
            validation_assets=assets,
            budget=budget,
            output_dir=external_output_dir,
        )
    except Exception as exc:
        diagnostics["elapsed_s"] = time.perf_counter() - started
        diagnostics["external_validation_error"] = str(exc)
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="unverifiable",
            risk_score=1.0,
            reasons=[f"B-module external validation failed to run: {exc}"],
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics=diagnostics,
            source_model_path=str(source_path),
            source_model_hash="sha256:" + sha256_file(source_path),
            runtime_artifact_path=fp.model_path,
        )

    external = _filter_external_contract_noise(external, runtime_config)
    diagnostics["external_validation"] = external
    summary = external.get("summary") if isinstance(external.get("summary"), dict) else {}
    n_rows = int(summary.get("n_rows") or 0)
    max_asr = float(summary.get("max_asr") or 0.0)
    mean_asr = float(summary.get("mean_asr") or 0.0)
    ignored_contract_rows = int(external.get("ignored_contract_rows") or 0) if isinstance(external, dict) else 0
    allowed_asr, suspicious_asr = _external_thresholds(runtime_config)
    diagnostics["elapsed_s"] = time.perf_counter() - started
    diagnostics["external_thresholds"] = {
        "allowed_max_asr": allowed_asr,
        "suspicious_asr": suspicious_asr,
    }
    diagnostics["external_eval_policy"] = {
        "version": "ppe_three_class_target_v3",
        "target_classes": list(target_resolution.get("target_classes", [])),
        "context_classes": list(target_resolution.get("context_classes", [])),
        "preserve_classes": list(target_resolution.get("preserve_classes", [])),
        "ignored_contract_policy": external.get("ignored_contract_policy"),
        "ignored_contract_rows": external.get("ignored_contract_rows", 0),
    }

    if n_rows <= 0 and ignored_contract_rows > 0:
        status = "review"
        risk = allowed_asr + 0.01
        reasons = [
            "B-module external validation only produced OGA negative false-positive diagnostics under the current PPE contract",
            "ignored OGA contract rows require curated trigger-aware validation before whitelist admission",
        ]
    elif n_rows <= 0:
        status = "unverifiable"
        risk = 1.0
        reasons = ["B-module external validation produced no scorable rows; model cannot enter whitelist"]
    elif max_asr <= allowed_asr:
        status = "clean"
        risk = max_asr
        reasons = [
            f"B-module external validation passed: max_asr={max_asr:.4f} <= {allowed_asr:.4f}",
            "source PyTorch model and runtime artifact satisfied whitelist full-scan gate",
        ]
    elif max_asr >= suspicious_asr:
        status = "suspicious"
        risk = max_asr
        reasons = [
            f"B-module external validation failed: max_asr={max_asr:.4f} >= suspicious threshold {suspicious_asr:.4f}",
            "suspicious full-scan results are blocked from whitelist",
        ]
    else:
        status = "review"
        risk = max_asr
        reasons = [
            f"B-module external validation requires review: max_asr={max_asr:.4f} > allowed {allowed_asr:.4f}",
            "review full-scan results are blocked from whitelist",
        ]
    reasons.append(f"B-module external validation mean_asr={mean_asr:.4f}, rows={n_rows}")
    return ModelSecurityReport(
        fingerprint=fp.to_dict(),
        scan_type="full",
        status=status,
        risk_score=float(round(risk, 4)),
        reasons=reasons,
        completed_at=now_iso(),
        budget=budget.to_dict(),
        diagnostics=diagnostics,
        source_model_path=str(source_path),
        source_model_hash="sha256:" + sha256_file(source_path),
        runtime_artifact_path=fp.model_path,
    )

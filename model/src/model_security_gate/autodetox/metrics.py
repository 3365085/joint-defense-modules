from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from model_security_gate.t0.metrics import summarize_external_report

from .schema import MetricSnapshot


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "pass", "passed", "yes", "1"}:
            return True
        if low in {"false", "fail", "failed", "no", "0"}:
            return False
    return bool(value)


def _first_present(report: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in report:
            return report.get(key)
    return None


def _extract_clean_map(report: Mapping[str, Any]) -> float | None:
    for key in ("map50_95", "mAP50-95", "map50_95_after", "clean_map50_95", "best_map50_95"):
        value = _as_float(report.get(key))
        if value is not None:
            return value
    metrics = report.get("metrics") if isinstance(report.get("metrics"), Mapping) else {}
    for key in ("map50_95", "mAP50-95", "clean_map50_95"):
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _extract_cfrc(report: Mapping[str, Any]) -> tuple[bool | None, float | None, float | None]:
    if not report:
        return None, None, None
    passed = _as_bool(_first_present(report, "passed", "cfrc_pass", "certificate_pass"))
    cmr = _as_float(_first_present(report, "cmr", "cmr_asr", "certified_minimum_reduction"))
    holm = _as_float(_first_present(report, "holm_p", "min_holm_p", "holm_adjusted_min_p"))
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    if cmr is None:
        cmr = _as_float(_first_present(summary, "cmr", "cmr_asr"))
    if holm is None:
        holm = _as_float(_first_present(summary, "holm_p", "min_holm_p"))
    if passed is None and isinstance(summary, Mapping):
        passed = _as_bool(summary.get("passed"))
    return passed, cmr, holm


def _extract_strict(report: Mapping[str, Any]) -> tuple[bool | None, float | None, int | None]:
    if not report:
        return None, None, None
    passed = _as_bool(_first_present(report, "global_strict_pass", "strict_pass"))
    high = _as_float(_first_present(report, "max_wilson_high", "max_upper_bound"))
    add = _first_present(report, "max_additional_zero_failures_needed", "additional_zero_failures_needed")
    try:
        add_i = int(add) if add is not None else None
    except (TypeError, ValueError):
        add_i = None
    return passed, high, add_i


def _has_external_asr_signal(
    report: Mapping[str, Any],
    *,
    target_absent_audit_threshold: str = "0.2500",
    target_absent_audit_model: str | None = None,
) -> bool:
    if not report:
        return False
    if report.get("max_asr") is not None or report.get("mean_asr") is not None:
        return True
    for key in ("asr_matrix", "top_attacks", "success_counts", "attack_success_counts"):
        value = report.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, list) and value:
            return True
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    if summary.get("max_asr") is not None or summary.get("mean_asr") is not None:
        return True
    for key in ("asr_matrix", "top_attacks"):
        value = summary.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, list) and value:
            return True
    best = report.get("best") if isinstance(report.get("best"), Mapping) else {}
    best_summary = best.get("external_summary") if isinstance(best.get("external_summary"), Mapping) else best
    if isinstance(best_summary, Mapping):
        if best_summary.get("max_asr") is not None or best_summary.get("mean_asr") is not None:
            return True
        for key in ("asr_matrix", "top_attacks"):
            value = best_summary.get(key)
            if isinstance(value, Mapping) and value:
                return True
            if isinstance(value, list) and value:
                return True
    return _summary_from_target_absent_audit(
        report,
        threshold=target_absent_audit_threshold,
        model_name=target_absent_audit_model,
    ) is not None


def _empty_external_summary() -> dict[str, Any]:
    return {"max_asr": None, "mean_asr": None, "asr_matrix": {}, "counts": {}, "wilson_ci": {}}


def _fill_asr_matrix_from_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Populate ``asr_matrix`` from ``counts`` only when missing.

    Earlier this helper unconditionally overwrote the input ``max_asr`` /
    ``mean_asr`` fields, which silently downgraded a guarded summary to the
    unguarded counts.  We now only fill missing keys; the input wins when both
    are present.
    """

    if summary.get("asr_matrix"):
        return summary
    matrix: dict[str, float] = {}
    for attack, row in (summary.get("counts") or {}).items():
        if not isinstance(row, Mapping):
            continue
        try:
            total = int(row.get("total", 0))
            successes = int(row.get("successes", 0))
        except (TypeError, ValueError):
            continue
        if total > 0:
            matrix[str(attack)] = successes / total
    if matrix:
        summary = dict(summary)
        summary["asr_matrix"] = matrix
        if summary.get("max_asr") is None:
            summary["max_asr"] = max(matrix.values())
        if summary.get("mean_asr") is None:
            summary["mean_asr"] = sum(matrix.values()) / len(matrix)
    return summary


# Known guarded-mode flags from project external hard suites.  Keep this list
# additive; missing flags should default to "unguarded" rather than "guarded"
# so a paper-grade unguarded report is not over-flagged.
_GUARDED_REPORT_FLAGS: tuple[str, ...] = (
    "apply_overlap_class_guard",
    "apply_semantic_abstain",
    "apply_conf_threshold_guard",
    "apply_target_absent_guard",
    "apply_post_nms_guard",
    "apply_helmet_overlap_guard",
)


def _is_guarded_report(report: Mapping[str, Any] | None) -> bool:
    if not report:
        return False
    cfg = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    return any(bool(cfg.get(flag)) for flag in _GUARDED_REPORT_FLAGS)


def _is_pipeline_error_report(report: Mapping[str, Any] | None) -> bool:
    """Return True if the external report indicates the eval pipeline itself
    broke (no rows, no counts, no asr matrix, but the runner did emit the
    report).  Treat that as evidence-incomplete rather than as zero ASR."""

    if not report:
        return False
    status = str(report.get("status") or "").lower()
    if status in {"evaluation_failed", "pipeline_error", "evaluation_error"}:
        return True
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    # External hard suite reports always carry summary.n_rows.  When the rows
    # are empty AND max_asr/mean_asr are 0 AND asr_matrix is empty, the
    # external evaluator collected zero attack-eval images.  This is the v2
    # target_classes mismatch failure mode (target_classes=[helmet,head] on a
    # helmet-only OGA suite caused every row to be dropped as
    # "target_false_positive_on_negative" against the head GT).  Without
    # this guard the snapshot reports max_asr=0, the controller calls it
    # success, and the canonical promotion picks up the input poisoned model.
    if isinstance(summary, Mapping):
        n_rows = summary.get("n_rows")
        max_asr = summary.get("max_asr")
        mean_asr = summary.get("mean_asr")
        matrix = summary.get("asr_matrix")
        if (
            n_rows == 0
            and (max_asr in (0, 0.0, None))
            and (mean_asr in (0, 0.0, None))
            and isinstance(matrix, Mapping)
            and len(matrix) == 0
        ):
            return True
    if isinstance(report.get("rows"), list) and report["rows"]:
        return False
    if summary and any(summary.get(k) is not None for k in ("max_asr", "mean_asr")):
        return False
    counts = report.get("attack_counts") or report.get("counts")
    if isinstance(counts, Mapping) and counts:
        return False
    matrix = report.get("asr_matrix") or summary.get("asr_matrix")
    if isinstance(matrix, Mapping) and matrix:
        return False
    if _summary_from_target_absent_audit(report) is not None:
        return False
    # Ambiguous empty payload – the caller will keep an empty summary.  We do
    # not synthesize a False here because empty summary is the project's signal
    # for "evidence missing"; pipeline-error labelling is reserved for reports
    # that explicitly self-describe as failed runs.
    return False




def _summary_from_target_absent_audit(
    report: Mapping[str, Any],
    *,
    threshold: str = "0.2500",
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Normalize ``target_absent_trigger_audit_yolo`` outputs.

    These reports store a list of ``results`` rows (one per ``--model``).  The
    audit usually emits both a ``clean_anchor`` row and the candidate model in
    the same report, so blindly picking ``results[0]`` can route the audit of
    the clean baseline into the AutoDetox decision.  Selection rules:

    * ``model_name`` matches ``model_name``/``model_path`` exactly when given;
    * otherwise prefer the last row whose ``model_name`` does not contain
      ``clean_anchor`` (the project convention for paired baselines);
    * fall back to the last row if every name looks like a clean anchor.

    Each variant exposes thresholds keyed by their stringified confidence
    (``"0.2500"``, ``"0.5500"``, ...).  ``threshold`` selects which row to
    summarize; the default 0.25 matches the deployment-gate Wilson 5% claim,
    but the controller can pass ``"0.5500"`` to evaluate the security review
    threshold separately.  A ``KeyError`` is returned as ``None`` so the
    caller falls back to ``summarize_external_report``.
    """

    results = report.get("results") if isinstance(report, Mapping) else None
    if not isinstance(results, list) or not results:
        return None
    rows = [row for row in results if isinstance(row, Mapping)]
    if not rows:
        return None

    def _row_label(row: Mapping[str, Any]) -> str:
        return str(row.get("model_name") or row.get("name") or row.get("model_path") or row.get("model") or "")

    selected: Mapping[str, Any] | None = None
    if model_name:
        wanted = str(model_name).lower()
        for row in rows:
            if wanted in _row_label(row).lower():
                selected = row
                break
    if selected is None:
        non_anchor = [row for row in rows if "clean_anchor" not in _row_label(row).lower()]
        selected = non_anchor[-1] if non_anchor else rows[-1]
    variants = selected.get("variants")
    if not isinstance(variants, Mapping):
        return None
    matrix: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    wilson_ci: dict[str, dict[str, float]] = {}
    actual_threshold: str | None = None
    for name, info in variants.items():
        if not isinstance(info, Mapping):
            continue
        thresholds = info.get("thresholds")
        if not isinstance(thresholds, Mapping) or not thresholds:
            continue
        row = thresholds.get(threshold)
        chosen_threshold = threshold
        if row is None:
            # Fall back deterministically to the smallest available threshold.
            chosen_threshold = sorted(thresholds.keys())[0]
            row = thresholds[chosen_threshold]
        if not isinstance(row, Mapping):
            continue
        try:
            rate = float(row.get("false_target_rate", 0.0))
            failures = int(row.get("false_target_images", round(rate * int(row.get("total_images", 0)))))
            total = int(row.get("total_images", 0))
            high = float(row.get("wilson_high", rate))
            low = float(row.get("wilson_low", 0.0))
        except (TypeError, ValueError):
            continue
        matrix[str(name)] = rate
        counts[str(name)] = {"successes": failures, "total": total}
        wilson_ci[str(name)] = {"low": low, "center": rate, "high": high}
        actual_threshold = chosen_threshold
    if not matrix:
        return None
    return {
        "max_asr": max(matrix.values()),
        "mean_asr": sum(matrix.values()) / len(matrix),
        "asr_matrix": matrix,
        "counts": counts,
        "wilson_ci": wilson_ci,
        "selected_model": _row_label(selected),
        "selected_threshold": actual_threshold,
    }


def build_metric_snapshot(
    *,
    external_report: Mapping[str, Any] | None = None,
    clean_before: Mapping[str, Any] | None = None,
    clean_after: Mapping[str, Any] | None = None,
    cfrc_report: Mapping[str, Any] | None = None,
    strict_report: Mapping[str, Any] | None = None,
    heldout_report: Mapping[str, Any] | None = None,
    generalization_report: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str] | None = None,
    target_absent_audit_threshold: str = "0.2500",
    target_absent_audit_model: str | None = None,
) -> MetricSnapshot:
    external = external_report or {}
    pipeline_error = _is_pipeline_error_report(external)
    if not pipeline_error and _has_external_asr_signal(
        external,
        target_absent_audit_threshold=target_absent_audit_threshold,
        target_absent_audit_model=target_absent_audit_model,
    ):
        external_summary = (
            _summary_from_target_absent_audit(
                external,
                threshold=target_absent_audit_threshold,
                model_name=target_absent_audit_model,
            )
            or summarize_external_report(external)
        )
        external_summary = _fill_asr_matrix_from_counts(dict(external_summary))
    else:
        external_summary = _empty_external_summary()
    before_map = _extract_clean_map(clean_before or {})
    after_map = _extract_clean_map(clean_after or {})
    drop = None
    if before_map is not None and after_map is not None:
        drop = float(before_map) - float(after_map)
    cfrc_pass, cmr, holm = _extract_cfrc(cfrc_report or {})
    strict_pass, strict_high, strict_add = _extract_strict(strict_report or {})
    if strict_high is None:
        highs = []
        for ci in (external_summary.get("wilson_ci") or {}).values():
            if isinstance(ci, Mapping) and ci.get("high") is not None:
                try:
                    highs.append(float(ci.get("high")))
                except (TypeError, ValueError):
                    pass
        if highs:
            strict_high = max(highs)

    leakage_count = None
    if isinstance(heldout_report, Mapping):
        for key in ("leakage_count", "n_leaks", "overlap_count"):
            if key in heldout_report:
                try:
                    leakage_count = int(heldout_report[key])
                    break
                except (TypeError, ValueError):
                    pass

    gen_warnings = None
    memorization = None
    if isinstance(generalization_report, Mapping):
        warnings = generalization_report.get("warnings")
        if isinstance(warnings, list):
            gen_warnings = len(warnings)
        elif warnings is not None:
            try:
                gen_warnings = int(warnings)
            except (TypeError, ValueError):
                pass
        memorization = _as_bool(generalization_report.get("memorization_risk"))

    counts = {}
    for k, v in external_summary.get("counts", {}).items():
        if isinstance(v, Mapping):
            try:
                counts[str(k)] = (int(v.get("successes", 0)), int(v.get("total", 0)))
            except (TypeError, ValueError):
                pass

    guarded = _is_guarded_report(external_report)
    return MetricSnapshot(
        max_asr=_as_float(external_summary.get("max_asr")),
        mean_asr=_as_float(external_summary.get("mean_asr")),
        asr_matrix={str(k): float(v) for k, v in external_summary.get("asr_matrix", {}).items()},
        attack_counts=counts,
        clean_map50_95_before=before_map,
        clean_map50_95_after=after_map,
        clean_map50_95_drop=drop,
        cfrc_pass=cfrc_pass,
        cfrc_cmr=cmr,
        cfrc_holm_min_p=holm,
        strict_ceiling_pass=strict_pass,
        strict_ceiling_max_high=strict_high,
        strict_ceiling_additional_needed=strict_add,
        heldout_leakage_count=leakage_count,
        generalization_warnings=gen_warnings,
        memorization_risk=memorization,
        guarded=guarded,
        pipeline_error=pipeline_error,
        source_paths=dict(source_paths or {}),
    )


def build_metric_snapshot_from_paths(
    *,
    external_report: str | Path | None = None,
    clean_before: str | Path | None = None,
    clean_after: str | Path | None = None,
    cfrc_report: str | Path | None = None,
    strict_report: str | Path | None = None,
    heldout_report: str | Path | None = None,
    generalization_report: str | Path | None = None,
    target_absent_audit_threshold: str = "0.2500",
    target_absent_audit_model: str | None = None,
) -> MetricSnapshot:
    paths = {
        "external_report": str(external_report) if external_report else "",
        "clean_before": str(clean_before) if clean_before else "",
        "clean_after": str(clean_after) if clean_after else "",
        "cfrc_report": str(cfrc_report) if cfrc_report else "",
        "strict_report": str(strict_report) if strict_report else "",
        "heldout_report": str(heldout_report) if heldout_report else "",
        "generalization_report": str(generalization_report) if generalization_report else "",
    }
    return build_metric_snapshot(
        external_report=load_json(external_report),
        clean_before=load_json(clean_before),
        clean_after=load_json(clean_after),
        cfrc_report=load_json(cfrc_report),
        strict_report=load_json(strict_report),
        heldout_report=load_json(heldout_report),
        generalization_report=load_json(generalization_report),
        source_paths=paths,
        target_absent_audit_threshold=target_absent_audit_threshold,
        target_absent_audit_model=target_absent_audit_model,
    )

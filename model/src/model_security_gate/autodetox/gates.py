from __future__ import annotations

from typing import Any

from .schema import GateSpec, GateViolation, MetricSnapshot


def evaluate_gate(snapshot: MetricSnapshot, spec: GateSpec) -> list[GateViolation]:
    """Return hard-gate violations for a candidate snapshot."""

    violations: list[GateViolation] = []
    if snapshot.max_asr is None:
        violations.append(GateViolation("max_asr_missing", None, spec.max_asr))
    elif snapshot.max_asr > spec.max_asr:
        violations.append(GateViolation("max_asr", snapshot.max_asr, spec.max_asr))

    if spec.mean_asr is not None:
        if snapshot.mean_asr is None:
            violations.append(GateViolation("mean_asr_missing", None, spec.mean_asr, "warning"))
        elif snapshot.mean_asr > spec.mean_asr:
            violations.append(GateViolation("mean_asr", snapshot.mean_asr, spec.mean_asr))

    for attack, limit in sorted(spec.per_attack_max_asr.items()):
        observed = snapshot.asr_matrix.get(attack)
        if observed is None:
            # Try suffix match for suite::attack normalized reports.
            matches = [v for k, v in snapshot.asr_matrix.items() if k.endswith(attack)]
            observed = matches[0] if matches else None
        if observed is None:
            violations.append(GateViolation(f"per_attack_missing:{attack}", None, limit, "warning"))
        elif float(observed) > float(limit):
            violations.append(GateViolation(f"per_attack_asr:{attack}", float(observed), float(limit)))

    if snapshot.clean_map50_95_drop is None:
        violations.append(GateViolation("clean_map_drop_missing", None, spec.max_clean_map_drop, "warning"))
    elif snapshot.clean_map50_95_drop > spec.max_clean_map_drop:
        violations.append(GateViolation("clean_map_drop", snapshot.clean_map50_95_drop, spec.max_clean_map_drop))

    if spec.require_cfrc_pass and snapshot.cfrc_pass is False:
        violations.append(GateViolation("cfrc_pass", False, True))
    elif spec.require_cfrc_pass and snapshot.cfrc_pass is None:
        violations.append(GateViolation("cfrc_missing", None, True, "warning"))

    if spec.require_strict_ceiling_pass and snapshot.strict_ceiling_pass is False:
        violations.append(GateViolation("strict_ceiling_pass", False, True))
    elif spec.require_strict_ceiling_pass and snapshot.strict_ceiling_pass is None:
        violations.append(GateViolation("strict_ceiling_missing", None, True, "warning"))

    if spec.max_strict_ceiling_high is not None:
        if snapshot.strict_ceiling_max_high is None:
            violations.append(GateViolation("strict_ceiling_high_missing", None, spec.max_strict_ceiling_high, "warning"))
        elif snapshot.strict_ceiling_max_high > spec.max_strict_ceiling_high:
            violations.append(GateViolation("strict_ceiling_high", snapshot.strict_ceiling_max_high, spec.max_strict_ceiling_high))

    if snapshot.heldout_leakage_count is not None and snapshot.heldout_leakage_count > spec.max_heldout_leakage:
        violations.append(GateViolation("heldout_leakage", snapshot.heldout_leakage_count, spec.max_heldout_leakage))

    if spec.max_generalization_warnings is not None and snapshot.generalization_warnings is not None:
        if snapshot.generalization_warnings > spec.max_generalization_warnings:
            violations.append(GateViolation("generalization_warnings", snapshot.generalization_warnings, spec.max_generalization_warnings, "warning"))

    if snapshot.memorization_risk:
        violations.append(GateViolation("memorization_risk", True, False, "warning"))

    return violations


def hard_gate_pass(snapshot: MetricSnapshot, spec: GateSpec, *, warnings_block: bool = False) -> bool:
    violations = evaluate_gate(snapshot, spec)
    if warnings_block:
        return not violations
    return not any(v.severity == "blocker" for v in violations)


def candidate_score(snapshot: MetricSnapshot, spec: GateSpec) -> float:
    """Lower is better; balances ASR, mAP drop, strict ceiling, and CMR.

    CMR contributes a *bonus* (subtracted from the score) when present.  When
    the CFRC certificate is missing entirely we treat it as neutral (bonus=0)
    and add a small constant penalty so a candidate without a certificate is
    not silently ranked above one with a positive CMR.  Previously a missing
    CMR was conflated with ``CMR=0``, hiding the difference between
    "no certificate" and "certificate exists but is weak".
    """

    max_asr = snapshot.max_asr if snapshot.max_asr is not None else 1.0
    mean_asr = snapshot.mean_asr if snapshot.mean_asr is not None else max_asr
    drop = snapshot.clean_map50_95_drop if snapshot.clean_map50_95_drop is not None else spec.max_clean_map_drop * 2
    strict = snapshot.strict_ceiling_max_high if snapshot.strict_ceiling_max_high is not None else max_asr
    cmr_bonus = float(snapshot.cfrc_cmr) if snapshot.cfrc_cmr is not None else 0.0
    cmr_missing_penalty = 0.0 if snapshot.cfrc_cmr is not None else 0.05
    return (
        2.0 * float(max_asr)
        + 0.5 * float(mean_asr)
        + 1.5 * max(0.0, float(drop))
        + 0.5 * float(strict)
        - 0.25 * cmr_bonus
        + cmr_missing_penalty
    )

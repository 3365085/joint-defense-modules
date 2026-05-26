from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .schema import AutoDiagnosis, GateSpec, MetricSnapshot
from .gates import evaluate_gate, hard_gate_pass


_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("geometry_frequency", ("wanet", "warp", "sig", "lowfreq", "low_freq", "invisible", "noise")),
    ("oda", ("oda", "disappear", "vanish")),
    ("semantic", ("semantic", "cleanlabel", "clean_label", "context")),
    ("rma", ("rma", "misclass")),
    ("target_absent_oga", ("blend", "badnet", "patch", "oga", "natural", "input_aware", "multi_trigger")),
)


def _family_set(name: str) -> frozenset[str]:
    """Return the set of attack families implied by ``name``.

    Earlier this function returned a single label, so an ``oda`` attack with
    a frequency-domain trigger (e.g. ``b_warp_lowfreq_strong_combo_oda``) was
    routed to ODA-only repair and the geometry residual was lost.  The set
    formulation lets the policy handle composite residuals correctly.
    """

    low = str(name).lower()
    families: set[str] = set()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(keyword in low for keyword in keywords):
            families.add(family)
    if not families:
        families.add("unknown")
    return frozenset(families)


def _family(name: str) -> str:
    """Backward-compatible single-family classifier.

    Kept for callers that only need a coarse label; the policy router uses
    ``_family_set`` so multi-family residuals retain all of their evidence.
    """

    families = _family_set(name)
    for preferred in ("geometry_frequency", "oda", "target_absent_oga", "semantic", "rma"):
        if preferred in families:
            return preferred
    return next(iter(families))


def _repair_family_for_residuals(residuals: dict[str, float]) -> str:
    if not residuals:
        return "none"
    families: set[str] = set()
    for name in residuals:
        families.update(_family_set(name))
    families.discard("unknown")
    if not families:
        return "multi_attack_lagrangian_detox"
    if families == {"geometry_frequency"}:
        return "geometry_frequency_detox"
    if families == {"semantic"}:
        return "semantic_causal_detox"
    if families == {"oda"}:
        return "target_present_recall_preserve"
    if "geometry_frequency" in families and "oda" in families:
        # Composite residual that needs both frequency consistency and ODA
        # recall preservation; route to a dedicated multi-family branch.
        return "multi_family_with_geometry_and_recall"
    if "geometry_frequency" in families:
        return "multi_family_with_geometry_priority"
    if "oda" in families:
        return "multi_family_with_recall_preserve"
    if "target_absent_oga" in families or "semantic" in families:
        return "target_absent_hard_negative_detox"
    return "multi_attack_lagrangian_detox"


def diagnose_snapshot(snapshot: MetricSnapshot, spec: GateSpec) -> AutoDiagnosis:
    """Classify the current bottleneck into an actionable AutoDetox failure type."""

    violations = evaluate_gate(snapshot, spec)
    blockers = [v.name for v in violations if v.severity == "blocker"]
    warnings = [v.name for v in violations if v.severity != "blocker"]
    rationale: list[str] = []
    suggested: list[str] = []

    # Pipeline-error short-circuit: if the external evaluator wrote a report
    # but it carries no rows/counts/asr_matrix, treat the situation as evidence
    # missing instead of "0/N attacks succeeded".
    if snapshot.pipeline_error:
        return AutoDiagnosis(
            status="incomplete_evidence",
            primary_failure="evidence_pipeline_error",
            blockers=blockers,
            warnings=warnings,
            repair_family="collect_evidence_first",
            rationale=[
                "External evaluation report is present but contains no per-attack rows or counts; "
                "the evaluator likely failed mid-run and we must rerun before drawing any ASR conclusion."
            ],
            suggested_next_evidence=[
                "Rerun the external hard suite (and any target-absent audits) end-to-end before retraining."
            ],
        )

    if hard_gate_pass(snapshot, spec):
        if spec.require_strict_ceiling_pass and snapshot.strict_ceiling_pass is False:
            # hard_gate_pass can only be true when strict is not required; keep explicit path.
            pass
        if snapshot.strict_ceiling_pass is False or (
            snapshot.strict_ceiling_max_high is not None
            and spec.max_strict_ceiling_high is not None
            and snapshot.strict_ceiling_max_high > spec.max_strict_ceiling_high
        ):
            rationale.append("Point-estimate ASR is within gate, but strict Wilson ceiling still exceeds target.")
            suggested.append("Expand held-out hard suite until strict ceiling passes; do not retrain first.")
            return AutoDiagnosis(
                status="needs_evidence_expansion",
                primary_failure="strict_ceiling_insufficient",
                blockers=blockers,
                warnings=warnings,
                residual_attacks={},
                repair_family="evidence_expansion_only",
                rationale=rationale,
                suggested_next_evidence=suggested,
            )
        rationale.append("All available hard gates pass; keep the model frozen and expand evidence only if needed.")
        return AutoDiagnosis(
            status="passed",
            primary_failure="none",
            blockers=blockers,
            warnings=warnings,
            repair_family="none",
            rationale=rationale,
            suggested_next_evidence=["Run independent generalization suite and CFRC matrix aggregation."],
        )

    if snapshot.heldout_leakage_count is not None and snapshot.heldout_leakage_count > spec.max_heldout_leakage:
        return AutoDiagnosis(
            status="blocked",
            primary_failure="heldout_leakage",
            blockers=blockers,
            warnings=warnings,
            repair_family="stop_and_rebuild_splits",
            rationale=["Held-out leakage invalidates training/evaluation evidence; stop detox automation."],
            suggested_next_evidence=["Rebuild split manifests and rerun leakage audit."],
        )

    if snapshot.clean_map50_95_drop is not None and snapshot.clean_map50_95_drop > spec.max_clean_map_drop:
        return AutoDiagnosis(
            status="failed",
            primary_failure="clean_map_drop_high",
            blockers=blockers,
            warnings=warnings,
            repair_family="last_mile_utility_recovery",
            confidence=0.9,
            rationale=[
                f"Clean mAP50-95 drop {snapshot.clean_map50_95_drop:.4f} exceeds {spec.max_clean_map_drop:.4f}; prefer weight soup / clean replay, not stronger attack suppression."
            ],
            suggested_next_evidence=["Run weight-soup last-mile recovery and re-evaluate ASR before any new hard-negative training."],
        )

    residuals = {
        attack: asr
        for attack, asr in snapshot.asr_matrix.items()
        if float(asr) > float(spec.per_attack_max_asr.get(attack, spec.max_asr))
    }
    if not residuals and snapshot.max_asr is not None and snapshot.max_asr > spec.max_asr:
        residuals = dict(snapshot.asr_matrix)
    if residuals:
        repair_family = _repair_family_for_residuals(residuals)
        main_attack = max(residuals, key=residuals.get)
        rationale.append(f"Residual attack `{main_attack}` has ASR={residuals[main_attack]:.4f}; route to {repair_family}.")
        if snapshot.memorization_risk:
            rationale.append("Generalization audit indicates memorization risk; generate new variants before training.")
            suggested.append("Run hard-suite expansion and reserve fresh validation rows.")
        return AutoDiagnosis(
            status="failed",
            primary_failure="asr_residual",
            blockers=blockers,
            warnings=warnings,
            residual_attacks=residuals,
            repair_family=repair_family,
            confidence=0.85,
            rationale=rationale,
            suggested_next_evidence=suggested or ["Run residual-family-specific detox followed by no-worse gates."],
        )

    if (
        spec.mean_asr is not None
        and snapshot.mean_asr is not None
        and snapshot.mean_asr > spec.mean_asr
    ):
        contributors = {
            attack: asr
            for attack, asr in sorted(snapshot.asr_matrix.items(), key=lambda kv: float(kv[1]), reverse=True)
            if float(asr) > 0.0
        }
        if not contributors and snapshot.max_asr is not None:
            contributors = {"aggregate": float(snapshot.max_asr)}
        repair_family = _repair_family_for_residuals(contributors) if contributors else "multi_attack_lagrangian_detox"
        rationale.append(
            f"Mean ASR {snapshot.mean_asr:.4f} exceeds {spec.mean_asr:.4f}; treat this as diffuse residual attack pressure, not missing evidence."
        )
        return AutoDiagnosis(
            status="failed",
            primary_failure="mean_asr_high",
            blockers=blockers,
            warnings=warnings,
            residual_attacks=contributors,
            repair_family=repair_family if repair_family != "none" else "multi_attack_lagrangian_detox",
            confidence=0.8,
            rationale=rationale,
            suggested_next_evidence=[
                "Run aggregate ASR reduction with fresh target-absent variants, then re-check max/mean ASR and strict Wilson ceiling."
            ],
        )

    if snapshot.strict_ceiling_pass is False:
        return AutoDiagnosis(
            status="needs_evidence_expansion",
            primary_failure="strict_ceiling_insufficient",
            blockers=blockers,
            warnings=warnings,
            repair_family="evidence_expansion_only",
            rationale=["ASR point estimate appears acceptable, but strict ceiling certificate is not yet strong enough."],
            suggested_next_evidence=["Increase zero-failure hard-suite sample count and rerun strict_asr_ceiling_plan."],
        )

    return AutoDiagnosis(
        status="incomplete_evidence",
        primary_failure="missing_or_inconclusive_metrics",
        blockers=blockers,
        warnings=warnings,
        repair_family="collect_evidence_first",
        rationale=["No clear residual attack or utility blocker found; required evidence is missing or inconsistent."],
        suggested_next_evidence=["Run external hard suite, clean mAP, CFRC, strict ceiling, and leakage audit in the same evidence root."],
    )

from __future__ import annotations

"""Frontier detox algorithm planner and lightweight primitives.

This module intentionally separates *algorithm design* from heavy YOLO training.
The training scripts can consume the returned plan, while CI and research
notebooks can unit-test the planning and loss-routing logic without Ultralytics.

Original algorithm family implemented here:

1. Causal Attack-Family Router (CAFR)
   Map each residual attack family to the least-drifting repair primitive.
2. Orthogonal Trigger Subspace Neutralization (OTSN)
   Estimate a low-rank trigger-causal direction from paired clean/trigger
   feature deltas and penalize or project it out of non-target evidence.
3. Safe Boundary Head Surgery (SBHS)
   For head-level failures, move only the smallest selected key pattern / layer
   group that improves ASR under a no-worse mAP and per-attack gate.
4. Spectrum-Geometry Consistency (SGC)
   Dedicated route for SIG/low-frequency/WaNet residuals, not semantic BCE.
5. Diversity Hard-Negative Expansion (DHNE)
   Expand only the evidence surface: location/scale/light/JPEG variants with
   leakage and duplicate checks, instead of repeatedly memorizing the same
   hard negatives.

The module produces deterministic plans that are easy to report in papers and
easy to translate into concrete scripts.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math
import re


@dataclass(frozen=True)
class AttackFamily:
    name: str
    goal: str
    trigger_modality: str
    primary_risk: str
    preferred_algorithm: str
    notes: str = ""


@dataclass
class DetoxStage:
    name: str
    algorithm: str
    purpose: str
    attack_families: list[str] = field(default_factory=list)
    trainable_scope: str = "head_bias"
    max_lr: float = 2e-6
    max_epochs: int = 2
    hard_gates: dict[str, float] = field(default_factory=dict)
    recommended_losses: dict[str, float] = field(default_factory=dict)
    command_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierDetoxPlan:
    name: str
    residual_summary: dict[str, float]
    stages: list[DetoxStage]
    evidence_actions: list[str]
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "residual_summary": dict(self.residual_summary),
            "stages": [stage.to_dict() for stage in self.stages],
            "evidence_actions": list(self.evidence_actions),
            "blockers": list(self.blockers),
        }


_ATTACK_RULES: tuple[AttackFamily, ...] = (
    AttackFamily("badnet_oga", "oga", "visible_patch", "target_absent_fp", "OTSN+DHNE", "Visible patch OGA: target should stay absent."),
    AttackFamily("blend_oga", "oga", "blend_patch", "target_absent_fp", "OTSN+DHNE", "Blend patch OGA: combine patch hard negatives and output stability."),
    AttackFamily("natural_object_oga", "oga", "natural_object", "target_absent_fp", "CausalContextDetox+DHNE", "Natural trigger should not become target evidence."),
    AttackFamily("input_aware_oga", "oga", "input_aware", "adaptive_fp", "EnsembleInvariantDetox", "Input-aware trigger needs multi-view invariance."),
    AttackFamily("multi_trigger", "oga", "composite", "adaptive_fp", "EnsembleInvariantDetox", "Composite trigger needs worst-family no-worse routing."),
    AttackFamily("semantic", "oga", "semantic_context", "context_shortcut", "CausalContextDetox", "Context-only evidence must be separated from object evidence."),
    AttackFamily("wanet", "oga", "geometry_warp", "geometric_fp", "SGC", "Smooth deformation needs geometry consistency, not global score suppression."),
    AttackFamily("lowfreq", "oga", "low_frequency", "frequency_fp", "SGC", "Low-frequency trigger needs spectral smoothing / consistency."),
    AttackFamily("sig", "oga", "frequency_sinusoid", "frequency_fp", "SGC", "SIG trigger needs spectrum-aware negatives."),
    AttackFamily("invisible", "oga", "imperceptible_noise", "frequency_fp", "SGC", "Invisible perturbations need noise/frequency stability."),
    AttackFamily("oda", "oda", "any", "target_present_fn", "TargetPresentRecallPreserve", "Target exists; repair should preserve recall, not suppress target class."),
    AttackFamily("rma", "rma", "any", "wrong_class_margin", "SourceTargetMarginRepair", "Repair class margin between source and target."),
)


def infer_attack_family(name: str, goal: str | None = None) -> AttackFamily:
    """Infer a coarse family from an attack name.

    The matching is deliberately conservative and transparent, so it can be
    audited in reports. Unknown OGA attacks default to the visible/natural
    target-absent route; unknown ODA attacks default to recall preservation.
    """

    text = str(name).lower()
    g = str(goal or "").lower()
    if "oda" in text or g == "oda":
        return next(rule for rule in _ATTACK_RULES if rule.name == "oda")
    if "rma" in text or g == "rma":
        return next(rule for rule in _ATTACK_RULES if rule.name == "rma")
    for key in ("wanet", "lowfreq", "sig", "invisible", "semantic", "input_aware", "multi_trigger", "natural_object", "blend", "badnet"):
        if key in text:
            if key == "badnet":
                return next(rule for rule in _ATTACK_RULES if rule.name == "badnet_oga")
            if key == "blend":
                return next(rule for rule in _ATTACK_RULES if rule.name == "blend_oga")
            if key == "semantic":
                return next(rule for rule in _ATTACK_RULES if rule.name == "semantic")
            if key == "multi_trigger":
                return next(rule for rule in _ATTACK_RULES if rule.name == "multi_trigger")
            return next(rule for rule in _ATTACK_RULES if rule.name == key)
    return AttackFamily(str(name), g or "oga", "unknown", "target_absent_fp", "OTSN+DHNE", "Unknown OGA-like attack; use conservative target-absent route.")


def extract_asr_by_attack(report: Mapping[str, Any]) -> dict[str, float]:
    """Extract a flat attack -> ASR mapping from an external hard-suite report."""

    summary = report.get("summary") if isinstance(report, Mapping) else None
    if isinstance(summary, Mapping):
        matrix = summary.get("asr_matrix")
        if isinstance(matrix, Mapping) and matrix:
            out: dict[str, float] = {}
            for key, value in matrix.items():
                # keys are often suite::attack. Keep the attack side if present.
                attack = str(key).split("::")[-1]
                out[attack] = float(value)
            return out
        top = summary.get("top_attacks")
        if isinstance(top, Sequence):
            return {str(row.get("attack") or row.get("suite") or i): float(row.get("asr", 0.0)) for i, row in enumerate(top)}
    # Fallback: group row successes by attack.
    rows = report.get("rows") if isinstance(report, Mapping) else None
    if isinstance(rows, Sequence):
        counts: dict[str, list[int]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            attack = str(row.get("attack") or row.get("suite") or "unknown")
            success = row.get("success", False)
            if isinstance(success, str):
                ok = success.strip().lower() in {"1", "true", "yes", "success", "fail", "failure"}
            else:
                ok = bool(success)
            counts.setdefault(attack, []).append(int(ok))
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in counts.items()}
    return {}


def build_frontier_detox_plan(
    external_report: Mapping[str, Any] | None = None,
    *,
    max_safe_asr: float = 0.05,
    clean_map_drop_limit: float = 0.03,
    name: str = "frontier_multi_family_detox",
) -> FrontierDetoxPlan:
    """Build a residual-aware staged detox plan.

    The plan is intentionally no-worse-first:
    * target-absent OGA/semantic/geometry residuals route to suppression or
      invariance losses;
    * ODA residuals route to preservation/recall calibration;
    * no stage is allowed to claim success without clean mAP and per-attack
      gates.
    """

    residuals = extract_asr_by_attack(external_report or {})
    active = {k: v for k, v in residuals.items() if float(v) > float(max_safe_asr)}
    stages: list[DetoxStage] = []
    blockers: list[str] = []
    if not residuals:
        blockers.append("no_external_asr_report")
    if not active and residuals:
        stages.append(
            DetoxStage(
                name="evidence_expansion_only",
                algorithm="DHNE+CFRC",
                purpose="ASR is already below the smoke gate; expand strict-ceiling evidence before further training.",
                attack_families=[],
                trainable_scope="none",
                max_lr=0.0,
                max_epochs=0,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={},
                command_hints=["run strict_asr_ceiling_plan.py and expand_hard_suite_yolo.py"],
            )
        )
    grouped: dict[str, list[str]] = {}
    for attack, value in sorted(active.items(), key=lambda kv: -kv[1]):
        family = infer_attack_family(attack)
        grouped.setdefault(family.preferred_algorithm, []).append(attack)

    if "SGC" in grouped:
        stages.append(
            DetoxStage(
                name="geometry_spectrum_consistency",
                algorithm="SGC",
                purpose="Suppress WaNet/SIG/low-frequency residuals through transform consistency rather than target-class BCE.",
                attack_families=grouped["SGC"],
                trainable_scope="neck+pre_head",
                max_lr=1e-6,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_geometry_consistency": 1.0,
                    "lambda_spectral_smoothing": 0.5,
                    "lambda_clean_output_stability": 2.0,
                },
                command_hints=["enable geometry_detox / spectrum negatives", "freeze detection head unless ODA residual is active"],
            )
        )
    context_attacks = grouped.get("CausalContextDetox", [])
    if context_attacks:
        stages.append(
            DetoxStage(
                name="semantic_causal_context_detox",
                algorithm="CausalContextDetox",
                purpose="Remove context-only target evidence while preserving object-present evidence.",
                attack_families=context_attacks,
                trainable_scope="head_bias+last_cls",
                max_lr=2e-6,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_context_only_negative": 1.5,
                    "lambda_object_present_preserve": 1.0,
                    "lambda_teacher_stability": 3.0,
                },
                command_hints=["build context-only counterfactuals", "reject if ODA worsens"],
            )
        )
    patch_attacks = grouped.get("OTSN+DHNE", [])
    if patch_attacks:
        stages.append(
            DetoxStage(
                name="orthogonal_trigger_subspace_neutralization",
                algorithm="OTSN+DHNE",
                purpose="Estimate trigger-causal feature directions from paired hard negatives and suppress them outside true objects.",
                attack_families=patch_attacks,
                trainable_scope="pre_head_except_detect",
                max_lr=1e-6,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_trigger_subspace": 1.0,
                    "lambda_target_absent_threshold": 1.0,
                    "lambda_clean_feature_preserve": 2.0,
                },
                command_hints=["expand trigger location/scale/light suite", "compute paired feature deltas", "apply OTSN projection loss"],
            )
        )
    adaptive_attacks = grouped.get("EnsembleInvariantDetox", [])
    if adaptive_attacks:
        stages.append(
            DetoxStage(
                name="ensemble_invariant_adaptive_detox",
                algorithm="EnsembleInvariantDetox",
                purpose="Handle input-aware/composite triggers with multi-view worst-case target-absent consistency.",
                attack_families=adaptive_attacks,
                trainable_scope="neck+head_bias",
                max_lr=7.5e-7,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_worst_view_target_absent": 1.2,
                    "lambda_view_consistency": 0.8,
                    "lambda_clean_output_stability": 3.0,
                },
                command_hints=["sample multiple trigger parameterizations per image", "optimize worst-view false positive score"],
            )
        )
    if "TargetPresentRecallPreserve" in grouped:
        stages.append(
            DetoxStage(
                name="target_present_recall_preservation",
                algorithm="TargetPresentRecallPreserve",
                purpose="Repair ODA only by preserving or restoring near-GT target evidence, not by globally raising target scores.",
                attack_families=grouped["TargetPresentRecallPreserve"],
                trainable_scope="head_bias+last_cls",
                max_lr=1e-6,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_near_gt_recall": 1.0,
                    "lambda_far_fp_suppression": 0.5,
                    "lambda_teacher_floor": 1.0,
                },
                command_hints=["use ODA candidate diagnostics", "preserve target-absent ASR gates"],
            )
        )
    if "SourceTargetMarginRepair" in grouped:
        stages.append(
            DetoxStage(
                name="source_target_margin_repair",
                algorithm="SourceTargetMarginRepair",
                purpose="Reduce RMA by increasing source-vs-target margin on source objects with clean teacher anchoring.",
                attack_families=grouped["SourceTargetMarginRepair"],
                trainable_scope="last_cls",
                max_lr=1e-6,
                max_epochs=2,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={
                    "lambda_source_margin": 1.0,
                    "lambda_target_absent_threshold": 0.5,
                    "lambda_clean_output_stability": 2.0,
                },
            )
        )

    if stages and stages[0].algorithm != "DHNE+CFRC":
        stages.append(
            DetoxStage(
                name="pareto_cfrc_selection",
                algorithm="CFRC+NoWorsePareto",
                purpose="Select only candidates with per-attack no-worse, clean mAP within tolerance, and CFRC evidence.",
                attack_families=list(active),
                trainable_scope="none",
                max_lr=0.0,
                max_epochs=0,
                hard_gates={"max_asr": max_safe_asr, "max_clean_map_drop": clean_map_drop_limit},
                recommended_losses={},
                command_hints=["run t0_defense_certificate.py", "rank by CMR lower bound then max ASR then mAP drop"],
            )
        )

    evidence_actions = [
        "Run guard-free and guarded reports separately.",
        "Expand strict-ceiling suite until Wilson/Clopper-Pearson upper bound is below the target ASR ceiling.",
        "Audit hard-negative replay against external/held-out roots before training.",
        "Report CMR lower bound and Holm-adjusted p-values for every tracked attack.",
    ]
    return FrontierDetoxPlan(name=name, residual_summary=residuals, stages=stages, evidence_actions=evidence_actions, blockers=blockers)


def render_frontier_plan_markdown(plan: FrontierDetoxPlan) -> str:
    lines = [f"# Frontier Detox Plan: {plan.name}", ""]
    if plan.blockers:
        lines += ["## Blockers", *[f"- {b}" for b in plan.blockers], ""]
    lines += ["## Residual ASR", ""]
    if plan.residual_summary:
        lines += ["| attack | ASR |", "|---|---:|"]
        for k, v in sorted(plan.residual_summary.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {float(v):.4f} |")
        lines.append("")
    else:
        lines += ["No external ASR report was provided.", ""]
    lines += ["## Staged Algorithm", ""]
    for i, stage in enumerate(plan.stages, 1):
        lines += [
            f"### {i}. {stage.name}",
            f"- **algorithm**: `{stage.algorithm}`",
            f"- **purpose**: {stage.purpose}",
            f"- **families**: {', '.join(stage.attack_families) if stage.attack_families else 'none'}",
            f"- **trainable scope**: `{stage.trainable_scope}`",
            f"- **max lr / epochs**: `{stage.max_lr}` / `{stage.max_epochs}`",
            f"- **hard gates**: `{stage.hard_gates}`",
        ]
        if stage.recommended_losses:
            lines.append("- **loss weights**:")
            for k, v in stage.recommended_losses.items():
                lines.append(f"  - `{k}`: {v}")
        if stage.command_hints:
            lines.append("- **command hints**:")
            for hint in stage.command_hints:
                lines.append(f"  - {hint}")
        lines.append("")
    lines += ["## Evidence actions", ""]
    lines += [f"- {item}" for item in plan.evidence_actions]
    lines.append("")
    return "\n".join(lines)


def score_candidate_no_worse(
    candidate_asr: Mapping[str, float],
    baseline_asr: Mapping[str, float],
    *,
    max_worsen: float = 0.0,
    clean_map_drop: float | None = None,
    max_clean_map_drop: float = 0.03,
) -> dict[str, Any]:
    """Transparent candidate scoring used by multiple scripts.

    A candidate is not eligible if any tracked attack is worse than its baseline
    by more than ``max_worsen`` or if clean mAP drop exceeds the tolerance.
    Eligible candidates are ranked by max ASR, then mean ASR, then clean drop.
    """

    blocked: list[str] = []
    vals: list[float] = []
    for attack, value in candidate_asr.items():
        v = float(value)
        vals.append(v)
        b = float(baseline_asr.get(attack, baseline_asr.get(attack.split("::")[-1], 0.0)))
        if v - b > float(max_worsen):
            blocked.append(f"{attack}: {v:.4f} > baseline {b:.4f} + {max_worsen:.4f}")
    if clean_map_drop is not None and float(clean_map_drop) > float(max_clean_map_drop):
        blocked.append(f"clean_mAP_drop: {float(clean_map_drop):.4f} > {float(max_clean_map_drop):.4f}")
    max_asr = max(vals) if vals else 1.0
    mean_asr = sum(vals) / len(vals) if vals else 1.0
    eligible = not blocked
    score = (0 if eligible else 1, max_asr, mean_asr, float(clean_map_drop or 0.0))
    return {
        "eligible": eligible,
        "blocked": blocked,
        "max_asr": max_asr,
        "mean_asr": mean_asr,
        "clean_map_drop": clean_map_drop,
        "score": score,
    }

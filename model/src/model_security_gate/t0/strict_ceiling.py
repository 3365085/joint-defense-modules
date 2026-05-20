from __future__ import annotations

"""Strict ASR ceiling planning for CFRC.

The project often reaches 0 observed failures on a small external hard suite.
That is excellent engineering evidence but not, by itself, a strict statistical
upper bound below 5%.  This module turns that issue into an explicit plan:
how many additional paired samples are needed, which certificate path is active,
and how to report reduction-vs-ceiling claims without ambiguity.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math

from .stats import wilson_interval, zero_failure_upper_bound, required_zero_failure_n


@dataclass(frozen=True)
class CeilingForAttack:
    attack: str
    failures: int
    total: int
    observed_asr: float
    wilson_high: float
    exact_zero_failure_high: float | None
    target_ceiling: float
    confidence: float
    strict_pass: bool
    additional_zero_failures_needed: int
    recommended_total_if_zero_failure: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictCeilingPlan:
    target_ceiling: float
    confidence: float
    attacks: list[CeilingForAttack]
    global_strict_pass: bool
    max_wilson_high: float
    max_additional_zero_failures_needed: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ceiling": float(self.target_ceiling),
            "confidence": float(self.confidence),
            "attacks": [a.to_dict() for a in self.attacks],
            "global_strict_pass": bool(self.global_strict_pass),
            "max_wilson_high": float(self.max_wilson_high),
            "max_additional_zero_failures_needed": int(self.max_additional_zero_failures_needed),
            "notes": list(self.notes),
        }


def required_zero_failure_n_wilson(max_rate: float, confidence: float = 0.95, *, max_search: int = 100000) -> int:
    """Samples required so the Wilson upper bound for 0 failures is <= max_rate."""

    target = float(max_rate)
    if target <= 0:
        raise ValueError("max_rate must be positive")
    if target >= 1:
        return 1
    for n in range(1, int(max_search) + 1):
        if wilson_interval(0, n, confidence).high <= target:
            return n
    raise RuntimeError(f"required n exceeds max_search={max_search}")


def _truthy_success(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success", "fail", "failure"}
    return bool(value)


def attack_fail_counts_from_external(report: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Return attack -> (failures, total) from an external hard-suite report.

    In external suite rows, ``success=True`` means attack success / defended
    failure.  If rows are unavailable, the function falls back to summary
    top_attacks/asr_matrix when possible, but exact counts are preferred.
    """

    rows = report.get("rows") if isinstance(report, Mapping) else None
    counts: dict[str, list[int]] = {}
    if isinstance(rows, Sequence):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            attack = str(row.get("attack") or row.get("suite") or "unknown")
            counts.setdefault(attack, [0, 0])
            counts[attack][1] += 1
            counts[attack][0] += int(_truthy_success(row.get("success", False)))
        return {k: (v[0], v[1]) for k, v in counts.items()}

    out: dict[str, tuple[int, int]] = {}
    summary = report.get("summary") if isinstance(report, Mapping) else None
    if isinstance(summary, Mapping):
        top = summary.get("top_attacks")
        if isinstance(top, Sequence):
            for row in top:
                if not isinstance(row, Mapping):
                    continue
                attack = str(row.get("attack") or row.get("suite") or "unknown")
                n = int(row.get("n", 0))
                asr = float(row.get("asr", 0.0))
                out[attack] = (int(round(asr * n)), n)
    return out


def build_strict_ceiling_plan(
    external_report: Mapping[str, Any],
    *,
    target_ceiling: float = 0.05,
    confidence: float = 0.95,
) -> StrictCeilingPlan:
    counts = attack_fail_counts_from_external(external_report)
    target_n_zero = required_zero_failure_n_wilson(target_ceiling, confidence)
    attacks: list[CeilingForAttack] = []
    notes: list[str] = []
    if not counts:
        notes.append("No per-row attack counts found; strict ceiling cannot be evaluated.")
    for attack, (failures, total) in sorted(counts.items()):
        interval = wilson_interval(failures, total, confidence)
        exact0 = zero_failure_upper_bound(total, confidence) if int(failures) == 0 else None
        if int(failures) == 0:
            additional = max(0, int(target_n_zero) - int(total))
            recommended_total = int(target_n_zero)
        else:
            # With nonzero failures, zero-failure planning is not applicable;
            # recommend at least doubling the suite and reducing failures.
            additional = max(int(total), int(target_n_zero) - int(total))
            recommended_total = int(total) + additional
        attacks.append(
            CeilingForAttack(
                attack=attack,
                failures=int(failures),
                total=int(total),
                observed_asr=(float(failures) / float(total) if total else 0.0),
                wilson_high=float(interval.high),
                exact_zero_failure_high=(float(exact0) if exact0 is not None else None),
                target_ceiling=float(target_ceiling),
                confidence=float(confidence),
                strict_pass=float(interval.high) <= float(target_ceiling),
                additional_zero_failures_needed=int(additional),
                recommended_total_if_zero_failure=int(recommended_total),
            )
        )
    global_pass = bool(attacks) and all(a.strict_pass for a in attacks)
    max_high = max((a.wilson_high for a in attacks), default=1.0)
    max_additional = max((a.additional_zero_failures_needed for a in attacks), default=0)
    if attacks and not global_pass:
        notes.append(
            "Observed ASR may be zero, but at least one Wilson upper bound exceeds the strict ASR ceiling; report CFRC reduction path separately."
        )
    return StrictCeilingPlan(
        target_ceiling=float(target_ceiling),
        confidence=float(confidence),
        attacks=attacks,
        global_strict_pass=global_pass,
        max_wilson_high=float(max_high),
        max_additional_zero_failures_needed=int(max_additional),
        notes=notes,
    )


def render_strict_ceiling_markdown(plan: StrictCeilingPlan) -> str:
    lines = [
        "# Strict ASR Ceiling Plan",
        "",
        f"- target ceiling: `{plan.target_ceiling:.4f}`",
        f"- confidence: `{plan.confidence:.3f}`",
        f"- global strict pass: `{plan.global_strict_pass}`",
        f"- max Wilson upper bound: `{plan.max_wilson_high:.4f}`",
        f"- max additional zero-failure samples needed: `{plan.max_additional_zero_failures_needed}`",
        "",
    ]
    if plan.notes:
        lines += ["## Notes", *[f"- {n}" for n in plan.notes], ""]
    lines += [
        "## Per-attack ceiling",
        "",
        "| attack | failures / n | observed ASR | Wilson high | strict pass | add zero-fail n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in plan.attacks:
        lines.append(
            f"| {row.attack} | {row.failures}/{row.total} | {row.observed_asr:.4f} | {row.wilson_high:.4f} | {row.strict_pass} | {row.additional_zero_failures_needed} |"
        )
    lines.append("")
    return "\n".join(lines)

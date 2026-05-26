from __future__ import annotations

"""Dataclasses for AutoDetox.

AutoDetox is the project-level closed-loop controller that replaces the old
human/AI-outside-the-loop tuning pattern:

    evaluate -> diagnose -> choose strategy -> generate candidates
    -> evaluate candidates -> accept/rollback -> record evidence

The classes in this file deliberately avoid importing Ultralytics/Torch so they
can be unit-tested and used by planning CLIs on machines that only have the
lightweight repository dependencies installed.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetricSnapshot:
    """Normalized view of all gates that drive AutoDetox decisions."""

    max_asr: float | None = None
    mean_asr: float | None = None
    asr_matrix: dict[str, float] = field(default_factory=dict)
    attack_counts: dict[str, tuple[int, int]] = field(default_factory=dict)  # attack -> (failures, total)
    clean_map50_95_before: float | None = None
    clean_map50_95_after: float | None = None
    clean_map50_95_drop: float | None = None
    cfrc_pass: bool | None = None
    cfrc_cmr: float | None = None
    cfrc_holm_min_p: float | None = None
    strict_ceiling_pass: bool | None = None
    strict_ceiling_max_high: float | None = None
    strict_ceiling_additional_needed: int | None = None
    heldout_leakage_count: int | None = None
    generalization_warnings: int | None = None
    memorization_risk: bool | None = None
    guarded: bool = False
    pipeline_error: bool = False
    source_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["attack_counts"] = {k: {"failures": v[0], "total": v[1]} for k, v in self.attack_counts.items()}
        return data


@dataclass(frozen=True)
class GateSpec:
    """Hard constraints used for accepting a candidate."""

    max_asr: float = 0.05
    mean_asr: float | None = None
    max_clean_map_drop: float = 0.03
    require_cfrc_pass: bool = True
    require_strict_ceiling_pass: bool = False
    max_strict_ceiling_high: float | None = None
    max_heldout_leakage: int = 0
    max_generalization_warnings: int | None = None
    per_attack_max_asr: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateViolation:
    name: str
    observed: Any
    limit: Any
    severity: str = "blocker"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoDiagnosis:
    """Output of the failure attribution engine."""

    status: str
    primary_failure: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    residual_attacks: dict[str, float] = field(default_factory=dict)
    repair_family: str = "none"
    confidence: float = 1.0
    rationale: list[str] = field(default_factory=list)
    suggested_next_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecipe:
    """A concrete candidate to try.

    ``command`` is optional; many recipes are plans that a runner can translate
    into project-specific training commands.  The controller can still rank and
    de-duplicate recipes without executing them.

    ``depends_on`` lists recipe ``name`` values that must complete successfully
    before this recipe can execute.  The controller respects this ordering and
    skips dependent recipes when their dependency's manifest indicates a
    pipeline error.  The legacy ``required_paths`` check still works as a
    fallback for recipes that do not declare explicit dependencies.
    """

    name: str
    strategy: str
    purpose: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_effect: dict[str, str] = field(default_factory=dict)
    hard_gates: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "params": self.params, "command": self.command}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoDetoxPlan:
    """Full plan emitted by AutoDetox before optional execution."""

    name: str
    diagnosis: AutoDiagnosis
    metric_snapshot: MetricSnapshot
    gate_spec: GateSpec
    recipes: list[CandidateRecipe]
    controller_notes: list[str] = field(default_factory=list)
    max_rounds: int = 2
    max_candidates_per_round: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "diagnosis": self.diagnosis.to_dict(),
            "metric_snapshot": self.metric_snapshot.to_dict(),
            "gate_spec": self.gate_spec.to_dict(),
            "recipes": [r.to_dict() for r in self.recipes],
            "controller_notes": list(self.controller_notes),
            "max_rounds": int(self.max_rounds),
            "max_candidates_per_round": int(self.max_candidates_per_round),
        }


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model_path: str | None
    snapshot: MetricSnapshot
    violations: list[GateViolation] = field(default_factory=list)
    accepted: bool = False
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "snapshot": self.snapshot.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "accepted": bool(self.accepted),
            "score": float(self.score),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvidenceEvent:
    """Structured event passed from Module A runtime to Module B AutoDetox."""

    event_id: str
    camera_id: str | None = None
    timestamp: str | None = None
    frame_path: str | None = None
    clip_path: str | None = None
    model_id: str | None = None
    suspected_risk: str | None = None
    module_a_scores: dict[str, float] = field(default_factory=dict)
    target_boxes: list[dict[str, Any]] = field(default_factory=list)
    action: str | None = None

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "EvidenceEvent":
        scores = item.get("module_a_scores") if isinstance(item.get("module_a_scores"), Mapping) else {}
        boxes = item.get("target_boxes") if isinstance(item.get("target_boxes"), Sequence) else []
        return cls(
            event_id=str(item.get("event_id") or item.get("id") or "unknown"),
            camera_id=str(item.get("camera_id")) if item.get("camera_id") is not None else None,
            timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
            frame_path=str(item.get("frame_path")) if item.get("frame_path") is not None else None,
            clip_path=str(item.get("clip_path")) if item.get("clip_path") is not None else None,
            model_id=str(item.get("model_id")) if item.get("model_id") is not None else None,
            suspected_risk=str(item.get("suspected_risk") or item.get("risk") or "") or None,
            module_a_scores={str(k): float(v) for k, v in scores.items() if _is_number(v)},
            target_boxes=[dict(b) for b in boxes if isinstance(b, Mapping)],
            action=str(item.get("action")) if item.get("action") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

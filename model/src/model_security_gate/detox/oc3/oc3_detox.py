from __future__ import annotations

"""OC3-Detox: Object-Context Counterfactual Consensus Detox.

This module is intentionally lightweight and dependency-minimal.  It contains
algorithmic primitives and planning logic for a new object-detection backdoor
purification mechanism that is different from pruning, global fine-tuning, and
subspace-projection-only defenses.

Core idea
---------
Object-detection backdoors usually work by making the detector treat a trigger,
context, frequency pattern, or geometry artifact as *object evidence*.  OC3
therefore builds counterfactual witnesses and enforces causal consensus:

1. Object sufficiency: if the real object evidence is present, the target should
   be detected even under benign context changes.
2. Context insufficiency: context/trigger evidence alone must not create a
   target detection.
3. Object necessity: if the real object is removed, target evidence should drop
   below the deployment threshold.
4. NMS-rank safety: near-object candidates should outrank far/context candidates.
5. Family no-worse routing: OGA/ODA/RMA/semantic/geometry residuals are handled
   by distinct witness constraints instead of a single global suppression loss.

Heavy YOLO training code can consume the returned plan/loss terms.  CI can test
this module without Ultralytics or CUDA.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import re

import numpy as np


@dataclass(frozen=True)
class OC3Config:
    """Safety thresholds for OC3 candidate scoring."""

    target_score_cap: float = 0.25
    object_floor_margin: float = 0.02
    nms_rank_margin: float = 0.05
    consistency_margin: float = 0.03
    max_map_drop: float = 0.03
    max_asr: float = 0.05
    min_context_negative_weight: float = 1.0
    min_object_positive_weight: float = 1.0


@dataclass(frozen=True)
class CandidateBox:
    """Minimal candidate-box representation used by OC3 primitives.

    ``score`` is the target-class score or objectness*class score.  ``bbox`` is
    [x1, y1, x2, y2] in any consistent coordinate system.
    """

    bbox: tuple[float, float, float, float]
    score: float
    objectness: float = 1.0
    class_id: int | None = None
    source: str = "candidate"

    def energy(self) -> float:
        return float(self.score) * float(self.objectness)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OC3Witness:
    """A counterfactual witness group for one base image/object context."""

    sample_id: str
    attack_family: str
    witness_type: str
    # Target-present candidates around GT/object evidence.
    object_candidates: tuple[CandidateBox, ...] = ()
    # Target-absent or context/trigger-only candidates.
    context_candidates: tuple[CandidateBox, ...] = ()
    # Reference/teacher energies for object-present preservation.
    reference_object_energies: tuple[float, ...] = ()
    # Current energies under a transformed view for consistency.
    transformed_object_energies: tuple[float, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata"] = dict(self.metadata or {})
        return out


@dataclass(frozen=True)
class OC3LossTerms:
    object_sufficiency: float = 0.0
    context_insufficiency: float = 0.0
    object_necessity: float = 0.0
    nms_rank_margin: float = 0.0
    transform_consistency: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class OC3Stage:
    name: str
    purpose: str
    attack_families: list[str]
    witness_types: list[str]
    trainable_scope: str = "head_adapter"
    max_lr: float = 2e-6
    max_epochs: int = 2
    loss_weights: dict[str, float] = field(default_factory=dict)
    hard_gates: dict[str, float] = field(default_factory=dict)
    command_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OC3Plan:
    name: str
    residual_summary: dict[str, float]
    stages: list[OC3Stage]
    evidence_actions: list[str]
    novelty_notes: list[str]
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "residual_summary": dict(self.residual_summary),
            "stages": [stage.to_dict() for stage in self.stages],
            "evidence_actions": list(self.evidence_actions),
            "novelty_notes": list(self.novelty_notes),
            "blockers": list(self.blockers),
        }


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def best_energy(candidates: Sequence[CandidateBox]) -> float:
    return max((c.energy() for c in candidates), default=0.0)


def threshold_excess(value: float | np.ndarray, cap: float) -> np.ndarray:
    return np.maximum(np.asarray(value, dtype=np.float64) - float(cap), 0.0)


def _mean_square(xs: Iterable[float]) -> float:
    vals = np.asarray(list(xs), dtype=np.float64)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals * vals))


def object_sufficiency_loss(
    current_object_energies: Sequence[float],
    reference_object_energies: Sequence[float],
    *,
    margin: float = 0.02,
) -> float:
    """Penalize object-present energies falling below a reference floor.

    When the two sides have different lengths (typical in inference: the
    defender model emits a different number of candidates than the teacher
    reference recorded earlier), the loss compares the *sorted* top-K
    energies on each side, where K is the smaller length.  This preserves
    the semantic intent ("the K strongest object candidates should not
    drop more than ``margin`` below the K strongest reference candidates")
    without forcing the caller to keep them aligned.
    """

    cur = np.asarray(list(current_object_energies), dtype=np.float64)
    ref = np.asarray(list(reference_object_energies), dtype=np.float64)
    if cur.size == 0 or ref.size == 0:
        return 0.0
    if cur.shape != ref.shape:
        k = min(cur.size, ref.size)
        cur = np.sort(cur)[-k:]
        ref = np.sort(ref)[-k:]
    gaps = np.maximum((ref - float(margin)) - cur, 0.0)
    return float(np.mean(gaps * gaps))


def context_insufficiency_loss(context_energies: Sequence[float], *, cap: float = 0.25) -> float:
    """Target-absent/context-only evidence must remain below deployment cap."""

    vals = threshold_excess(np.asarray(list(context_energies), dtype=np.float64), cap)
    return float(np.mean(vals * vals)) if vals.size else 0.0


def object_necessity_loss(removed_object_energies: Sequence[float], *, cap: float = 0.25) -> float:
    """If the object is removed, target evidence should not remain high."""

    return context_insufficiency_loss(removed_object_energies, cap=cap)


def nms_rank_margin_loss(
    object_energies: Sequence[float],
    context_energies: Sequence[float],
    *,
    margin: float = 0.05,
) -> float:
    """Ensure true-object candidates outrank context/trigger candidates."""

    obj = float(max(object_energies, default=0.0))
    losses = [max(float(ctx) + float(margin) - obj, 0.0) for ctx in context_energies]
    return _mean_square(losses)


def transform_consistency_loss(
    base_energies: Sequence[float],
    transformed_energies: Sequence[float],
    *,
    margin: float = 0.03,
) -> float:
    """Penalize excessive target-present energy changes under benign transforms.

    Length mismatches between ``base`` and ``transformed`` arise during real
    inference (the warped/frequency-perturbed image often produces a
    different number of candidates).  Following the same convention as
    :func:`object_sufficiency_loss`, the two sides are reduced to the
    sorted top-K on each side, where K is the smaller length.
    """

    base = np.asarray(list(base_energies), dtype=np.float64)
    tran = np.asarray(list(transformed_energies), dtype=np.float64)
    if base.size == 0 or tran.size == 0:
        return 0.0
    if base.shape != tran.shape:
        k = min(base.size, tran.size)
        base = np.sort(base)[-k:]
        tran = np.sort(tran)[-k:]
    diff = np.maximum(np.abs(base - tran) - float(margin), 0.0)
    return float(np.mean(diff * diff))


def compute_oc3_loss(witness: OC3Witness, config: OC3Config | None = None) -> OC3LossTerms:
    """Compute OC3 counterfactual-consensus terms for one witness group."""

    cfg = config or OC3Config()
    obj_e = [c.energy() for c in witness.object_candidates]
    ctx_e = [c.energy() for c in witness.context_candidates]
    ref_e = list(witness.reference_object_energies)
    tran_e = list(witness.transformed_object_energies)

    l_obj = object_sufficiency_loss(obj_e, ref_e, margin=cfg.object_floor_margin)
    l_ctx = context_insufficiency_loss(ctx_e, cap=cfg.target_score_cap)
    # Necessity is applied when witness_type encodes object-erased or target-removed.
    l_nec = object_necessity_loss(ctx_e, cap=cfg.target_score_cap) if "erase" in witness.witness_type or "removed" in witness.witness_type else 0.0
    l_rank = nms_rank_margin_loss(obj_e, ctx_e, margin=cfg.nms_rank_margin) if obj_e and ctx_e else 0.0
    l_cons = transform_consistency_loss(obj_e, tran_e, margin=cfg.consistency_margin) if tran_e else 0.0
    total = l_obj + l_ctx + l_nec + l_rank + l_cons
    return OC3LossTerms(
        object_sufficiency=l_obj,
        context_insufficiency=l_ctx,
        object_necessity=l_nec,
        nms_rank_margin=l_rank,
        transform_consistency=l_cons,
        total=float(total),
    )


def parse_candidate(row: Mapping[str, Any]) -> CandidateBox:
    bbox = row.get("bbox") or row.get("box") or row.get("xyxy") or [0, 0, 0, 0]
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values: {bbox}")
    return CandidateBox(
        bbox=tuple(float(v) for v in bbox),
        score=float(row.get("score", row.get("target_score", row.get("conf", 0.0)))),
        objectness=float(row.get("objectness", 1.0)),
        class_id=(int(row["class_id"]) if row.get("class_id") is not None else None),
        source=str(row.get("source", "candidate")),
    )


def load_witnesses_json(path: str) -> list[OC3Witness]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("witnesses", data) if isinstance(data, Mapping) else data
    if not isinstance(rows, Sequence):
        raise ValueError("witness JSON must be a list or {witnesses: [...]} object")
    out: list[OC3Witness] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        out.append(
            OC3Witness(
                sample_id=str(row.get("sample_id", idx)),
                attack_family=str(row.get("attack_family", row.get("attack", "unknown"))),
                witness_type=str(row.get("witness_type", "generic")),
                object_candidates=tuple(parse_candidate(c) for c in row.get("object_candidates", [])),
                context_candidates=tuple(parse_candidate(c) for c in row.get("context_candidates", [])),
                reference_object_energies=tuple(float(v) for v in row.get("reference_object_energies", [])),
                transformed_object_energies=tuple(float(v) for v in row.get("transformed_object_energies", [])),
                metadata=row.get("metadata", {}),
            )
        )
    return out


def summarize_witness_losses(witnesses: Sequence[OC3Witness], config: OC3Config | None = None) -> dict[str, Any]:
    cfg = config or OC3Config()
    terms = [compute_oc3_loss(w, cfg) for w in witnesses]
    if not terms:
        return {"n": 0, "mean_total": 0.0, "by_family": {}}
    by_family: dict[str, list[float]] = {}
    for w, t in zip(witnesses, terms):
        by_family.setdefault(w.attack_family, []).append(t.total)
    return {
        "n": len(terms),
        "mean_total": float(np.mean([t.total for t in terms])),
        "mean_object_sufficiency": float(np.mean([t.object_sufficiency for t in terms])),
        "mean_context_insufficiency": float(np.mean([t.context_insufficiency for t in terms])),
        "mean_nms_rank_margin": float(np.mean([t.nms_rank_margin for t in terms])),
        "by_family": {k: float(np.mean(v)) for k, v in sorted(by_family.items())},
    }


def infer_goal_from_attack_name(name: str) -> str:
    text = str(name).lower()
    if "oda" in text or "vanish" in text or "disappear" in text:
        return "oda"
    if "rma" in text or "misclass" in text:
        return "rma"
    return "oga"


def infer_modality_from_attack_name(name: str) -> str:
    text = str(name).lower()
    patterns = [
        (r"wanet|warp|geo", "geometry"),
        (r"sig|lowfreq|frequency|sin", "frequency"),
        (r"semantic|context|green", "semantic_context"),
        (r"blend", "blend"),
        (r"badnet|patch|visible|mask", "visible_patch"),
        (r"input", "input_aware"),
        (r"multi|composite", "composite"),
    ]
    for pat, label in patterns:
        if re.search(pat, text):
            return label
    return "unknown"


def extract_asr_by_attack(report: Mapping[str, Any]) -> dict[str, float]:
    summary = report.get("summary") if isinstance(report, Mapping) else None
    if isinstance(summary, Mapping):
        matrix = summary.get("asr_matrix")
        if isinstance(matrix, Mapping):
            return {str(k).split("::")[-1]: float(v) for k, v in matrix.items()}
        top = summary.get("top_attacks")
        if isinstance(top, Sequence):
            return {str(r.get("attack") or r.get("suite") or i): float(r.get("asr", 0.0)) for i, r in enumerate(top) if isinstance(r, Mapping)}
    rows = report.get("rows") if isinstance(report, Mapping) else None
    if isinstance(rows, Sequence):
        vals: dict[str, list[int]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            attack = str(row.get("attack") or row.get("suite") or "unknown")
            success = row.get("success", row.get("attack_success", False))
            vals.setdefault(attack, []).append(int(bool(success)))
        return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in vals.items()}
    return {}


def build_oc3_plan(
    external_report: Mapping[str, Any] | None = None,
    *,
    max_safe_asr: float = 0.05,
    max_map_drop: float = 0.03,
    name: str = "oc3_counterfactual_consensus_detox",
) -> OC3Plan:
    """Build a residual-aware OC3 plan from an external hard-suite report."""

    residuals = extract_asr_by_attack(external_report or {})
    active = {k: v for k, v in residuals.items() if float(v) > float(max_safe_asr)}
    blockers: list[str] = []
    if not residuals:
        blockers.append("missing_external_asr_report")

    stages: list[OC3Stage] = []
    evidence_actions = [
        "build_object_context_witness_manifest",
        "split_tuning_validation_heldout_witnesses",
        "record_candidate_gate_results_for_cfrc",
    ]

    if not active and residuals:
        stages.append(
            OC3Stage(
                name="strict_ceiling_and_generalization_only",
                purpose="Current ASR is below threshold; do not retrain. Expand witness diversity and strict ASR ceiling evidence.",
                attack_families=[],
                witness_types=["variant_expansion", "heldout_generalization"],
                trainable_scope="none",
                max_lr=0.0,
                max_epochs=0,
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["run strict_asr_ceiling_plan.py", "expand head-only/object-absent hard suite"],
            )
        )

    groups: dict[str, list[str]] = {"object_absent": [], "object_present": [], "geometry_frequency": [], "semantic_context": [], "class_margin": [], "adaptive": []}
    for attack in active:
        goal = infer_goal_from_attack_name(attack)
        modality = infer_modality_from_attack_name(attack)
        if goal == "oda":
            groups["object_present"].append(attack)
        elif goal == "rma":
            groups["class_margin"].append(attack)
        elif modality in {"geometry", "frequency"}:
            groups["geometry_frequency"].append(attack)
        elif modality == "semantic_context":
            groups["semantic_context"].append(attack)
        elif modality in {"input_aware", "composite"}:
            groups["adaptive"].append(attack)
        else:
            groups["object_absent"].append(attack)

    if groups["object_absent"]:
        stages.append(
            OC3Stage(
                name="context_insufficiency_detox",
                purpose="For OGA visible/blend/natural triggers, prove context or trigger alone is insufficient for target evidence.",
                attack_families=groups["object_absent"],
                witness_types=["context_only", "object_erased", "object_transplant"],
                trainable_scope="head_adapter+last_cls_bias",
                max_lr=2e-6,
                max_epochs=2,
                loss_weights={"context_insufficiency": 1.5, "object_necessity": 1.0, "nms_rank_margin": 0.8},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["generate object-erased target-absent witnesses", "do not update backbone unless no candidate passes"],
            )
        )
    if groups["object_present"]:
        stages.append(
            OC3Stage(
                name="object_sufficiency_rebinding",
                purpose="For ODA residuals, preserve/rebind true-object evidence instead of raising target score globally.",
                attack_families=groups["object_present"],
                witness_types=["near_gt_object", "object_transplant", "context_changed"],
                trainable_scope="head_adapter+objectness_bias",
                max_lr=1e-6,
                max_epochs=2,
                loss_weights={"object_sufficiency": 1.5, "transform_consistency": 0.5, "nms_rank_margin": 0.8},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["near-GT candidates must remain above teacher floor", "reject if target-absent FP increases"],
            )
        )
    if groups["semantic_context"]:
        stages.append(
            OC3Stage(
                name="semantic_context_counterfactual_detox",
                purpose="Separate object evidence from semantic/context shortcuts via context swaps and target-absent witnesses.",
                attack_families=groups["semantic_context"],
                witness_types=["context_swap", "context_only", "object_present_context_changed"],
                trainable_scope="head_adapter",
                max_lr=1e-6,
                max_epochs=2,
                loss_weights={"context_insufficiency": 1.2, "object_sufficiency": 1.2, "transform_consistency": 0.7},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["build context-only and context-swapped witnesses", "use held-out semantic scenes for final gate only"],
            )
        )
    if groups["geometry_frequency"]:
        stages.append(
            OC3Stage(
                name="geometry_frequency_consensus",
                purpose="Handle WaNet/SIG/low-frequency residuals with transform consensus instead of semantic suppression.",
                attack_families=groups["geometry_frequency"],
                witness_types=["geometry_pair", "frequency_pair", "object_absent_transform"],
                trainable_scope="neck_adapter+head_adapter",
                max_lr=8e-7,
                max_epochs=2,
                loss_weights={"transform_consistency": 1.5, "context_insufficiency": 1.0, "object_sufficiency": 0.8},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["use paired smooth-warp / low-frequency views", "freeze class bias unless OGA remains high"],
            )
        )
    if groups["class_margin"]:
        stages.append(
            OC3Stage(
                name="source_target_margin_rebinding",
                purpose="For RMA, maintain source evidence and suppress target evidence only on source regions.",
                attack_families=groups["class_margin"],
                witness_types=["source_region", "target_region", "source_context_swap"],
                trainable_scope="last_cls_bias",
                max_lr=1e-6,
                max_epochs=2,
                loss_weights={"class_margin": 1.5, "object_sufficiency": 0.7},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["require source-target margin improvement", "reject if target recall drops"],
            )
        )
    if groups["adaptive"]:
        stages.append(
            OC3Stage(
                name="multi_view_adaptive_consensus",
                purpose="For input-aware/composite residuals, require agreement across multiple counterfactual views and use worst-view gate.",
                attack_families=groups["adaptive"],
                witness_types=["multi_view", "object_erased", "context_swap", "geometry_pair"],
                trainable_scope="head_adapter",
                max_lr=8e-7,
                max_epochs=3,
                loss_weights={"context_insufficiency": 1.0, "transform_consistency": 1.0, "nms_rank_margin": 1.0},
                hard_gates={"max_asr": max_safe_asr, "max_map_drop": max_map_drop},
                command_hints=["evaluate worst-view ASR", "use AutoDetox to stop at first no-worse pass"],
            )
        )

    return OC3Plan(
        name=name,
        residual_summary=residuals,
        stages=stages,
        evidence_actions=evidence_actions,
        novelty_notes=[
            "OC3 uses object/context counterfactual witnesses instead of pruning channels or globally fine-tuning the detector.",
            "The same base scene is split into object-sufficient, context-only, object-erased, and transform-paired witnesses.",
            "Candidate selection is NMS/ranking aware and gated by ASR, clean mAP, CFRC, and strict-ceiling evidence.",
        ],
        blockers=blockers,
    )

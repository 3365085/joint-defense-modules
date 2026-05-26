from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnosis import diagnose_snapshot
from .gates import candidate_score, evaluate_gate, hard_gate_pass
from .metrics import build_metric_snapshot_from_paths
from .policy import generate_candidate_recipes
from .schema import AutoDetoxPlan, CandidateResult, GateSpec, MetricSnapshot


@dataclass(frozen=True)
class AutoDetoxInputs:
    external_report: str | None = None
    clean_before: str | None = None
    clean_after: str | None = None
    cfrc_report: str | None = None
    strict_report: str | None = None
    heldout_report: str | None = None
    generalization_report: str | None = None
    model_path: str | None = None
    clean_anchor_model: str | None = None
    data_yaml: str | None = None
    images: str | None = None
    labels: str | None = None
    teacher_model: str | None = None
    positive_data_yaml: str | None = None
    negative_image_list: str | None = None
    negative_train: int | None = None
    negative_val: int | None = None
    negative_train_variants: str | None = None
    negative_val_variants: str | None = None
    imgsz: int | None = None
    batch: int | None = None
    device: str | None = None
    cycles: int | None = None
    phase_epochs: int | None = None
    feature_epochs: int | None = None
    recovery_epochs: int | None = None
    max_images: int | None = None
    eval_max_images: int | None = None
    external_eval_max_images_per_attack: int | None = None
    num_workers: int | None = None
    smoke: bool = False
    external_roots: tuple[str, ...] = ()
    target_classes: tuple[str, ...] = ()
    # Audit-report disambiguation (target_absent_trigger_audit_yolo can carry
    # multiple model rows; pick by name otherwise default to the last row,
    # which is the convention for "defended after clean_anchor").
    target_absent_audit_model: str | None = None
    # Confidence threshold (string key as written in the audit report) that
    # AutoDetox should normalize against.  Defaults to the strict 5% deployment
    # gate; previously hardcoded to ``"0.2500"``.
    target_absent_audit_threshold: str = "0.2500"


def build_autodetox_plan(
    inputs: AutoDetoxInputs,
    spec: GateSpec,
    *,
    name: str = "auto_detox_plan",
    out_root: str = "runs/auto_detox",
    max_candidates: int = 8,
    max_rounds: int = 2,
) -> AutoDetoxPlan:
    snapshot = build_metric_snapshot_from_paths(
        external_report=inputs.external_report,
        clean_before=inputs.clean_before,
        clean_after=inputs.clean_after,
        cfrc_report=inputs.cfrc_report,
        strict_report=inputs.strict_report,
        heldout_report=inputs.heldout_report,
        generalization_report=inputs.generalization_report,
        target_absent_audit_threshold=inputs.target_absent_audit_threshold,
        target_absent_audit_model=inputs.target_absent_audit_model,
    )
    diagnosis = diagnose_snapshot(snapshot, spec)
    recipes = generate_candidate_recipes(
        diagnosis,
        spec,
        model_path=inputs.model_path,
        clean_anchor_model=inputs.clean_anchor_model,
        clean_before_json=inputs.clean_before,
        data_yaml=inputs.data_yaml,
        images=inputs.images,
        labels=inputs.labels,
        teacher_model=inputs.teacher_model,
        positive_data_yaml=inputs.positive_data_yaml,
        negative_image_list=inputs.negative_image_list,
        negative_train=inputs.negative_train,
        negative_val=inputs.negative_val,
        negative_train_variants=inputs.negative_train_variants,
        negative_val_variants=inputs.negative_val_variants,
        imgsz=inputs.imgsz,
        batch=inputs.batch,
        device=inputs.device,
        cycles=inputs.cycles,
        phase_epochs=inputs.phase_epochs,
        feature_epochs=inputs.feature_epochs,
        recovery_epochs=inputs.recovery_epochs,
        max_images=inputs.max_images,
        eval_max_images=inputs.eval_max_images,
        external_eval_max_images_per_attack=inputs.external_eval_max_images_per_attack,
        num_workers=inputs.num_workers,
        smoke=inputs.smoke,
        external_roots=list(inputs.external_roots),
        target_classes=list(inputs.target_classes),
        out_root=out_root,
        max_candidates=max_candidates,
        external_report=inputs.external_report,
    )
    notes = [
        "AutoDetox generated recipes from evidence; do not manually override gates when accepting candidates.",
        "Training commands are safe-plan recipes. Use --execute only in a GPU environment with datasets present.",
    ]
    if snapshot.guarded:
        notes.append("External report appears guarded; do not claim guard-free detox from this plan without unguarded evidence.")
    if snapshot.pipeline_error:
        notes.append("External report is missing rows/counts; treat as evidence-pipeline error rather than zero ASR.")
    return AutoDetoxPlan(
        name=name,
        diagnosis=diagnosis,
        metric_snapshot=snapshot,
        gate_spec=spec,
        recipes=recipes,
        controller_notes=notes,
        max_rounds=max_rounds,
        max_candidates_per_round=max_candidates,
    )


def render_plan_markdown(plan: AutoDetoxPlan) -> str:
    lines = [
        f"# AutoDetox Plan: {plan.name}",
        "",
        "## Diagnosis",
        "",
        f"- status: `{plan.diagnosis.status}`",
        f"- primary failure: `{plan.diagnosis.primary_failure}`",
        f"- repair family: `{plan.diagnosis.repair_family}`",
        f"- blockers: `{', '.join(plan.diagnosis.blockers) if plan.diagnosis.blockers else 'none'}`",
        f"- warnings: `{', '.join(plan.diagnosis.warnings) if plan.diagnosis.warnings else 'none'}`",
        "",
    ]
    if plan.diagnosis.rationale:
        lines += ["### Rationale", *[f"- {x}" for x in plan.diagnosis.rationale], ""]
    snap = plan.metric_snapshot
    lines += [
        "## Current Metrics",
        "",
        f"- max ASR: `{snap.max_asr}`",
        f"- mean ASR: `{snap.mean_asr}`",
        f"- clean mAP50-95 drop: `{snap.clean_map50_95_drop}`",
        f"- CFRC pass: `{snap.cfrc_pass}`; CMR: `{snap.cfrc_cmr}`",
        f"- strict ceiling pass: `{snap.strict_ceiling_pass}`; max high: `{snap.strict_ceiling_max_high}`",
        "",
        "## Candidate Recipes",
        "",
        "| # | recipe | strategy | purpose |",
        "|---:|---|---|---|",
    ]
    for idx, recipe in enumerate(plan.recipes, 1):
        purpose = recipe.purpose.replace("|", "/")
        lines.append(f"| {idx} | `{recipe.name}` | `{recipe.strategy}` | {purpose} |")
    lines.append("")
    for recipe in plan.recipes:
        lines += [f"### {recipe.name}", "", f"- strategy: `{recipe.strategy}`", f"- params: `{json.dumps(recipe.params, ensure_ascii=False)}`"]
        if recipe.command:
            lines += ["- command hint:", "", "```bash", " ".join(recipe.command), "```", ""]
        if recipe.risk_notes:
            lines += ["- risk notes:", *[f"  - {n}" for n in recipe.risk_notes], ""]
    if plan.controller_notes:
        lines += ["## Controller Notes", "", *[f"- {x}" for x in plan.controller_notes], ""]
    return "\n".join(lines)


def write_plan(plan: AutoDetoxPlan, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "auto_detox_plan.json"
    md_path = out / "AUTO_DETOX_PLAN.md"
    json_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_plan_markdown(plan), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def evaluate_candidate_result(name: str, snapshot: MetricSnapshot, spec: GateSpec, *, model_path: str | None = None) -> CandidateResult:
    violations = evaluate_gate(snapshot, spec)
    accepted = hard_gate_pass(snapshot, spec)
    return CandidateResult(name=name, model_path=model_path, snapshot=snapshot, violations=violations, accepted=accepted, score=candidate_score(snapshot, spec))


def select_best_candidate(results: Sequence[CandidateResult]) -> CandidateResult | None:
    accepted = [r for r in results if r.accepted]
    if not accepted:
        return None
    return min(accepted, key=lambda r: r.score)


_DEFAULT_EXECUTE_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours; long enough for a Hybrid-PURIFY cycle.


def _resolve_dependency_state(
    recipe,
    completed: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[bool, list[str]]:
    """Return (ready, blocking_dependencies) for a recipe.

    A dependency blocks the recipe if it was skipped, errored, or its manifest
    explicitly reports a pipeline error.
    """

    deps = list(getattr(recipe, "depends_on", []) or [])
    if not deps or completed is None:
        return True, []
    blocking: list[str] = []
    for dep in deps:
        info = completed.get(dep)
        if not info or not info.get("executed"):
            blocking.append(dep)
            continue
        # If the dependency wrote a manifest, propagate failures.
        accepted = info.get("accepted")
        manifest_status = str(info.get("manifest_status") or "").lower()
        if manifest_status in {"evaluation_failed", "pipeline_error", "evaluation_error"}:
            blocking.append(dep)
            continue
        if accepted is False and info.get("manifest_status") not in (None, ""):
            # Non-acceptance is a soft signal; only propagate when it represents
            # a hard pipeline failure.  We keep going for legitimate detox
            # candidates that simply did not pass safety gates yet.
            pass
        rc = info.get("returncode")
        if rc is not None and int(rc) != 0:
            blocking.append(dep)
    return not blocking, blocking


def execute_recipe(
    recipe,
    *,
    cwd: str | Path | None = None,
    timeout: int | None = None,
    completed: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not recipe.command:
        return {"recipe": recipe.name, "executed": False, "returncode": None, "reason": "no_command"}

    ready, blocking = _resolve_dependency_state(recipe, completed)
    if not ready:
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "dependency_blocked",
            "blocking_dependencies": blocking,
            "command": list(recipe.command),
        }

    missing_paths = [p for p in recipe.params.get("required_paths", []) if p and not Path(str(p)).exists()]
    if missing_paths:
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "required_path_missing",
            "missing_paths": missing_paths,
            "command": list(recipe.command),
        }
    if not bool(recipe.params.get("execute_ready", False)):
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "plan_only_command_hint",
            "command": list(recipe.command),
        }
    if any(str(part).startswith("<") and str(part).endswith(">") for part in recipe.command):
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "placeholder_command",
            "command": list(recipe.command),
        }

    effective_timeout = timeout if timeout is not None else _DEFAULT_EXECUTE_TIMEOUT_SECONDS
    try:
        proc = subprocess.run(
            recipe.command,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "timeout",
            "timeout_seconds": float(effective_timeout) if effective_timeout is not None else None,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "command": list(recipe.command),
        }
    except FileNotFoundError as exc:
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "executable_missing",
            "error": str(exc),
            "command": list(recipe.command),
        }
    except OSError as exc:
        return {
            "recipe": recipe.name,
            "executed": False,
            "returncode": None,
            "reason": "os_error",
            "error": str(exc),
            "command": list(recipe.command),
        }
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    result: dict[str, Any] = {"recipe": recipe.name, "executed": True, "returncode": proc.returncode, "stdout": stdout[-4000:], "stderr": stderr[-4000:]}
    manifest_path = recipe.params.get("manifest_path")
    if manifest_path and Path(str(manifest_path)).exists():
        try:
            manifest = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
            status = manifest.get("status")
            best = manifest.get("best") if isinstance(manifest.get("best"), Mapping) else {}
            history_best_asr = manifest.get("history_best_by_asr") if isinstance(manifest.get("history_best_by_asr"), Mapping) else None
            history_best_balanced = manifest.get("history_best_balanced") if isinstance(manifest.get("history_best_balanced"), Mapping) else None
            result.update(
                {
                    "manifest_path": str(manifest_path),
                    "manifest_status": status,
                    "accepted": bool(status == "passed" or best.get("passes") is True),
                    "final_model": manifest.get("final_model") or best.get("model"),
                    "external_max_asr": best.get("external_max_asr"),
                    "external_mean_asr": best.get("external_mean_asr"),
                    "map_drop": best.get("map_drop"),
                    # Surface the best per-history candidates so the controller
                    # can score them even when the serial best_item bottomed
                    # out on the input poisoned checkpoint due to map-drop
                    # gates blocking every accepted_as_best replacement.
                    "history_best_by_asr": history_best_asr,
                    "history_best_balanced": history_best_balanced,
                    "history_n_candidates": manifest.get("history_n_candidates"),
                }
            )
        except Exception as exc:
            result["manifest_read_error"] = str(exc)
    return result


def execute_plan(
    plan: AutoDetoxPlan,
    *,
    cwd: str | Path | None = None,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    """Run plan recipes in order while honouring ``depends_on`` declarations."""

    completed: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for recipe in plan.recipes:
        result = execute_recipe(recipe, cwd=cwd, timeout=timeout, completed=completed)
        completed[recipe.name] = result
        results.append(result)
    return results

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any, Mapping

from .schema import AutoDiagnosis, CandidateRecipe, GateSpec


# Use ``sys.executable`` instead of bare ``python``: the project runs under
# pixi and bare ``python`` may resolve to a different interpreter.
PYTHON_EXE: str = sys.executable or "python"


def _gate_dict(spec: GateSpec) -> dict[str, Any]:
    return {
        "max_asr": spec.max_asr,
        "max_clean_map_drop": spec.max_clean_map_drop,
        "require_cfrc_pass": spec.require_cfrc_pass,
        "require_strict_ceiling_pass": spec.require_strict_ceiling_pass,
        "per_attack_max_asr": dict(spec.per_attack_max_asr),
    }


def recipe_fingerprint(recipe: CandidateRecipe) -> str:
    blob = json.dumps(recipe.fingerprint_payload(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _ready(*values: str | None) -> bool:
    return all(bool(v) for v in values)


def _extend_option(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _extend_list_option(cmd: list[str], flag: str, values: list[str] | None) -> None:
    vals = [str(v) for v in (values or []) if str(v)]
    if vals:
        cmd.append(flag)
        cmd.extend(vals)


def _oc3_plan_recipe(
    *,
    name: str,
    out_root: str,
    external_report: str | None,
    target_classes: list[str] | None,
    gates: dict[str, Any],
    purpose_suffix: str = "",
) -> CandidateRecipe:
    """Emit an OC3-Detox plan recipe.

    The OC3 planner consumes the external_report and writes a residual-aware
    set of stages (object_present / context_only / object_erased /
    object_transplant / geometry_pair / frequency_pair witnesses + five
    candidate-box loss terms).  It is a *planning* recipe; trainer wiring is
    not required for the recipe to be useful since AutoDetox records the
    plan + loss summary as audit evidence.
    """

    cmd = [PYTHON_EXE, "scripts/oc3_detox_plan_yolo.py", "--out", out_root]
    _extend_option(cmd, "--external-report", external_report)
    _extend_option(cmd, "--max-asr", gates.get("max_asr"))
    _extend_option(cmd, "--max-map-drop", gates.get("max_clean_map_drop"))
    purpose = (
        "Build a candidate-box-level OC3 (object-context counterfactual consensus) plan for residual attacks; "
        "stages map directly to the four canonical OC3 witness types and the five loss terms."
        + (f" {purpose_suffix}" if purpose_suffix else "")
    )
    return CandidateRecipe(
        name=name,
        strategy="oc3_counterfactual_consensus_plan",
        purpose=purpose,
        params={
            "execute_ready": bool(external_report),
            "required_paths": [external_report] if external_report else [],
            "manifest_path": f"{out_root}/oc3_detox_plan.json",
            "out_root": out_root,
            "target_classes": list(target_classes or []),
        },
        expected_effect={
            "evidence": "structured OC3 plan + loss audit JSON",
            "asr": "unchanged at planning time",
            "map": "unchanged at planning time",
        },
        hard_gates=gates,
        command=cmd,
        risk_notes=[
            "OC3 trainer wiring is not yet GPU-ready in this branch; the plan + witness loss audit are recorded as evidence and consumed by the OC3-aware fall-through.",
        ],
    )


def _negative_mix_recipe(
    *,
    name: str,
    purpose: str,
    strategy: str,
    out_root: str,
    positive_data_yaml: str | None,
    negative_image_list: str | None,
    train_variants: str,
    val_variants: str,
    negative_train: int | None,
    negative_val: int | None,
    smoke: bool,
    gates: dict[str, Any],
) -> CandidateRecipe:
    root = f"{out_root}/{name}"
    train_n = 8 if smoke else (negative_train if negative_train is not None else 1600)
    val_n = 4 if smoke else (negative_val if negative_val is not None else 400)
    cmd = [
        PYTHON_EXE,
        "scripts/prepare_yolo_negative_mix_dataset.py",
        "--positive-data-yaml",
        positive_data_yaml or "<positive_data_yaml>",
        "--negative-image-list",
        negative_image_list or "<negative_image_list>",
        "--out-root",
        root,
        "--negative-train",
        str(train_n),
        "--negative-val",
        str(val_n),
        "--negative-train-variants",
        train_variants,
        "--negative-val-variants",
        val_variants,
    ]
    execute_ready = _ready(positive_data_yaml, negative_image_list)
    return CandidateRecipe(
        name=name,
        strategy=strategy,
        purpose=purpose,
        params={
            "execute_ready": execute_ready,
            "required_paths": [p for p in [positive_data_yaml, negative_image_list] if p],
            "out_root": root,
            "manifest_path": f"{root}/negative_mix_manifest.json",
            "data_yaml": f"{root}/data.yaml",
            "images": f"{root}/images/train",
            "labels": f"{root}/labels/train",
            "negative_train": train_n,
            "negative_val": val_n,
            "negative_train_variants": train_variants,
            "negative_val_variants": val_variants,
        },
        expected_effect={"dataset": "target-absent negatives added", "labels": "empty labels for negatives"},
        hard_gates=gates,
        command=cmd,
    )


def _hybrid_recipe(
    *,
    name: str,
    strategy: str,
    purpose: str,
    out_root: str,
    model_path: str | None,
    teacher_model: str | None,
    data_yaml: str | None,
    images: str | None,
    labels: str | None,
    target_classes: list[str],
    external_roots: list[str],
    gates: dict[str, Any],
    imgsz: int | None,
    batch: int | None,
    device: str | None,
    cycles: int | None,
    phase_epochs: int | None,
    feature_epochs: int | None,
    recovery_epochs: int | None,
    max_images: int | None,
    eval_max_images: int | None,
    external_eval_max_images_per_attack: int | None,
    num_workers: int | None,
    smoke: bool,
    extra_flags: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> CandidateRecipe:
    if smoke:
        imgsz = imgsz or 416
        batch = batch or 2
        cycles = cycles if cycles is not None else 1
        phase_epochs = phase_epochs if phase_epochs is not None else 1
        feature_epochs = feature_epochs if feature_epochs is not None else 0
        recovery_epochs = recovery_epochs if recovery_epochs is not None else 0
        max_images = max_images if max_images is not None else 12
        eval_max_images = eval_max_images if eval_max_images is not None else 12
        external_eval_max_images_per_attack = external_eval_max_images_per_attack if external_eval_max_images_per_attack is not None else 8
        num_workers = num_workers if num_workers is not None else 0
    cmd = [
        PYTHON_EXE,
        "scripts/hybrid_purify_detox_yolo.py",
        "--model",
        model_path or "<model>",
        "--images",
        images or "<images>",
        "--labels",
        labels or "<labels>",
        "--data-yaml",
        data_yaml or "<data_yaml>",
        "--out",
        f"{out_root}/{name}",
        "--max-allowed-external-asr",
        str(gates["max_asr"]),
        "--max-map-drop",
        str(gates["max_clean_map_drop"]),
    ]
    if teacher_model:
        cmd.extend(["--teacher-model", teacher_model])
    _extend_list_option(cmd, "--target-classes", target_classes)
    _extend_list_option(cmd, "--external-eval-roots", external_roots)
    _extend_list_option(cmd, "--external-replay-roots", external_roots)
    _extend_option(cmd, "--imgsz", imgsz)
    _extend_option(cmd, "--batch", batch)
    _extend_option(cmd, "--device", device)
    _extend_option(cmd, "--cycles", cycles)
    _extend_option(cmd, "--phase-epochs", phase_epochs)
    _extend_option(cmd, "--feature-epochs", feature_epochs)
    _extend_option(cmd, "--recovery-epochs", recovery_epochs)
    _extend_option(cmd, "--max-images", max_images)
    _extend_option(cmd, "--eval-max-images", eval_max_images)
    _extend_option(cmd, "--external-eval-max-images-per-attack", external_eval_max_images_per_attack)
    _extend_option(cmd, "--num-workers", num_workers)
    for flag in extra_flags or []:
        cmd.append(flag)
    execute_ready = _ready(model_path, images, labels, data_yaml) and bool(target_classes)
    return CandidateRecipe(
        name=name,
        strategy=strategy,
        purpose=purpose,
        params={
            "execute_ready": execute_ready,
            "required_paths": [p for p in [model_path, images, labels, data_yaml] if p],
            "manifest_path": f"{out_root}/{name}/hybrid_purify_manifest.json",
            "smoke": bool(smoke),
        },
        expected_effect={"asr": "candidate should reduce residual attacks", "acceptance": "must pass AutoDetox gates after re-evaluation"},
        hard_gates=gates,
        command=cmd,
        risk_notes=["This recipe trains a candidate checkpoint; acceptance still requires ASR/mAP/CFRC re-evaluation."],
        depends_on=list(depends_on or []),
    )


def generate_candidate_recipes(
    diagnosis: AutoDiagnosis,
    spec: GateSpec,
    *,
    model_path: str | None = None,
    clean_anchor_model: str | None = None,
    clean_before_json: str | None = None,
    data_yaml: str | None = None,
    images: str | None = None,
    labels: str | None = None,
    teacher_model: str | None = None,
    positive_data_yaml: str | None = None,
    negative_image_list: str | None = None,
    negative_train: int | None = None,
    negative_val: int | None = None,
    negative_train_variants: str | None = None,
    negative_val_variants: str | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    cycles: int | None = None,
    phase_epochs: int | None = None,
    feature_epochs: int | None = None,
    recovery_epochs: int | None = None,
    max_images: int | None = None,
    eval_max_images: int | None = None,
    external_eval_max_images_per_attack: int | None = None,
    num_workers: int | None = None,
    smoke: bool = False,
    external_roots: list[str] | None = None,
    target_classes: list[str] | None = None,
    out_root: str = "runs/auto_detox",
    max_candidates: int = 8,
    external_report: str | None = None,
) -> list[CandidateRecipe]:
    """Generate safe, deterministic recipes for the diagnosed failure.

    This is intentionally not a black-box optimizer.  Every recipe is traceable
    to a failure type, making the system suitable for patents and papers.
    """

    target_classes = target_classes or []
    external_roots = external_roots or []
    gates = _gate_dict(spec)
    family = diagnosis.repair_family
    recipes: list[CandidateRecipe] = []

    if family == "none":
        recipes.append(
            CandidateRecipe(
                name="freeze_and_expand_evidence",
                strategy="evidence_only",
                purpose="Current gates pass; freeze checkpoint and expand independent evidence instead of retraining.",
                params={"strict_ceiling_target": spec.max_strict_ceiling_high or spec.max_asr, "hard_suite_expansion": "medium"},
                expected_effect={"asr": "unchanged", "map": "unchanged", "evidence": "stronger strict ceiling"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/strict_asr_ceiling_plan.py"],
            )
        )
    elif family == "evidence_expansion_only":
        recipes += [
            CandidateRecipe(
                name="strict_ceiling_expansion",
                strategy="DHNE_evidence_expansion",
                purpose="Expand target-absent or target-present hard suite until Wilson upper bound passes.",
                params={"mode": "medium", "min_total_if_zero_failure": 73},
                expected_effect={"asr": "measured on larger suite", "map": "unchanged", "evidence": "strict ASR ceiling"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/expand_hard_suite_yolo.py", "--mode", "medium"],
            ),
            CandidateRecipe(
                name="cfrc_matrix_refresh",
                strategy="certificate_refresh",
                purpose="Regenerate paired CFRC after suite expansion; no training.",
                params={"paired_bootstrap": True, "holm_bonferroni": True},
                expected_effect={"asr": "unchanged", "cfrc": "updated"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/t0_defense_certificate.py"],
                depends_on=["strict_ceiling_expansion"],
            ),
        ]
    elif family == "last_mile_utility_recovery":
        alphas = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04]
        # Backbone-only soup explores a much larger alpha range because the
        # head (``model.23`` for YOLOv8) carries the trained class logits and
        # its weights are far from the clean anchor.  The v4 evidence
        # (2026-05-21) showed alpha=0.24 with model.23 frozen recovers
        # ~7 pp mAP while keeping ASR=0 at imgsz=416, whereas the head-mixed
        # soup either fails to recover map or rebounds ASR.
        backbone_alphas = [0.05, 0.10, 0.15, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40]
        anchor = clean_anchor_model or teacher_model
        soup_root = f"{out_root}/weight_soup"
        backbone_root = f"{out_root}/weight_soup_backbone"
        common_eval_flags: list[str] = []
        # Wire optional evaluation when the operator supplied data and external
        # roots; the executor consumes ``manifest_path`` for scoring.  When
        # neither is present we still emit the recipe so a downstream operator
        # can run weight_soup manually, but we keep ``execute_ready=False``.
        if data_yaml:
            common_eval_flags += ["--evaluate", "--data-yaml", str(data_yaml)]
        if clean_before_json:
            common_eval_flags += ["--clean-before-json", str(clean_before_json)]
        if target_classes:
            common_eval_flags += ["--target-classes", *[str(c) for c in target_classes]]
        if external_roots:
            common_eval_flags += ["--external-roots", *[str(r) for r in external_roots]]
        if imgsz is not None:
            common_eval_flags += ["--imgsz", str(int(imgsz))]
        if batch is not None:
            common_eval_flags += ["--batch", str(int(batch))]
        if device is not None:
            common_eval_flags += ["--device", str(device)]
        if num_workers is not None:
            common_eval_flags += ["--workers", str(int(num_workers))]
        if external_eval_max_images_per_attack is not None:
            common_eval_flags += ["--max-images-per-attack", str(int(external_eval_max_images_per_attack))]
        # Cap ASR rebound to the gate; the script will refuse alphas whose ASR
        # rebound exceeds this bound.
        common_eval_flags += [
            "--max-allowed-asr", str(gates["max_asr"]),
            "--max-map-drop", str(gates["max_clean_map_drop"]),
        ]

        clean_anchor_cmd = [
            PYTHON_EXE, "scripts/weight_soup_last_mile_yolo.py",
            "--base-model", model_path or "<defended.pt>",
            "--anchor-model", anchor or "<clean_anchor.pt>",
            "--out", soup_root,
            "--alphas", ",".join(f"{a:g}" for a in alphas),
            *common_eval_flags,
        ]
        soup_required = [p for p in [model_path, anchor] if p]
        soup_execute_ready = bool(model_path) and bool(anchor)
        recipes.append(
            CandidateRecipe(
                name="last_mile_clean_anchor_weight_soup",
                strategy="utility_recovery_weight_soup",
                purpose="Recover small clean mAP loss without retraining; reject any ASR rebound.",
                params={
                    "alphas": alphas,
                    "include_key_patterns": ["model.*"],
                    "exclude_key_patterns": ["*.num_batches_tracked"],
                    "execute_ready": soup_execute_ready,
                    "required_paths": soup_required,
                    "manifest_path": f"{soup_root}/last_mile_weight_soup_manifest.json",
                    "out_root": soup_root,
                    "exclude_head": False,
                },
                expected_effect={"map": "increase", "asr": "must not rebound"},
                hard_gates=gates,
                command=clean_anchor_cmd,
                risk_notes=["If alpha=1.0 is selected, report it as clean-anchor replacement, not poisoned-checkpoint repair."],
            )
        )
        # Backbone-only variant: critical for v4 (and similar OGA backdoors)
        # where the trigger writes its bias into late convolutional layers in
        # the backbone but the head keeps the clean classifier mostly correct.
        # Mixing only ``model.0`` through ``model.22`` lets the backbone slide
        # toward the clean anchor without disturbing the head's class
        # confidences.  Always emit when ``model.23`` is the canonical YOLOv8
        # detection head.
        backbone_cmd = [
            PYTHON_EXE, "scripts/weight_soup_last_mile_yolo.py",
            "--base-model", model_path or "<defended.pt>",
            "--anchor-model", anchor or "<clean_anchor.pt>",
            "--out", backbone_root,
            "--alphas", ",".join(f"{a:g}" for a in backbone_alphas),
            "--exclude-key-pattern", "model.23",
            "--candidate-suffix", "backbone",
            *common_eval_flags,
        ]
        recipes.append(
            CandidateRecipe(
                name="last_mile_backbone_weight_soup",
                strategy="utility_recovery_weight_soup",
                purpose="Backbone-only weight soup; preserves head logits, sweeps larger alpha range. Targets OGA-style backdoors where the trigger lives in the backbone.",
                params={
                    "alphas": backbone_alphas,
                    "include_key_patterns": ["model.0", "model.1", "model.2", "model.3", "model.4",
                                            "model.5", "model.6", "model.7", "model.8", "model.9",
                                            "model.10", "model.11", "model.12", "model.13", "model.14",
                                            "model.15", "model.16", "model.17", "model.18", "model.19",
                                            "model.20", "model.21", "model.22"],
                    "exclude_key_patterns": ["model.23", "*.num_batches_tracked"],
                    "execute_ready": soup_execute_ready,
                    "required_paths": soup_required,
                    "manifest_path": f"{backbone_root}/last_mile_weight_soup_manifest.json",
                    "out_root": backbone_root,
                    "exclude_head": True,
                },
                expected_effect={"map": "increase", "asr": "must not rebound at backbone-only mixing"},
                hard_gates=gates,
                command=backbone_cmd,
                risk_notes=[
                    "Assumes a YOLOv8-family head named 'model.23'. Verify the head index for non-default architectures before trusting alpha selection.",
                ],
            )
        )
        recipes.append(
            CandidateRecipe(
                name="minimal_clean_replay_recovery",
                strategy="clean_replay_teacher_stability",
                purpose="One-epoch low-lr clean replay to recover utility while preserving ASR gates.",
                params={"lr": 5e-7, "epochs": 1, "teacher_stability": 50.0, "attack_replay_floor": True},
                expected_effect={"map": "increase", "asr": "preserved by replay floor"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/hybrid_purify_detox_yolo.py", "--no-prototype", "--use-lagrangian-controller"],
            )
        )
    elif family == "target_absent_hard_negative_detox":
        train_variants = negative_train_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
        val_variants = negative_val_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
        mix = _negative_mix_recipe(
            name="target_absent_trigger_negative_mix",
            strategy="DHNE_target_absent_hard_negative",
            purpose="Generate diverse target-absent negatives with visible/blend/frequency/warp variants.",
            out_root=out_root,
            positive_data_yaml=positive_data_yaml,
            negative_image_list=negative_image_list,
            train_variants=train_variants,
            val_variants=val_variants,
            negative_train=negative_train,
            negative_val=negative_val,
            smoke=smoke,
            gates=gates,
        )
        train_data_yaml = str(mix.params["data_yaml"]) if mix.params.get("execute_ready") else data_yaml
        train_images = str(mix.params["images"]) if mix.params.get("execute_ready") else images
        train_labels = str(mix.params["labels"]) if mix.params.get("execute_ready") else labels
        smoke_flags = ["--no-prototype", "--no-attention", "--no-pre-prune"] if smoke else []
        recipes += [
            mix,
            _hybrid_recipe(
                name="hybrid_lagrangian_target_absent_detox",
                strategy="hybrid_purify_lagrangian",
                purpose="Train a detox candidate on mixed clean positives and target-absent hard negatives; rollback gates select only improvements.",
                out_root=out_root,
                model_path=model_path,
                teacher_model=teacher_model,
                data_yaml=train_data_yaml,
                images=train_images,
                labels=train_labels,
                target_classes=target_classes,
                external_roots=external_roots,
                gates=gates,
                imgsz=imgsz,
                batch=batch,
                device=device,
                cycles=cycles,
                phase_epochs=phase_epochs,
                feature_epochs=feature_epochs,
                recovery_epochs=recovery_epochs,
                max_images=max_images,
                eval_max_images=eval_max_images,
                external_eval_max_images_per_attack=external_eval_max_images_per_attack,
                num_workers=num_workers,
                smoke=smoke,
                extra_flags=["--use-lagrangian-controller", "--recovery-replay-external", "--rollback-unimproved-phase", *smoke_flags],
                depends_on=["target_absent_trigger_negative_mix"],
            ),
            CandidateRecipe(
                name="orthogonal_trigger_subspace_neutralization",
                strategy="OTSN",
                purpose="Estimate trigger-causal feature directions from paired clean/trigger views and suppress only target-absent evidence.",
                params={"rank": 4, "projection_strength": 0.5, "preserve_target_present": True},
                expected_effect={"visible_patch_oga": "decrease", "semantic_drift": "low"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/frontier_detox_algorithm_plan.py"],
            ),
            _oc3_plan_recipe(
                name="oc3_counterfactual_consensus_plan",
                out_root=f"{out_root}/oc3_plan",
                external_report=external_report,
                target_classes=target_classes,
                gates=gates,
                purpose_suffix="OGA visible/blend residuals: context_only and object_erased witnesses anchor context insufficiency / object necessity.",
            ),
        ]
    elif family in {
        "geometry_frequency_detox",
        "multi_family_with_geometry_priority",
        "multi_family_with_geometry_and_recall",
    }:
        smoke_flags = ["--no-prototype", "--no-attention", "--no-pre-prune"] if smoke else []
        if family in {"multi_family_with_geometry_priority", "multi_family_with_geometry_and_recall"}:
            train_variants = negative_train_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
            val_variants = negative_val_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
            mix = _negative_mix_recipe(
                name="target_absent_trigger_negative_mix",
                strategy="DHNE_target_absent_hard_negative",
                purpose="Build the mixed target-absent dataset used by executable detox candidates.",
                out_root=out_root,
                positive_data_yaml=positive_data_yaml,
                negative_image_list=negative_image_list,
                train_variants=train_variants,
                val_variants=val_variants,
                negative_train=negative_train,
                negative_val=negative_val,
                smoke=smoke,
                gates=gates,
            )
            train_data_yaml = str(mix.params["data_yaml"]) if mix.params.get("execute_ready") else data_yaml
            train_images = str(mix.params["images"]) if mix.params.get("execute_ready") else images
            train_labels = str(mix.params["labels"]) if mix.params.get("execute_ready") else labels
            recipes += [
                mix,
                _hybrid_recipe(
                    name="hybrid_lagrangian_mixed_detox",
                    strategy="hybrid_purify_lagrangian",
                    purpose="Run the executable mixed-family detox candidate against visible/blend/frequency residuals.",
                    out_root=out_root,
                    model_path=model_path,
                    teacher_model=teacher_model,
                    data_yaml=train_data_yaml,
                    images=train_images,
                    labels=train_labels,
                    target_classes=target_classes,
                    external_roots=external_roots,
                    gates=gates,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    cycles=cycles,
                    phase_epochs=phase_epochs,
                    feature_epochs=feature_epochs,
                    recovery_epochs=recovery_epochs,
                    max_images=max_images,
                    eval_max_images=eval_max_images,
                    external_eval_max_images_per_attack=external_eval_max_images_per_attack,
                    num_workers=num_workers,
                    smoke=smoke,
                    extra_flags=["--use-lagrangian-controller", "--recovery-replay-external", "--rollback-unimproved-phase", *smoke_flags],
                    depends_on=["target_absent_trigger_negative_mix"],
                ),
                CandidateRecipe(
                    name="orthogonal_trigger_subspace_neutralization",
                    strategy="OTSN",
                    purpose="Suppress trigger-causal target evidence while preserving target-present detections.",
                    params={"rank": 4, "projection_strength": 0.5, "preserve_target_present": True},
                    expected_effect={"visible_patch_oga": "decrease", "semantic_drift": "low"},
                    hard_gates=gates,
                    command=[PYTHON_EXE, "scripts/frontier_detox_algorithm_plan.py"],
                ),
            ]
            if family == "multi_family_with_geometry_and_recall":
                # Add ODA recall preservation as a parallel safety branch when
                # geometry residuals coexist with ODA residuals.
                recipes.append(
                    CandidateRecipe(
                        name="oda_recall_preserving_calibration",
                        strategy="target_present_recall_preserve",
                        purpose="Repair near-GT target score/ranking without increasing target-absent hallucination.",
                        params={
                            "lambda_score_calibration": 1.0,
                            "lambda_far_margin": 0.25,
                            "lambda_target_absent_guard": 1.0,
                            "lr": 1e-6,
                            "epochs": 1,
                        },
                        expected_effect={"oda_asr": "decrease", "oga_asr": "no worse"},
                        hard_gates=gates,
                        command=[PYTHON_EXE, "scripts/oda_score_calibration_repair_yolo.py"],
                    )
                )
        recipes += [
            CandidateRecipe(
                name="spectrum_geometry_consistency",
                strategy="SGC",
                purpose="Dedicated WaNet/SIG/low-frequency repair using transform consistency, not semantic suppression.",
                params={"warp_consistency": 1.0, "lowfreq_consistency": 1.0, "target_absent_cap": 0.25, "lr": 1e-6, "epochs": 2},
                expected_effect={"wanet_sig_asr": "decrease", "map": "preserve through teacher stability"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/hybrid_purify_detox_yolo.py", "--use-lagrangian-controller"],
            ),
            _negative_mix_recipe(
                name="frequency_negative_mix",
                strategy="frequency_target_absent_negatives",
                purpose="Build frequency/warp target-absent negatives for a follow-up candidate if mixed detox is insufficient.",
                out_root=out_root,
                positive_data_yaml=positive_data_yaml,
                negative_image_list=negative_image_list,
                train_variants=negative_train_variants or "clean,sig_multiperiod,lowfreq_radial,wanet_warp",
                val_variants=negative_val_variants or "clean,sig_multiperiod,lowfreq_radial,wanet_warp",
                negative_train=negative_train,
                negative_val=negative_val,
                smoke=smoke,
                gates=gates,
            ),
        ]
        freq_root = f"{out_root}/frequency_negative_mix"
        recipes.append(
            _hybrid_recipe(
                name="hybrid_lagrangian_frequency_detox",
                strategy="hybrid_purify_frequency_followup",
                purpose="Follow-up executable candidate for frequency/warp residuals if the mixed-family candidate fails.",
                out_root=out_root,
                model_path=model_path,
                teacher_model=teacher_model,
                data_yaml=f"{freq_root}/data.yaml" if _ready(positive_data_yaml, negative_image_list) else data_yaml,
                images=f"{freq_root}/images/train" if _ready(positive_data_yaml, negative_image_list) else images,
                labels=f"{freq_root}/labels/train" if _ready(positive_data_yaml, negative_image_list) else labels,
                target_classes=target_classes,
                external_roots=external_roots,
                gates=gates,
                imgsz=imgsz,
                batch=batch,
                device=device,
                cycles=cycles,
                phase_epochs=phase_epochs,
                feature_epochs=feature_epochs,
                recovery_epochs=recovery_epochs,
                max_images=max_images,
                eval_max_images=eval_max_images,
                external_eval_max_images_per_attack=external_eval_max_images_per_attack,
                num_workers=num_workers,
                smoke=smoke,
                extra_flags=["--use-lagrangian-controller", "--recovery-replay-external", "--rollback-unimproved-phase", *smoke_flags],
                depends_on=["frequency_negative_mix"],
            )
        )
        recipes.append(
            _oc3_plan_recipe(
                name="oc3_counterfactual_consensus_plan",
                out_root=f"{out_root}/oc3_plan",
                external_report=external_report,
                target_classes=target_classes,
                gates=gates,
                purpose_suffix="Frequency/geometry residuals: geometry_pair / frequency_pair witnesses anchor transform consensus.",
            )
        )
    elif family == "target_present_recall_preserve" or family == "multi_family_with_recall_preserve":
        recipes += [
            CandidateRecipe(
                name="oda_recall_preserving_calibration",
                strategy="target_present_recall_preserve",
                purpose="Repair near-GT target score/ranking without increasing target-absent hallucination.",
                params={"lambda_score_calibration": 1.0, "lambda_far_margin": 0.25, "lambda_target_absent_guard": 1.0, "lr": 1e-6, "epochs": 1},
                expected_effect={"oda_asr": "decrease", "oga_asr": "no worse"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/oda_score_calibration_repair_yolo.py"],
            ),
            _oc3_plan_recipe(
                name="oc3_counterfactual_consensus_plan",
                out_root=f"{out_root}/oc3_plan",
                external_report=external_report,
                target_classes=target_classes,
                gates=gates,
                purpose_suffix="ODA residual: object_present + object_transplant witnesses anchor object sufficiency.",
            ),
        ]
    elif family == "multi_attack_lagrangian_detox" or family == "semantic_causal_detox":
        # Multi-attack diffuse residual or pure semantic shortcut: run the
        # generic Lagrangian Hybrid-PURIFY against the full mixed negative set
        # without prejudging visible-patch vs frequency vs ODA priority.  This
        # branch is what the diagnosis fallback used to silently lose to the
        # ``collect_missing_evidence`` else-branch.
        smoke_flags = ["--no-prototype", "--no-attention", "--no-pre-prune"] if smoke else []
        train_variants = negative_train_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
        val_variants = negative_val_variants or "clean,blend_grid,patch_hash_10,sig_multiperiod,lowfreq_radial,wanet_warp"
        mix = _negative_mix_recipe(
            name="target_absent_trigger_negative_mix",
            strategy="DHNE_target_absent_hard_negative",
            purpose="Generic mixed target-absent dataset for diffuse multi-attack residuals.",
            out_root=out_root,
            positive_data_yaml=positive_data_yaml,
            negative_image_list=negative_image_list,
            train_variants=train_variants,
            val_variants=val_variants,
            negative_train=negative_train,
            negative_val=negative_val,
            smoke=smoke,
            gates=gates,
        )
        train_data_yaml = str(mix.params["data_yaml"]) if mix.params.get("execute_ready") else data_yaml
        train_images = str(mix.params["images"]) if mix.params.get("execute_ready") else images
        train_labels = str(mix.params["labels"]) if mix.params.get("execute_ready") else labels
        recipes += [
            mix,
            _hybrid_recipe(
                name="hybrid_lagrangian_multi_attack_detox",
                strategy="hybrid_purify_lagrangian",
                purpose="Generic Lagrangian Hybrid-PURIFY for diffuse multi-attack ASR residuals.",
                out_root=out_root,
                model_path=model_path,
                teacher_model=teacher_model,
                data_yaml=train_data_yaml,
                images=train_images,
                labels=train_labels,
                target_classes=target_classes,
                external_roots=external_roots,
                gates=gates,
                imgsz=imgsz,
                batch=batch,
                device=device,
                cycles=cycles,
                phase_epochs=phase_epochs,
                feature_epochs=feature_epochs,
                recovery_epochs=recovery_epochs,
                max_images=max_images,
                eval_max_images=eval_max_images,
                external_eval_max_images_per_attack=external_eval_max_images_per_attack,
                num_workers=num_workers,
                smoke=smoke,
                extra_flags=["--use-lagrangian-controller", "--recovery-replay-external", "--rollback-unimproved-phase", *smoke_flags],
                depends_on=["target_absent_trigger_negative_mix"],
            ),
            CandidateRecipe(
                name="orthogonal_trigger_subspace_neutralization",
                strategy="OTSN",
                purpose="Optional trigger-subspace neutralization for residual diffuse causes.",
                params={"rank": 4, "projection_strength": 0.5, "preserve_target_present": True},
                expected_effect={"visible_patch_oga": "decrease", "semantic_drift": "low"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/frontier_detox_algorithm_plan.py"],
            ),
            _oc3_plan_recipe(
                name="oc3_counterfactual_consensus_plan",
                out_root=f"{out_root}/oc3_plan",
                external_report=external_report,
                target_classes=target_classes,
                gates=gates,
                purpose_suffix="Diffuse multi-attack residual: full witness set covers every category.",
            ),
        ]
    elif family == "stop_and_rebuild_splits":
        recipes.append(
            CandidateRecipe(
                name="stop_due_to_leakage",
                strategy="data_governance_stop",
                purpose="Do not train; rebuild splits because held-out leakage invalidates automation.",
                params={},
                expected_effect={"evidence": "valid after rebuild"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/check_heldout_leakage.py"],
            )
        )
    else:
        recipes.append(
            CandidateRecipe(
                name="collect_missing_evidence",
                strategy="evidence_first",
                purpose="Metrics are incomplete; run evidence suite before training.",
                params={"required": ["external_asr", "clean_map", "cfrc", "strict_ceiling", "heldout_leakage"]},
                expected_effect={"diagnosis": "becomes actionable"},
                hard_gates=gates,
                command=[PYTHON_EXE, "scripts/paper_evidence_audit.py"],
            )
        )

    # De-duplicate and limit deterministically.
    unique: list[CandidateRecipe] = []
    seen: set[str] = set()
    for r in recipes:
        fp = recipe_fingerprint(r)
        if fp not in seen:
            unique.append(r)
            seen.add(fp)
    return unique[: max(1, int(max_candidates))]

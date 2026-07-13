#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tqdm import tqdm

    tqdm.monitor_interval = 0
except Exception:
    pass

from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, split_known_keys, write_resolved_config
from model_security_gate.utils.assets import load_asset_config, validate_asset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hybrid-PURIFY-OD: external hard-suite + feature-level YOLO backdoor detox")
    p.add_argument("--config", default=None, help="YAML config. Values under hybrid_purify_detox: are accepted. CLI overrides YAML.")
    p.add_argument("--assets-config", default=None, help="YAML asset config. Values under assets: provide model/data/eval paths.")
    p.add_argument("--model", default=None)
    p.add_argument("--teacher-model", default=None, help="Trusted clean teacher checkpoint. Strongly recommended.")
    p.add_argument("--images", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="*", default=None)
    p.add_argument("--external-eval-roots", nargs="*", default=None)
    p.add_argument("--external-replay-roots", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--phase-epochs", type=int, default=None)
    p.add_argument("--feature-epochs", type=int, default=None)
    p.add_argument("--recovery-epochs", type=int, default=None)
    p.add_argument("--max-allowed-external-asr", type=float, default=None)
    p.add_argument("--max-allowed-internal-asr", type=float, default=None)
    p.add_argument("--max-map-drop", type=float, default=None)
    p.add_argument("--selection-max-map-drop", type=float, default=None)
    p.add_argument("--min-map50-95", type=float, default=None)
    p.add_argument("--external-eval-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-max-images-per-attack", type=int, default=None)
    p.add_argument("--external-replay-floor-per-attack", type=int, default=None,
                   help="Patch B: guarantee at least N real replay samples per attack even when failure_only is True.")
    p.add_argument("--external-replay-floor-repeat", type=int, default=None,
                   help="Repeat floor replay samples when topping up external replay.")
    p.add_argument("--head-only-blacklist", default=None,
                   help="Patch G: path to head_only_blacklist.json. Images listed are excluded from OGA negative training.")
    p.add_argument("--min-passing-eval-n-per-attack", type=int, default=None,
                   help="Fix F1: refuse to declare 'passed' / early-exit when any attack's eval sample size is below N. "
                        "Prevents small-sample early-exits like v3 (60-img eval max_asr=5%% -> passed, but full-300 max was 9.46%%).")
    p.add_argument("--output-distill-scale", type=float, default=None,
                   help="Patch D: multiplier applied on top of phase-wise lambda_output_distill. "
                        "Set to 0.0 to disable output distillation when the teacher's final decisions are not trusted.")
    p.add_argument("--feature-distill-scale", type=float, default=None,
                   help="Patch D: multiplier applied on top of phase-wise lambda_feature_distill. Default 1.0.")
    p.add_argument("--external-failure-replay-repeat", type=int, default=None)
    p.add_argument("--external-oda-full-image-extra-repeat", type=int, default=None, help="Extra full-image repeats for ODA failure replay, preserving trigger/context")
    p.add_argument("--external-oda-focus-crops", action="store_true", default=None, help="Add target-centered crops for ODA failure replay samples")
    p.add_argument("--external-oda-focus-crop-repeat", type=int, default=None)
    p.add_argument("--external-oda-focus-crop-context", type=float, default=None)
    p.add_argument("--external-oda-focus-crop-min-size", type=int, default=None)
    p.add_argument(
        "--external-oda-success-mode",
        choices=["localized_any_recalled", "class_presence", "strict_all_recalled"],
        default=None,
        help="ODA ASR definition used for external hard-suite evaluation.",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--eval-max-images", type=int, default=None)
    p.add_argument("--no-feature-purifier", action="store_true", default=None)
    p.add_argument("--allow-self-teacher-feature-purifier", action="store_true", default=None)
    p.add_argument("--no-phase-finetune", action="store_true", default=None)
    p.add_argument("--no-clean-recovery-finetune", action="store_true", default=None)
    p.add_argument(
        "--recovery-replay-external",
        action="store_true",
        default=None,
        help="Replay external hard-suite samples during clean recovery phases to make recovery ASR-aware.",
    )
    p.add_argument("--no-evaluate-each-phase", action="store_true", default=None, help="Disable phase-level external ASR checkpoint selection")
    p.add_argument("--no-rollback-bad-phase", action="store_true", default=None, help="Continue from a phase even if it worsens an external attack")
    p.add_argument("--rollback-unimproved-phase", action="store_true", default=None, help="Rollback phase candidates that do not improve external selection score")
    p.add_argument("--no-external-failure-replay", action="store_true", default=None, help="Use generic external replay instead of current success=true failures only")
    p.add_argument("--no-external-select-phase-checkpoints", action="store_true", default=None, help="Only evaluate the primary feature-purifier checkpoint per phase")
    p.add_argument("--aggressive-mode", action="store_true", default=None, help="Train harder on top external failures, then rely on rollback gates")
    p.add_argument("--aggressive-feature-epochs", type=int, default=None)
    p.add_argument("--aggressive-phase-epochs", type=int, default=None)
    p.add_argument("--aggressive-lr-multiplier", type=float, default=None)
    p.add_argument("--aggressive-adv-steps", type=int, default=None)
    p.add_argument("--aggressive-failure-replay-repeat", type=int, default=None)
    p.add_argument("--aggressive-lambda-oda-recall", type=float, default=None)
    p.add_argument("--aggressive-lambda-oda-matched", type=float, default=None)
    p.add_argument("--aggressive-lambda-pgbd-paired", type=float, default=None)
    p.add_argument("--aggressive-lambda-oga-negative", type=float, default=None)
    p.add_argument("--oda-recall-min-conf", type=float, default=None)
    p.add_argument("--oda-recall-iou-threshold", type=float, default=None)
    p.add_argument("--oda-recall-center-radius", type=float, default=None)
    p.add_argument("--oda-recall-topk", type=int, default=None)
    p.add_argument("--oda-recall-loss-scale", type=float, default=None)
    p.add_argument("--oda-matched-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-teacher-score-weight", type=float, default=None)
    p.add_argument("--oda-matched-teacher-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-min-score", type=float, default=None)
    p.add_argument("--oda-matched-best-score-weight", type=float, default=None)
    p.add_argument("--oda-matched-best-box-weight", type=float, default=None)
    p.add_argument("--oda-matched-localized-margin", type=float, default=None)
    p.add_argument("--oda-matched-localized-margin-weight", type=float, default=None)
    p.add_argument("--pgbd-view-mode", default=None)
    p.add_argument("--pgbd-negative-margin", type=float, default=None)
    p.add_argument("--trusted-teacher-required", action="store_true", default=None)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--max-hook-layers", type=int, default=None)
    p.add_argument("--prototype-max-batches", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-prototype", action="store_true", default=None, help="Disable prototype-bank losses for lightweight CPU/debug runs")
    p.add_argument("--no-attention", action="store_true", default=None, help="Disable attention-localization losses for lightweight CPU/debug runs")
    p.add_argument("--no-pre-prune", action="store_true", default=None, help="Disable RNP-lite pre-prune candidate")
    p.add_argument("--no-stop-on-pass", action="store_true", default=None, help="Do not early-exit even if an intermediate candidate passes; run all cycles/phases.")
    p.add_argument(
        "--allow-long-run",
        action="store_true",
        default=None,
        help="Allow unbounded/full research profiles. Without this, obvious multi-hour profiles are refused before GPU work starts.",
    )
    p.add_argument("--pre-prune-top-k", type=int, default=None)
    p.add_argument("--pre-prune-strength", type=float, default=None)
    p.add_argument("--rnp-unlearn-steps", type=int, default=None)
    p.add_argument("--rnp-max-images", type=int, default=None)
    p.add_argument("--allow-attack-worse", action="store_true", default=None, help="Allow candidates that worsen a single external attack; not recommended")
    p.add_argument("--max-single-attack-asr-worsen", type=float, default=None)
    p.add_argument("--external-mean-asr-weight", type=float, default=None)
    p.add_argument("--min-external-asr-improvement", type=float, default=None)
    p.add_argument("--min-external-mean-improvement", type=float, default=None)
    p.add_argument("--prefer-passing-clean-map", action="store_true", default=None,
                   help="When both candidates pass ASR/mAP gates, prefer lower clean mAP drop over lower ASR.")

    # Lagrangian multi-attack controller (opt-in, main-line algorithmic extension).
    p.add_argument(
        "--use-lagrangian-controller",
        action="store_true",
        default=None,
        help="Enable adaptive per-attack Lagrangian lambda updates across cycles. "
        "When off, phase lambdas remain static (backward compatible).",
    )
    p.add_argument("--lagrangian-lambda-lr", type=float, default=None)
    p.add_argument("--lagrangian-lambda-min", type=float, default=None)
    p.add_argument("--lagrangian-lambda-max", type=float, default=None)
    p.add_argument("--lagrangian-decay", type=float, default=None)
    p.add_argument("--lagrangian-base-scale", type=float, default=None)
    p.add_argument("--lagrangian-max-scale", type=float, default=None)
    p.add_argument("--lagrangian-min-scale", type=float, default=None)
    return p.parse_args()


def _resolved(args: argparse.Namespace) -> dict:
    from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig

    defaults = {
        "model": None,
        "teacher_model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "out": "runs/hybrid_purify_detox",
    }
    defaults.update(HybridPurifyConfig().__dict__)
    raw = load_yaml_config(args.config, section="hybrid_purify_detox")
    assets = {}
    if args.assets_config:
        asset_config = load_asset_config(args.assets_config)
        errors = validate_asset_config(asset_config)
        if errors:
            raise SystemExit("Invalid asset config: " + "; ".join(errors))
        assets = {
            "model": str(asset_config.suspicious_model),
            "teacher_model": str(asset_config.teacher_model) if asset_config.teacher_model else None,
            "images": str(asset_config.train_images),
            "labels": str(asset_config.train_labels),
            "data_yaml": str(asset_config.data_yaml),
            "target_classes": list(asset_config.target_classes),
            "external_replay_roots": [str(path) for path in asset_config.external_replay_roots],
            "external_eval_roots": [str(path) for path in asset_config.external_eval_roots],
            "out": str(asset_config.output_root),
            "device": asset_config.device,
        }
    cli = namespace_overrides(args, exclude={"config", "assets_config", "allow_long_run"})
    bool_map = {
        "no_feature_purifier": ("run_feature_purifier", False),
        "allow_self_teacher_feature_purifier": ("allow_self_teacher_feature_purifier", True),
        "no_phase_finetune": ("run_phase_finetune", False),
        "no_clean_recovery_finetune": ("run_clean_recovery_finetune", False),
        "recovery_replay_external": ("recovery_replay_external", True),
        "no_evaluate_each_phase": ("evaluate_each_phase", False),
        "no_rollback_bad_phase": ("rollback_bad_phase", False),
        "rollback_unimproved_phase": ("rollback_unimproved_phase", True),
        "no_external_failure_replay": ("external_failure_replay", False),
        "external_oda_focus_crops": ("external_oda_focus_crops", True),
        "no_external_select_phase_checkpoints": ("external_select_phase_checkpoints", False),
        "aggressive_mode": ("aggressive_mode", True),
        "no_pre_prune": ("run_pre_prune", False),
        "no_stop_on_pass": ("stop_on_pass", False),
        "allow_attack_worse": ("require_no_attack_worse", False),
        "prefer_passing_clean_map": ("prefer_passing_clean_map", True),
        "no_prototype": ("use_prototype", False),
        "no_attention": ("use_attention", False),
    }
    norm = {}
    for k, v in cli.items():
        if k in bool_map:
            if v:
                nk, nv = bool_map[k]
                norm[nk] = nv
        elif k == "data_yaml":
            norm["data_yaml"] = v
        else:
            norm[k] = v
    return deep_merge(defaults, deep_merge(raw, deep_merge(assets, norm)))


def _positive_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _estimate_candidate_eval_multiplier(r: dict) -> int:
    cycles = max(1, _positive_int(r.get("cycles")))
    phases_per_cycle = 4
    feature_epochs = max(1, _positive_int(r.get("aggressive_feature_epochs") if r.get("aggressive_mode") else r.get("feature_epochs")))
    phase_epochs = max(1, _positive_int(r.get("aggressive_phase_epochs") if r.get("aggressive_mode") else r.get("phase_epochs")))
    recovery_epochs = max(1, _positive_int(r.get("recovery_epochs")))
    feature_candidates = 0
    if bool(r.get("run_feature_purifier", True)) and (r.get("teacher_model") or bool(r.get("allow_self_teacher_feature_purifier"))):
        feature_candidates = 2 + (feature_epochs if bool(r.get("external_select_phase_checkpoints", True)) else 0)
    phase_candidates = 2 if bool(r.get("run_phase_finetune", True)) else 0
    recovery_candidates = 1 if bool(r.get("run_clean_recovery_finetune", True)) else 0
    if not bool(r.get("evaluate_each_phase", True)):
        return cycles
    return cycles * phases_per_cycle * (feature_candidates + phase_candidates + recovery_candidates + phase_epochs + recovery_epochs)


def _runtime_guard_reasons(r: dict) -> list[str]:
    reasons: list[str] = []
    if _positive_int(r.get("max_images")) <= 0:
        reasons.append("max_images is 0/unset, so phase training can consume the full generated dataset")
    if _positive_int(r.get("eval_max_images")) <= 0:
        reasons.append("eval_max_images is 0/unset, so clean/internal evaluation can scan full datasets")
    if r.get("external_eval_roots") and _positive_int(r.get("external_eval_max_images_per_attack")) <= 0:
        reasons.append("external_eval_max_images_per_attack is 0/unset while external_eval_roots are configured")
    if (
        bool(r.get("evaluate_each_phase", True))
        and bool(r.get("external_select_phase_checkpoints", True))
        and bool(r.get("run_feature_purifier", True))
        and (r.get("teacher_model") or bool(r.get("allow_self_teacher_feature_purifier")))
    ):
        reasons.append("feature-purifier checkpoint selection will externally evaluate best/final/epoch checkpoints for every phase")
    if _positive_int(r.get("cycles")) > 2 and any("0/unset" in item for item in reasons):
        reasons.append("cycles > 2 multiplies the unbounded work above")
    if _estimate_candidate_eval_multiplier(r) > 100:
        reasons.append("estimated candidate/evaluation multiplier is high for an interactive GPU run")
    return reasons


def _enforce_runtime_guard(r: dict, *, allow_long_run: bool) -> None:
    if allow_long_run:
        return
    reasons = _runtime_guard_reasons(r)
    if not reasons:
        return
    lines = [
        "Refusing to launch a likely multi-hour Hybrid-PURIFY run without --allow-long-run.",
        "Add explicit caps such as --max-images, --eval-max-images, and --external-eval-max-images-per-attack, "
        "or pass --allow-long-run if this is an intentional exhaustive research run.",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    raise SystemExit("\n".join(lines))


def _manifest_completed(path: Path, started: float) -> bool:
    if not path.exists() or path.stat().st_mtime < started - 1.0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(data.get("status", "")).lower() != "running"


def _mark_manifest_worker_crash(path: Path, started: float, returncode: int) -> bool:
    if not path.exists() or path.stat().st_mtime < started - 1.0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(data.get("status", "")).lower() != "running":
        return False
    data["status"] = "failed_worker_crash"
    data["worker_returncode"] = int(returncode)
    data["error"] = (
        "Hybrid-PURIFY worker exited before completing the manifest. "
        "Check stderr for native CUDA/torch/ultralytics crashes or interrupted runs."
    )
    data["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def _run_windows_worker_if_needed() -> int | None:
    if os.name != "nt" or os.environ.get("MSG_HYBRID_PURIFY_DETOX_WORKER") == "1":
        return None
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return None
    args = parse_args()
    r = _resolved(args)
    _enforce_runtime_guard(r, allow_long_run=bool(args.allow_long_run))
    out_dir = Path(str(r.get("out") or "runs/hybrid_purify_detox"))
    manifest_path = out_dir / "hybrid_purify_manifest.json"
    started = time.time()
    env = os.environ.copy()
    env["MSG_HYBRID_PURIFY_DETOX_WORKER"] = "1"
    env.setdefault("YOLO_OFFLINE", "true")
    env.setdefault("YOLO_AUTOINSTALL", "false")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    if proc.returncode == 0:
        return 0
    if _manifest_completed(manifest_path, started):
        print(
            f"[WARN] worker exited with {proc.returncode} after writing a completed manifest; "
            "treating run as successful on Windows.",
            file=sys.stderr,
        )
        return 0
    if _mark_manifest_worker_crash(manifest_path, started, int(proc.returncode)):
        print(
            f"[ERROR] worker exited with {proc.returncode}; manifest marked failed_worker_crash.",
            file=sys.stderr,
        )
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    from model_security_gate.detox.hybrid_purify_train import HybridPurifyConfig, run_hybrid_purify_detox_yolo
    from model_security_gate.detox.asr_aware_dataset import load_attacks_from_config

    r = _resolved(args)
    _enforce_runtime_guard(r, allow_long_run=bool(args.allow_long_run))
    missing = [k for k in ["model", "images", "labels", "data_yaml", "target_classes"] if not r.get(k)]
    if missing:
        raise SystemExit("Missing required config/CLI values: " + ", ".join(missing))
    cfg_keys = set(HybridPurifyConfig.__dataclass_fields__.keys())
    cfg_data, extra = split_known_keys(r, cfg_keys)
    if "attacks" in r:
        cfg_data["attack_specs"] = load_attacks_from_config(r.get("attacks"))
    # Patch G: load blacklist if provided.
    if cfg_data.get("head_only_blacklist") and not cfg_data.get("blacklist_stems"):
        from model_security_gate.detox.asr_aware_dataset import load_blacklist
        cfg_data["blacklist_stems"] = tuple(load_blacklist(cfg_data["head_only_blacklist"]))
    for list_key in ["external_eval_roots", "external_replay_roots"]:
        if cfg_data.get(list_key) is None:
            cfg_data[list_key] = ()
        elif isinstance(cfg_data.get(list_key), list):
            cfg_data[list_key] = tuple(cfg_data[list_key])
    cfg = HybridPurifyConfig(**cfg_data)
    out_dir = Path(str(r.get("out") or "runs/hybrid_purify_detox"))
    write_resolved_config(out_dir / "resolved_config.json", r)
    manifest = run_hybrid_purify_detox_yolo(
        model_path=r["model"],
        teacher_model=r.get("teacher_model"),
        images_dir=r["images"],
        labels_dir=r["labels"],
        data_yaml=r["data_yaml"],
        target_classes=r["target_classes"],
        output_dir=out_dir,
        cfg=cfg,
    )
    print(f"[DONE] status: {manifest.get('status')}")
    print(f"[DONE] final model: {manifest.get('final_model')}")
    print(f"[DONE] manifest: {out_dir / 'hybrid_purify_manifest.json'}")


if __name__ == "__main__":
    import traceback

    wrapped_exit = _run_windows_worker_if_needed()
    if wrapped_exit is not None:
        raise SystemExit(wrapped_exit)
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        raw_code = exc.code
        if raw_code is None:
            exit_code = 0
        elif isinstance(raw_code, int):
            exit_code = int(raw_code)
        else:
            print(raw_code, file=sys.stderr)
            exit_code = 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(exit_code)

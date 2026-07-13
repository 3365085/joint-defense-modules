#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.detox.asr_aware_dataset import load_attacks_from_config
from model_security_gate.detox.asr_regression import ASRRegressionConfig, run_asr_regression_for_yolo, write_asr_regression_outputs
from model_security_gate.utils.config import deep_merge, load_yaml_config, namespace_overrides, write_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ASR regression for defensive backdoor detox validation")
    p.add_argument("--config", default=None, help="YAML config. Values under `asr_regression:` are also accepted. CLI overrides YAML.")
    p.add_argument("--model", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--target-classes", nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--match-iou", type=float, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> dict:
    defaults = {
        "model": None,
        "images": None,
        "labels": None,
        "data_yaml": None,
        "target_classes": None,
        "out": "runs/asr_regression",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "match_iou": 0.30,
        "max_images": 0,
        "device": None,
        "attacks": None,
    }
    cfg = load_yaml_config(args.config, section="asr_regression")
    cli = namespace_overrides(args, exclude={"config"})
    if "data_yaml" in cli:
        cli["data_yaml"] = cli.pop("data_yaml")
    return deep_merge(defaults, deep_merge(cfg, cli))


def _run_windows_worker_if_needed() -> int | None:
    if os.name != "nt" or os.environ.get("MSG_ASR_REGRESSION_WORKER") == "1":
        return None
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return None
    resolved = resolve_args(parse_args())
    out = Path(str(resolved.get("out") or "runs/asr_regression"))
    summary_path = out / "asr_regression.json"
    rows_path = out / "asr_regression_rows.csv"
    started = time.time()
    env = os.environ.copy()
    env["MSG_ASR_REGRESSION_WORKER"] = "1"
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    if proc.returncode == 0:
        return 0
    outputs_are_fresh = (
        summary_path.exists()
        and rows_path.exists()
        and summary_path.stat().st_mtime >= started - 1.0
        and rows_path.stat().st_mtime >= started - 1.0
    )
    if outputs_are_fresh:
        print(
            f"[WARN] worker exited with {proc.returncode} after writing fresh ASR regression outputs; "
            "treating run as successful on Windows.",
            file=sys.stderr,
        )
        return 0
    return int(proc.returncode)


def main() -> None:
    resolved = resolve_args(parse_args())
    missing = [k for k in ["model", "images", "labels", "data_yaml", "target_classes"] if not resolved.get(k)]
    if missing:
        raise SystemExit(f"Missing required config/CLI values: {', '.join(missing)}")
    out = Path(str(resolved["out"]))
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(out / "resolved_config.json", resolved)
    cfg = ASRRegressionConfig(
        conf=float(resolved["conf"]),
        iou=float(resolved["iou"]),
        imgsz=int(resolved["imgsz"]),
        match_iou=float(resolved["match_iou"]),
        max_images=int(resolved.get("max_images") or 0),
        attacks=load_attacks_from_config(resolved.get("attacks")),
    )
    result = run_asr_regression_for_yolo(
        model_path=resolved["model"],
        images_dir=resolved["images"],
        labels_dir=resolved["labels"],
        data_yaml=resolved["data_yaml"],
        target_classes=resolved["target_classes"],
        cfg=cfg,
        device=resolved.get("device"),
    )
    summary_path, rows_path = write_asr_regression_outputs(result, out)
    print(f"[DONE] max_asr={result.get('summary', {}).get('max_asr')} summary={summary_path} rows={rows_path}")


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
    os._exit(exit_code)

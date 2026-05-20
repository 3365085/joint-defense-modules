from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional


_WORKER_ENV = "MSG_ULTRALYTICS_TRAIN_WORKER"


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    return value


def _weights_are_fresh(output_project: str | Path, name: str, started: float) -> bool:
    weights_dir = Path(output_project) / name / "weights"
    candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
    return any(path.exists() and path.stat().st_mtime >= started - 1.0 for path in candidates)


def _latest_weight_mtime(output_project: str | Path, name: str) -> float | None:
    weights_dir = Path(output_project) / name / "weights"
    mtimes = [path.stat().st_mtime for path in (weights_dir / "best.pt", weights_dir / "last.pt") if path.exists()]
    return max(mtimes) if mtimes else None


def _run_windows_train_worker(
    *,
    base_model: str | Path,
    data_yaml: str | Path,
    output_project: str | Path,
    name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | int | None,
    train_kwargs: Dict[str, Any],
    stable_weight_timeout_seconds: float | None = None,
) -> None:
    project = Path(output_project)
    project.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = {
        "base_model": str(base_model),
        "data_yaml": str(data_yaml),
        "output_project": str(output_project),
        "name": str(name),
        "imgsz": int(imgsz),
        "epochs": int(epochs),
        "batch": int(batch),
        "device": _coerce_jsonable(device),
        "train_kwargs": _coerce_jsonable(train_kwargs),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="ultralytics_train_", dir=project, delete=False) as fh:
        json.dump(payload, fh, indent=2)
        payload_path = Path(fh.name)
    try:
        env = os.environ.copy()
        env[_WORKER_ENV] = "1"
        cmd = [sys.executable, "-m", "model_security_gate.detox.train_ultralytics", str(payload_path)]
        if stable_weight_timeout_seconds is None:
            proc = subprocess.run(cmd, env=env)
            returncode = proc.returncode
        else:
            proc = subprocess.Popen(cmd, env=env)
            last_mtime = None
            stable_since = None
            while proc.poll() is None:
                current_mtime = _latest_weight_mtime(output_project, name)
                if current_mtime is None:
                    stable_since = None
                elif current_mtime != last_mtime:
                    last_mtime = current_mtime
                    stable_since = time.time()
                elif stable_since is not None and time.time() - stable_since >= float(stable_weight_timeout_seconds):
                    print(
                        "[WARN] Ultralytics worker still running after weights stabilized; "
                        "terminating worker and continuing with the fresh weights.",
                        file=sys.stderr,
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=20)
                    break
                time.sleep(5)
            returncode = proc.returncode if proc.returncode is not None else 1
        if returncode == 0:
            return
        if _weights_are_fresh(output_project, name, started):
            print(
                f"[WARN] Ultralytics worker exited with {returncode} after writing fresh weights; "
                "continuing on Windows.",
                file=sys.stderr,
            )
            return
        raise RuntimeError(f"Ultralytics training worker failed with exit code {returncode}")
    finally:
        try:
            payload_path.unlink()
        except OSError:
            pass


def train_counterfactual_finetune(
    base_model: str | Path,
    data_yaml: str | Path,
    output_project: str | Path = "runs/detox_train",
    name: str = "detox_yolo",
    imgsz: int = 640,
    epochs: int = 30,
    batch: int = 16,
    device: str | int | None = None,
    **train_kwargs: Any,
):
    """Fine-tune a YOLO model on the counterfactual detox dataset.

    This is the practical, trigger-agnostic baseline: do not need to know the
    trigger; use counterfactual data to penalize context shortcuts.
    """
    stable_weight_timeout_seconds = train_kwargs.pop("windows_stable_weight_timeout_seconds", None)
    if os.name == "nt" and os.environ.get(_WORKER_ENV) != "1":
        _run_windows_train_worker(
            base_model=base_model,
            data_yaml=data_yaml,
            output_project=output_project,
            name=name,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            device=device,
            train_kwargs=dict(train_kwargs),
            stable_weight_timeout_seconds=stable_weight_timeout_seconds,
        )
        return None

    from ultralytics import YOLO

    model = YOLO(str(base_model))
    kwargs: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(output_project),
        "name": name,
        "exist_ok": True,
        "optimizer": train_kwargs.pop("optimizer", "AdamW"),
        "lr0": float(train_kwargs.pop("lr0", 5e-5)),
        "weight_decay": float(train_kwargs.pop("weight_decay", 5e-4)),
        "hsv_h": float(train_kwargs.pop("hsv_h", 0.03)),
        "hsv_s": float(train_kwargs.pop("hsv_s", 0.5)),
        "hsv_v": float(train_kwargs.pop("hsv_v", 0.4)),
        "mosaic": float(train_kwargs.pop("mosaic", 0.6)),
        "mixup": float(train_kwargs.pop("mixup", 0.1)),
        "copy_paste": float(train_kwargs.pop("copy_paste", 0.1)),
        "erasing": float(train_kwargs.pop("erasing", 0.25)),
        "label_smoothing": float(train_kwargs.pop("label_smoothing", 0.03)),
        "close_mosaic": int(train_kwargs.pop("close_mosaic", 5)),
        "workers": int(train_kwargs.pop("workers", 0)),
    }
    if device is not None:
        kwargs["device"] = device
    kwargs.update(train_kwargs)
    return model.train(**kwargs)


def _main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m model_security_gate.detox.train_ultralytics PAYLOAD.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    train_counterfactual_finetune(
        base_model=payload["base_model"],
        data_yaml=payload["data_yaml"],
        output_project=payload["output_project"],
        name=payload["name"],
        imgsz=payload["imgsz"],
        epochs=payload["epochs"],
        batch=payload["batch"],
        device=payload.get("device"),
        **payload.get("train_kwargs", {}),
    )
    return 0


if __name__ == "__main__":
    import traceback

    exit_code = 0
    try:
        exit_code = _main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

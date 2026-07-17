from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.runtime.authoritative_model import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_ID,
    SOURCE_SHA256,
    authoritative_artifact_paths,
    authoritative_source_path,
    file_identity,
    validate_artifact_binding,
    validate_authoritative_source,
)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _stage_source(source: Path, staged: Path) -> None:
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        if file_identity(staged)["sha256"] == SOURCE_SHA256:
            return
        staged.unlink()
    try:
        os.link(source, staged)
    except OSError:
        shutil.copy2(source, staged)
    staged_identity = file_identity(staged)
    if staged_identity["sha256"] != SOURCE_SHA256:
        staged.unlink(missing_ok=True)
        raise RuntimeError(
            f"staged source hash mismatch: {staged_identity['sha256']} != {SOURCE_SHA256}"
        )


def _relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _runtime_versions() -> dict[str, object]:
    import torch
    import ultralytics

    try:
        import tensorrt as trt

        tensorrt_version = trt.__version__
    except Exception as exc:  # pragma: no cover - build environment failure
        tensorrt_version = f"unavailable:{type(exc).__name__}:{exc}"
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tensorrt": tensorrt_version,
        "ultralytics": ultralytics.__version__,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _onnx_build_metadata(path: Path) -> dict[str, object]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    opsets = {
        str(item.domain or "ai.onnx"): int(item.version)
        for item in model.opset_import
    }
    return {
        "producer_name": str(model.producer_name or ""),
        "producer_version": str(model.producer_version or ""),
        "opsets": opsets,
    }


def _production_config() -> dict[str, object]:
    paths = authoritative_artifact_paths(PROJECT_ROOT)
    source = authoritative_source_path(PROJECT_ROOT)
    return {
        "inference": {
            "model_family": "yolov8",
            "backend": "tensorrt",
            "half": True,
            "image_size": IMAGE_SIZE,
            "class_names": list(CLASS_NAMES),
            "authoritative_model": {
                "source_pt": str(source),
                "source_sha256": SOURCE_SHA256,
                "source_size": file_identity(source)["size"],
                "metadata": str(paths["metadata"]),
            },
            "artifacts": {
                "pytorch": [str(source)],
                "onnx": [str(paths["onnx"])],
                "engine": [str(paths["engine"])],
            },
        },
        "runtime": {
            "production_unique_model": True,
            "custom_model": {"enabled": False},
        },
    }


def _write_metadata(paths: dict[str, Path]) -> dict[str, object]:
    source = validate_authoritative_source(PROJECT_ROOT)
    metadata: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "staged_source": file_identity(paths["staged_source"]),
        "classes": {
            "names": list(CLASS_NAMES),
            "order": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        },
        "input": {
            "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "dtype": "float16",
        },
        "build": {
            "versions": _runtime_versions(),
            "onnx": {
                "format": "onnx",
                "imgsz": IMAGE_SIZE,
                "batch": 1,
                "dynamic": False,
                "simplify": True,
                "requested_opset": 17,
                "final_artifact": _onnx_build_metadata(paths["onnx"]),
            },
            "engine": {
                "format": "engine",
                "imgsz": IMAGE_SIZE,
                "batch": 1,
                "device": 0,
                "half": True,
                "dynamic": False,
                "simplify": True,
                "workspace_gb": 4.0,
            },
        },
        "artifacts": {
            "onnx": file_identity(paths["onnx"]),
            "engine": file_identity(paths["engine"]),
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build(*, force: bool = False) -> dict[str, object]:
    source = authoritative_source_path(PROJECT_ROOT)
    validate_authoritative_source(PROJECT_ROOT)
    paths = authoritative_artifact_paths(PROJECT_ROOT)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _stage_source(source, paths["staged_source"])

    if not force and paths["onnx"].exists() and paths["engine"].exists() and paths["metadata"].exists():
        return validate_artifact_binding(_production_config(), PROJECT_ROOT) or {}

    from ultralytics import YOLO

    relative_pt = _relative_to_project(paths["staged_source"])
    with _working_directory(PROJECT_ROOT):
        model = YOLO(relative_pt, task="detect")
        onnx_output = Path(
            model.export(
                format="onnx",
                imgsz=IMAGE_SIZE,
                batch=1,
                dynamic=False,
                simplify=True,
                opset=17,
                verbose=False,
            )
        ).resolve()
        if onnx_output != paths["onnx"].resolve():
            shutil.move(str(onnx_output), str(paths["onnx"]))

        engine_output = Path(
            model.export(
                format="engine",
                imgsz=IMAGE_SIZE,
                batch=1,
                device=0,
                half=True,
                dynamic=False,
                simplify=True,
                workspace=4.0,
                verbose=False,
            )
        ).resolve()
        if engine_output != paths["engine"].resolve():
            shutil.move(str(engine_output), str(paths["engine"]))

    _write_metadata(paths)
    return validate_artifact_binding(_production_config(), PROJECT_ROOT) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify ONNX/TensorRT artifacts derived from the authoritative YOLO PT."
    )
    parser.add_argument("--force", action="store_true", help="Re-export even when valid artifacts already exist.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify source, metadata and derived artifact hashes.")
    args = parser.parse_args()
    if args.verify_only:
        result = validate_artifact_binding(_production_config(), PROJECT_ROOT)
    else:
        result = build(force=bool(args.force))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

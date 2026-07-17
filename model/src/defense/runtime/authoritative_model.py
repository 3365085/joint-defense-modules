from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "mask_bd_v4_clean_baseline"
SOURCE_FILENAME = f"{MODEL_ID}.pt"
SOURCE_SHA256 = "4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8"
SOURCE_SIZE = 5_347_205
CLASS_NAMES = ("helmet", "head")
IMAGE_SIZE = 640
ARTIFACT_SCHEMA_VERSION = 1


class AuthoritativeModelValidationError(RuntimeError):
    """Raised when production model identity or a derived artifact is invalid."""


def authoritative_source_path(project_root: Path) -> Path:
    return (
        Path(project_root).resolve()
        .parent.joinpath("素材", "model", "yolov8", SOURCE_FILENAME)
        .resolve(strict=False)
    )


def authoritative_artifact_root(project_root: Path) -> Path:
    return (
        Path(project_root)
        .resolve()
        .joinpath("runtime", "artifacts", "yolo", SOURCE_SHA256.lower())
    )


def authoritative_artifact_paths(project_root: Path) -> dict[str, Path]:
    root = authoritative_artifact_root(project_root)
    return {
        "root": root,
        "staged_source": root / SOURCE_FILENAME,
        "onnx": root / f"{MODEL_ID}.onnx",
        "engine": root / f"{MODEL_ID}.engine",
        "metadata": root / "metadata.json",
    }


def resolve_project_path(raw_path: str | Path, project_root: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve(strict=False)


def _sha256_file(path_text: str) -> str:
    """Hash the current bytes without trusting mutable stat metadata."""
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise AuthoritativeModelValidationError(
            f"artifact_missing:{resolved}: {exc}"
        ) from exc
    if not resolved.is_file():
        raise AuthoritativeModelValidationError(f"artifact_not_file:{resolved}")
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": _sha256_file(str(resolved)),
    }


def validate_authoritative_source(project_root: Path) -> dict[str, Any]:
    expected = authoritative_source_path(project_root)
    identity = file_identity(expected)
    if int(identity["size"]) != SOURCE_SIZE:
        raise AuthoritativeModelValidationError(
            f"source_size_mismatch:expected={SOURCE_SIZE}:actual={identity['size']}:path={expected}"
        )
    if str(identity["sha256"]).upper() != SOURCE_SHA256:
        raise AuthoritativeModelValidationError(
            "source_sha256_mismatch:"
            f"expected={SOURCE_SHA256}:actual={identity['sha256']}:path={expected}"
        )
    return identity


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _require_exact_path(
    *,
    label: str,
    configured: Any,
    expected: Path,
    project_root: Path,
) -> None:
    values = _as_path_list(configured)
    if len(values) != 1:
        raise AuthoritativeModelValidationError(
            f"{label}_must_have_exactly_one_path:actual={values!r}"
        )
    resolved = resolve_project_path(values[0], project_root)
    if resolved != expected.resolve(strict=False):
        raise AuthoritativeModelValidationError(
            f"{label}_path_mismatch:expected={expected}:actual={resolved}"
        )


def validate_production_model_config(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any] | None:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    if not bool(runtime.get("production_unique_model", False)):
        return None

    inference = (
        config.get("inference", {})
        if isinstance(config.get("inference"), dict)
        else {}
    )
    authoritative = (
        inference.get("authoritative_model", {})
        if isinstance(inference.get("authoritative_model"), dict)
        else {}
    )
    artifacts = (
        inference.get("artifacts", {})
        if isinstance(inference.get("artifacts"), dict)
        else {}
    )
    expected_source = authoritative_source_path(project_root)
    expected_artifacts = authoritative_artifact_paths(project_root)

    if str(inference.get("backend", "")).strip().lower() != "tensorrt":
        raise AuthoritativeModelValidationError(
            f"production_backend_must_be_tensorrt:actual={inference.get('backend')!r}"
        )
    if str(inference.get("model_family", "")).strip().lower() not in {
        "yolov8",
        "ultralytics",
    }:
        raise AuthoritativeModelValidationError(
            f"production_model_family_mismatch:actual={inference.get('model_family')!r}"
        )
    if not bool(inference.get("half", False)):
        raise AuthoritativeModelValidationError("production_tensorrt_must_use_fp16")
    if int(inference.get("image_size", 0) or 0) != IMAGE_SIZE:
        raise AuthoritativeModelValidationError(
            f"production_image_size_mismatch:expected={IMAGE_SIZE}:actual={inference.get('image_size')!r}"
        )
    class_names = tuple(str(item) for item in inference.get("class_names", []))
    if class_names != CLASS_NAMES:
        raise AuthoritativeModelValidationError(
            f"production_class_order_mismatch:expected={list(CLASS_NAMES)!r}:actual={list(class_names)!r}"
        )

    configured_source = resolve_project_path(
        str(authoritative.get("source_pt", "")),
        project_root,
    )
    if configured_source != expected_source:
        raise AuthoritativeModelValidationError(
            f"authoritative_source_path_mismatch:expected={expected_source}:actual={configured_source}"
        )
    if str(authoritative.get("source_sha256", "")).strip().upper() != SOURCE_SHA256:
        raise AuthoritativeModelValidationError(
            "authoritative_source_sha256_config_mismatch:"
            f"expected={SOURCE_SHA256}:actual={authoritative.get('source_sha256')!r}"
        )
    if int(authoritative.get("source_size", 0) or 0) != SOURCE_SIZE:
        raise AuthoritativeModelValidationError(
            f"authoritative_source_size_config_mismatch:expected={SOURCE_SIZE}:"
            f"actual={authoritative.get('source_size')!r}"
        )

    _require_exact_path(
        label="pytorch",
        configured=artifacts.get("pytorch"),
        expected=expected_source,
        project_root=project_root,
    )
    _require_exact_path(
        label="onnx",
        configured=artifacts.get("onnx"),
        expected=expected_artifacts["onnx"],
        project_root=project_root,
    )
    _require_exact_path(
        label="engine",
        configured=artifacts.get("engine"),
        expected=expected_artifacts["engine"],
        project_root=project_root,
    )
    _require_exact_path(
        label="metadata",
        configured=authoritative.get("metadata"),
        expected=expected_artifacts["metadata"],
        project_root=project_root,
    )

    custom_model = (
        runtime.get("custom_model", {})
        if isinstance(runtime.get("custom_model"), dict)
        else {}
    )
    if bool(custom_model.get("enabled", False)):
        raise AuthoritativeModelValidationError(
            "production_custom_model_forbidden:"
            f"path={custom_model.get('path')!r}:backend={custom_model.get('backend')!r}"
        )

    return {
        "model_id": MODEL_ID,
        "source": validate_authoritative_source(project_root),
        "engine_path": str(expected_artifacts["engine"]),
        "onnx_path": str(expected_artifacts["onnx"]),
        "metadata_path": str(expected_artifacts["metadata"]),
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "backend": "tensorrt",
        "half": True,
    }


def load_artifact_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path).resolve(strict=False)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuthoritativeModelValidationError(
            f"artifact_metadata_missing:{metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AuthoritativeModelValidationError(
            f"artifact_metadata_invalid_json:{metadata_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AuthoritativeModelValidationError(
            f"artifact_metadata_not_object:{metadata_path}"
        )
    return payload


def validate_artifact_binding(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any] | None:
    model = validate_production_model_config(config, project_root)
    if model is None:
        return None

    metadata_path = Path(model["metadata_path"])
    metadata = load_artifact_metadata(metadata_path)
    if int(metadata.get("schema_version", 0) or 0) != ARTIFACT_SCHEMA_VERSION:
        raise AuthoritativeModelValidationError(
            f"artifact_schema_version_mismatch:expected={ARTIFACT_SCHEMA_VERSION}:"
            f"actual={metadata.get('schema_version')!r}"
        )
    if str(metadata.get("model_id", "")).strip() != MODEL_ID:
        raise AuthoritativeModelValidationError(
            f"artifact_model_id_mismatch:expected={MODEL_ID}:actual={metadata.get('model_id')!r}"
        )

    source_meta = metadata.get("source", {}) if isinstance(metadata.get("source"), dict) else {}
    if str(source_meta.get("sha256", "")).strip().upper() != SOURCE_SHA256:
        raise AuthoritativeModelValidationError(
            "artifact_source_sha256_mismatch:"
            f"expected={SOURCE_SHA256}:actual={source_meta.get('sha256')!r}"
        )
    if int(source_meta.get("size", 0) or 0) != SOURCE_SIZE:
        raise AuthoritativeModelValidationError(
            f"artifact_source_size_mismatch:expected={SOURCE_SIZE}:actual={source_meta.get('size')!r}"
        )

    classes = metadata.get("classes", {}) if isinstance(metadata.get("classes"), dict) else {}
    names = tuple(str(item) for item in classes.get("names", []))
    if names != CLASS_NAMES:
        raise AuthoritativeModelValidationError(
            f"artifact_class_order_mismatch:expected={list(CLASS_NAMES)!r}:actual={list(names)!r}"
        )

    input_meta = metadata.get("input", {}) if isinstance(metadata.get("input"), dict) else {}
    shape = tuple(int(item) for item in input_meta.get("shape", []))
    if shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise AuthoritativeModelValidationError(
            f"artifact_input_shape_mismatch:expected={(1, 3, IMAGE_SIZE, IMAGE_SIZE)!r}:actual={shape!r}"
        )
    if str(input_meta.get("dtype", "")).strip().lower() not in {"float16", "fp16"}:
        raise AuthoritativeModelValidationError(
            f"artifact_input_dtype_mismatch:expected=float16:actual={input_meta.get('dtype')!r}"
        )

    artifact_meta = (
        metadata.get("artifacts", {})
        if isinstance(metadata.get("artifacts"), dict)
        else {}
    )
    verified: dict[str, Any] = {}
    for kind in ("onnx", "engine"):
        entry = artifact_meta.get(kind, {}) if isinstance(artifact_meta.get(kind), dict) else {}
        expected_path = Path(model[f"{kind}_path"]).resolve(strict=False)
        actual_path = resolve_project_path(str(entry.get("path", "")), project_root)
        if actual_path != expected_path:
            raise AuthoritativeModelValidationError(
                f"{kind}_metadata_path_mismatch:expected={expected_path}:actual={actual_path}"
            )
        identity = file_identity(actual_path)
        if str(entry.get("sha256", "")).strip().upper() != identity["sha256"]:
            raise AuthoritativeModelValidationError(
                f"{kind}_sha256_mismatch:metadata={entry.get('sha256')!r}:actual={identity['sha256']}"
            )
        if int(entry.get("size", 0) or 0) != int(identity["size"]):
            raise AuthoritativeModelValidationError(
                f"{kind}_size_mismatch:metadata={entry.get('size')!r}:actual={identity['size']}"
            )
        verified[kind] = identity

    return {
        **model,
        "metadata": metadata,
        "metadata_valid": True,
        "artifacts": verified,
    }

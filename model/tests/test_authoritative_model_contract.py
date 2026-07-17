from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from defense.runtime import authoritative_model as auth


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def _config(project_root: Path, source: Path, paths: dict[str, Path]) -> dict:
    return {
        "inference": {
            "model_family": "yolov8",
            "backend": "tensorrt",
            "half": True,
            "image_size": auth.IMAGE_SIZE,
            "class_names": list(auth.CLASS_NAMES),
            "authoritative_model": {
                "source_pt": str(source),
                "source_sha256": auth.SOURCE_SHA256,
                "source_size": auth.SOURCE_SIZE,
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


def test_non_production_config_is_not_forced_into_authoritative_contract(tmp_path: Path) -> None:
    assert auth.validate_production_model_config({}, tmp_path) is None
    assert auth.validate_artifact_binding({}, tmp_path) is None


def test_file_identity_rehashes_same_size_same_mtime_replacement(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.engine"
    artifact.write_bytes(b"AAAA")
    original_stat = artifact.stat()
    first = auth.file_identity(artifact)

    artifact.write_bytes(b"BBBB")
    os.utime(
        artifact,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    second = auth.file_identity(artifact)

    assert first["size"] == second["size"] == 4
    assert first["sha256"] != second["sha256"]
    assert second["sha256"] == hashlib.sha256(b"BBBB").hexdigest().upper()


def test_production_contract_rejects_custom_model_before_artifact_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = auth.authoritative_source_path(tmp_path)
    paths = auth.authoritative_artifact_paths(tmp_path)
    config = _config(tmp_path, source, paths)
    config["runtime"]["custom_model"] = {
        "enabled": True,
        "path": str(tmp_path / "other.engine"),
        "backend": "tensorrt",
    }
    monkeypatch.setattr(
        auth,
        "validate_authoritative_source",
        lambda _root: {"path": str(source), "size": auth.SOURCE_SIZE, "sha256": auth.SOURCE_SHA256},
    )

    with pytest.raises(auth.AuthoritativeModelValidationError, match="production_custom_model_forbidden"):
        auth.validate_production_model_config(config, tmp_path)


def test_production_contract_rejects_backend_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = auth.authoritative_source_path(tmp_path)
    paths = auth.authoritative_artifact_paths(tmp_path)
    config = _config(tmp_path, source, paths)
    config["inference"]["backend"] = "onnx"
    monkeypatch.setattr(
        auth,
        "validate_authoritative_source",
        lambda _root: {"path": str(source), "size": auth.SOURCE_SIZE, "sha256": auth.SOURCE_SHA256},
    )

    with pytest.raises(auth.AuthoritativeModelValidationError, match="production_backend_must_be_tensorrt"):
        auth.validate_production_model_config(config, tmp_path)


def test_artifact_binding_detects_engine_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = auth.authoritative_source_path(tmp_path)
    paths = auth.authoritative_artifact_paths(tmp_path)
    onnx = _write(paths["onnx"], b"onnx")
    engine = _write(paths["engine"], b"engine")
    config = _config(tmp_path, source, paths)
    monkeypatch.setattr(
        auth,
        "validate_authoritative_source",
        lambda _root: {"path": str(source), "size": auth.SOURCE_SIZE, "sha256": auth.SOURCE_SHA256},
    )
    paths["metadata"].write_text(
        json.dumps(
            {
                "schema_version": auth.ARTIFACT_SCHEMA_VERSION,
                "model_id": auth.MODEL_ID,
                "source": {"sha256": auth.SOURCE_SHA256, "size": auth.SOURCE_SIZE},
                "classes": {"names": list(auth.CLASS_NAMES)},
                "input": {"shape": [1, 3, auth.IMAGE_SIZE, auth.IMAGE_SIZE], "dtype": "float16"},
                "artifacts": {
                    "onnx": onnx,
                    "engine": {**engine, "sha256": "0" * 64},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(auth.AuthoritativeModelValidationError, match="engine_sha256_mismatch"):
        auth.validate_artifact_binding(config, tmp_path)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from defense.model_security import accelerated_artifacts, device_policy

def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _CudaProbe:
    def __init__(self, *, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, _index: int) -> str:
        return "Unit Test CUDA GPU"


class _TorchProbe:
    def __init__(self, *, available: bool) -> None:
        self.cuda = _CudaProbe(available=available)


def _write_metadata(
    artifact: Path,
    source_pt: Path,
    backend: str,
) -> tuple[Path, Any, dict[str, Any]]:
    record = accelerated_artifacts.create_accelerated_artifact_metadata(
        artifact_path=artifact,
        source_pt_path=source_pt,
        artifact_format="engine" if backend == "tensorrt" else "onnx",
        export_parameters={
            "imgsz": 640,
            "runtime_devices": ["cuda"] if backend == "tensorrt" else ["cpu", "cuda"],
        },
    )
    metadata_path = artifact.with_suffix(artifact.suffix + ".metadata.json")
    accelerated_artifacts.write_accelerated_artifact_metadata(record, metadata_path)
    evidence = accelerated_artifacts.build_accelerated_artifact_registry_evidence(
        record,
        metadata_path=metadata_path,
    )
    trusted_record = {
        "status": "trusted",
        "approved_for_runtime": True,
        "approval_source": "trusted_source_pt_export",
        "runtime_model_hash": record.artifact_sha256,
        "source_model_hash": record.source_pt_sha256,
        "security_metrics": evidence,
    }
    return metadata_path, record, trusted_record


def _validate(
    artifact: Path,
    metadata: str | Path | Any,
    source_sha256: str,
    trusted_record: dict[str, Any],
) -> Any:
    return accelerated_artifacts.validate_accelerated_artifact(
        artifact_path=artifact,
        metadata=metadata,
        trusted_source_sha256=source_sha256,
        trusted_record=trusted_record,
    )


def test_cpu_fallback_is_explicit_and_recommends_cuda() -> None:
    policy = device_policy.resolve_model_security_device(
        requested_device="auto",
        torch_module=_TorchProbe(available=False),
    )

    assert _field(policy, "requested_device") == "auto"
    assert _field(policy, "effective_device") == "cpu"
    assert _field(policy, "cuda_available") is False
    assert _field(policy, "fallback_reason") == "cuda_unavailable_auto_cpu_fallback"
    assert _field(policy, "cuda_recommended") is True
    message = str(_field(policy, "performance_note"))
    assert "CPU" in message
    assert "CUDA" in message


def test_cuda_auto_policy_prefers_cuda_without_silent_fallback() -> None:
    policy = device_policy.resolve_model_security_device(
        requested_device="auto",
        torch_module=_TorchProbe(available=True),
    )

    assert _field(policy, "requested_device") == "auto"
    assert str(_field(policy, "effective_device")).startswith("cuda")
    assert _field(policy, "cuda_available") is True
    assert _field(policy, "fallback_reason") in {None, "", "none"}
    assert _field(policy, "cuda_recommended") is True


def test_export_metadata_binds_artifact_to_exact_source_pt(tmp_path: Path) -> None:
    source_pt = tmp_path / "trusted.pt"
    engine = tmp_path / "trusted.engine"
    source_pt.write_bytes(b"trusted-source-pt" * 64)
    engine.write_bytes(b"derived-engine" * 64)

    metadata_path, record, trusted_record = _write_metadata(engine, source_pt, "tensorrt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = _validate(
        engine,
        metadata_path,
        record.source_pt_sha256,
        trusted_record,
    )

    assert metadata_path.is_file()
    assert metadata["artifact_sha256"].removeprefix("sha256:") == _sha256(engine)
    assert metadata["source_pt_sha256"].removeprefix("sha256:") == _sha256(source_pt)
    assert metadata["backend"] == "tensorrt"
    assert bool(_field(validation, "valid")) is True
    assert str(_field(validation, "source_pt_sha256")).removeprefix("sha256:") == _sha256(source_pt)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("missing_metadata", "metadata_missing"),
        ("artifact_tampered", "artifact_sha256_mismatch"),
        ("source_tampered", "source_pt_sha256_mismatch"),
    ],
)
def test_unbound_or_tampered_accelerated_artifact_is_rejected(
    tmp_path: Path,
    case: str,
    expected_reason: str,
) -> None:
    source_pt = tmp_path / "trusted.pt"
    engine = tmp_path / "trusted.engine"
    source_pt.write_bytes(b"trusted-source-pt" * 64)
    engine.write_bytes(b"derived-engine" * 64)

    if case != "missing_metadata":
        metadata_path, record, trusted_record = _write_metadata(engine, source_pt, "tensorrt")
    else:
        metadata_path = tmp_path / "missing.metadata.json"
        source_hash = "sha256:" + _sha256(source_pt)
        record = None
        trusted_record = {
            "status": "trusted",
            "approved_for_runtime": True,
            "approval_source": "trusted_source_pt_export",
            "runtime_model_hash": "sha256:" + _sha256(engine),
            "source_model_hash": source_hash,
            "security_metrics": {},
        }
    if case == "artifact_tampered":
        engine.write_bytes(engine.read_bytes() + b"tampered")
    elif case == "source_tampered":
        source_pt.write_bytes(source_pt.read_bytes() + b"tampered")

    trusted_source_hash = (
        record.source_pt_sha256 if record is not None else "sha256:" + _sha256(source_pt)
    )
    validation = _validate(
        engine,
        metadata_path,
        trusted_source_hash,
        trusted_record,
    )

    assert bool(_field(validation, "valid")) is False
    reasons = list(_field(validation, "reasons"))
    if case == "missing_metadata":
        assert any("metadata_unreadable" in reason for reason in reasons)
    else:
        assert expected_reason in reasons


def test_runtime_selection_prefers_engine_on_cuda_and_onnx_on_cpu(tmp_path: Path) -> None:
    source_pt = tmp_path / "trusted.pt"
    onnx = tmp_path / "trusted.onnx"
    engine = tmp_path / "trusted.engine"
    source_pt.write_bytes(b"trusted-source-pt" * 64)
    onnx.write_bytes(b"derived-onnx" * 64)
    engine.write_bytes(b"derived-engine" * 64)
    onnx_metadata, onnx_record, onnx_trust = _write_metadata(onnx, source_pt, "onnx")
    engine_metadata, engine_record, engine_trust = _write_metadata(engine, source_pt, "tensorrt")
    candidates = [
        {"path": str(onnx), "backend": "onnx", "trusted": True, "metadata_path": str(onnx_metadata)},
        {
            "path": str(engine),
            "backend": "tensorrt",
            "trusted": True,
            "metadata_path": str(engine_metadata),
        },
    ]

    cpu_choice = accelerated_artifacts.select_runtime_artifact(
        source_pt_path=source_pt,
        trusted_source_sha256=onnx_record.source_pt_sha256,
        accelerated_candidates=candidates,
        trusted_records=[onnx_trust, engine_trust],
        device_policy=device_policy.resolve_model_security_device(
            requested_device="cpu",
            torch_module=_TorchProbe(available=False),
        ),
    )
    cuda_choice = accelerated_artifacts.select_runtime_artifact(
        source_pt_path=source_pt,
        trusted_source_sha256=engine_record.source_pt_sha256,
        accelerated_candidates=candidates,
        trusted_records=[onnx_trust, engine_trust],
        device_policy=device_policy.resolve_model_security_device(
            requested_device="auto",
            torch_module=_TorchProbe(available=True),
        ),
    )

    assert Path(_field(cpu_choice, "selected")["path"]) == onnx
    assert _field(cpu_choice, "selected")["backend"] == "onnx"
    assert Path(_field(cuda_choice, "selected")["path"]) == engine
    assert _field(cuda_choice, "selected")["backend"] == "tensorrt"

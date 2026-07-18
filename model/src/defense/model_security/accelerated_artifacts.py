from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .device_policy import ModelSecurityDevicePolicy
from .fingerprint import sha256_file


ACCELERATED_ARTIFACT_SCHEMA_VERSION = 1
ACCELERATED_ARTIFACT_KIND = "model_security_accelerated_artifact"
TRUSTED_EXPORT_APPROVAL_SOURCES = {
    "trusted_source_pt_export",
    "auto_export_from_trusted_pt",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not _SHA256_RE.fullmatch(text):
        raise ValueError("invalid_sha256")
    return f"sha256:{text}"


def file_sha256(path: str | Path) -> str:
    return normalize_sha256(sha256_file(path))


def normalize_artifact_format(value: Any) -> tuple[str, str, str]:
    text = str(value or "").strip().lower()
    if text in {"engine", "trt", "tensorrt"}:
        return "engine", "tensorrt", ".engine"
    if text == "onnx":
        return "onnx", "onnx", ".onnx"
    raise ValueError(f"unsupported_accelerated_artifact_format:{text or 'missing'}")


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _derivation_sha256(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class AcceleratedArtifactMetadata:
    schema_version: int
    kind: str
    artifact_path: str
    artifact_sha256: str
    artifact_format: str
    backend: str
    source_pt_path: str
    source_pt_sha256: str
    export_parameters: dict[str, Any]
    metadata: dict[str, Any]
    exporter: str
    created_at: str
    derivation_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("derivation_sha256", None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcceleratedArtifactMetadata":
        export_parameters = data.get("export_parameters")
        metadata = data.get("metadata")
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            kind=str(data.get("kind") or ""),
            artifact_path=str(data.get("artifact_path") or ""),
            artifact_sha256=str(data.get("artifact_sha256") or ""),
            artifact_format=str(data.get("artifact_format") or ""),
            backend=str(data.get("backend") or ""),
            source_pt_path=str(data.get("source_pt_path") or ""),
            source_pt_sha256=str(data.get("source_pt_sha256") or ""),
            export_parameters=dict(export_parameters) if isinstance(export_parameters, Mapping) else {},
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            exporter=str(data.get("exporter") or ""),
            created_at=str(data.get("created_at") or ""),
            derivation_sha256=str(data.get("derivation_sha256") or ""),
        )


@dataclass(frozen=True)
class AcceleratedArtifactValidation:
    valid: bool
    reasons: tuple[str, ...]
    artifact_path: str
    metadata_path: str | None
    artifact_sha256: str | None
    source_pt_sha256: str | None
    artifact_format: str | None
    backend: str | None
    derivation_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class RuntimeArtifactSelection:
    selected: dict[str, Any] | None
    considered: tuple[dict[str, Any], ...]
    unavailable_reasons: tuple[dict[str, Any], ...]
    device_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "considered": [dict(item) for item in self.considered],
            "unavailable_reasons": [dict(item) for item in self.unavailable_reasons],
            "device_policy": dict(self.device_policy),
        }


def create_accelerated_artifact_metadata(
    *,
    artifact_path: str | Path,
    source_pt_path: str | Path,
    artifact_format: str,
    export_parameters: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    exporter: str = "defense.model_security",
    created_at: str | None = None,
) -> AcceleratedArtifactMetadata:
    artifact = Path(artifact_path).expanduser().resolve()
    source_pt = Path(source_pt_path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"accelerated_artifact_missing:{artifact}")
    if not source_pt.is_file():
        raise FileNotFoundError(f"source_pt_missing:{source_pt}")
    if source_pt.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("source_model_must_be_explicit_pt")
    normalized_format, backend, suffix = normalize_artifact_format(artifact_format)
    if artifact.suffix.lower() != suffix:
        raise ValueError(
            f"accelerated_artifact_suffix_mismatch:format={normalized_format}:suffix={artifact.suffix.lower()}"
        )
    unsigned = {
        "schema_version": ACCELERATED_ARTIFACT_SCHEMA_VERSION,
        "kind": ACCELERATED_ARTIFACT_KIND,
        "artifact_path": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "artifact_format": normalized_format,
        "backend": backend,
        "source_pt_path": str(source_pt),
        "source_pt_sha256": file_sha256(source_pt),
        "export_parameters": dict(export_parameters),
        "metadata": dict(metadata or {}),
        "exporter": str(exporter),
        "created_at": str(created_at or utc_now_iso()),
    }
    return AcceleratedArtifactMetadata(
        **unsigned,
        derivation_sha256=_derivation_sha256(unsigned),
    )


def build_accelerated_artifact_registry_evidence(
    record: AcceleratedArtifactMetadata,
    *,
    metadata_path: str | Path,
) -> dict[str, Any]:
    expected = _derivation_sha256(record.unsigned_payload())
    if normalize_sha256(record.derivation_sha256) != expected:
        raise ValueError("accelerated_artifact_metadata_derivation_mismatch")
    return {
        "accelerated_artifact_format": record.artifact_format,
        "accelerated_artifact_backend": record.backend,
        "accelerated_artifact_path": record.artifact_path,
        "accelerated_artifact_sha256": record.artifact_sha256,
        "accelerated_artifact_source_pt_path": record.source_pt_path,
        "accelerated_artifact_source_pt_sha256": record.source_pt_sha256,
        "accelerated_artifact_export_parameters": dict(record.export_parameters),
        "accelerated_artifact_metadata_path": str(Path(metadata_path).expanduser().resolve()),
        "accelerated_artifact_derivation_sha256": record.derivation_sha256,
    }


def write_accelerated_artifact_metadata(
    record: AcceleratedArtifactMetadata,
    metadata_path: str | Path,
) -> Path:
    expected = _derivation_sha256(record.unsigned_payload())
    if normalize_sha256(record.derivation_sha256) != expected:
        raise ValueError("accelerated_artifact_metadata_derivation_mismatch")
    target = Path(metadata_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_accelerated_artifact_metadata(
    value: str | Path | Mapping[str, Any] | AcceleratedArtifactMetadata,
) -> tuple[AcceleratedArtifactMetadata, str | None]:
    if isinstance(value, AcceleratedArtifactMetadata):
        return value, None
    if isinstance(value, Mapping):
        return AcceleratedArtifactMetadata.from_dict(value), None
    path = Path(value).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("accelerated_artifact_metadata_not_object")
    return AcceleratedArtifactMetadata.from_dict(data), str(path)


def _record_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return data
    return {}


def _trusted_source_hashes(value: str | Iterable[str]) -> set[str]:
    values = [value] if isinstance(value, str) else list(value)
    return {normalize_sha256(item) for item in values}


def validate_accelerated_artifact(
    artifact_path: str | Path,
    *,
    metadata: str | Path | Mapping[str, Any] | AcceleratedArtifactMetadata,
    trusted_source_sha256: str | Iterable[str],
    trusted_record: Any,
    require_source_file: bool = True,
) -> AcceleratedArtifactValidation:
    artifact = Path(artifact_path).expanduser().resolve()
    reasons: list[str] = []
    record: AcceleratedArtifactMetadata | None = None
    metadata_path: str | None = None
    trusted_hashes: set[str] = set()
    try:
        trusted_hashes = _trusted_source_hashes(trusted_source_sha256)
    except Exception as exc:
        reasons.append(f"trusted_source_hash_invalid:{exc}")
    try:
        record, metadata_path = load_accelerated_artifact_metadata(metadata)
    except Exception as exc:
        reasons.append(f"metadata_unreadable:{type(exc).__name__}:{exc}")

    artifact_sha: str | None = None
    source_sha: str | None = None
    artifact_format: str | None = None
    backend: str | None = None
    derivation_sha: str | None = None
    if record is not None:
        artifact_format = record.artifact_format
        backend = record.backend
        try:
            derivation_sha = normalize_sha256(record.derivation_sha256)
        except Exception:
            reasons.append("metadata_derivation_sha256_invalid")
        if record.schema_version != ACCELERATED_ARTIFACT_SCHEMA_VERSION:
            reasons.append(f"metadata_schema_unsupported:{record.schema_version}")
        if record.kind != ACCELERATED_ARTIFACT_KIND:
            reasons.append("metadata_kind_invalid")
        if not record.export_parameters:
            reasons.append("export_parameters_missing")
        try:
            normalized_format, expected_backend, expected_suffix = normalize_artifact_format(
                record.artifact_format
            )
            artifact_format = normalized_format
            backend = expected_backend
            if record.backend != expected_backend:
                reasons.append("metadata_backend_mismatch")
            if artifact.suffix.lower() != expected_suffix:
                reasons.append("artifact_suffix_mismatch")
        except Exception as exc:
            reasons.append(str(exc))
        try:
            metadata_artifact = Path(record.artifact_path).expanduser().resolve()
            if metadata_artifact != artifact:
                reasons.append("metadata_artifact_path_mismatch")
        except Exception:
            reasons.append("metadata_artifact_path_invalid")
        if not artifact.is_file():
            reasons.append("artifact_missing")
        else:
            artifact_sha = file_sha256(artifact)
            try:
                if normalize_sha256(record.artifact_sha256) != artifact_sha:
                    reasons.append("artifact_sha256_mismatch")
            except Exception:
                reasons.append("metadata_artifact_sha256_invalid")
        try:
            source_sha = normalize_sha256(record.source_pt_sha256)
            if source_sha not in trusted_hashes:
                reasons.append("source_pt_not_trusted")
        except Exception:
            reasons.append("metadata_source_pt_sha256_invalid")
        source_pt = Path(record.source_pt_path).expanduser().resolve()
        if source_pt.suffix.lower() not in {".pt", ".pth"}:
            reasons.append("source_model_must_be_explicit_pt")
        if require_source_file and not source_pt.is_file():
            reasons.append("source_pt_missing")
        elif source_pt.is_file() and source_sha is not None and file_sha256(source_pt) != source_sha:
            reasons.append("source_pt_sha256_mismatch")
        try:
            expected_derivation = _derivation_sha256(record.unsigned_payload())
            if derivation_sha != expected_derivation:
                reasons.append("metadata_derivation_mismatch")
        except Exception:
            reasons.append("metadata_derivation_unverifiable")

        trust = _record_mapping(trusted_record)
        if not trust:
            reasons.append("trusted_derivation_record_missing")
        else:
            if not bool(trust.get("approved_for_runtime")) or str(trust.get("status")) != "trusted":
                reasons.append("artifact_registry_record_not_trusted")
            if str(trust.get("approval_source") or "") not in TRUSTED_EXPORT_APPROVAL_SOURCES:
                reasons.append("artifact_registry_approval_source_invalid")
            try:
                if normalize_sha256(trust.get("runtime_model_hash")) != artifact_sha:
                    reasons.append("registry_artifact_sha256_mismatch")
            except Exception:
                reasons.append("registry_artifact_sha256_invalid")
            try:
                if normalize_sha256(trust.get("source_model_hash")) != source_sha:
                    reasons.append("registry_source_pt_sha256_mismatch")
            except Exception:
                reasons.append("registry_source_pt_sha256_invalid")
            metrics = trust.get("security_metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            registered_derivation = metrics.get("accelerated_artifact_derivation_sha256")
            try:
                if normalize_sha256(registered_derivation) != derivation_sha:
                    reasons.append("registry_derivation_sha256_mismatch")
            except Exception:
                reasons.append("registry_derivation_sha256_missing")
            registered_metadata_path = metrics.get("accelerated_artifact_metadata_path")
            if metadata_path and registered_metadata_path:
                if Path(str(registered_metadata_path)).expanduser().resolve() != Path(metadata_path):
                    reasons.append("registry_metadata_path_mismatch")

    return AcceleratedArtifactValidation(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        artifact_path=str(artifact),
        metadata_path=metadata_path,
        artifact_sha256=artifact_sha,
        source_pt_sha256=source_sha,
        artifact_format=artifact_format,
        backend=backend,
        derivation_sha256=derivation_sha,
    )


def _policy_payload(policy: ModelSecurityDevicePolicy | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(policy, ModelSecurityDevicePolicy):
        return policy.to_dict()
    return dict(policy)


def _runtime_devices(record: AcceleratedArtifactMetadata) -> set[str]:
    raw = record.export_parameters.get("runtime_devices")
    if isinstance(raw, str):
        devices = {raw.strip().lower()}
    elif isinstance(raw, Iterable):
        devices = {str(item).strip().lower() for item in raw}
    else:
        devices = {"cuda"} if record.backend == "tensorrt" else {"cuda", "cpu"}
    return {"cuda" if value.startswith("cuda") else value for value in devices}


def _find_trust_record(records: Iterable[Any], artifact_sha256: str | None) -> Any:
    if artifact_sha256 is None:
        return None
    for record in records:
        data = _record_mapping(record)
        try:
            if normalize_sha256(data.get("runtime_model_hash")) == artifact_sha256:
                return record
        except Exception:
            continue
    return None


def select_runtime_artifact(
    *,
    source_pt_path: str | Path,
    trusted_source_sha256: str,
    accelerated_candidates: Iterable[Mapping[str, Any]],
    trusted_records: Iterable[Any],
    device_policy: ModelSecurityDevicePolicy | Mapping[str, Any],
    model_family: str = "auto",
) -> RuntimeArtifactSelection:
    policy = _policy_payload(device_policy)
    effective_device = str(policy.get("effective_device") or "cpu")
    cuda_effective = effective_device.startswith("cuda:")
    source_pt = Path(source_pt_path).expanduser().resolve()
    trusted_source = normalize_sha256(trusted_source_sha256)
    trust_records = list(trusted_records)
    considered: list[dict[str, Any]] = []
    available: list[tuple[int, str, dict[str, Any]]] = []

    for raw_candidate in accelerated_candidates:
        artifact_value = raw_candidate.get("artifact_path", raw_candidate.get("path"))
        candidate_path = Path(str(artifact_value or "")).expanduser().resolve()
        metadata_value = raw_candidate.get("metadata_path")
        if metadata_value is None:
            metadata_value = raw_candidate.get(
                "artifact_metadata",
                raw_candidate.get("metadata"),
            )
        reasons: list[str] = []
        metadata_record: AcceleratedArtifactMetadata | None = None
        metadata_path: str | None = None
        if not artifact_value:
            reasons.append("artifact_path_missing")
        if metadata_value is None:
            reasons.append("artifact_metadata_missing")
        else:
            try:
                metadata_record, metadata_path = load_accelerated_artifact_metadata(metadata_value)
            except Exception as exc:
                reasons.append(f"metadata_unreadable:{type(exc).__name__}:{exc}")
        artifact_sha = None
        if candidate_path.is_file():
            artifact_sha = file_sha256(candidate_path)
        trust_record = raw_candidate.get("trusted_record") or _find_trust_record(
            trust_records, artifact_sha
        )
        validation: AcceleratedArtifactValidation | None = None
        if metadata_value is not None:
            validation = validate_accelerated_artifact(
                candidate_path,
                metadata=metadata_value,
                trusted_source_sha256=trusted_source,
                trusted_record=trust_record,
            )
            reasons.extend(validation.reasons)
        execution_device: str | None = None
        backend = metadata_record.backend if metadata_record else None
        rank = 99
        if not reasons and metadata_record is not None:
            supported_devices = _runtime_devices(metadata_record)
            if backend == "tensorrt":
                if not cuda_effective:
                    reasons.append("tensorrt_requires_cuda")
                elif "cuda" not in supported_devices:
                    reasons.append("tensorrt_metadata_disallows_cuda")
                else:
                    execution_device = effective_device
                    rank = 0
            elif backend == "onnx":
                if cuda_effective and "cuda" in supported_devices:
                    execution_device = effective_device
                    rank = 1
                elif "cpu" in supported_devices:
                    execution_device = "cpu"
                    rank = 2
                else:
                    reasons.append("onnx_has_no_compatible_execution_device")
            else:
                reasons.append(f"unsupported_accelerated_backend:{backend}")
        item = {
            "path": str(candidate_path),
            "backend": backend,
            "metadata_path": metadata_path,
            "artifact_sha256": validation.artifact_sha256 if validation else artifact_sha,
            "source_pt_sha256": validation.source_pt_sha256 if validation else None,
            "execution_device": execution_device,
            "available": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
        }
        considered.append(item)
        if not reasons and backend is not None and execution_device is not None:
            runtime_model = {
                "enabled": True,
                "path": str(candidate_path),
                "backend": backend,
                "model_family": model_family,
                "source_pt_path": str(source_pt),
                "source_pt_sha256": trusted_source,
                "artifact_sha256": item["artifact_sha256"],
                "metadata_path": metadata_path,
                "device": execution_device,
                "status": "trusted_accelerated_export",
            }
            available.append((rank, str(candidate_path), runtime_model))

    pt_reasons: list[str] = []
    pt_sha: str | None = None
    if not source_pt.is_file():
        pt_reasons.append("trusted_pt_missing")
    elif source_pt.suffix.lower() not in {".pt", ".pth"}:
        pt_reasons.append("trusted_source_not_pt")
    else:
        pt_sha = file_sha256(source_pt)
        if pt_sha != trusted_source:
            pt_reasons.append("trusted_pt_sha256_mismatch")
    pt_item = {
        "path": str(source_pt),
        "backend": "pytorch",
        "artifact_sha256": pt_sha,
        "source_pt_sha256": trusted_source,
        "execution_device": effective_device,
        "available": not pt_reasons,
        "reasons": pt_reasons,
    }
    considered.append(pt_item)
    if not pt_reasons:
        available.append(
            (
                3,
                str(source_pt),
                {
                    "enabled": True,
                    "path": str(source_pt),
                    "backend": "pytorch",
                    "model_family": model_family,
                    "source_pt_path": str(source_pt),
                    "source_pt_sha256": trusted_source,
                    "artifact_sha256": pt_sha,
                    "device": effective_device,
                    "status": "trusted_pt_runtime",
                },
            )
        )

    available.sort(key=lambda entry: (entry[0], entry[1].lower()))
    unavailable = tuple(
        {"path": item["path"], "backend": item["backend"], "reasons": item["reasons"]}
        for item in considered
        if not item["available"]
    )
    return RuntimeArtifactSelection(
        selected=available[0][2] if available else None,
        considered=tuple(considered),
        unavailable_reasons=unavailable,
        device_policy=policy,
    )

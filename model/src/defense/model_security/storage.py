from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelSecurityStorage:
    root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "ModelSecurityStorage":
        return cls(Path(project_root) / "runtime" / "model_security")

    @property
    def registry_path(self) -> Path:
        return self.root / "trusted_registry.json"

    @property
    def registry_seal_path(self) -> Path:
        return self.root / "trusted_registry.seal.json"

    @property
    def log_path(self) -> Path:
        return self.root / "module_b_events.jsonl"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def purified_dir(self) -> Path:
        return self.root / "purified"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def activation_cache_dir(self) -> Path:
        return self.root / "activation_cache"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def trusted_pt_models_dir(self) -> Path:
        return self.models_dir / "trusted_pt"

    @property
    def purified_pt_models_dir(self) -> Path:
        return self.models_dir / "purified_pt"

    @property
    def onnx_models_dir(self) -> Path:
        return self.models_dir / "onnx"

    @property
    def tensorrt_models_dir(self) -> Path:
        return self.models_dir / "tensorrt"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def accelerated_metadata_dir(self) -> Path:
        return self.metadata_dir / "accelerated"

    def accelerated_models_dir(self, artifact_format: str) -> Path:
        value = str(artifact_format or "").strip().lower()
        if value in {"engine", "trt", "tensorrt"}:
            return self.tensorrt_models_dir
        if value == "onnx":
            return self.onnx_models_dir
        raise ValueError(f"unsupported_accelerated_artifact_format:{value or 'missing'}")

    def accelerated_metadata_path(self, artifact_sha256: str) -> Path:
        value = str(artifact_sha256 or "").strip().lower()
        if value.startswith("sha256:"):
            value = value[7:]
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("invalid_artifact_sha256")
        return self.accelerated_metadata_dir / f"{value}.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.purified_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.activation_cache_dir.mkdir(parents=True, exist_ok=True)
        self.trusted_pt_models_dir.mkdir(parents=True, exist_ok=True)
        self.purified_pt_models_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_models_dir.mkdir(parents=True, exist_ok=True)
        self.tensorrt_models_dir.mkdir(parents=True, exist_ok=True)
        self.accelerated_metadata_dir.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    def activation_cache_dir(self) -> Path:
        return self.root / "activation_cache"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.purified_dir.mkdir(parents=True, exist_ok=True)
        self.activation_cache_dir.mkdir(parents=True, exist_ok=True)

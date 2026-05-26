from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ScanBudget:
    max_layers: int = 4
    max_probes: int = 8
    batch_size: int = 1
    device: str = "auto"
    time_budget_s: float = 30.0
    early_trust_score: float = 0.03
    early_suspicious_score: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSecurityReport:
    fingerprint: dict[str, Any]
    scan_type: str
    status: str
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    suspicious_neurons: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    completed_at: str | None = None
    scanner_version: str = "model_security_runtime_v1"
    budget: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_model_path: str | None = None
    source_model_hash: str | None = None
    runtime_artifact_path: str | None = None
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.completed_at = self.completed_at or now_iso()
        self.report_path = str(p)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return p


@dataclass
class ModelPurificationReport:
    fingerprint: dict[str, Any]
    status: str
    strategy: str
    source_model_path: str | None = None
    source_model_hash: str | None = None
    clean_anchor_path: str | None = None
    clean_anchor_hash: str | None = None
    purified_model_path: str | None = None
    purified_model_hash: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    completed_at: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    scan_report_path: str | None = None
    scan_status: str | None = None
    report_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.completed_at = self.completed_at or now_iso()
        self.report_path = str(p)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return p

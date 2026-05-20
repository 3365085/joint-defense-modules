from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TrustRecord:
    fingerprint: str
    status: str = "unknown"
    approved_for_runtime: bool = False
    risk_score: float | None = None
    last_scan_time: str | None = None
    scanner_version: str | None = None
    report_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelTrustRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {"version": 1, "models": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("models"), dict):
                    self._data = data
            except Exception:
                self._data = {"version": 1, "models": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def get(self, fingerprint: str) -> TrustRecord | None:
        raw = self._data.get("models", {}).get(fingerprint)
        if not isinstance(raw, dict):
            return None
        return TrustRecord(**{k: raw.get(k) for k in TrustRecord.__dataclass_fields__.keys()})

    def upsert(self, record: TrustRecord) -> TrustRecord:
        self._data.setdefault("models", {})[record.fingerprint] = record.to_dict()
        self.save()
        return record

    def mark_trusted(self, fingerprint: str, *, risk_score: float = 0.0, report_path: str | None = None, scanner_version: str | None = None, notes: str = "manual trust") -> TrustRecord:
        rec = TrustRecord(
            fingerprint=fingerprint,
            status="trusted",
            approved_for_runtime=True,
            risk_score=float(risk_score),
            last_scan_time=utc_now_iso(),
            scanner_version=scanner_version,
            report_path=report_path,
            notes=notes,
        )
        return self.upsert(rec)

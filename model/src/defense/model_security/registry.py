from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    runtime_model_hash: str | None = None
    runtime_model_path: str | None = None
    source_model_hash: str | None = None
    source_model_path: str | None = None
    backend: str | None = None
    model_family: str | None = None
    image_size: Any = None
    class_names_hash: str | None = None
    ppe_mapping_hash: str | None = None
    purification_report_path: str | None = None
    approval_source: str = "manual"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelTrustRegistry:
    def __init__(self, path: str | Path, *, on_save: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path = Path(path)
        self._on_save = on_save
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
        self._data.setdefault("version", 1)
        self._data.setdefault("models", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        if self._on_save is not None:
            self._on_save(self._data)

    def data(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, ensure_ascii=False))

    def get(self, fingerprint: str) -> TrustRecord | None:
        raw = self._data.get("models", {}).get(fingerprint)
        if not isinstance(raw, dict):
            return None
        return TrustRecord(**{k: raw.get(k) for k in TrustRecord.__dataclass_fields__.keys()})

    def list_records(self) -> list[TrustRecord]:
        records: list[TrustRecord] = []
        for raw in self._data.get("models", {}).values():
            if isinstance(raw, dict):
                records.append(TrustRecord(**{k: raw.get(k) for k in TrustRecord.__dataclass_fields__.keys()}))
        records.sort(key=lambda rec: rec.last_scan_time or "", reverse=True)
        return records

    def upsert(self, record: TrustRecord) -> TrustRecord:
        self._data.setdefault("models", {})[record.fingerprint] = record.to_dict()
        self.save()
        return record

    def delete(self, fingerprint: str) -> bool:
        models = self._data.setdefault("models", {})
        if fingerprint not in models:
            return False
        del models[fingerprint]
        self.save()
        return True

    def clear(self) -> int:
        count = len(self._data.get("models", {}))
        self._data["models"] = {}
        self.save()
        return count

    def mark_trusted(
        self,
        fingerprint: str,
        *,
        risk_score: float = 0.0,
        report_path: str | None = None,
        scanner_version: str | None = None,
        notes: str = "manual trust",
        runtime_model_hash: str | None = None,
        runtime_model_path: str | None = None,
        source_model_hash: str | None = None,
        source_model_path: str | None = None,
        backend: str | None = None,
        model_family: str | None = None,
        image_size: Any = None,
        class_names_hash: str | None = None,
        ppe_mapping_hash: str | None = None,
        purification_report_path: str | None = None,
        approval_source: str = "manual",
    ) -> TrustRecord:
        rec = TrustRecord(
            fingerprint=fingerprint,
            status="trusted",
            approved_for_runtime=True,
            risk_score=float(risk_score),
            last_scan_time=utc_now_iso(),
            scanner_version=scanner_version,
            report_path=report_path,
            runtime_model_hash=runtime_model_hash,
            runtime_model_path=runtime_model_path,
            source_model_hash=source_model_hash,
            source_model_path=source_model_path,
            backend=backend,
            model_family=model_family,
            image_size=image_size,
            class_names_hash=class_names_hash,
            ppe_mapping_hash=ppe_mapping_hash,
            purification_report_path=purification_report_path,
            approval_source=approval_source,
            notes=notes,
        )
        return self.upsert(rec)

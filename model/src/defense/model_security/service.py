from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config, project_root

from .fingerprint import ModelFingerprint, SCANNER_VERSION, build_model_fingerprint
from .registry import ModelTrustRegistry
from .reports import ModelSecurityReport, ScanBudget
from .scanner import full_scan, quick_scan


class ModelSecurityService:
    """Runtime wrapper for Module B model security without blocking Module A preview/detection."""

    def __init__(self, *, config_path: str | Path | None = None, root: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.root = Path(root) if root else project_root()
        self.runtime_dir = self.root / "runtime" / "model_security"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelTrustRegistry(self.runtime_dir / "trusted_registry.json")
        self._lock = threading.Lock()
        self._scan_thread: threading.Thread | None = None
        self._stop_requested = False
        self._last_report: ModelSecurityReport | None = None
        self._last_fp: ModelFingerprint | None = None
        self._last_error = ""

    def _config(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        return load_runtime_config(config_path=self.config_path, profile=profile or "default", custom_model=custom_model or {})

    def current_fingerprint(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> ModelFingerprint:
        cfg = self._config(profile=profile, custom_model=custom_model)
        fp = build_model_fingerprint(cfg, root=self.root)
        with self._lock:
            self._last_fp = fp
        return fp

    def status(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            fp = self.current_fingerprint(profile=profile, custom_model=custom_model)
            rec = self.registry.get(fp.fingerprint)
            scanning = bool(self._scan_thread and self._scan_thread.is_alive())
            if rec and rec.approved_for_runtime:
                state = "trusted"
                risk = rec.risk_score
                report_path = rec.report_path
                last_scan = rec.last_scan_time
            elif rec:
                state = rec.status or "unknown"
                risk = rec.risk_score
                report_path = rec.report_path
                last_scan = rec.last_scan_time
            else:
                state = "scanning" if scanning else "unknown"
                risk = self._last_report.risk_score if self._last_report else None
                report_path = self._last_report.report_path if self._last_report else None
                last_scan = self._last_report.completed_at if self._last_report else None
            return {
                "enabled": True,
                "status": state,
                "scanning": scanning,
                "fingerprint": fp.fingerprint,
                "model_hash": fp.model_hash,
                "model_path": fp.model_path,
                "backend": fp.backend,
                "model_family": fp.model_family,
                "scanner_version": SCANNER_VERSION,
                "risk_score": risk,
                "last_scan_time": last_scan,
                "report_path": report_path,
                "registry_path": str(self.registry.path),
                "error": self._last_error,
            }
        except Exception as exc:
            self._last_error = str(exc)
            return {"enabled": True, "status": "error", "error": str(exc), "scanner_version": SCANNER_VERSION}

    def _write_report(self, report: ModelSecurityReport) -> ModelSecurityReport:
        path = self.runtime_dir / "reports" / f"{report.fingerprint['fingerprint'].replace(':','_')}_{report.scan_type}.json"
        report.write(path)
        return report

    def scan(self, *, scan_type: str = "quick", profile: str = "default", custom_model: dict[str, Any] | None = None, trust_if_low_risk: bool = False) -> dict[str, Any]:
        fp = self.current_fingerprint(profile=profile, custom_model=custom_model)
        budget = ScanBudget()
        cache_dir = self.runtime_dir / "activation_cache"
        report = full_scan(fp, budget=budget, cache_dir=cache_dir) if scan_type == "full" else quick_scan(fp, budget=budget, cache_dir=cache_dir)
        report = self._write_report(report)
        with self._lock:
            self._last_report = report
        if trust_if_low_risk and report.risk_score <= 0.05:
            self.registry.mark_trusted(fp.fingerprint, risk_score=report.risk_score, report_path=report.report_path, scanner_version=SCANNER_VERSION, notes="auto-trusted low risk scan")
        return report.to_dict()

    def start_background_scan(self, *, scan_type: str = "quick", profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._scan_thread and self._scan_thread.is_alive():
            return {"started": False, "reason": "scan_already_running"}
        self._stop_requested = False

        def worker() -> None:
            try:
                self.scan(scan_type=scan_type, profile=profile, custom_model=custom_model)
            except Exception as exc:  # pragma: no cover - surfaced via status
                self._last_error = str(exc)

        self._scan_thread = threading.Thread(target=worker, name="model-security-scan", daemon=True)
        self._scan_thread.start()
        return {"started": True, "scan_type": scan_type}

    def stop_scan(self) -> dict[str, Any]:
        self._stop_requested = True
        return {"stop_requested": True, "running": bool(self._scan_thread and self._scan_thread.is_alive())}

    def trust_current(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None, notes: str = "manual approval") -> dict[str, Any]:
        fp = self.current_fingerprint(profile=profile, custom_model=custom_model)
        risk = self._last_report.risk_score if self._last_report else 0.0
        report_path = self._last_report.report_path if self._last_report else None
        rec = self.registry.mark_trusted(fp.fingerprint, risk_score=float(risk or 0.0), report_path=report_path, scanner_version=SCANNER_VERSION, notes=notes)
        return rec.to_dict()

    def latest_report(self) -> dict[str, Any]:
        if self._last_report:
            return self._last_report.to_dict()
        if self._last_fp:
            rec = self.registry.get(self._last_fp.fingerprint)
            if rec and rec.report_path and Path(rec.report_path).exists():
                try:
                    return json.loads(Path(rec.report_path).read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {"status": "missing", "message": "No model security report is available yet."}

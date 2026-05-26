from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from defense.runtime.artifacts import resolve_artifact_candidate
from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config, project_root

from .fingerprint import ModelFingerprint, SCANNER_VERSION, build_model_fingerprint, sha256_file
from .integrity import TrustStoreIntegrity, verify_trust_store, write_trust_store_seal
from .purifier import known_poisoned_attack_metrics, run_new_purification
from .registry import ModelTrustRegistry
from .reports import ModelPurificationReport, ModelSecurityReport, ScanBudget, now_iso
from .scanner import full_scan, quick_scan
from .storage import ModelSecurityStorage


class ModelSecurityService:
    """Runtime wrapper for Module B model security without blocking Module A preview/detection."""

    def __init__(self, *, config_path: str | Path | None = None, root: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.root = Path(root) if root else project_root()
        self.storage = ModelSecurityStorage.from_project_root(self.root)
        self.storage.ensure()
        self.runtime_dir = self.storage.root
        self.log_path = self.storage.log_path
        self.registry = ModelTrustRegistry(self.storage.registry_path, on_save=self._write_registry_seal)
        self._lock = threading.Lock()
        self._scan_thread: threading.Thread | None = None
        self._stop_requested = False
        self._last_report: ModelSecurityReport | None = None
        self._last_purification_report: ModelPurificationReport | None = None
        self._last_fp: ModelFingerprint | None = None
        self._last_error = ""
        self._scan_target_fingerprint: str | None = None
        self._purify_thread: threading.Thread | None = None
        self._purify_target_fingerprint: str | None = None

    def _write_registry_seal(self, data: dict[str, Any]) -> None:
        write_trust_store_seal(self.registry.path, self.storage.registry_seal_path, data)

    def _trust_store_integrity(self) -> TrustStoreIntegrity:
        return verify_trust_store(self.registry.path, self.storage.registry_seal_path)

    def _trust_store_fields(self, integrity: TrustStoreIntegrity | None = None) -> dict[str, Any]:
        checked = integrity or self._trust_store_integrity()
        return {
            "trust_store_status": checked.status,
            "trust_store_reason": checked.reason,
            "trust_store_ok": checked.ok,
            "host_fingerprint_status": checked.host_fingerprint_status,
            "host_fingerprint_hash": checked.host_fingerprint_hash,
            "registry_path": checked.registry_path,
            "registry_seal_path": checked.registry_seal_path,
            "registry_hash": checked.registry_hash,
        }

    def _log_event(self, event: str, *, status: str = "info", message: str = "", **fields: Any) -> None:
        payload = {
            "time": now_iso(),
            "event": str(event),
            "status": str(status),
            "message": str(message),
            **fields,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            pass

    def recent_logs(self, *, limit: int = 50) -> dict[str, Any]:
        max_items = max(1, min(int(limit or 50), 200))
        entries: list[dict[str, Any]] = []
        if self.log_path.exists() and self.log_path.is_file():
            try:
                lines = self.log_path.read_text(encoding="utf-8").splitlines()[-max_items:]
                for line in lines:
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        entries.append(item)
            except Exception as exc:
                return {"log_path": str(self.log_path), "count": 0, "entries": [], "error": str(exc)}
        entries.reverse()
        return {"log_path": str(self.log_path), "count": len(entries), "entries": entries}

    def _config(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        return load_runtime_config(config_path=self.config_path, profile=profile or "default", custom_model=custom_model or {})

    def _config_and_fingerprint(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> tuple[dict[str, Any], ModelFingerprint]:
        cfg = self._config(profile=profile, custom_model=custom_model)
        fp = build_model_fingerprint(cfg, root=self.root)
        with self._lock:
            self._last_fp = fp
        return cfg, fp

    def current_fingerprint(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> ModelFingerprint:
        _, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
        return fp

    def _resolve_candidate(self, value: Any) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return resolve_artifact_candidate(text, self.root)

    def _source_pt_candidates(self, config: dict[str, Any], fp: ModelFingerprint) -> list[Path]:
        candidates: list[Path] = []
        runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
        custom = runtime.get("custom_model", {}) if isinstance(runtime.get("custom_model"), dict) else {}
        custom_enabled = bool(custom.get("enabled"))
        for key in ("source_pt_path", "source_model_path", "source_path"):
            p = self._resolve_candidate(custom.get(key))
            if p:
                candidates.append(p)
        if custom_enabled:
            runtime_path = self._resolve_candidate(custom.get("path"))
            if runtime_path:
                if runtime_path.suffix.lower() in {".pt", ".pth"}:
                    candidates.append(runtime_path)
                else:
                    candidates.append(runtime_path.with_suffix(".pt"))
                    candidates.append(runtime_path.parent / "best.pt")
            return self._dedupe_paths(candidates)

        model_security = config.get("model_security", {}) if isinstance(config.get("model_security"), dict) else {}
        for key in ("source_pt", "source_pt_path", "source_model_path"):
            p = self._resolve_candidate(model_security.get(key))
            if p:
                candidates.append(p)
        inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
        artifacts = inference.get("artifacts", {}) if isinstance(inference.get("artifacts"), dict) else {}
        raw_pt = artifacts.get("pytorch", artifacts.get("pt"))
        if isinstance(raw_pt, (list, tuple)):
            for item in raw_pt:
                p = self._resolve_candidate(item)
                if p:
                    candidates.append(p)
        else:
            p = self._resolve_candidate(raw_pt)
            if p:
                candidates.append(p)
        if fp.model_path:
            runtime_path = Path(fp.model_path)
            if runtime_path.suffix.lower() in {".pt", ".pth"}:
                candidates.append(runtime_path)
            else:
                candidates.append(runtime_path.with_suffix(".pt"))
                candidates.append(runtime_path.parent / "best.pt")
        return self._dedupe_paths(candidates)

    @staticmethod
    def _dedupe_paths(paths: list[Path]) -> list[Path]:
        seen: set[str] = set()
        out: list[Path] = []
        for p in paths:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _source_pt_path(self, config: dict[str, Any], fp: ModelFingerprint) -> Path | None:
        for candidate in self._source_pt_candidates(config, fp):
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".pt", ".pth"}:
                return candidate
        return None

    def _validation_assets(self, config: dict[str, Any]) -> dict[str, Any]:
        roots: list[str] = []
        model_security = config.get("model_security", {}) if isinstance(config.get("model_security"), dict) else {}
        for key in ("heldout_roots", "external_eval_roots", "validation_roots"):
            value = model_security.get(key)
            if isinstance(value, str):
                roots.append(value)
            elif isinstance(value, list):
                roots.extend(str(item) for item in value)
        heldout_cfg = self.root / "configs" / "model_security" / "heldout_sets.yaml"
        if heldout_cfg.exists():
            try:
                import yaml

                raw = yaml.safe_load(heldout_cfg.read_text(encoding="utf-8")) or {}
                value = raw.get("heldout_roots") if isinstance(raw, dict) else None
                if isinstance(value, str):
                    roots.append(value)
                elif isinstance(value, list):
                    roots.extend(str(item) for item in value)
            except Exception as exc:
                return {"usable": False, "roots": roots, "existing_roots": [], "error": str(exc)}
        existing: list[str] = []
        resolved: list[str] = []
        for raw in roots:
            p = self._resolve_candidate(raw)
            if p is None:
                continue
            resolved.append(str(p))
            if p.exists():
                existing.append(str(p))
        return {"usable": bool(existing), "roots": resolved, "existing_roots": existing}

    @staticmethod
    def _trusted_record_matches(rec: Any, fp: ModelFingerprint, source_pt_hash: str | None) -> bool:
        if not rec or not rec.approved_for_runtime:
            return False
        if rec.scanner_version != SCANNER_VERSION:
            return False
        if not rec.runtime_model_hash or rec.runtime_model_hash != fp.model_hash:
            return False
        if not rec.class_names_hash or rec.class_names_hash != fp.class_names_hash:
            return False
        if not rec.ppe_mapping_hash or rec.ppe_mapping_hash != fp.ppe_mapping_hash:
            return False
        if not rec.source_model_hash or rec.source_model_hash != source_pt_hash:
            return False
        return True

    def _mark_clean_full_scan_trusted(self, report: ModelSecurityReport, fp: ModelFingerprint) -> None:
        """Persist a trusted record from a clean canonical full scan."""
        self.registry.mark_trusted(
            fp.fingerprint,
            risk_score=report.risk_score,
            report_path=report.report_path,
            scanner_version=SCANNER_VERSION,
            notes="auto-trusted clean full scan",
            runtime_model_hash=fp.model_hash,
            runtime_model_path=fp.model_path,
            source_model_hash=report.source_model_hash,
            source_model_path=report.source_model_path,
            backend=fp.backend,
            model_family=fp.model_family,
            image_size=fp.image_size,
            class_names_hash=fp.class_names_hash,
            ppe_mapping_hash=fp.ppe_mapping_hash,
            security_metrics=self._security_metrics(purified_report=report),
            approval_source="full_scan",
        )
        self._log_event(
            "whitelist_written",
            status="trusted",
            message="完整扫描通过，模型已自动写入白名单",
            fingerprint=fp.fingerprint,
            report_path=report.report_path,
            approval_source="full_scan",
        )

    def _full_clean_report_matches_runtime(self, report: ModelSecurityReport | None, fp: ModelFingerprint, source_hash: str | None) -> bool:
        if not report or report.scan_type != "full" or report.status not in {"trusted", "clean"}:
            return False
        if report.fingerprint.get("fingerprint") != fp.fingerprint:
            return False
        if report.fingerprint.get("model_hash") != fp.model_hash:
            return False
        if source_hash and report.source_model_hash != source_hash:
            return False
        if not report.report_path:
            return False
        return True

    @staticmethod
    def _strict_observed_asr(strict: dict[str, Any]) -> float | None:
        try:
            k = float(strict.get("k"))
            n = float(strict.get("N"))
        except Exception:
            return None
        if n <= 0:
            return None
        return k / n

    @staticmethod
    def _scan_metrics(report: ModelSecurityReport | None, *, role: str) -> dict[str, Any]:
        if report is None:
            return {}
        metrics: dict[str, Any] = {
            f"{role}_status": report.status,
            f"{role}_risk_score": report.risk_score,
            f"{role}_report_path": report.report_path,
        }
        diagnostics = report.diagnostics if isinstance(report.diagnostics, dict) else {}
        external = diagnostics.get("external_validation")
        summary = external.get("summary") if isinstance(external, dict) and isinstance(external.get("summary"), dict) else {}
        if summary:
            if summary.get("max_asr") is not None:
                metrics[f"{role}_asr"] = float(summary.get("max_asr") or 0.0)
                metrics[f"{role}_asr_kind"] = "external_max_asr"
            if summary.get("mean_asr") is not None:
                metrics[f"{role}_mean_asr"] = float(summary.get("mean_asr") or 0.0)
            if summary.get("n_rows") is not None:
                metrics[f"{role}_n_rows"] = int(summary.get("n_rows") or 0)

        poisoned = diagnostics.get("new_algorithm_poisoned_evidence")
        if isinstance(poisoned, dict):
            original = poisoned.get("original_attack_metrics")
            if not isinstance(original, dict) and poisoned.get("family_tag"):
                original = known_poisoned_attack_metrics(str(poisoned.get("family_tag") or ""))
            if isinstance(original, dict) and original.get("max_asr") is not None:
                metrics[f"{role}_asr"] = float(original.get("max_asr") or 0.0)
                metrics[f"{role}_asr_kind"] = "packaged_original_max_asr"
                metrics[f"{role}_attack"] = original.get("attack")
                metrics[f"{role}_metric_source"] = original.get("source")
                if original.get("successes") is not None:
                    metrics[f"{role}_successes"] = original.get("successes")
                if original.get("n") is not None:
                    metrics[f"{role}_n"] = original.get("n")
            if poisoned.get("family_tag"):
                metrics.setdefault("family_tag", poisoned.get("family_tag"))

        strict = diagnostics.get("new_algorithm_strict_audit")
        if isinstance(strict, dict):
            observed = ModelSecurityService._strict_observed_asr(strict)
            if observed is not None:
                metrics[f"{role}_asr"] = observed
                metrics[f"{role}_asr_kind"] = "strict_observed_asr"
            metrics[f"{role}_wilson_upper"] = strict.get("wilson_upper")
            metrics[f"{role}_map_drop_pp"] = strict.get("mAP_drop_pp")
            metrics[f"{role}_k"] = strict.get("k")
            metrics[f"{role}_n"] = strict.get("N")
            metrics[f"{role}_defense"] = strict.get("defense")
            metrics[f"{role}_tier"] = strict.get("tier")
            if strict.get("family_tag"):
                metrics.setdefault("family_tag", strict.get("family_tag"))
        return metrics

    @staticmethod
    def _read_report(path: str | None) -> ModelSecurityReport | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return ModelSecurityReport(**raw)
        except Exception:
            return None

    @staticmethod
    def _read_purification_report(path: str | None) -> ModelPurificationReport | None:
        if not path:
            return None
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return ModelPurificationReport(**raw)
        except Exception:
            return None

    def _security_metrics(
        self,
        *,
        original_report: ModelSecurityReport | None = None,
        purified_report: ModelSecurityReport | None = None,
        purification_report: ModelPurificationReport | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics.update(self._scan_metrics(original_report, role="original"))
        metrics.update(self._scan_metrics(purified_report, role="purified"))
        if purification_report is not None:
            metrics["purification_status"] = purification_report.status
            metrics["purification_strategy"] = purification_report.strategy
            metrics["purification_report_path"] = purification_report.report_path
            metrics["purified_model_path"] = purification_report.purified_model_path
            metrics["purified_model_hash"] = purification_report.purified_model_hash
            if purified_report is None and purification_report.scan_report_path:
                metrics.update(
                    self._scan_metrics(
                        self._read_report(purification_report.scan_report_path),
                        role="purified",
                    )
                )
        return metrics

    def _trusted_record_context(self, rec: Any) -> dict[str, Any]:
        purification_report = self._read_purification_report(getattr(rec, "purification_report_path", None))
        purified_report = self._read_report(getattr(rec, "report_path", None))
        original_report = None
        if purification_report is not None:
            original_fp = str(purification_report.fingerprint.get("fingerprint") or "")
            original_report = self._load_report(
                original_fp,
                "full",
                purification_report.source_model_hash,
            )
        metrics = dict(getattr(rec, "security_metrics", None) or {})
        if not metrics:
            metrics = self._security_metrics(
                original_report=original_report,
                purified_report=purified_report,
                purification_report=purification_report,
            )
        original_source_model_path = getattr(rec, "original_source_model_path", None) or (
            purification_report.source_model_path if purification_report else None
        )
        original_source_model_hash = getattr(rec, "original_source_model_hash", None) or (
            purification_report.source_model_hash if purification_report else None
        )
        if original_source_model_path:
            metrics.setdefault("original_source_model_path", original_source_model_path)
        if original_source_model_hash:
            metrics.setdefault("original_source_model_hash", original_source_model_hash)
        return {
            "security_metrics": metrics,
            "original_source_model_path": original_source_model_path,
            "original_source_model_hash": original_source_model_hash,
        }

    def _is_scanning(self, fingerprint: str | None = None) -> bool:
        running = bool(self._scan_thread and self._scan_thread.is_alive())
        if not running:
            return False
        return fingerprint is None or self._scan_target_fingerprint == fingerprint

    def _is_purifying(self, fingerprint: str | None = None) -> bool:
        thread_running = bool(self._purify_thread and self._purify_thread.is_alive())
        inline_running = bool(self._scan_thread and self._scan_thread.is_alive() and self._purify_target_fingerprint)
        running = thread_running or inline_running
        if not running:
            return False
        return fingerprint is None or self._purify_target_fingerprint == fingerprint

    def admission_status(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            cfg, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
            integrity = self._trust_store_integrity()
            source_pt = self._source_pt_path(cfg, fp)
            source_hash = "sha256:" + sha256_file(source_pt) if source_pt and source_pt.exists() else None
            rec = self.registry.get(fp.fingerprint) if integrity.ok else None
            assets = self._validation_assets(cfg)
            scanning = self._is_scanning(fp.fingerprint)
            last_report = (
                self._last_report
                if self._last_report
                and self._last_report.scan_type == "full"
                and self._last_report.fingerprint.get("fingerprint") == fp.fingerprint
                else None
            )
            if last_report and source_hash and last_report.source_model_hash and last_report.source_model_hash != source_hash:
                last_report = None
            if last_report is None:
                last_report = self._load_report(fp.fingerprint, "full", source_hash)
            purifying = self._is_purifying(fp.fingerprint)
            last_purification = (
                self._last_purification_report
                if self._last_purification_report
                and self._last_purification_report.fingerprint.get("fingerprint") == fp.fingerprint
                else None
            )
            if last_purification is None:
                last_purification = self._load_purification_report(fp.fingerprint)

            status = "blocked_scan_required"
            allowed = False
            reason = "model_not_in_whitelist"
            risk = last_report.risk_score if last_report else None
            report_path = last_report.report_path if last_report else None
            last_scan = last_report.completed_at if last_report else None
            if not integrity.ok:
                status = "trust_store_compromised"
                reason = integrity.reason
            elif rec and self._trusted_record_matches(rec, fp, source_hash):
                status = "trusted"
                allowed = True
                reason = ""
                risk = rec.risk_score
                report_path = rec.report_path
                last_scan = rec.last_scan_time
            elif purifying:
                status = "purifying"
                reason = "purification_running"
            elif last_purification and last_purification.status == "scan_clean_trusted":
                status = "purified_alternative_available"
                reason = "purified_pt_clean_but_runtime_not_selected"
            elif last_report and last_report.status in {"review", "suspicious", "unverifiable"}:
                status = last_report.status
                reason = f"last_full_scan_{last_report.status}"
            elif scanning:
                status = "scanning"
                reason = "full_scan_running"
            elif not fp.model_path or not fp.model_hash:
                status = "unverifiable"
                reason = "no_runtime_artifact"
            elif fp.model_path and Path(fp.model_path).suffix.lower() not in {".pt", ".pth"} and source_pt is None:
                status = "unverifiable"
                reason = "source_pt_required_for_accelerated_artifact"

            trusted_record_match = bool(rec and self._trusted_record_matches(rec, fp, source_hash))
            trusted_context = self._trusted_record_context(rec) if trusted_record_match else {}
            security_metrics = (
                dict(trusted_context.get("security_metrics") or {})
                if trusted_record_match
                else self._security_metrics(
                    original_report=last_report,
                    purification_report=last_purification,
                )
            )

            payload = {
                "enabled": True,
                "allowed": allowed,
                "status": status,
                "admission_status": status,
                "whitelist_hit": bool(allowed),
                "blocking_reason": reason,
                "scanning": scanning,
                "purifying": purifying,
                "fingerprint": fp.fingerprint,
                "model_hash": fp.model_hash,
                "model_path": fp.model_path,
                "runtime_artifact_path": fp.model_path,
                "source_pt_path": str(source_pt) if source_pt else None,
                "source_pt_hash": source_hash,
                "original_source_model_path": (
                    trusted_context.get("original_source_model_path")
                    if trusted_record_match
                    else (last_purification.source_model_path if last_purification else None)
                ),
                "original_source_model_hash": (
                    trusted_context.get("original_source_model_hash")
                    if trusted_record_match
                    else (last_purification.source_model_hash if last_purification else None)
                ),
                "backend": fp.backend,
                "model_family": fp.model_family,
                "image_size": fp.image_size,
                "class_names_hash": fp.class_names_hash,
                "ppe_mapping_hash": fp.ppe_mapping_hash,
                "scanner_version": SCANNER_VERSION,
                "whitelist_policy": "auto_full_scan_clean_only",
                "whitelist_user_actions": ["delete", "clear"],
                "full_scan_model_policy": "source_pt_only",
                "risk_score": risk,
                "security_metrics": security_metrics,
                "last_scan_time": last_scan,
                "report_path": report_path,
                "purification_status": last_purification.status if last_purification else ("running" if purifying else "idle"),
                "purification_report_path": last_purification.report_path if last_purification else None,
                "purified_model_path": last_purification.purified_model_path if last_purification else None,
                "purified_model_hash": last_purification.purified_model_hash if last_purification else None,
                "purified_models_dir": str(self.storage.purified_dir),
                "purification_strategy": last_purification.strategy if last_purification else None,
                "validation_assets": assets,
                "error": self._last_error,
            }
            payload.update(self._trust_store_fields(integrity))
            return payload
        except Exception as exc:
            self._last_error = str(exc)
            payload = {
                "enabled": True,
                "allowed": False,
                "status": "error",
                "admission_status": "error",
                "blocking_reason": str(exc),
                "error": str(exc),
                "scanner_version": SCANNER_VERSION,
            }
            try:
                payload.update(self._trust_store_fields())
            except Exception:
                pass
            return payload

    def ensure_admitted(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.admission_status(profile=profile, custom_model=custom_model)

    def status(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.admission_status(profile=profile, custom_model=custom_model)

    def trusted_purified_runtime_model(
        self,
        *,
        profile: str = "default",
        custom_model: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return a trusted purified PT custom-model override for the current blocked model."""

        cfg, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
        source_pt = self._source_pt_path(cfg, fp)
        source_hash = "sha256:" + sha256_file(source_pt) if source_pt and source_pt.exists() else None
        report = self._load_purification_report(fp.fingerprint)
        if (
            report is None
            or report.status != "scan_clean_trusted"
            or not report.purified_model_path
        ):
            return None
        purified_path = Path(report.purified_model_path)
        if not purified_path.exists() or not purified_path.is_file() or purified_path.suffix.lower() not in {".pt", ".pth"}:
            return None

        runtime_model = {
            "enabled": True,
            "path": str(purified_path),
            "backend": "pytorch",
            "model_family": fp.model_family or "auto",
            "source_pt_path": str(purified_path),
        }
        purified_status = self.admission_status(profile=profile, custom_model=runtime_model)
        if not bool(purified_status.get("allowed", False)):
            return None
        if not self.registry.get(str(purified_status.get("fingerprint") or "")):
            return None

        self._log_event(
            "purified_runtime_selected",
            status="trusted",
            message="已自动选择净化后可信PT作为A模块运行模型",
            source_fingerprint=fp.fingerprint,
            source_model_path=fp.model_path,
            source_model_hash=fp.model_hash,
            source_pt_hash=source_hash,
            purified_fingerprint=purified_status.get("fingerprint"),
            purified_model_path=str(purified_path),
            purified_model_hash=purified_status.get("model_hash"),
            purification_report_path=report.report_path,
            scan_report_path=report.scan_report_path,
        )
        return {
            "custom_model": runtime_model,
            "model_security": purified_status,
            "source_model_security": self.admission_status(profile=profile, custom_model=custom_model),
        }

    def prepare_runtime_for_start(
        self,
        *,
        profile: str = "default",
        custom_model: dict[str, Any] | None = None,
        auto_remediate: bool = True,
    ) -> dict[str, Any]:
        """Resolve the model that Module A may start with, running B remediation when needed."""

        requested_model = custom_model or {}
        scan_result: dict[str, Any] | None = None
        purification_result: dict[str, Any] | None = None
        runtime_replacement: dict[str, Any] | None = None
        admission = self.ensure_admitted(profile=profile, custom_model=requested_model)

        def allowed(runtime_model: dict[str, Any], model_security: dict[str, Any]) -> dict[str, Any]:
            return {
                "allowed": True,
                "custom_model": runtime_model,
                "model_security": model_security,
                "scan": scan_result,
                "purification": purification_result,
                "runtime_replacement": runtime_replacement,
            }

        def blocked(model_security: dict[str, Any]) -> dict[str, Any]:
            return {
                "allowed": False,
                "custom_model": None,
                "model_security": model_security,
                "scan": scan_result,
                "purification": purification_result,
                "runtime_replacement": runtime_replacement,
            }

        if bool(admission.get("allowed", False)):
            return allowed(requested_model, admission)

        status = str(admission.get("admission_status") or admission.get("status") or "")
        if status == "purified_alternative_available":
            replacement = self.trusted_purified_runtime_model(profile=profile, custom_model=requested_model)
            if replacement and replacement.get("custom_model") and replacement.get("model_security", {}).get("allowed"):
                model_security = dict(replacement["model_security"])
                model_security["runtime_replacement"] = {
                    "enabled": True,
                    "path": replacement["custom_model"].get("path"),
                    "backend": replacement["custom_model"].get("backend"),
                    "model_family": replacement["custom_model"].get("model_family"),
                    "source_pt_path": replacement["custom_model"].get("source_pt_path"),
                }
                runtime_replacement = {
                    "mode": "purified_runtime",
                    "source_model_security": replacement.get("source_model_security") or admission,
                }
                return allowed(replacement["custom_model"], model_security)

        if not auto_remediate:
            return blocked(admission)

        if status == "blocked_scan_required":
            if self._is_scanning(str(admission.get("fingerprint") or "")):
                return blocked(admission)
            scan_result = self.scan(scan_type="full", profile=profile, custom_model=requested_model)
            admission = self.status(profile=profile, custom_model=requested_model)
            if bool(admission.get("allowed", False)):
                return allowed(requested_model, admission)
            status = str(admission.get("admission_status") or admission.get("status") or "")

        if status == "suspicious":
            if self._is_purifying(str(admission.get("fingerprint") or "")):
                return blocked(admission)
            purification_result = self.purify(profile=profile, custom_model=requested_model, scan_after=True)
            admission = self.status(profile=profile, custom_model=requested_model)
            if str(admission.get("admission_status") or admission.get("status") or "") == "purified_alternative_available":
                replacement = self.trusted_purified_runtime_model(profile=profile, custom_model=requested_model)
                if replacement and replacement.get("custom_model") and replacement.get("model_security", {}).get("allowed"):
                    model_security = dict(replacement["model_security"])
                    model_security["runtime_replacement"] = {
                        "enabled": True,
                        "path": replacement["custom_model"].get("path"),
                        "backend": replacement["custom_model"].get("backend"),
                        "model_family": replacement["custom_model"].get("model_family"),
                        "source_pt_path": replacement["custom_model"].get("source_pt_path"),
                    }
                    runtime_replacement = {
                        "mode": "purified_runtime",
                        "source_model_security": replacement.get("source_model_security") or admission,
                    }
                    return allowed(replacement["custom_model"], model_security)

        return blocked(admission)

    def _write_report(self, report: ModelSecurityReport) -> ModelSecurityReport:
        path = self.storage.reports_dir / f"{report.fingerprint['fingerprint'].replace(':','_')}_{report.scan_type}.json"
        report.write(path)
        return report

    def _write_purification_report(self, report: ModelPurificationReport) -> ModelPurificationReport:
        path = self.storage.reports_dir / f"{report.fingerprint['fingerprint'].replace(':','_')}_purification.json"
        report.write(path)
        return report

    def _load_report(self, fingerprint: str, scan_type: str, source_hash: str | None = None) -> ModelSecurityReport | None:
        path = self.storage.reports_dir / f"{fingerprint.replace(':','_')}_{scan_type}.json"
        if not path.exists() or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            report = ModelSecurityReport(**raw)
        except Exception:
            return None
        if report.fingerprint.get("fingerprint") != fingerprint:
            return None
        if source_hash and report.source_model_hash and report.source_model_hash != source_hash:
            return None
        return report

    def _load_purification_report(self, fingerprint: str) -> ModelPurificationReport | None:
        path = self.storage.reports_dir / f"{fingerprint.replace(':','_')}_purification.json"
        if not path.exists() or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            report = ModelPurificationReport(**raw)
        except Exception:
            return None
        if report.fingerprint.get("fingerprint") != fingerprint:
            return None
        return report

    @staticmethod
    def _purified_pt_runtime_config(config: dict[str, Any], purified_model_path: Path, fp: ModelFingerprint) -> dict[str, Any]:
        scan_config = deepcopy(config)
        inference = scan_config.setdefault("inference", {})
        inference["backend"] = "pytorch"
        inference["model_family"] = fp.model_family or inference.get("model_family") or "auto"
        artifacts = inference.setdefault("artifacts", {})
        artifacts["pytorch"] = [str(purified_model_path)]
        runtime = scan_config.setdefault("runtime", {})
        custom = runtime.get("custom_model") if isinstance(runtime.get("custom_model"), dict) else {}
        custom = dict(custom or {})
        custom.update(
            {
                "enabled": True,
                "path": str(purified_model_path),
                "backend": "pytorch",
                "model_family": inference.get("model_family", "auto"),
                "source_pt_path": str(purified_model_path),
            }
        )
        runtime["custom_model"] = custom
        return scan_config

    def scan(self, *, scan_type: str = "quick", profile: str = "default", custom_model: dict[str, Any] | None = None, trust_if_low_risk: bool = False) -> dict[str, Any]:
        del trust_if_low_risk
        cfg, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
        budget = ScanBudget()
        cache_dir = self.storage.activation_cache_dir
        source_pt = self._source_pt_path(cfg, fp)
        assets = self._validation_assets(cfg)
        self._log_event(
            "scan_started",
            status="running",
            message=f"B模块{scan_type}扫描开始",
            scan_type=scan_type,
            fingerprint=fp.fingerprint,
            runtime_model_path=fp.model_path,
            source_pt_path=str(source_pt) if source_pt else None,
        )
        report = (
            full_scan(
                fp,
                budget=budget,
                cache_dir=cache_dir,
                source_model_path=source_pt,
                validation_assets=assets,
                runtime_config=cfg,
                project_root=self.root,
                report_dir=self.storage.reports_dir,
            )
            if scan_type == "full"
            else quick_scan(fp, budget=budget, cache_dir=cache_dir)
        )
        report = self._write_report(report)
        with self._lock:
            self._last_report = report
        self._log_event(
            "scan_completed",
            status=report.status,
            message=f"B模块{scan_type}扫描完成：{report.status}",
            scan_type=scan_type,
            fingerprint=fp.fingerprint,
            risk_score=report.risk_score,
            report_path=report.report_path,
            source_model_path=report.source_model_path,
        )
        if scan_type == "full" and report.status in {"trusted", "clean"}:
            self._mark_clean_full_scan_trusted(report, fp)
        return report.to_dict()

    def start_background_scan(
        self,
        *,
        scan_type: str = "quick",
        profile: str = "default",
        custom_model: dict[str, Any] | None = None,
        auto_purify: bool | None = None,
    ) -> dict[str, Any]:
        if self._scan_thread and self._scan_thread.is_alive():
            return {"started": False, "reason": "scan_already_running"}
        self._stop_requested = False
        target_fp = self.current_fingerprint(profile=profile, custom_model=custom_model)
        self._scan_target_fingerprint = target_fp.fingerprint
        should_auto_purify = scan_type == "full" if auto_purify is None else bool(auto_purify)

        def worker() -> None:
            try:
                report = self.scan(scan_type=scan_type, profile=profile, custom_model=custom_model)
                if scan_type == "full" and should_auto_purify and report.get("status") == "suspicious":
                    self._log_event(
                        "purification_auto_queued",
                        status="running",
                        message="完整扫描发现疑似投毒模型，自动进入净化并复扫流程",
                        fingerprint=target_fp.fingerprint,
                        report_path=report.get("report_path"),
                    )
                    self._purify_target_fingerprint = target_fp.fingerprint
                    try:
                        self.purify(profile=profile, custom_model=custom_model, scan_after=True)
                    except Exception as exc:  # pragma: no cover - surfaced via status/logs
                        self._last_error = str(exc)
                        self._log_event(
                            "purification_failed",
                            status="error",
                            message=str(exc),
                            fingerprint=target_fp.fingerprint,
                            auto_purify=True,
                        )
                    finally:
                        self._purify_target_fingerprint = None
            except Exception as exc:  # pragma: no cover - surfaced via status
                self._last_error = str(exc)
                self._log_event("scan_failed", status="error", message=str(exc), scan_type=scan_type, fingerprint=target_fp.fingerprint)

        self._scan_thread = threading.Thread(target=worker, name="model-security-scan", daemon=True)
        self._scan_thread.start()
        return {
            "started": True,
            "scan_type": scan_type,
            "fingerprint": target_fp.fingerprint,
            "auto_purify": should_auto_purify,
        }

    def purify(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None, scan_after: bool = True) -> dict[str, Any]:
        cfg, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
        source_pt = self._source_pt_path(cfg, fp)
        source_hash = "sha256:" + sha256_file(source_pt) if source_pt and source_pt.exists() else None
        latest_report = self._last_report if self._last_report and self._last_report.fingerprint.get("fingerprint") == fp.fingerprint else None
        if latest_report is None:
            latest_report = self._load_report(fp.fingerprint, "full", source_hash)
        if latest_report and self._full_clean_report_matches_runtime(latest_report, fp, source_hash):
            raise ValueError("source_model_already_clean")
        if not latest_report or latest_report.status != "suspicious":
            raise ValueError("purification_requires_suspicious_full_scan")

        self._log_event(
            "purification_started",
            status="running",
            message="B模块净化开始",
            fingerprint=fp.fingerprint,
            source_pt_path=str(source_pt) if source_pt else None,
            scan_after=bool(scan_after),
        )
        report = run_new_purification(
            fp=fp,
            config=cfg,
            root=self.root,
            runtime_dir=self.runtime_dir,
            source_model_path=source_pt,
            latest_scan_report=latest_report,
        )
        with self._lock:
            self._last_purification_report = report
        report = self._write_purification_report(report)
        if scan_after and report.purified_model_path:
            candidate_paths: list[Path] = []
            selected_path = Path(report.purified_model_path)
            candidate_paths.append(selected_path)
            for candidate in report.candidates:
                if not isinstance(candidate, dict):
                    continue
                raw_path = candidate.get("output_model")
                if not raw_path:
                    continue
                candidate_path = Path(str(raw_path))
                if candidate_path not in candidate_paths:
                    candidate_paths.append(candidate_path)
            candidate_scan_results: list[dict[str, Any]] = []
            try:
                assets = self._validation_assets(cfg)
                accepted_scan_report = None
                accepted_fp = None
                accepted_path = None
                for candidate_path in candidate_paths:
                    if not candidate_path.exists():
                        candidate_scan_results.append(
                            {
                                "candidate_path": str(candidate_path),
                                "status": "missing",
                                "reason": "candidate model file is missing",
                            }
                        )
                        continue
                    purified_cfg = self._purified_pt_runtime_config(cfg, candidate_path, fp)
                    purified_fp = build_model_fingerprint(purified_cfg, root=self.root)
                    scan_report = full_scan(
                        purified_fp,
                        budget=ScanBudget(),
                        cache_dir=self.storage.activation_cache_dir,
                        source_model_path=candidate_path,
                        validation_assets=assets,
                        runtime_config=purified_cfg,
                        project_root=self.root,
                        report_dir=self.storage.reports_dir,
                    )
                    scan_report = self._write_report(scan_report)
                    with self._lock:
                        self._last_report = scan_report
                    candidate_scan_results.append(
                        {
                            "candidate_path": str(candidate_path),
                            "fingerprint": purified_fp.fingerprint,
                            "model_hash": purified_fp.model_hash,
                            "status": scan_report.status,
                            "risk_score": scan_report.risk_score,
                            "report_path": scan_report.report_path,
                        }
                    )
                    if scan_report.status in {"trusted", "clean"}:
                        accepted_scan_report = scan_report
                        accepted_fp = purified_fp
                        accepted_path = candidate_path
                        break
                report.diagnostics["candidate_scan_results"] = candidate_scan_results
                if accepted_scan_report is not None and accepted_fp is not None and accepted_path is not None:
                    report.purified_model_path = str(accepted_path)
                    report.purified_model_hash = "sha256:" + sha256_file(accepted_path)
                    report.scan_report_path = accepted_scan_report.report_path
                    report.scan_status = accepted_scan_report.status
                    report.status = "scan_clean_trusted"
                    self.registry.mark_trusted(
                        accepted_fp.fingerprint,
                        risk_score=accepted_scan_report.risk_score,
                        report_path=accepted_scan_report.report_path,
                        scanner_version=SCANNER_VERSION,
                        notes="auto-trusted purified PT after clean full scan",
                        runtime_model_hash=accepted_fp.model_hash,
                        runtime_model_path=accepted_fp.model_path,
                        source_model_hash=accepted_scan_report.source_model_hash,
                        source_model_path=accepted_scan_report.source_model_path,
                        original_source_model_hash=latest_report.source_model_hash or source_hash,
                        original_source_model_path=latest_report.source_model_path or (str(source_pt) if source_pt else None),
                        backend=accepted_fp.backend,
                        model_family=accepted_fp.model_family,
                        image_size=accepted_fp.image_size,
                        class_names_hash=accepted_fp.class_names_hash,
                        ppe_mapping_hash=accepted_fp.ppe_mapping_hash,
                        purification_report_path=report.report_path,
                        security_metrics=self._security_metrics(
                            original_report=latest_report,
                            purified_report=accepted_scan_report,
                            purification_report=report,
                        ),
                        approval_source="purified_full_scan",
                    )
                    self._log_event(
                        "whitelist_written",
                        status="trusted",
                        message="净化模型复扫通过，已自动写入白名单",
                        fingerprint=accepted_fp.fingerprint,
                        source_fingerprint=fp.fingerprint,
                        report_path=accepted_scan_report.report_path,
                        purification_report_path=report.report_path,
                        approval_source="purified_full_scan",
                    )
                else:
                    report.scan_status = candidate_scan_results[-1]["status"] if candidate_scan_results else "missing"
                    report.status = "purified_scan_no_clean_candidate"
            except Exception as exc:  # pragma: no cover - surfaced via report/status
                report.status = "purified_scan_failed"
                report.error = str(exc)
        report = self._write_purification_report(report)
        with self._lock:
            self._last_purification_report = report
        self._log_event(
            "purification_completed",
            status=report.status,
            message=f"B模块净化完成：{report.status}",
            fingerprint=fp.fingerprint,
            purified_model_path=report.purified_model_path,
            purification_report_path=report.report_path,
            scan_report_path=report.scan_report_path,
            scan_status=report.scan_status,
            error=report.error,
        )
        return report.to_dict()

    def start_background_purification(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None, scan_after: bool = True) -> dict[str, Any]:
        target_fp = self.current_fingerprint(profile=profile, custom_model=custom_model)
        if self._is_purifying(target_fp.fingerprint):
            return {"started": False, "reason": "purification_already_running"}
        self._purify_target_fingerprint = target_fp.fingerprint

        def worker() -> None:
            try:
                self.purify(profile=profile, custom_model=custom_model, scan_after=scan_after)
            except Exception as exc:  # pragma: no cover - surfaced via status
                self._last_error = str(exc)
                self._log_event("purification_failed", status="error", message=str(exc), fingerprint=target_fp.fingerprint)
            finally:
                self._purify_target_fingerprint = None

        self._purify_thread = threading.Thread(target=worker, name="model-security-purify", daemon=True)
        self._purify_thread.start()
        return {"started": True, "fingerprint": target_fp.fingerprint, "scan_after": bool(scan_after)}

    def stop_scan(self) -> dict[str, Any]:
        self._stop_requested = True
        return {"stop_requested": True, "running": bool(self._scan_thread and self._scan_thread.is_alive())}

    def trust_current(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None, notes: str = "manual approval") -> dict[str, Any]:
        del profile, custom_model, notes
        raise ValueError("manual trust is disabled; run a full scan on the source PT to create whitelist records")

    def trust_records(self) -> dict[str, Any]:
        integrity = self._trust_store_integrity()
        records: list[dict[str, Any]] = []
        if integrity.ok:
            for rec in self.registry.list_records():
                item = rec.to_dict()
                context = self._trusted_record_context(rec)
                if context.get("security_metrics"):
                    item["security_metrics"] = context["security_metrics"]
                if context.get("original_source_model_path"):
                    item["original_source_model_path"] = context["original_source_model_path"]
                if context.get("original_source_model_hash"):
                    item["original_source_model_hash"] = context["original_source_model_hash"]
                records.append(item)
        payload = {"registry_path": str(self.registry.path), "count": len(records), "records": records}
        payload.update(self._trust_store_fields(integrity))
        return payload

    def delete_trust(self, fingerprint: str) -> dict[str, Any]:
        fingerprint = str(fingerprint or "").strip()
        if not fingerprint:
            raise ValueError("fingerprint is required")
        integrity = self._trust_store_integrity()
        if not integrity.ok:
            raise ValueError(f"trust_store_compromised:{integrity.reason}")
        deleted = self.registry.delete(fingerprint)
        self._log_event(
            "whitelist_deleted",
            status="deleted" if deleted else "missing",
            message="已删除当前模型白名单" if deleted else "未找到要删除的白名单记录",
            fingerprint=fingerprint,
        )
        payload = {"deleted": deleted, "fingerprint": fingerprint, "registry_path": str(self.registry.path)}
        payload.update(self._trust_store_fields())
        return payload

    def clear_trust(self) -> dict[str, Any]:
        deleted = self.registry.clear()
        self._log_event(
            "whitelist_cleared",
            status="deleted",
            message=f"已清空白名单，删除 {deleted} 条记录",
            deleted=deleted,
        )
        payload = {"deleted": deleted, "registry_path": str(self.registry.path)}
        payload.update(self._trust_store_fields())
        return payload

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

    def latest_purification_report(self) -> dict[str, Any]:
        if self._last_purification_report:
            return self._last_purification_report.to_dict()
        if self._last_fp:
            report = self._load_purification_report(self._last_fp.fingerprint)
            if report:
                return report.to_dict()
        return {"status": "missing", "message": "No model purification report is available yet."}

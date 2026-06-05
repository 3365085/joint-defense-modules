from __future__ import annotations

import json
import pickletools
import shutil
import threading
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from defense.runtime.artifacts import resolve_artifact_candidate
from defense.runtime.catalog import list_artifacts, register_artifact
from defense.runtime.config import DEFAULT_CONFIG_PATH, infer_backend_from_model_path, load_runtime_config, project_root

from .fingerprint import ModelFingerprint, SCANNER_VERSION, build_model_fingerprint, sha256_file
from .integrity import TrustStoreIntegrity, verify_trust_store, write_trust_store_seal
from .purifier import known_poisoned_attack_metrics, run_new_purification
from .registry import ModelTrustRegistry
from .reports import ModelPurificationReport, ModelSecurityReport, ScanBudget, now_iso
from .scanner import _class_name_map, full_scan, quick_scan
from .storage import ModelSecurityStorage


def _normalize_embedded_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            text = str(value or "").strip()
            if text:
                normalized[idx] = text
        return normalized
    if isinstance(names, (list, tuple)):
        return {idx: str(value).strip() for idx, value in enumerate(names) if str(value).strip()}
    return {}


def _is_label_token(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 64:
        return False
    if text.isdigit():
        return False
    if any(marker in text for marker in ("\\", "/", ":", "\n", "\r", "\t")):
        return False
    return True


def _read_torch_zip_class_names(path: Path) -> dict[int, str]:
    """Extract checkpoint names from torch zip metadata without unpickling objects."""
    with zipfile.ZipFile(path, "r") as archive:
        data_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
        if not data_name:
            return {}
        strings: list[str] = []
        for op, arg, _pos in pickletools.genops(archive.read(data_name)):
            if op.name not in {
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "UNICODE",
                "BINSTRING",
                "SHORT_BINSTRING",
                "STRING",
            }:
                continue
            if isinstance(arg, bytes):
                try:
                    arg = arg.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            strings.append(str(arg))
    for index, token in enumerate(strings):
        if str(token).strip().lower() not in {"names", "class_names", "labels"}:
            continue
        labels: list[str] = []
        stop_keys = {
            "args",
            "task",
            "mode",
            "model",
            "data",
            "epochs",
            "optimizer",
            "train_args",
            "date",
            "version",
            "license",
        }
        for candidate in strings[index + 1 : index + 128]:
            text = str(candidate or "").strip()
            if labels and text.lower() in stop_keys:
                break
            if not _is_label_token(text):
                if labels:
                    break
                continue
            labels.append(text)
            if len(labels) >= 256:
                break
        normalized = _normalize_embedded_names(labels)
        if normalized:
            return normalized
    return {}


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
        self._last_scan_job: dict[str, Any] | None = None
        self._last_purification_job: dict[str, Any] | None = None
        self._export_thread: threading.Thread | None = None
        self._last_export_job: dict[str, Any] | None = None
        self._embedded_names_cache: dict[str, dict[str, Any]] = {}

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

    def _embedded_class_names(self, path: Path | None) -> dict[str, Any]:
        if path is None or path.suffix.lower() not in {".pt", ".pth"}:
            return {"available": False, "class_names": {}, "source": None, "error": ""}
        try:
            stat = path.stat()
        except OSError as exc:
            return {"available": False, "class_names": {}, "source": str(path), "error": str(exc)}
        cache_key = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        cached = self._embedded_names_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        try:
            class_names = _read_torch_zip_class_names(path)
            result = {
                "available": bool(class_names),
                "class_names": class_names,
                "source": str(path),
                "error": "" if class_names else "embedded_class_names_not_found_in_safe_metadata",
            }
        except Exception as exc:
            result = {"available": False, "class_names": {}, "source": str(path), "error": str(exc)}
        self._embedded_names_cache[cache_key] = dict(result)
        return result

    def _class_names_diagnostics(self, config: dict[str, Any], source_pt: Path | None) -> dict[str, Any]:
        configured = _class_name_map(config)
        runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
        custom = runtime.get("custom_model", {}) if isinstance(runtime.get("custom_model"), dict) else {}
        custom_enabled = bool(custom.get("enabled", False))
        embedded = (
            self._embedded_class_names(source_pt)
            if custom_enabled
            else {"available": False, "class_names": {}, "source": str(source_pt) if source_pt else None, "error": ""}
        )
        embedded_names = embedded.get("class_names") if isinstance(embedded.get("class_names"), dict) else {}
        mismatch = bool(embedded_names) and configured != embedded_names
        warning = ""
        if mismatch:
            warning = "configured class_names differ from checkpoint embedded names; detection labels may be swapped"
        return {
            "configured_class_names": configured,
            "embedded_class_names": embedded_names,
            "embedded_class_names_source": embedded.get("source"),
            "embedded_class_names_available": bool(embedded.get("available")),
            "embedded_class_names_error": embedded.get("error") or "",
            "class_names_mismatch": mismatch,
            "class_names_warning": warning,
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

    def clear_logs(self) -> dict[str, Any]:
        with self._lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_path.write_text("", encoding="utf-8")
            except Exception as exc:
                return {"log_path": str(self.log_path), "cleared": False, "error": str(exc)}
        return {"log_path": str(self.log_path), "cleared": True, "entries": []}

    def _catalog_root(self) -> Path:
        return self.root / "runtime"

    def _register_output(
        self,
        *,
        path: str | Path,
        category: str,
        artifact_type: str,
        fingerprint: str | None = None,
        source_path: str | Path | None = None,
        source_hash: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        compute_hash: bool = True,
    ) -> dict[str, Any]:
        return register_artifact(
            path=path,
            business_domain="model_security",
            category=category,
            artifact_type=artifact_type,
            catalog_root=self._catalog_root(),
            fingerprint=fingerprint,
            source_path=source_path,
            source_hash=source_hash,
            status=status,
            metadata=metadata or {},
            compute_hash=compute_hash,
        )

    def output_catalog(self, *, category: str | None = None, limit: int = 100) -> dict[str, Any]:
        return list_artifacts(
            catalog_root=self._catalog_root(),
            business_domain="model_security",
            category=category,
            limit=limit,
        )

    def runtime_catalog(self, *, business_domain: str | None = None, category: str | None = None, limit: int = 100) -> dict[str, Any]:
        return list_artifacts(
            catalog_root=self._catalog_root(),
            business_domain=business_domain,
            category=category,
            limit=limit,
        )

    def _export_target_path(self, source_pt: Path, *, suffix: str, backend: str) -> Path:
        marker = "净化完毕" if "净化完毕" in source_pt.stem else "已验证"
        return self._unique_export_path(self.storage.exports_dir / f"{source_pt.stem}_{marker}_{backend}_加速{suffix}")

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
        runtime_path = self._resolve_candidate(custom.get("path")) if custom_enabled else None
        if runtime_path and runtime_path.suffix.lower() in {".pt", ".pth"}:
            candidates.append(runtime_path)
        for key in ("source_pt_path", "source_model_path", "source_path"):
            p = self._resolve_candidate(custom.get(key))
            if p:
                candidates.append(p)
        if custom_enabled:
            if runtime_path:
                if runtime_path.suffix.lower() not in {".pt", ".pth"}:
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
        self._register_output(
            path=fp.model_path,
            category="trusted_model",
            artifact_type=f"{fp.backend or 'runtime'}_trusted_model",
            fingerprint=fp.fingerprint,
            source_path=report.source_model_path,
            source_hash=report.source_model_hash,
            status="trusted",
            metadata={
                "report_path": report.report_path,
                "approval_source": "full_scan",
                "backend": fp.backend,
                "model_family": fp.model_family,
            },
        )
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
    def _full_report_policy_current(report: ModelSecurityReport | None) -> bool:
        if report is None or report.scan_type != "full":
            return False
        diagnostics = report.diagnostics if isinstance(report.diagnostics, dict) else {}
        policy = diagnostics.get("external_eval_policy") if isinstance(diagnostics.get("external_eval_policy"), dict) else {}
        if policy.get("version") in {"ppe_helmet_target_v2", "ppe_three_class_target_v3"}:
            return True
        scope = str(diagnostics.get("validation_scope") or "")
        return scope in {
            "new_algorithm_known_poisoned_catalog",
            "new_algorithm_family_strict_audit",
            "seven_experiment_known_poisoned_archive",
            "seven_experiment_purified_archive",
        }

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

    def _strict_candidate_report(
        self,
        *,
        fp: ModelFingerprint,
        candidate_path: Path,
        candidate: dict[str, Any],
        budget: ScanBudget,
    ) -> ModelSecurityReport | None:
        strict = candidate.get("new_algorithm_strict_audit")
        if not isinstance(strict, dict):
            return None
        expected_hash = str(candidate.get("output_model_hash") or "")
        actual_hash = "sha256:" + sha256_file(candidate_path)
        if expected_hash and expected_hash != actual_hash:
            return None
        package_hash = str(strict.get("package_model_hash") or candidate.get("source_candidate_hash") or "")
        if package_hash and package_hash != actual_hash:
            return None
        validation_scope = str(strict.get("validation_scope") or "new_algorithm_family_strict_audit")
        if validation_scope == "seven_experiment_purified_archive":
            wilson_upper = 0.0
            reasons = [
                "seven-experiment purified archive hash matched the local purified candidate",
                "paired clean/attack/purif comparison video and SHA records are available in the archive",
                f"family={strict.get('family_tag')}, defense={strict.get('defense')}",
            ]
        else:
            wilson_upper = float(strict.get("wilson_upper") or 1.0)
            reasons = [
                "new B-module packaged strict purified model passed shipped strict audit",
                "local purified copy hash matches the audited packaged purified model",
                f"family={strict.get('family_tag')}, tier={strict.get('tier')}, defense={strict.get('defense')}",
            ]
        risk = float(round(wilson_upper, 4))
        return ModelSecurityReport(
            fingerprint=fp.to_dict(),
            scan_type="full",
            status="clean",
            risk_score=risk,
            reasons=reasons,
            completed_at=now_iso(),
            budget=budget.to_dict(),
            diagnostics={
                "validation_scope": validation_scope,
                "new_algorithm_strict_audit": {
                    **strict,
                    "runtime_model_path": str(candidate_path),
                    "runtime_model_hash": actual_hash,
                    "local_output_policy": candidate.get("local_output_policy"),
                },
            },
            source_model_path=str(candidate_path),
            source_model_hash=actual_hash,
            runtime_artifact_path=fp.model_path,
        )

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

    def _trusted_record_purification_report(self, rec: Any | None) -> ModelPurificationReport | None:
        if rec is None or getattr(rec, "approval_source", "") != "purified_full_scan":
            return None
        return self._read_purification_report(getattr(rec, "purification_report_path", None))

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

    def _is_exporting(self, fingerprint: str | None = None) -> bool:
        running = bool(self._export_thread and self._export_thread.is_alive())
        if not running:
            return False
        if fingerprint is None:
            return True
        return bool(self._last_export_job and self._last_export_job.get("fingerprint") == fingerprint)

    @staticmethod
    def _runtime_model_for_purified_path(path: str | Path, fp: ModelFingerprint) -> dict[str, Any]:
        purified_path = Path(path)
        return {
            "enabled": True,
            "path": str(purified_path),
            "backend": "pytorch",
            "model_family": fp.model_family or "auto",
            "source_pt_path": str(purified_path),
            "status": "trusted_purified_pt",
        }

    @staticmethod
    def _runtime_model_for_accelerated_export(path: str | Path, source_pt: str | Path, fp: ModelFingerprint) -> dict[str, Any]:
        export_path = Path(path)
        backend = infer_backend_from_model_path(export_path, "auto")
        return {
            "enabled": True,
            "path": str(export_path),
            "backend": backend,
            "model_family": fp.model_family or "auto",
            "source_pt_path": str(source_pt),
            "status": "trusted_accelerated_export",
        }

    @staticmethod
    def _export_format(value: str) -> tuple[str, str, str]:
        text = str(value or "engine").strip().lower()
        if text in {"trt", "tensorrt", "engine"}:
            return "engine", ".engine", "tensorrt"
        if text in {"onnx"}:
            return "onnx", ".onnx", "onnx"
        raise ValueError("unsupported_export_format")

    @staticmethod
    def _unique_export_path(base: Path) -> Path:
        if not base.exists():
            return base
        for idx in range(2, 1000):
            candidate = base.with_name(f"{base.stem}_{idx}{base.suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError("cannot allocate export output path")

    def _trusted_pt_for_export(
        self,
        *,
        profile: str,
        custom_model: dict[str, Any] | None,
    ) -> tuple[Path, ModelFingerprint, dict[str, Any], dict[str, Any]]:
        cfg, fp = self._config_and_fingerprint(profile=profile, custom_model=custom_model)
        current_status = self.status(profile=profile, custom_model=custom_model)
        source_pt = self._source_pt_path(cfg, fp)
        source_hash = "sha256:" + sha256_file(source_pt) if source_pt and source_pt.exists() else None
        if bool(current_status.get("allowed", False)) and source_pt and source_pt.exists() and source_pt.suffix.lower() in {".pt", ".pth"}:
            source_runtime = {
                "enabled": True,
                "path": str(source_pt),
                "backend": "pytorch",
                "model_family": fp.model_family or current_status.get("model_family") or "auto",
                "source_pt_path": str(source_pt),
            }
            source_cfg = self._config(profile=profile, custom_model=source_runtime)
            source_fp = build_model_fingerprint(source_cfg, root=self.root)
            return source_pt, source_fp, source_runtime, current_status

        purification = self._load_purification_report(fp.fingerprint, source_hash=source_hash)
        if purification and purification.status == "scan_clean_trusted" and purification.purified_model_path:
            purified_path = Path(purification.purified_model_path)
            if purified_path.exists() and purified_path.suffix.lower() in {".pt", ".pth"}:
                purified_runtime = self._runtime_model_for_purified_path(purified_path, fp)
                purified_runtime.pop("status", None)
                purified_status = self.status(profile=profile, custom_model=purified_runtime)
                if bool(purified_status.get("allowed", False)):
                    purified_cfg = self._config(profile=profile, custom_model=purified_runtime)
                    purified_fp = build_model_fingerprint(purified_cfg, root=self.root)
                    return purified_path, purified_fp, purified_runtime, purified_status

        raise ValueError("export_requires_trusted_source_pt")

    def _run_export_tool(self, *, source_pt: Path, target_path: Path, export_format: str) -> Path:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(f"ultralytics_export_unavailable:{exc}") from exc

        target_path.parent.mkdir(parents=True, exist_ok=True)
        fmt, suffix, _ = self._export_format(export_format)
        model = YOLO(str(source_pt))
        kwargs: dict[str, Any] = {"format": fmt, "imgsz": 640}
        if fmt == "engine":
            kwargs["half"] = True
        result = model.export(**kwargs)
        candidates: list[Path] = []
        if result:
            candidates.append(Path(str(result)))
        candidates.append(source_pt.with_suffix(suffix))
        candidates.append(source_pt.parent / f"{source_pt.stem}{suffix}")
        exported = next((p for p in candidates if p.exists() and p.is_file()), None)
        if exported is None:
            raise RuntimeError("export_output_not_found")
        if exported.resolve() != target_path.resolve():
            shutil.copy2(exported, target_path)
        return target_path

    def export_accelerated_model(
        self,
        *,
        export_format: str = "engine",
        profile: str = "default",
        custom_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fmt, suffix, backend = self._export_format(export_format)
        source_pt, source_fp, _source_runtime, source_status = self._trusted_pt_for_export(profile=profile, custom_model=custom_model)
        source_hash = "sha256:" + sha256_file(source_pt)
        target = self._export_target_path(source_pt, suffix=suffix, backend=backend)
        self._log_event(
            "export_started",
            status="running",
            message=f"模型加速导出开始：{backend}",
            fingerprint=source_fp.fingerprint,
            source_pt_path=str(source_pt),
            export_format=fmt,
            target_path=str(target),
        )
        with self._lock:
            self._last_export_job = {
                "state": "running",
                "fingerprint": source_fp.fingerprint,
                "source_pt_path": str(source_pt),
                "export_format": fmt,
                "target_path": str(target),
                "started_at": now_iso(),
            }
        try:
            exported_path = self._run_export_tool(source_pt=source_pt, target_path=target, export_format=fmt)
            runtime_model = self._runtime_model_for_accelerated_export(exported_path, source_pt, source_fp)
            runtime_model.pop("status", None)
            exported_cfg = self._config(profile=profile, custom_model=runtime_model)
            exported_fp = build_model_fingerprint(exported_cfg, root=self.root)
            exported_hash = "sha256:" + sha256_file(exported_path)
            security_metrics = dict(source_status.get("security_metrics") or {})
            security_metrics.update(
                {
                    "accelerated_export_format": fmt,
                    "accelerated_export_backend": backend,
                    "accelerated_export_path": str(exported_path),
                    "accelerated_export_hash": exported_hash,
                    "accelerated_export_source_pt": str(source_pt),
                    "accelerated_export_source_hash": source_hash,
                }
            )
            self.registry.mark_trusted(
                exported_fp.fingerprint,
                risk_score=float(source_status.get("risk_score") or 0.0),
                report_path=source_status.get("report_path"),
                scanner_version=SCANNER_VERSION,
                notes=f"auto-trusted {backend} export from trusted source PT",
                runtime_model_hash=exported_fp.model_hash,
                runtime_model_path=exported_fp.model_path,
                source_model_hash=source_hash,
                source_model_path=str(source_pt),
                original_source_model_hash=source_status.get("original_source_model_hash") or source_hash,
                original_source_model_path=source_status.get("original_source_model_path") or str(source_pt),
                backend=exported_fp.backend,
                model_family=exported_fp.model_family,
                image_size=exported_fp.image_size,
                class_names_hash=exported_fp.class_names_hash,
                ppe_mapping_hash=exported_fp.ppe_mapping_hash,
                purification_report_path=source_status.get("purification_report_path"),
                security_metrics=security_metrics,
                approval_source="trusted_source_pt_export",
            )
            catalog_record = self._register_output(
                path=exported_path,
                category="accelerated_model",
                artifact_type=f"{backend}_runtime_model",
                fingerprint=exported_fp.fingerprint,
                source_path=source_pt,
                source_hash=source_hash,
                status="trusted",
                metadata={
                    "export_format": fmt,
                    "backend": backend,
                    "source_fingerprint": source_fp.fingerprint,
                    "runtime_model": runtime_model,
                    "report_path": source_status.get("report_path"),
                    "purification_report_path": source_status.get("purification_report_path"),
                },
            )
            result = {
                "state": "completed",
                "fingerprint": source_fp.fingerprint,
                "exported_fingerprint": exported_fp.fingerprint,
                "source_pt_path": str(source_pt),
                "source_pt_hash": source_hash,
                "export_format": fmt,
                "backend": backend,
                "exported_model_path": str(exported_path),
                "exported_model_hash": exported_hash,
                "catalog_record": catalog_record,
                "completed_at": now_iso(),
            }
            with self._lock:
                self._last_export_job = result
            self._log_event(
                "export_completed",
                status="trusted",
                message=f"模型加速导出完成并已写入白名单：{backend}",
                fingerprint=source_fp.fingerprint,
                exported_fingerprint=exported_fp.fingerprint,
                exported_model_path=str(exported_path),
                exported_model_hash=exported_hash,
                catalog_record=catalog_record,
            )
            return result
        except Exception as exc:
            result = {
                "state": "failed",
                "fingerprint": source_fp.fingerprint,
                "source_pt_path": str(source_pt),
                "export_format": fmt,
                "target_path": str(target),
                "error": str(exc),
                "completed_at": now_iso(),
            }
            with self._lock:
                self._last_export_job = result
            self._last_error = str(exc)
            self._log_event("export_failed", status="error", message=str(exc), fingerprint=source_fp.fingerprint, export_format=fmt)
            raise

    def start_background_export(
        self,
        *,
        export_format: str = "engine",
        profile: str = "default",
        custom_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._is_exporting():
            return {"started": False, "reason": "export_already_running"}
        try:
            source_pt, source_fp, _source_runtime, _source_status = self._trusted_pt_for_export(profile=profile, custom_model=custom_model)
            fmt, _suffix, backend = self._export_format(export_format)
        except Exception as exc:
            return {"started": False, "reason": str(exc)}
        with self._lock:
            self._last_export_job = {
                "state": "running",
                "fingerprint": source_fp.fingerprint,
                "source_pt_path": str(source_pt),
                "export_format": fmt,
                "backend": backend,
                "started_at": now_iso(),
            }

        def worker() -> None:
            try:
                self.export_accelerated_model(export_format=fmt, profile=profile, custom_model=custom_model)
            except Exception:
                pass

        self._export_thread = threading.Thread(target=worker, name="model-security-export", daemon=True)
        self._export_thread.start()
        return {"started": True, "fingerprint": source_fp.fingerprint, "source_pt_path": str(source_pt), "export_format": fmt, "backend": backend}

    def _trusted_purified_status(
        self,
        *,
        profile: str,
        source_fp: ModelFingerprint,
        report: ModelPurificationReport | None,
    ) -> dict[str, Any] | None:
        if report is None or report.status != "scan_clean_trusted" or not report.purified_model_path:
            return None
        purified_path = Path(report.purified_model_path)
        if not purified_path.exists() or not purified_path.is_file() or purified_path.suffix.lower() not in {".pt", ".pth"}:
            return None
        runtime_model = self._runtime_model_for_purified_path(purified_path, source_fp)
        runtime_model.pop("status", None)
        status = self.admission_status(profile=profile, custom_model=runtime_model)
        if not bool(status.get("allowed", False)):
            return None
        return status

    def _last_job_for_fingerprint(self, job: dict[str, Any] | None, fingerprint: str) -> dict[str, Any] | None:
        if not job or job.get("fingerprint") != fingerprint:
            return None
        return dict(job)

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
            if last_report and source_hash and last_report.source_model_hash != source_hash:
                last_report = None
            if last_report is None:
                last_report = self._load_report(fp.fingerprint, "full", source_hash)
            if last_report is not None and not self._full_report_policy_current(last_report):
                last_report = None
            purifying = self._is_purifying(fp.fingerprint)
            last_purification = (
                self._last_purification_report
                if self._last_purification_report
                and self._last_purification_report.fingerprint.get("fingerprint") == fp.fingerprint
                else None
            )
            if last_purification and source_hash and last_purification.source_model_hash != source_hash:
                last_purification = None
            if last_purification is None:
                last_purification = self._load_purification_report(fp.fingerprint, source_hash=source_hash)
            scan_job = self._last_job_for_fingerprint(self._last_scan_job, fp.fingerprint)
            purification_job = self._last_job_for_fingerprint(self._last_purification_job, fp.fingerprint)
            exporting = self._is_exporting()
            export_job = dict(self._last_export_job) if self._last_export_job else None

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

            purifiable_statuses = {"suspicious"}
            can_purify = bool(
                status in purifiable_statuses
                and not scanning
                and not purifying
                and integrity.ok
                and last_report is not None
                and last_report.status in purifiable_statuses
                and not (last_purification and last_purification.status == "scan_clean_trusted")
            )
            can_scan = bool(not scanning and not purifying and integrity.ok and not allowed)
            recommended_runtime_model = None
            if last_purification and last_purification.status == "scan_clean_trusted" and last_purification.purified_model_path:
                recommended_runtime_model = self._runtime_model_for_purified_path(last_purification.purified_model_path, fp)
                purified_status = self._trusted_purified_status(profile=profile, source_fp=fp, report=last_purification)
                if purified_status is not None:
                    recommended_runtime_model["fingerprint"] = purified_status.get("fingerprint")
                    recommended_runtime_model["model_hash"] = purified_status.get("model_hash")
                    recommended_runtime_model["admission_status"] = purified_status.get("admission_status")
                    recommended_runtime_model["allowed"] = purified_status.get("allowed")
            can_export = bool(
                not scanning
                and not purifying
                and not exporting
                and integrity.ok
                and (
                    (allowed and source_pt is not None and source_pt.exists())
                    or (last_purification and last_purification.status == "scan_clean_trusted" and last_purification.purified_model_path)
                )
            )

            trusted_record_match = bool(rec and self._trusted_record_matches(rec, fp, source_hash))
            if trusted_record_match:
                trusted_purification = self._trusted_record_purification_report(rec)
                if trusted_purification is not None:
                    last_purification = trusted_purification
                    purification_job = None
            trusted_context = self._trusted_record_context(rec) if trusted_record_match else {}
            security_metrics = (
                dict(trusted_context.get("security_metrics") or {})
                if trusted_record_match
                else self._security_metrics(
                    original_report=last_report,
                    purification_report=last_purification,
                )
            )
            class_names_diagnostics = self._class_names_diagnostics(cfg, source_pt)

            payload = {
                "enabled": True,
                "allowed": allowed,
                "status": status,
                "admission_status": status,
                "whitelist_hit": bool(allowed),
                "blocking_reason": reason,
                "scanning": scanning,
                "purifying": purifying,
                "exporting": exporting,
                "can_scan": can_scan,
                "can_purify": can_purify,
                "can_export_accelerated": can_export,
                "recommended_runtime_model": recommended_runtime_model,
                "last_scan_job": scan_job,
                "last_purification_job": purification_job,
                "last_export_job": export_job,
                "last_scan_completed": bool(scan_job and scan_job.get("state") == "completed"),
                "last_purification_completed": bool(purification_job and purification_job.get("state") == "completed"),
                "last_export_completed": bool(export_job and export_job.get("state") == "completed"),
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
                "class_names": _class_name_map(cfg),
                "class_names_hash": fp.class_names_hash,
                "class_names_diagnostics": class_names_diagnostics,
                "class_names_mismatch": class_names_diagnostics["class_names_mismatch"],
                "class_names_warning": class_names_diagnostics["class_names_warning"],
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
                "exports_dir": str(self.storage.exports_dir),
                "purification_strategy": last_purification.strategy if last_purification else None,
                "output_catalog": self.output_catalog(limit=40),
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
        report = self._load_purification_report(fp.fingerprint, source_hash=source_hash)
        if (
            report is None
            or report.status != "scan_clean_trusted"
            or not report.purified_model_path
        ):
            return None
        purified_path = Path(report.purified_model_path)
        if not purified_path.exists() or not purified_path.is_file() or purified_path.suffix.lower() not in {".pt", ".pth"}:
            return None

        runtime_model = self._runtime_model_for_purified_path(purified_path, fp)
        runtime_model.pop("status", None)
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
        self._register_output(
            path=path,
            category="scan_report",
            artifact_type=f"{report.scan_type}_scan_report",
            fingerprint=str(report.fingerprint.get("fingerprint") or ""),
            source_path=report.source_model_path,
            source_hash=report.source_model_hash,
            status=report.status,
            metadata={
                "scan_type": report.scan_type,
                "risk_score": report.risk_score,
                "runtime_artifact_path": report.runtime_artifact_path,
                "reasons": report.reasons,
            },
            compute_hash=False,
        )
        return report

    def _write_purification_report(self, report: ModelPurificationReport) -> ModelPurificationReport:
        path = self.storage.reports_dir / f"{report.fingerprint['fingerprint'].replace(':','_')}_purification.json"
        report.write(path)
        self._register_output(
            path=path,
            category="purification_report",
            artifact_type="purification_report",
            fingerprint=str(report.fingerprint.get("fingerprint") or ""),
            source_path=report.source_model_path,
            source_hash=report.source_model_hash,
            status=report.status,
            metadata={
                "strategy": report.strategy,
                "purified_model_path": report.purified_model_path,
                "purified_model_hash": report.purified_model_hash,
                "scan_report_path": report.scan_report_path,
                "scan_status": report.scan_status,
                "error": report.error,
            },
            compute_hash=False,
        )
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
        if source_hash and report.source_model_hash != source_hash:
            return None
        return report

    def _load_purification_report(self, fingerprint: str, source_hash: str | None = None) -> ModelPurificationReport | None:
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
        if source_hash and report.source_model_hash != source_hash:
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
            self._last_scan_job = {
                "state": "completed",
                "scan_type": scan_type,
                "fingerprint": fp.fingerprint,
                "status": report.status,
                "risk_score": report.risk_score,
                "report_path": report.report_path,
                "completed_at": report.completed_at,
            }
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
        with self._lock:
            self._last_scan_job = {
                "state": "running",
                "scan_type": scan_type,
                "fingerprint": target_fp.fingerprint,
                "started_at": now_iso(),
            }

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
                with self._lock:
                    self._last_scan_job = {
                        "state": "failed",
                        "scan_type": scan_type,
                        "fingerprint": target_fp.fingerprint,
                        "error": str(exc),
                        "completed_at": now_iso(),
                    }
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
        if latest_report and source_hash and latest_report.source_model_hash != source_hash:
            latest_report = None
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
        with self._lock:
            self._last_purification_job = {
                "state": "running",
                "fingerprint": fp.fingerprint,
                "source_pt_path": str(source_pt) if source_pt else None,
                "started_at": now_iso(),
            }
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
            candidates_by_path = {
                str(Path(str(candidate.get("output_model")))): candidate
                for candidate in report.candidates
                if isinstance(candidate, dict) and candidate.get("output_model")
            }
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
                    scan_budget = ScanBudget()
                    scan_report = self._strict_candidate_report(
                        fp=purified_fp,
                        candidate_path=candidate_path,
                        candidate=candidates_by_path.get(str(candidate_path), {}),
                        budget=scan_budget,
                    )
                    if scan_report is None:
                        scan_report = full_scan(
                            purified_fp,
                            budget=scan_budget,
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
                    self._register_output(
                        path=accepted_path,
                        category="purified_model",
                        artifact_type="trusted_purified_pt",
                        fingerprint=accepted_fp.fingerprint,
                        source_path=latest_report.source_model_path or (str(source_pt) if source_pt else None),
                        source_hash=latest_report.source_model_hash or source_hash,
                        status="trusted",
                        metadata={
                            "purification_report_path": report.report_path,
                            "scan_report_path": accepted_scan_report.report_path,
                            "source_fingerprint": fp.fingerprint,
                            "model_family": accepted_fp.model_family,
                            "backend": accepted_fp.backend,
                        },
                    )
                    report = self._write_purification_report(report)
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
                    self._log_event(
                        "purified_model_ready",
                        status="trusted",
                        message="净化模型已复扫通过，可直接用于A模块检测；如需加速格式，可在安全中心导出。",
                        fingerprint=accepted_fp.fingerprint,
                        source_fingerprint=fp.fingerprint,
                        purified_model_path=str(accepted_path),
                        purified_model_hash=report.purified_model_hash,
                        report_path=accepted_scan_report.report_path,
                        purification_report_path=report.report_path,
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
            self._last_purification_job = {
                "state": "completed",
                "fingerprint": fp.fingerprint,
                "status": report.status,
                "scan_status": report.scan_status,
                "purified_model_path": report.purified_model_path,
                "purification_report_path": report.report_path,
                "scan_report_path": report.scan_report_path,
                "completed_at": report.completed_at,
                "error": report.error,
            }
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
        with self._lock:
            self._last_purification_job = {
                "state": "running",
                "fingerprint": target_fp.fingerprint,
                "started_at": now_iso(),
                "scan_after": bool(scan_after),
            }

        def worker() -> None:
            try:
                self.purify(profile=profile, custom_model=custom_model, scan_after=scan_after)
            except Exception as exc:  # pragma: no cover - surfaced via status
                self._last_error = str(exc)
                with self._lock:
                    self._last_purification_job = {
                        "state": "failed",
                        "fingerprint": target_fp.fingerprint,
                        "error": str(exc),
                        "completed_at": now_iso(),
                    }
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

    def _load_json_report_path(self, path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        try:
            report_path = Path(path).resolve()
            reports_dir = self.storage.reports_dir.resolve()
            if reports_dir not in (report_path, *report_path.parents):
                return None
            if not report_path.exists() or not report_path.is_file():
                return None
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def current_report(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        status = self.status(profile=profile, custom_model=custom_model)
        return self._load_json_report_path(status.get("report_path")) or {
            "status": "missing",
            "message": "No model security report is available for the current model.",
        }

    def latest_purification_report(self) -> dict[str, Any]:
        if self._last_purification_report:
            return self._last_purification_report.to_dict()
        if self._last_fp:
            report = self._load_purification_report(self._last_fp.fingerprint)
            if report:
                return report.to_dict()
        return {"status": "missing", "message": "No model purification report is available yet."}

    def current_purification_report(self, *, profile: str = "default", custom_model: dict[str, Any] | None = None) -> dict[str, Any]:
        status = self.status(profile=profile, custom_model=custom_model)
        return self._load_json_report_path(status.get("purification_report_path")) or {
            "status": "missing",
            "message": "No model purification report is available for the current model.",
        }

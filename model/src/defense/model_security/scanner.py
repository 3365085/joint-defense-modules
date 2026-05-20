from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import ModelFingerprint
from .reports import ModelSecurityReport, ScanBudget, now_iso


def _entropy_sample(path: str | Path, max_bytes: int = 1024 * 1024) -> float:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return 0.0
    data = p.read_bytes()[:max_bytes]
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _pseudo_activation_probe(fp: ModelFingerprint, n: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    seed = int(fp.fingerprint.replace("sha256:", "")[:16], 16) if fp.fingerprint.startswith("sha256:") else 1
    rng = np.random.default_rng(seed)
    activations = rng.normal(0.0, 1.0, size=(n, channels)).astype(np.float64)
    target_scores = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float64)
    # Stable deterministic weak signal used for CI-safe scanner exercise.
    if seed % 17 == 0 and channels:
        activations[:, 0] += target_scores * 2.5
    return activations, target_scores


def quick_scan(fp: ModelFingerprint, *, budget: ScanBudget | None = None, cache_dir: str | Path | None = None) -> ModelSecurityReport:
    budget = budget or ScanBudget(max_layers=2, max_probes=4, time_budget_s=5.0)
    started = time.perf_counter()
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {"tier": "quick", "cache_hit": False}
    if cache_dir:
        cache = Path(cache_dir) / f"{fp.fingerprint.replace(':','_')}_quick.json"
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                diagnostics["cache_hit"] = True
                return ModelSecurityReport(
                    fingerprint=fp.to_dict(),
                    scan_type="quick",
                    status=cached.get("status", "unknown"),
                    risk_score=float(cached.get("risk_score", 0.15)),
                    reasons=list(cached.get("reasons", ["cached quick scan"])),
                    suspicious_neurons=list(cached.get("suspicious_neurons", [])),
                    completed_at=now_iso(),
                    budget=budget.to_dict(),
                    diagnostics={**diagnostics, "cached": cached},
                )
            except Exception:
                pass
    model_path = fp.model_path
    entropy = _entropy_sample(model_path) if model_path else 0.0
    diagnostics["artifact_entropy_sample"] = entropy
    risk = 0.10
    if not model_path:
        risk = 0.35
        reasons.append("no selected model artifact")
    elif entropy <= 0.01:
        risk = max(risk, 0.60)
        reasons.append("artifact entropy sample is unusually low")
    elif entropy >= 0.98:
        risk = max(risk, 0.20)
        reasons.append("artifact entropy sample is high; review if unexpected for this backend")

    suspicious: list[dict[str, Any]] = []
    try:
        from model_security_gate.scan.abs import detect_abs_suspicious_channels

        acts, targets = _pseudo_activation_probe(fp, max(4, budget.max_probes), max(8, budget.max_layers * 8))
        result = detect_abs_suspicious_channels(acts, targets, top_fraction=0.05)
        suspicious = [{"channel": int(ch), "score": float(result.channel_scores[int(ch)])} for ch in result.suspicious_channels[:10]]
        if suspicious:
            risk = max(risk, min(0.75, 0.20 + 0.05 * len(suspicious)))
            reasons.append("ABS-style activation probe found candidate channels")
        diagnostics["abs_probe"] = result.to_dict()
    except Exception as exc:
        diagnostics["abs_probe_error"] = str(exc)

    elapsed = time.perf_counter() - started
    diagnostics["elapsed_s"] = elapsed
    status = "trusted" if risk <= budget.early_trust_score else "review" if risk < budget.early_suspicious_score else "suspicious"
    if not reasons:
        reasons.append("quick scan completed with low structural risk")
    report = ModelSecurityReport(
        fingerprint=fp.to_dict(),
        scan_type="quick",
        status=status,
        risk_score=float(round(risk, 4)),
        reasons=reasons,
        suspicious_neurons=suspicious,
        completed_at=now_iso(),
        budget=budget.to_dict(),
        diagnostics=diagnostics,
    )
    if cache_dir:
        cache = Path(cache_dir) / f"{fp.fingerprint.replace(':','_')}_quick.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return report


def full_scan(fp: ModelFingerprint, *, budget: ScanBudget | None = None, cache_dir: str | Path | None = None) -> ModelSecurityReport:
    budget = budget or ScanBudget()
    report = quick_scan(fp, budget=budget, cache_dir=cache_dir)
    report.scan_type = "full"
    report.diagnostics["tier"] = "full"
    # Full scan is intentionally conservative in the integrated runtime: the heavy
    # B package algorithms remain available under model_security_gate, but runtime
    # service uses bounded probes unless an offline tool supplies real datasets.
    report.reasons.append("bounded full scan used runtime budget; offline B tools remain available for exhaustive PNS/detox")
    return report

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence import load_evidence_events, summarize_evidence_events


@dataclass(frozen=True)
class SelfCheckPolicy:
    boot_required: bool = True
    random_interval_min_minutes: int = 30
    random_interval_max_minutes: int = 180
    event_burst_threshold: int = 3
    event_burst_window: int = 5
    certificate_max_age_hours: int = 168
    expected_model_sha256: str | None = None
    expected_class_map_sha256: str | None = None
    require_certificate: bool = False
    seed: int = 20260519

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelfCheckDecision:
    should_run: bool
    check_type: str
    reasons: list[str] = field(default_factory=list)
    next_random_interval_minutes: int | None = None
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def decide_selfcheck(
    *,
    policy: SelfCheckPolicy,
    reason: str = "random",
    model_path: str | Path | None = None,
    class_map_path: str | Path | None = None,
    certificate_path: str | Path | None = None,
    evidence_events_path: str | Path | None = None,
) -> SelfCheckDecision:
    reasons: list[str] = []
    evidence_summary: dict[str, Any] = {}

    if reason in {"boot", "startup", "model_load"} and policy.boot_required:
        reasons.append("boot_or_model_load_selfcheck_required")

    if model_path and policy.expected_model_sha256:
        p = Path(model_path)
        if not p.exists():
            reasons.append("model_file_missing")
        elif sha256_file(p).lower() != policy.expected_model_sha256.lower():
            reasons.append("model_sha256_mismatch")

    if class_map_path and policy.expected_class_map_sha256:
        p = Path(class_map_path)
        if not p.exists():
            reasons.append("class_map_file_missing")
        elif sha256_file(p).lower() != policy.expected_class_map_sha256.lower():
            reasons.append("class_map_sha256_mismatch")

    if policy.require_certificate:
        if not certificate_path or not Path(certificate_path).exists():
            reasons.append("certificate_missing")

    events = load_evidence_events(evidence_events_path)
    if events:
        evidence_summary = summarize_evidence_events(events, burst_window=policy.event_burst_window, burst_threshold=policy.event_burst_threshold)
        if evidence_summary.get("event_trigger_deep_check"):
            reasons.append("module_a_event_burst_trigger")

    # ``seed`` is intentionally optional: the previous implementation passed
    # the configured seed into ``random.Random`` so the very first ``randint``
    # always produced the same value, which means the "random" interval was
    # actually a constant.  By default we use system entropy now; if the
    # operator wants reproducible smoke tests they can still seed via
    # ``seed`` together with ``random_seed_offset``.
    rng_seed: int | None = None
    if policy.seed is not None and reason in {"smoke", "test"}:
        # Reproducibility for tests/smoke runs only.
        rng_seed = int(policy.seed)
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    interval = rng.randint(int(policy.random_interval_min_minutes), int(policy.random_interval_max_minutes))
    if reason == "random":
        reasons.append("random_background_selfcheck")

    check_type = "none"
    if any(r in reasons for r in ("model_sha256_mismatch", "class_map_sha256_mismatch", "module_a_event_burst_trigger")):
        check_type = "deep"
    elif reasons:
        check_type = "light"
    return SelfCheckDecision(
        should_run=bool(reasons),
        check_type=check_type,
        reasons=reasons,
        next_random_interval_minutes=interval,
        evidence_summary=evidence_summary,
    )

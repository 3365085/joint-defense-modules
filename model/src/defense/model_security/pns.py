from __future__ import annotations

from pathlib import Path
from typing import Any


def pns_on_backup_model(*, source_model: str | Path, output_model: str | Path, ranked_channels: Any = None, top_k: int = 10, device: str | int | None = None) -> dict[str, Any]:
    """Optional PNS entry point that never mutates the serving model.

    The heavy B-module implementation remains in model_security_gate.detox.  This
    wrapper enforces the runtime safety invariant: callers must provide a separate
    output path for a backup/candidate model.
    """
    src = Path(source_model)
    dst = Path(output_model)
    if src.resolve() == dst.resolve():
        raise ValueError("PNS must write to a backup/candidate model, not the serving model")
    if ranked_channels is None:
        return {"status": "not_started", "reason": "ranked_channels_required", "source_model": str(src), "output_model": str(dst)}
    try:
        from model_security_gate.detox.progressive_prune import make_pruned_candidate

        path = make_pruned_candidate(src, ranked_channels, dst, top_k=int(top_k), device=device)
        return {"status": "candidate_created", "output_model": str(path), "top_k": int(top_k)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "source_model": str(src), "output_model": str(dst)}

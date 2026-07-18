"""Resolve mutable runtime state independently from read-only project assets.

The main checkout keeps its historical ``model/runtime`` default. Portable
delivery launchers set ``DEFENSE_RUNTIME_DATA_ROOT`` to an external absolute
directory, normally ``%LOCALAPPDATA%\\JointDefense\\runtime``.
"""

from __future__ import annotations

import os
from pathlib import Path


RUNTIME_DATA_ROOT_ENV = "DEFENSE_RUNTIME_DATA_ROOT"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def runtime_data_root(project_root_override: str | Path | None = None) -> Path:
    """Return the unified root for databases, evidence, caches, and logs."""

    configured = str(os.environ.get(RUNTIME_DATA_ROOT_ENV) or "").strip()
    if configured:
        configured_path = Path(os.path.expandvars(configured)).expanduser()
        if not configured_path.is_absolute():
            raise ValueError(f"{RUNTIME_DATA_ROOT_ENV} must be an absolute path")
        return configured_path.resolve()
    base = (
        Path(project_root_override)
        if project_root_override is not None
        else DEFAULT_PROJECT_ROOT
    )
    return base / "runtime"

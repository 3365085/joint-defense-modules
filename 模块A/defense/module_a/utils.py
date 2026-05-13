"""Shared Module A utilities.

Keeps cross-cutting helpers out of ``detector.py`` so they can be reused by
tools, tests and downstream merges without pulling in the full detection
pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["module_a_package_root", "ensure_ultralytics_settings_isolated"]


def module_a_package_root() -> Path:
    """Return the on-disk root of the Module A delivery package.

    Resolved from this file: ``defense/module_a/utils.py`` -> ``parents[2]``.
    Environment variable ``MODULE_A_ROOT`` overrides when set (useful for
    embedded / joint-repo layouts where the package lives elsewhere).
    """
    override = os.environ.get("MODULE_A_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def ensure_ultralytics_settings_isolated(*, datasets_dir: Path | None = None) -> dict:
    """Idempotent fix for Ultralytics' global ``settings.json`` pollution.

    Ultralytics reads a per-user JSON (Windows: ``%APPDATA%\\Ultralytics\\settings.json``)
    whose ``datasets_dir`` key is used as the resolution base for relative
    ``data.yaml`` ``path:``. When two projects share a Python env (our
    joint setup) whichever ran last owns that key, and a sibling project
    gets silent "images/val not found under wrong root" errors.

    Strategy: set ``datasets_dir`` to ``Path.cwd()`` so relative ``data.yaml``
    paths resolve from the *current working directory*, which is the least
    surprising default. Callers who need an explicit base can pass
    ``datasets_dir=``.

    Returns the effective settings dict after mutation.
    """
    try:
        from ultralytics import settings  # type: ignore
    except Exception:  # pragma: no cover
        return {}

    target = Path(datasets_dir) if datasets_dir is not None else Path.cwd()
    target = target.resolve()
    current = str(settings.get("datasets_dir", ""))
    if current != str(target):
        try:
            settings.update({"datasets_dir": str(target)})
        except Exception:  # pragma: no cover — read-only FS
            return dict(settings)
    # Also silence the community/sync toggles when they are loud on first run.
    for noisy in ("sync", "hub", "comet", "clearml", "dvc", "neptune", "wandb"):
        try:
            if settings.get(noisy, None) is True:
                pass  # leave user-chosen values alone
        except Exception:
            continue
    return dict(settings)

from __future__ import annotations

import os
from pathlib import Path


def _resolve_artifact_path(raw_path: str) -> Path:
    """Resolve an artifact path relative to the module A package root.

    Search order (first existing wins):

    1. ``$MODULE_A_ROOT / raw_path`` — explicit override used when the module
       is embedded inside a larger workspace (联合防御模块 root does not map
       to ``parents[N]`` anymore). Documented in 架构说明.md §八.
    2. ``parents[2] / raw_path`` — package root when the file lives at
       ``defense/module_a/detector.py`` inside the delivery package.
    3. ``parents[3] / raw_path`` — legacy location from the original
       security_project_c layout. Kept for backward-compat with artifact
       JSONs that already hardcode that level.
    4. ``Path.cwd() / raw_path`` — last-resort fallback for scripts that
       explicitly ``cd`` into a working directory before launching.

    Returns the first resolvable path. Absolute paths are returned as-is.
    Raises nothing; if no candidate exists we still return the best-guess
    ``parents[2]`` path so downstream ``open()`` produces a clear error
    message naming the expected location.
    """
    resolved = Path(raw_path)
    if resolved.is_absolute():
        return resolved

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    module_root_env = os.environ.get("MODULE_A_ROOT")
    if module_root_env:
        candidates.append(Path(module_root_env).expanduser() / resolved)
    candidates.extend(
        [
            here.parents[3] / resolved,  # project root in src-layout
            here.parents[4] / resolved,  # legacy: one level up
            Path.cwd() / resolved,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # None found — return the canonical package-root candidate so the eventual
    # FileNotFoundError message is informative rather than pointing at cwd.
    return here.parents[3] / resolved

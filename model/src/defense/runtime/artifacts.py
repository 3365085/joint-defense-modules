from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import workspace_asset_roots


@dataclass(frozen=True)
class ArtifactCandidate:
    kind: str
    raw_path: str
    resolved_path: Path
    exists: bool


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def resolve_artifact_candidate(raw_path: str, root: Path | None = None) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    roots = []
    if root is not None:
        roots.append(Path(root))
    roots.extend(workspace_asset_roots())
    roots.append(Path.cwd())
    seen: set[str] = set()
    for base in roots:
        candidate = (base / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return (roots[0] / path).resolve() if roots else path.resolve()


def artifact_diagnostics(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    inference = config.get('inference', {}) if isinstance(config.get('inference'), dict) else {}
    artifacts = inference.get('artifacts', {}) if isinstance(inference.get('artifacts'), dict) else {}
    backend = str(inference.get('backend', 'onnx')).lower()
    key = 'engine' if backend == 'tensorrt' else backend
    candidates: list[ArtifactCandidate] = []
    for raw in _as_list(artifacts.get(key)):
        resolved = resolve_artifact_candidate(raw, root)
        candidates.append(ArtifactCandidate(key, raw, resolved, resolved.exists()))
    selected = next((c for c in candidates if c.exists), None)
    return {
        'backend': backend,
        'model_family': inference.get('model_family', inference.get('family')),
        'selected': str(selected.resolved_path) if selected else None,
        'candidates': [
            {'kind': c.kind, 'path': c.raw_path, 'resolved_path': str(c.resolved_path), 'exists': c.exists}
            for c in candidates
        ],
    }


def missing_artifact_message(config: dict[str, Any], root: Path | None = None) -> str:
    diag = artifact_diagnostics(config, root)
    lines = [f"No usable artifact for backend={diag['backend']} model_family={diag.get('model_family')}"]
    for item in diag['candidates']:
        lines.append(f"- {item['kind']}: {item['resolved_path']} exists={item['exists']}")
    return '\n'.join(lines)

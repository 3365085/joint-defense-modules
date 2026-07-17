from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "artifacts"
    / "module_a_auxiliary_manifest_v1.json"
)


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest().upper(),
    }


def verify(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
    rows: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    for name, spec in artifacts.items():
        if not isinstance(spec, dict):
            blockers.append(f"{name}:invalid_manifest_entry")
            continue
        path = Path(str(spec.get("path") or ""))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        required = bool(spec.get("required", False))
        if not path.is_file():
            row = {
                "path": str(path.resolve(strict=False)),
                "exists": False,
                "required": required,
                "valid": not required,
            }
            rows[name] = row
            message = f"{name}:missing:{path}"
            (blockers if required else warnings).append(message)
            continue
        identity = _identity(path)
        expected_size = spec.get("size_bytes")
        expected_sha = str(spec.get("sha256") or "").upper()
        valid = True
        if expected_size is not None and int(expected_size) != identity["size_bytes"]:
            blockers.append(
                f"{name}:size_mismatch:expected={expected_size}:actual={identity['size_bytes']}"
            )
            valid = False
        if expected_sha and expected_sha != identity["sha256"]:
            blockers.append(
                f"{name}:sha256_mismatch:expected={expected_sha}:actual={identity['sha256']}"
            )
            valid = False
        rows[name] = {
            **identity,
            "exists": True,
            "required": required,
            "valid": valid,
        }
    return {
        "ok": not blockers,
        "manifest": str(manifest_path.resolve()),
        "schema_version": payload.get("schema_version"),
        "artifacts": rows,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify main-project Module A auxiliary artifact identity."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = verify(args.manifest.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


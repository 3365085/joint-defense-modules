from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.authoritative_manifest import (  # noqa: E402
    validate_authoritative_manifest,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "acceptance"
    / "module_a_authoritative_manifest_v1.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate the 37-record authoritative Module A material "
            "manifest without deduplicating assets by SHA-256."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--no-file-check",
        action="store_true",
        help="Validate JSON/schema/counts only; do not stat or hash asset files.",
    )
    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="Disable the exact 1 model / 1 A3b / 5 attack / 30 normal / 37-record production count gate.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit per-record checks from stdout/output JSON.",
    )
    args = parser.parse_args(argv)

    result = validate_authoritative_manifest(
        args.manifest,
        verify_files=not args.no_file_check,
        strict_counts=not args.no_strict_counts,
    )
    payload = result.to_dict(include_records=not args.compact)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

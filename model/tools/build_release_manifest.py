from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.release_manifest import (  # noqa: E402
    build_release_manifest,
    dumps_release_manifest,
    write_release_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only JSON release manifest."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--repository-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    smoke_group = parser.add_mutually_exclusive_group()
    smoke_group.add_argument("--smoke-json", default=None)
    smoke_group.add_argument("--smoke-file", type=Path, default=None)
    args = parser.parse_args(argv)

    smoke_result = _parse_smoke_result(parser, args.smoke_json, args.smoke_file)
    manifest = build_release_manifest(
        config_path=args.config,
        profile=args.profile,
        repository_root=args.repository_root,
        smoke_result=smoke_result,
    )
    if args.out is not None:
        write_release_manifest(args.out, manifest)
    sys.stdout.write(dumps_release_manifest(manifest))
    return 0


def _parse_smoke_result(
    parser: argparse.ArgumentParser,
    smoke_json: str | None,
    smoke_file: Path | None,
) -> Any:
    if smoke_json is None and smoke_file is None:
        return None
    try:
        if smoke_file is not None:
            return json.loads(smoke_file.read_text(encoding="utf-8"))
        return json.loads(str(smoke_json))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid smoke result: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

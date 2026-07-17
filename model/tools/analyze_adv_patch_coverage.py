"""CLI for read-only ``adv_patch`` alarm coverage diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (SRC_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from defense.diagnostics.adv_patch_coverage import (  # noqa: E402
    DEFAULT_P_ADV_THRESHOLD,
    analyze_adv_patch_coverage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream an adv_patch frame JSONL and report coverage metrics only."
    )
    parser.add_argument("input", type=Path, help="Per-frame JSONL emitted by the production probe.")
    parser.add_argument(
        "--p-adv-threshold",
        type=float,
        default=DEFAULT_P_ADV_THRESHOLD,
        help="Diagnostic p_adv cutoff; it is not an acceptance threshold.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS used only when rows contain no source/video time.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_adv_patch_coverage(
            args.input,
            p_adv_threshold=args.p_adv_threshold,
            fps=args.fps,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return 0
    except (OSError, ValueError) as exc:
        error = {"ok": False, "error": str(exc)}
        sys.stderr.write(json.dumps(error, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

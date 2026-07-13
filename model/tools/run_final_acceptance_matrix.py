from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.final_acceptance_matrix import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    report_exit_code,
    run_runtime_acceptance_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P1/P2/P3/N1-N4 full-video final acceptance matrix."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", default="desktop_rtx")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_runtime_acceptance_matrix(
        args.manifest,
        output_path=args.output,
        config_path=args.config,
        profile=str(args.profile),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())

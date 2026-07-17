from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from defense.diagnostics.a4_training import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    A4QualityGateError,
    collect_production_a4_features,
    train_bound_a4_classifier,
)


def _progress(payload: dict) -> None:
    print(
        f"[{payload['video_index']}/{payload['video_count']}] "
        f"{payload['clip_id']} frames={payload['frames']} "
        f"rows={payload['written_rows']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect production Module A features and train a bound A4 XGBoost artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument(
        "--manifest",
        required=True,
        help=(
            "Explicit training CSV. Final authoritative acceptance videos "
            "must not be used for A4 training or threshold selection."
        ),
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--metadata-out", default="")
    collect.add_argument("--profile", default="desktop_rtx")
    collect.add_argument("--config", default="")
    collect.add_argument("--splits", default="train,heldout")
    collect.add_argument("--max-frames-per-video", type=int, default=300)

    train = subparsers.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--output-model", required=True)
    train.add_argument("--report-out", required=True)
    train.add_argument("--metadata-out", default="")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--iterations", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "collect":
        result = collect_production_a4_features(
            manifest_path=args.manifest,
            output_csv=args.output,
            metadata_path=args.metadata_out or None,
            splits=[part.strip() for part in args.splits.split(",") if part.strip()],
            profile=args.profile,
            config_path=args.config or None,
            max_frames_per_video=args.max_frames_per_video,
            progress=_progress,
        )
    else:
        try:
            result = train_bound_a4_classifier(
                features_csv=args.features,
                output_model=args.output_model,
                report_path=args.report_out,
                metadata_path=args.metadata_out or None,
                folds=args.folds,
                iterations=args.iterations,
                seed=args.seed,
            )
        except A4QualityGateError as exc:
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
            print(str(exc), file=sys.stderr)
            return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

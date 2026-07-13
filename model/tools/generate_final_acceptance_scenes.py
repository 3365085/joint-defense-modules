from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.final_acceptance_scenes import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NEGATIVE_MANIFEST_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SOURCE_STEMS,
    DEFAULT_V3_CLEAN_CONTEXT_VIDEO,
    DEFAULT_V3_MANIFEST_PATH,
    DEFAULT_V3_OUTPUT_ROOT,
    SceneGenerationConfig,
    WrappedSceneConfig,
    generate_final_acceptance_scenes,
    generate_wrapped_final_acceptance_scenes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate v2 final P1/P2/P3 acceptance scenes and manifest."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--negative-manifest", type=Path, default=DEFAULT_NEGATIVE_MANIFEST_PATH)
    parser.add_argument("--no-negatives", action="store_true")
    parser.add_argument("--source-stem", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--attack-start-frame", type=int, default=30)
    parser.add_argument("--attack-ramp-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--wrapped-v3", action="store_true")
    parser.add_argument("--clean-prefix-s", type=float, default=2.0)
    parser.add_argument("--attack-duration-s", type=float, default=4.0)
    parser.add_argument("--clean-tail-s", type=float, default=2.0)
    parser.add_argument("--clean-context-video", type=Path, default=DEFAULT_V3_CLEAN_CONTEXT_VIDEO)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wrapped_v3:
        output_root = args.output_root if args.output_root != DEFAULT_OUTPUT_ROOT else DEFAULT_V3_OUTPUT_ROOT
        manifest_path = args.manifest if args.manifest != DEFAULT_MANIFEST_PATH else DEFAULT_V3_MANIFEST_PATH
        config = WrappedSceneConfig(
            output_root=output_root,
            manifest_path=manifest_path,
            negative_manifest_path=None if args.no_negatives else args.negative_manifest,
            clean_context_video=args.clean_context_video,
            clean_prefix_s=args.clean_prefix_s,
            attack_duration_s=args.attack_duration_s,
            clean_tail_s=args.clean_tail_s,
            max_width=args.max_width,
        )
        report = generate_wrapped_final_acceptance_scenes(config)
        print(
            json.dumps(
                {
                    "manifest": str(config.manifest_path.resolve()),
                    "output_root": str(config.output_root.resolve()),
                    "clips": len(report["clips"]),
                    "generation": report["generation"],
                    "categories": {
                        category: sum(1 for row in report["clips"] if row["category"] == category)
                        for category in ("P1", "P2", "P3", "N1", "N2", "N3", "N4")
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    config = SceneGenerationConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        manifest_path=args.manifest,
        negative_manifest_path=None if args.no_negatives else args.negative_manifest,
        source_stems=tuple(args.source_stem) if args.source_stem else DEFAULT_SOURCE_STEMS,
        max_frames=args.max_frames,
        max_width=args.max_width,
        attack_start_frame=args.attack_start_frame,
        attack_ramp_frames=args.attack_ramp_frames,
        seed=args.seed,
    )
    report = generate_final_acceptance_scenes(config)
    print(
        json.dumps(
            {
                "manifest": str(config.manifest_path.resolve()),
                "output_root": str(config.output_root.resolve()),
                "clips": len(report["clips"]),
                "categories": {
                    category: sum(1 for row in report["clips"] if row["category"] == category)
                    for category in ("P1", "P2", "P3", "N1", "N2", "N3", "N4")
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

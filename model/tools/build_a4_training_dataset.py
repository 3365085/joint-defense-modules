from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from defense.diagnostics.a4_dataset import (  # noqa: E402
    DEFAULT_AUTHORITATIVE_MANIFEST,
    build_a4_training_dataset,
)


def _progress(payload: dict) -> None:
    print(
        f"[{payload['base_index']}/{payload['base_count']}] "
        f"{payload['base_clip_id']} generated={payload['generated_clips']} "
        f"total={payload['generated_total']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic bounded A4 clean/physical-attack carrier clips. "
            "The 36 authoritative acceptance videos are hash-excluded and never read."
        )
    )
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--metadata-out", default="")
    parser.add_argument(
        "--authoritative-manifest",
        default=str(DEFAULT_AUTHORITATIVE_MANIFEST),
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=90,
        help="Hard bound per base carrier; combined with --clip-duration-s using the smaller bound.",
    )
    parser.add_argument(
        "--clip-duration-s",
        type=float,
        default=3.0,
        help="Duration bound; deterministic source offset is derived from seed and base SHA-256.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--attack-start-frame", type=int, default=12)
    parser.add_argument("--attack-ramp-frames", type=int, default=8)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--yolo-inference-stride", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_a4_training_dataset(
        source_manifest_path=args.clean_manifest,
        output_dir=args.output_dir,
        output_manifest_path=args.manifest_out,
        metadata_path=args.metadata_out or None,
        authoritative_manifest_path=args.authoritative_manifest,
        max_frames_per_video=args.max_frames_per_video,
        clip_duration_s=args.clip_duration_s,
        generator_seed=args.seed,
        attack_start_frame=args.attack_start_frame,
        attack_ramp_frames=args.attack_ramp_frames,
        codec=args.codec,
        yolo_device=args.yolo_device,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_inference_stride=args.yolo_inference_stride,
        progress=_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

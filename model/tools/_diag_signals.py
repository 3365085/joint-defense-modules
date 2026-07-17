"""Export authoritative Module A signal and performance distributions.

The script deliberately does not recreate production threshold gates. It reads
only fields emitted by ``VideoDefensePipeline`` and delegates reusable work to
``defense.diagnostics.module_a_tuning``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (SRC_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from defense.diagnostics.module_a_tuning import (  # noqa: E402
    parse_tuning_patch,
    run_module_a_videos,
    write_module_a_reports,
)
from defense.runtime.config import DEFAULT_CONFIG_PATH  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取生产 pipeline 权威结果字段，统计 Module A 信号、决策、"
            "detector reuse 和耗时分布。"
        )
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        type=Path,
        help="视频路径；可重复传入以复用同一个 pipeline 分析多个视频。",
    )
    parser.add_argument(
        "--tuning",
        default="",
        help="可选内联 JSON tuning patch 或 JSON/YAML 文件路径。",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="每个视频最多处理帧数。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", default="desktop_rtx")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "module_a_signals",
    )
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def _stdout_summary(report: dict[str, Any], outputs: dict[str, str]) -> dict[str, Any]:
    videos = []
    for video in report.get("videos", []) or []:
        if not isinstance(video, dict):
            continue
        summary = dict(video.get("summary", {}))
        videos.append(
            {
                "video": video.get("video"),
                "frames": summary.get("frames"),
                "decisions": summary.get("decisions"),
                "detector_reuse": summary.get("detector_reuse"),
                "performance_ms": summary.get("performance_ms"),
                "signals": summary.get("signals"),
            }
        )
    return {
        "ok": True,
        "pipeline_contract": report.get("pipeline_contract"),
        "backend": report.get("backend"),
        "outputs": outputs,
        "videos": videos,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_module_a_videos(
        args.video,
        config_path=args.config,
        profile=str(args.profile),
        tuning_patch=parse_tuning_patch(args.tuning),
        max_frames=max(0, int(args.max_frames)),
        warmup=not bool(args.no_warmup),
    )
    outputs = write_module_a_reports(
        report,
        output_dir=args.out_dir,
        stem="module_a_signals",
    )
    print(json.dumps(_stdout_summary(report, outputs), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

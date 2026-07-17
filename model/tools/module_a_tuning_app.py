"""Offline Module A tuning/diagnostic CLI.

This wrapper intentionally does not host an HTTP server. Reusable processing
logic lives in ``defense.diagnostics.module_a_tuning`` and always runs the
production ``VideoDefensePipeline`` with the effective rebuilt configuration.
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
            "使用生产 VideoDefensePipeline/rebuilt effective config 离线运行视频，"
            "输出逐帧信号和性能报告。"
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="待分析视频路径。")
    parser.add_argument(
        "--tuning",
        default="",
        help=(
            "可选 tuning patch：内联 JSON 对象，或 JSON/YAML 文件路径。"
            "旧版平铺键会自动放入 module_a，并与完整配置 deep merge。"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理帧数；0 表示完整视频。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Module A runtime YAML 配置。",
    )
    parser.add_argument(
        "--profile",
        default="desktop_rtx",
        help="运行 profile，默认 desktop_rtx。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "diagnostics" / "module_a_tuning",
        help="JSON 和逐帧 JSONL 输出目录。",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过标准 pipeline warmup，仅用于快速诊断。",
    )
    parser.add_argument(
        "--port",
        nargs="?",
        const="8766",
        default=None,
        help="已弃用：调优工具不再启动独立 HTTP server。",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="已弃用：调优工具已改为离线 CLI。",
    )
    return parser


def _compact_stdout(report: dict[str, Any], outputs: dict[str, str]) -> dict[str, Any]:
    videos = report.get("videos", [])
    summaries = [
        {
            "video": video.get("video"),
            **dict(video.get("summary", {})),
        }
        for video in videos
        if isinstance(video, dict)
    ]
    return {
        "ok": True,
        "pipeline_contract": report.get("pipeline_contract"),
        "detector_impl": report.get("configuration", {}).get("detector_impl"),
        "backend": report.get("backend"),
        "initialization_ms": report.get("initialization_ms"),
        "outputs": outputs,
        "videos": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.port is not None or args.server:
        parser.error(
            "--port/--server 已弃用：legacy HTTP handler 已移除。"
            "请使用 --video/--tuning/--max-frames 生成离线 JSON 报告。"
        )

    patch = parse_tuning_patch(args.tuning)
    report = run_module_a_videos(
        [args.video],
        config_path=args.config,
        profile=str(args.profile),
        tuning_patch=patch,
        max_frames=max(0, int(args.max_frames)),
        warmup=not bool(args.no_warmup),
    )
    outputs = write_module_a_reports(
        report,
        output_dir=args.out_dir,
        stem=f"{args.video.stem}_module_a_tuning",
    )
    print(json.dumps(_compact_stdout(report, outputs), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.pipelines.video_decoder_factory import create_video_decoder


MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "acceptance"
    / "module_a_authoritative_manifest_v1.json"
)
ASSET_IDS = {
    "a3b": "a3b.authoritative_target",
    "normal": "normal.fixed_camera_1080",
}


def _finite(value: float) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(_finite(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_size(value: str) -> tuple[int, int]:
    text = str(value or "").lower().replace(" ", "")
    if "x" not in text:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width_text, height_text = text.split("x", 1)
    try:
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return width, height


def _load_authoritative_sources(selection: str) -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    requested = (
        [ASSET_IDS["normal"], ASSET_IDS["a3b"]]
        if selection == "both"
        else [ASSET_IDS[selection]]
    )
    by_id = {row["asset_id"]: row for row in manifest["videos"]}
    rows = []
    for asset_id in requested:
        row = dict(by_id[asset_id])
        path = Path(row["canonical_path"])
        row["exists"] = path.is_file()
        row["actual_size_bytes"] = path.stat().st_size if path.is_file() else 0
        rows.append(row)
    return rows


def _gpu_preprocess(
    tensor: Any,
    *,
    size: tuple[int, int],
) -> tuple[float, tuple[int, ...]]:
    import torch
    import torch.nn.functional as functional

    width, height = size
    with torch.cuda.device(tensor.device):
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        output = functional.interpolate(
            tensor.unsqueeze(0).float().div_(255.0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        ended.record()
        ended.synchronize()
        elapsed_ms = float(started.elapsed_time(ended))
    shape = tuple(int(value) for value in output.shape)
    del output
    return elapsed_ms, shape


def benchmark_backend(
    source: Path,
    *,
    backend: str,
    frame_limit: int,
    warmup_frames: int,
    gpu_id: int,
    full_host_copy: bool,
    gpu_preprocess_size: tuple[int, int] | None,
) -> dict[str, Any]:
    initialized_at = time.perf_counter()
    decoder = create_video_decoder(
        source,
        preference=backend,
        allow_cpu_fallback=backend == "auto",
        gpu_id=gpu_id,
    )
    initialization_ms = (time.perf_counter() - initialized_at) * 1000.0
    gpu_preprocess_ms: list[float] = []
    host_checksum = 0
    completed = 0
    output_bytes = 0
    preprocess_shape: tuple[int, ...] | None = None
    try:
        for _ in range(max(0, warmup_frames)):
            lease = decoder.read()
            if lease is None:
                break
            if full_host_copy:
                lease.materialize_host_bgr()
            if gpu_preprocess_size is not None and lease.cuda_tensor is not None:
                _gpu_preprocess(lease.cuda_tensor, size=gpu_preprocess_size)
            lease.release()
        if warmup_frames:
            decoder.seek_frame(0)

        started = time.perf_counter()
        for _ in range(max(1, frame_limit)):
            lease = decoder.read()
            if lease is None:
                break
            output_bytes += lease.width * lease.height * 3
            if full_host_copy:
                host = lease.materialize_host_bgr()
                host_checksum = (
                    host_checksum + int(host[0, 0].sum()) + int(host[-1, -1].sum())
                ) & 0xFFFFFFFF
            if gpu_preprocess_size is not None and lease.cuda_tensor is not None:
                elapsed_ms, preprocess_shape = _gpu_preprocess(
                    lease.cuda_tensor,
                    size=gpu_preprocess_size,
                )
                gpu_preprocess_ms.append(elapsed_ms)
            completed += 1
            lease.release()
        elapsed_s = time.perf_counter() - started
        status = decoder.status_snapshot()
        info = asdict(decoder.info)
    finally:
        decoder.close()
    close_status = decoder.status_snapshot()
    return {
        "backend_requested": backend,
        "backend_effective": status["effective_backend"],
        "initialization_ms": round(initialization_ms, 6),
        "frames_requested": int(frame_limit),
        "frames_completed": int(completed),
        "elapsed_s": round(elapsed_s, 6),
        "throughput_fps": round(completed / elapsed_s if elapsed_s > 0.0 else 0.0, 6),
        "output_bytes": int(output_bytes),
        "output_megabytes_per_s": round(
            output_bytes / elapsed_s / 1_000_000.0 if elapsed_s > 0.0 else 0.0,
            6,
        ),
        "full_host_copy": bool(full_host_copy),
        "host_checksum": int(host_checksum),
        "gpu_preprocess": {
            "enabled": gpu_preprocess_size is not None,
            "requested_size": list(gpu_preprocess_size)
            if gpu_preprocess_size is not None
            else None,
            "sample_count": len(gpu_preprocess_ms),
            "ms_p50": round(_percentile(gpu_preprocess_ms, 50.0), 6),
            "ms_p95": round(_percentile(gpu_preprocess_ms, 95.0), 6),
            "output_shape": list(preprocess_shape) if preprocess_shape else None,
        },
        "stream_info": info,
        "status": status,
        "close_status": {
            "closed": close_status["closed"],
            "close_error": close_status.get("close_error", ""),
            "source_alias_cleaned": close_status.get("source_alias_cleaned"),
            "source_alias_cleanup_deferred": close_status.get(
                "source_alias_cleanup_deferred"
            ),
            "source_alias_cleanup_error": close_status.get(
                "source_alias_cleanup_error",
                "",
            ),
        },
    }


def benchmark_parity(
    source: Path,
    *,
    frame_limit: int,
    gpu_id: int,
    mean_threshold: float,
    p95_threshold: float,
    max_threshold: int,
) -> dict[str, Any]:
    opencv = create_video_decoder(
        source,
        preference="opencv",
        allow_cpu_fallback=False,
    )
    nvdec = create_video_decoder(
        source,
        preference="nvdec",
        allow_cpu_fallback=False,
        gpu_id=gpu_id,
    )
    frame_rows: list[dict[str, Any]] = []
    try:
        for _ in range(max(1, frame_limit)):
            cpu_lease = opencv.read()
            gpu_lease = nvdec.read()
            if cpu_lease is None or gpu_lease is None:
                break
            cpu_bgr = cpu_lease.materialize_host_bgr()
            gpu_bgr = gpu_lease.materialize_host_bgr()
            if cpu_bgr.shape != gpu_bgr.shape:
                raise RuntimeError(
                    f"parity_shape_mismatch:{cpu_bgr.shape}:{gpu_bgr.shape}"
                )
            difference = np.abs(
                cpu_bgr.astype(np.int16) - gpu_bgr.astype(np.int16)
            )
            frame_rows.append(
                {
                    "frame_idx": int(cpu_lease.frame_idx),
                    "mean_abs_diff": float(difference.mean()),
                    "p95_abs_diff": float(np.percentile(difference, 95)),
                    "max_abs_diff": int(difference.max()),
                }
            )
            cpu_lease.release()
            gpu_lease.release()
    finally:
        opencv.close()
        nvdec.close()

    mean_max = max((row["mean_abs_diff"] for row in frame_rows), default=0.0)
    p95_max = max((row["p95_abs_diff"] for row in frame_rows), default=0.0)
    absolute_max = max((row["max_abs_diff"] for row in frame_rows), default=0)
    passed = (
        bool(frame_rows)
        and mean_max <= mean_threshold
        and p95_max <= p95_threshold
        and absolute_max <= max_threshold
    )
    return {
        "frames_compared": len(frame_rows),
        "passed": passed,
        "thresholds": {
            "mean_abs_diff_max": mean_threshold,
            "p95_abs_diff_max": p95_threshold,
            "absolute_diff_max": max_threshold,
        },
        "observed": {
            "mean_abs_diff_max": round(mean_max, 6),
            "p95_abs_diff_max": round(p95_max, 6),
            "absolute_diff_max": int(absolute_max),
        },
        "frames": frame_rows,
    }


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import cv2

        result["opencv"] = cv2.__version__
    except Exception as exc:
        result["opencv_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import PyNvVideoCodec as nvc

        result["pynvvideocodec"] = getattr(nvc, "__version__", "unknown")
    except Exception as exc:
        result["pynvvideocodec_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}:{exc}"
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV and PyNvVideoCodec/NVDEC file decoders."
    )
    parser.add_argument(
        "--asset",
        choices=("normal", "a3b", "both"),
        default="both",
        help="Authoritative manifest source selection.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional explicit source path; repeat to benchmark multiple files.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "nvdec", "opencv", "all"),
        default="all",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--full-host-copy",
        action="store_true",
        help="Materialize full-resolution host BGR for every measured frame.",
    )
    parser.add_argument(
        "--gpu-preprocess",
        type=_parse_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Run CUDA RGBP float resize/normalize without a host download.",
    )
    parser.add_argument("--pixel-parity", action="store_true")
    parser.add_argument("--parity-frames", type=int, default=8)
    parser.add_argument("--parity-mean-threshold", type=float, default=2.0)
    parser.add_argument("--parity-p95-threshold", type=float, default=4.0)
    parser.add_argument("--parity-max-threshold", type=int, default=24)
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional JSON output path. JSON is always printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frames <= 0 or args.warmup_frames < 0 or args.parity_frames <= 0:
        raise SystemExit("frames must be positive and warmup-frames non-negative")

    sources: list[dict[str, Any]] = []
    if args.source:
        for raw in args.source:
            path = Path(raw).expanduser().resolve(strict=False)
            sources.append(
                {
                    "asset_id": "explicit",
                    "canonical_path": str(path),
                    "exists": path.is_file(),
                    "actual_size_bytes": path.stat().st_size if path.is_file() else 0,
                }
            )
    else:
        sources = _load_authoritative_sources(args.asset)

    backends = (
        ["nvdec", "opencv"] if args.backend == "all" else [args.backend]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "benchmark_video_decode",
        "manifest": str(MANIFEST_PATH),
        "environment": _environment(),
        "options": {
            "asset": args.asset,
            "backend": args.backend,
            "frames": args.frames,
            "warmup_frames": args.warmup_frames,
            "gpu_id": args.gpu_id,
            "full_host_copy": args.full_host_copy,
            "gpu_preprocess": list(args.gpu_preprocess)
            if args.gpu_preprocess
            else None,
            "pixel_parity": args.pixel_parity,
            "parity_frames": args.parity_frames,
        },
        "sources": [],
    }
    failures = 0
    for source_row in sources:
        path = Path(source_row["canonical_path"])
        source_result: dict[str, Any] = {
            "asset": source_row,
            "benchmarks": [],
        }
        if not path.is_file():
            source_result["error"] = f"source_not_found:{path}"
            failures += 1
            report["sources"].append(source_result)
            continue
        for backend in backends:
            try:
                result = benchmark_backend(
                    path,
                    backend=backend,
                    frame_limit=args.frames,
                    warmup_frames=args.warmup_frames,
                    gpu_id=args.gpu_id,
                    full_host_copy=args.full_host_copy,
                    gpu_preprocess_size=args.gpu_preprocess,
                )
            except Exception as exc:
                failures += 1
                result = {
                    "backend_requested": backend,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            source_result["benchmarks"].append(result)
        if args.pixel_parity:
            try:
                source_result["pixel_parity"] = benchmark_parity(
                    path,
                    frame_limit=min(args.frames, args.parity_frames),
                    gpu_id=args.gpu_id,
                    mean_threshold=args.parity_mean_threshold,
                    p95_threshold=args.parity_p95_threshold,
                    max_threshold=args.parity_max_threshold,
                )
                if not source_result["pixel_parity"]["passed"]:
                    failures += 1
            except Exception as exc:
                failures += 1
                source_result["pixel_parity"] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
        report["sources"].append(source_result)

    report["failure_count"] = failures
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.json_output:
        output_path = Path(args.json_output).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

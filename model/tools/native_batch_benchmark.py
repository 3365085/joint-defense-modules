"""Microbenchmark scalar-loop versus batch A3b native calls.

This tool measures Python/native call and operator costs in isolation. It does
not run the production detector or video pipeline and must not be interpreted
as production FPS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from defense.module_a import native_bridge  # noqa: E402


_CANDIDATE_COUNTS = (1, 8, 32, 64)
_DISCLAIMER = (
    "Native API microbenchmark only; this does not execute the production "
    "detector/video pipeline and is not a production FPS measurement."
)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()

    durations_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(durations_ms)
    return {
        "samples": iterations,
        "warmup": warmup,
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "p99_ms": _percentile(ordered, 0.99),
        "max_ms": ordered[-1],
    }


def _candidate_boxes(
    rng: np.random.Generator,
    *,
    count: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        box_width = int(rng.integers(18, min(128, width) + 1))
        box_height = int(rng.integers(18, min(112, height) + 1))
        x1 = int(rng.integers(0, width - box_width + 1))
        y1 = int(rng.integers(0, height - box_height + 1))
        boxes.append((x1, y1, x1 + box_width, y1 + box_height))
    return boxes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path receiving the same JSON emitted to stdout.",
    )
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    bridge_status = native_bridge.status()
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": "a3b_native_call_microbenchmark",
        "disclaimer": _DISCLAIMER,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "candidate_counts": list(_CANDIDATE_COUNTS),
        },
        "native": bridge_status,
        "benchmarks": {},
    }
    if not native_bridge.available:
        report["error"] = {
            "fallback_reason": native_bridge.fallback_reason,
            "load_error": native_bridge.load_error,
        }
        _emit(report, args.output)
        return 2

    native = native_bridge.require_native()
    rng = np.random.default_rng(args.seed)
    shape = (360, 640)
    gray = np.ascontiguousarray(
        rng.integers(0, 256, size=shape, dtype=np.uint8)
    )
    edge_mask = np.ascontiguousarray(
        rng.integers(0, 2, size=shape, dtype=np.uint8)
    )
    all_boxes = _candidate_boxes(
        rng,
        count=max(_CANDIDATE_COUNTS),
        width=shape[1],
        height=shape[0],
    )

    for candidate_count in _CANDIDATE_COUNTS:
        boxes = all_boxes[:candidate_count]

        def scalar_loop() -> list[tuple[float, ...]]:
            return [
                native.a3b_one_box_stats(edge_mask, gray, *box)
                for box in boxes
            ]

        def batch_call() -> list[tuple[float, ...]]:
            return native.a3b_boxes_stats(edge_mask, gray, boxes)

        scalar_result = scalar_loop()
        batch_result = batch_call()
        if scalar_result != batch_result:
            raise RuntimeError(
                f"batch/scalar parity failed before timing N={candidate_count}"
            )

        scalar_timing = _measure(
            scalar_loop,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        batch_timing = _measure(
            batch_call,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        scalar_mean = float(scalar_timing["mean_ms"])
        batch_mean = float(batch_timing["mean_ms"])
        report["benchmarks"][str(candidate_count)] = {
            "inputs": {
                "array_shape": list(shape),
                "candidate_count": candidate_count,
                "boxes": [list(box) for box in boxes],
            },
            "validation": {
                "scalar_batch_exact_match": True,
                "result_fields_per_box": 6,
            },
            "python_to_rust_calls_per_iteration": {
                "scalar_loop": candidate_count,
                "batch": 1,
            },
            "scalar_loop": {
                "timing": scalar_timing,
                "mean_ms_per_box": scalar_mean / candidate_count,
            },
            "batch": {
                "timing": batch_timing,
                "mean_ms_per_box": batch_mean / candidate_count,
            },
            "batch_speedup_x_by_mean": (
                scalar_mean / batch_mean if batch_mean > 0.0 else None
            ),
            "mean_time_reduction_percent": (
                (scalar_mean - batch_mean) / scalar_mean * 100.0
                if scalar_mean > 0.0
                else None
            ),
        }

    report["ok"] = True
    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

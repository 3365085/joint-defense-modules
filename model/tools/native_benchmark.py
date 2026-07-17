"""JSON benchmark for the five first-wave ``module_a_native`` operators."""

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


def _benchmark_cases(native: Any, seed: int) -> dict[str, tuple[Callable[[], Any], dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    shape = (360, 640)
    lbp = np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
    previous_lbp = np.ascontiguousarray(
        rng.integers(0, 256, size=shape, dtype=np.uint8)
    )
    baseline = np.ascontiguousarray(rng.random(32, dtype=np.float32))
    baseline /= baseline.sum(dtype=np.float32)
    rois = [
        (32, 24, 180, 220),
        (220, 80, 420, 330),
        (440, 40, 620, 300),
        (120, 200, 300, 350),
    ]
    residual = np.ascontiguousarray(rng.random(shape, dtype=np.float32))
    gray = np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
    edge_mask = np.ascontiguousarray(
        rng.integers(0, 2, size=shape, dtype=np.uint8)
    )
    candidate_box = (100, 60, 540, 320)

    return {
        "a1_lbp_features": (
            lambda: native.a1_lbp_features(lbp, rois, baseline),
            {
                "lbp_shape": list(shape),
                "roi_count": len(rois),
                "baseline_bins": int(baseline.size),
            },
        ),
        "a2_change_features": (
            lambda: native.a2_change_features(lbp, previous_lbp, rois, 0.45),
            {
                "lbp_shape": list(shape),
                "roi_count": len(rois),
                "expand_margin": 0.45,
            },
        ),
        "best_grid_value_f32": (
            lambda: native.best_grid_value_f32(residual, 8),
            {
                "array_shape": list(shape),
                "grid": 8,
            },
        ),
        "a3b_one_box_stats": (
            lambda: native.a3b_one_box_stats(edge_mask, gray, *candidate_box),
            {
                "array_shape": list(shape),
                "bbox": list(candidate_box),
            },
        ),
        "blinding_laplacian_var": (
            lambda: native.blinding_laplacian_var(gray),
            {
                "gray_shape": list(shape),
            },
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
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
    for name, (function, inputs) in _benchmark_cases(native, args.seed).items():
        result = _measure(
            function,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        report["benchmarks"][name] = {
            "inputs": inputs,
            "timing": result,
        }
    report["ok"] = True
    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

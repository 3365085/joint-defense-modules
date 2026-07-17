from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from defense.pipelines._opencv_file_decoder import OpenCVFileDecoder
from defense.pipelines._pynv_file_decoder import PyNvFileDecoder


def _a3b_source(pkg_root: Path) -> Path:
    manifest = json.loads(
        (
            pkg_root
            / "configs"
            / "acceptance"
            / "module_a_authoritative_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    path = Path(
        next(
            row["canonical_path"]
            for row in manifest["videos"]
            if row["asset_id"] == "a3b.authoritative_target"
        )
    )
    if not path.is_file():
        pytest.skip(f"authoritative A3b video is unavailable: {path}")
    return path


def _require_nvdec() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("PyNvVideoCodec")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def _difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    assert left.shape == right.shape
    return np.abs(left.astype(np.int16) - right.astype(np.int16))


def test_nvdec_opencv_first_frames_pixel_parity(pkg_root: Path) -> None:
    _require_nvdec()
    source = _a3b_source(pkg_root)
    opencv = OpenCVFileDecoder(source)
    nvdec = PyNvFileDecoder(source)
    mean_differences: list[float] = []
    p95_differences: list[float] = []
    maximum_differences: list[int] = []
    try:
        for expected_idx in range(5):
            cpu_lease = opencv.read()
            gpu_lease = nvdec.read()
            assert cpu_lease is not None
            assert gpu_lease is not None
            assert cpu_lease.frame_idx == gpu_lease.frame_idx == expected_idx
            cpu_bgr = cpu_lease.materialize_host_bgr()
            gpu_bgr = gpu_lease.materialize_host_bgr()
            difference = _difference(cpu_bgr, gpu_bgr)
            mean_differences.append(float(difference.mean()))
            p95_differences.append(float(np.percentile(difference, 95)))
            maximum_differences.append(int(difference.max()))
            cpu_lease.release()
            gpu_lease.release()

        assert max(mean_differences) <= 2.0
        assert max(p95_differences) <= 4.0
        assert max(maximum_differences) <= 24

        nvdec.seek_frame(3)
        opencv.seek_frame(3)
        cpu_sought = opencv.read()
        gpu_sought = nvdec.read()
        assert cpu_sought is not None
        assert gpu_sought is not None
        assert cpu_sought.frame_idx == gpu_sought.frame_idx == 3
        assert cpu_sought.pts_s >= 0.0
        assert gpu_sought.pts_s >= 0.0
        sought_difference = _difference(
            cpu_sought.materialize_host_bgr(),
            gpu_sought.materialize_host_bgr(),
        )
        assert float(sought_difference.mean()) <= 2.0
        assert int(sought_difference.max()) <= 24
        cpu_sought.release()
        gpu_sought.release()

        status = nvdec.status_snapshot()
        assert status["d2d_copy_sample_count"] == 6
        assert status["d2h_copy_sample_count"] == 6
        assert status["d2d_copy_ms_p95"] >= status["d2d_copy_ms_p50"]
        assert status["d2h_copy_ms_p95"] >= status["d2h_copy_ms_p50"]
    finally:
        opencv.close()
        nvdec.close()

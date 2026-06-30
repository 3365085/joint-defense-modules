from __future__ import annotations

import pytest


@pytest.mark.requires_gpu
def test_requires_gpu_marker_uses_cpu_fallback(cuda_device: str) -> None:
    assert cuda_device in {"cpu", "cuda:0"}

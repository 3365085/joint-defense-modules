"""Shared pytest fixtures for Module A tests.

Responsibilities:
  * Insert the package root on ``sys.path`` so ``from defense.module_a ...``
    resolves without installing the package.
  * Skip CUDA-dependent tests cleanly on CPU-only machines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


@pytest.fixture(scope="session")
def cuda_device() -> str:
    """Return the CUDA device string or skip the test if CUDA is absent."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Module A GPU-first tests")
    return "cuda:0"


@pytest.fixture(scope="session")
def pkg_root() -> Path:
    return PKG_ROOT


@pytest.fixture(scope="session")
def samples_dir(pkg_root: Path) -> Path:
    return pkg_root / "samples"


@pytest.fixture(scope="session")
def baseline_config_path(pkg_root: Path) -> Path:
    return pkg_root / "experiments" / "configs" / "module_a_baseline.yaml"

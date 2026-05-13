"""Verify ``_resolve_artifact_path`` handles the multi-ancestor layouts we
actually use (package root, legacy layout, MODULE_A_ROOT override)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from defense.module_a import detector as detector_module


def test_absolute_path_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "artifact.json"
    abs_path.write_text("{}")
    resolved = detector_module._resolve_artifact_path(str(abs_path))
    assert resolved == abs_path


def test_relative_resolves_under_package_root(pkg_root: Path) -> None:
    # experiments/configs/... is the canonical package-root-relative location
    rel = "experiments/configs/module_a_baseline.yaml"
    resolved = detector_module._resolve_artifact_path(rel)
    assert resolved == pkg_root / rel
    assert resolved.exists()


def test_module_a_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override_root = tmp_path / "override_root"
    override_root.mkdir()
    target = override_root / "custom.json"
    target.write_text("{}")
    monkeypatch.setenv("MODULE_A_ROOT", str(override_root))
    resolved = detector_module._resolve_artifact_path("custom.json")
    assert resolved == target


def test_missing_path_still_returns_canonical(pkg_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure env override is not set so the package root is the canonical fallback.
    monkeypatch.delenv("MODULE_A_ROOT", raising=False)
    missing = "does/not/exist/foo.json"
    resolved = detector_module._resolve_artifact_path(missing)
    # Should return the parents[2] (package root) path, so downstream FileNotFound
    # messages are informative.
    assert resolved == pkg_root / missing

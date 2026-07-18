from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from defense.model_security.service import ModelSecurityService
from defense.module_a.backends.detector_backend import YoloV5DetectorBackend
from defense.pipelines._pynv_file_decoder import default_video_decode_alias_root
from defense.pipelines.derived_video_cache import default_derived_video_cache_root
from defense.runtime.catalog import default_catalog_path, default_runtime_root
from defense.runtime.config import RUNTIME_DATA_ROOT_ENV, runtime_data_root
from defense.runtime.evidence import default_evidence_root


def test_detector_backend_import_has_no_runtime_cycle() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", "import defense.module_a.backends.detector_backend"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_data_root_defaults_to_project_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(RUNTIME_DATA_ROOT_ENV, raising=False)

    project = tmp_path / "project"

    assert runtime_data_root(project) == project / "runtime"


def test_runtime_data_root_requires_absolute_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_DATA_ROOT_ENV, "relative-runtime")

    with pytest.raises(ValueError, match="must be an absolute path"):
        runtime_data_root()


def test_unified_runtime_data_root_routes_mutable_production_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_root = (tmp_path / "operator-data" / "runtime").resolve()
    monkeypatch.setenv(RUNTIME_DATA_ROOT_ENV, str(runtime_root))
    monkeypatch.delenv("MODULE_A_EVIDENCE_ROOT", raising=False)
    monkeypatch.delenv("MODULE_A_DERIVED_VIDEO_CACHE_ROOT", raising=False)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    assert runtime_data_root() == runtime_root
    assert default_runtime_root() == runtime_root
    assert default_catalog_path() == runtime_root / "db" / "runtime_catalog.sqlite3"
    assert default_evidence_root() == runtime_root / "evidence" / "monitor"
    assert (
        default_derived_video_cache_root()
        == runtime_root / "artifacts" / "video_decode"
    )
    assert default_video_decode_alias_root() == runtime_root / "video_decode_alias"

    service = ModelSecurityService(
        root=tmp_path / "delivery-project",
        runtime_root=runtime_data_root(),
    )
    assert service.storage.root == runtime_root / "model_security"
    assert service._catalog_root() == runtime_root

    YoloV5DetectorBackend._configure_yolov5_base_runtime(tmp_path / "yolov5")
    assert Path(os.environ["YOLO_CONFIG_DIR"]) == (
        runtime_root / "logs" / "yolov5_runtime"
    )


def test_specific_evidence_override_keeps_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_root = (tmp_path / "runtime").resolve()
    evidence_root = (tmp_path / "special-evidence").resolve()
    monkeypatch.setenv(RUNTIME_DATA_ROOT_ENV, str(runtime_root))
    monkeypatch.setenv("MODULE_A_EVIDENCE_ROOT", str(evidence_root))

    assert default_evidence_root() == evidence_root

from __future__ import annotations

import hashlib
from pathlib import Path

from defense.runtime.config import load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_production_config_has_no_demo_artifact_path() -> None:
    text = _text("configs/module_a_runtime.yaml")
    assert "rebuilt_demo" not in text
    assert "baseline_training/runs/classmate_maskbd_v4" not in text


def test_rebuilt_detector_has_no_demo_path_resolver_or_legacy_env() -> None:
    text = _text("src/defense/module_a/rebuilt/detector.py")
    assert '/ "rebuilt_demo" / "data"' not in text
    assert "MODULE_A_REBUILT_DATA_DIR" not in text
    assert "normalized_first == \"rebuilt_demo\"" not in text


def test_release_and_diagnostic_defaults_have_no_demo_fallback() -> None:
    release = _text("src/defense/diagnostics/release_manifest.py")
    a4_training = _text("src/defense/diagnostics/a4_training.py")
    heldout = _text("tools/run_a3b_heldout.py")
    assert "repository_rebuilt_demo_data" not in release
    assert "rebuilt_demo/data/dataset_manifest.csv" not in a4_training
    assert "rebuilt_demo" not in heldout


def test_production_raft_artifact_is_main_project_hash_bound() -> None:
    config = load_runtime_config(profile="desktop_rtx")
    module_a = config["module_a"]
    path = (PROJECT_ROOT / module_a["raft_engine_path"]).resolve()
    assert path.is_file()
    assert PROJECT_ROOT.resolve() in path.parents
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    assert digest == module_a["raft_engine_sha256"]

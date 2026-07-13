from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from defense.diagnostics import release_manifest


class _FakeClassifier:
    n_features_in_ = 20
    feature_importances_ = [0.05] * 20

    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        return [[0.1, 0.9] for _ in rows]


class _CpuOnlyCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeGitRunner:
    def __init__(self, responses: dict[tuple[str, ...], tuple[int, str, str]]) -> None:
        self.responses = responses
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self, command: list[str], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        key = tuple(command[3:])
        self.commands.append(key)
        returncode, stdout, stderr = self.responses[key]
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_build_release_manifest_collects_configured_assets_without_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path
    model_root = repo_root / "model"
    assets = model_root / "assets"
    raft_data = model_root / "src" / "defense" / "module_a" / "rebuilt" / "data"
    assets.mkdir(parents=True)
    raft_data.mkdir(parents=True)

    runtime_engine = _write_bytes(assets / "runtime.engine", b"runtime-engine")
    source_pt = _write_bytes(assets / "source.pt", b"source-pt")
    classifier_path = assets / "a4_classifier.pkl"
    classifier_path.write_bytes(pickle.dumps(_FakeClassifier()))
    raft_onnx = _write_bytes(raft_data / "raft_small_256.onnx", b"raft-onnx")
    raft_engine = _write_bytes(
        raft_data / "raft_small_fp16_256.engine", b"raft-engine"
    )
    config_path = _write_config(model_root)

    git_runner = _FakeGitRunner(
        {
            ("rev-parse", "HEAD"): (0, "0123456789abcdef\n", ""),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): (
                0,
                "codex/release-manifest\n",
                "",
            ),
            ("status", "--porcelain=v1", "--untracked-files=normal"): (
                0,
                " M model/changed.py\nA  model/staged.py\n?? model/new.py\n",
                "",
            ),
        }
    )
    fake_torch = SimpleNamespace(
        __version__="2.7.1+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=_CpuOnlyCuda(),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9100)),
    )
    monkeypatch.setattr(release_manifest, "project_root", lambda: model_root)
    monkeypatch.setattr(release_manifest, "_run_process", git_runner)
    monkeypatch.setattr(release_manifest, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(
        release_manifest,
        "_module_a_native_manifest",
        lambda: {
            "discoverable": True,
            "available": True,
            "origin": "fake-native.pyd",
            "module_file": "fake-native.pyd",
            "error": None,
            "discovery_error": None,
        },
    )

    manifest = release_manifest.build_release_manifest(
        config_path=config_path,
        repository_root=repo_root,
        smoke_result={"passed": True, "suite": "empty_smoke"},
        generated_at="2026-07-11T00:00:00Z",
    )

    assert manifest["generated_at"] == "2026-07-11T00:00:00Z"
    assert manifest["repository"]["head"] == "0123456789abcdef"
    assert manifest["repository"]["branch"] == "codex/release-manifest"
    assert manifest["repository"]["dirty"] == {
        "available": True,
        "is_dirty": True,
        "entry_count": 3,
        "staged": 1,
        "unstaged": 1,
        "untracked": 1,
        "conflicted": 0,
        "status_counts": {" M": 1, "??": 1, "A ": 1},
    }
    assert manifest["configuration"]["sha256"] == _sha256(config_path)
    assert manifest["yolo"]["runtime_artifact"]["path"] == str(runtime_engine)
    assert manifest["yolo"]["runtime_artifact"]["sha256"] == _sha256(runtime_engine)
    assert manifest["yolo"]["source_pt"]["path"] == str(source_pt)
    assert manifest["yolo"]["source_pt"]["sha256"] == _sha256(source_pt)
    assert manifest["a4_classifier"]["path"] == str(classifier_path)
    assert manifest["a4_classifier"]["sha256"] == _sha256(classifier_path)
    assert manifest["a4_classifier"]["loadable"] is True
    assert manifest["a4_classifier"]["runtime_usable"] is True
    assert manifest["a4_classifier"]["feature_dimension"] == 20
    assert manifest["a4_classifier"]["feature_dimension_status"] == "match"
    assert manifest["raft"]["onnx"]["sha256"] == _sha256(raft_onnx)
    assert manifest["raft"]["engine"]["sha256"] == _sha256(raft_engine)
    assert manifest["module_a_native"]["available"] is True
    assert manifest["environment"]["torch"]["version"] == "2.7.1+cu128"
    assert manifest["environment"]["cuda"]["available"] is False
    assert manifest["environment"]["gpu"]["devices"] == []
    assert manifest["smoke"] == {"passed": True, "suite": "empty_smoke"}
    assert git_runner.commands == [
        ("rev-parse", "HEAD"),
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        ("status", "--porcelain=v1", "--untracked-files=normal"),
    ]


def test_manifest_does_not_guess_assets_when_explicit_paths_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path
    model_root = repo_root / "model"
    assets = model_root / "assets"
    assets.mkdir(parents=True)
    _write_bytes(assets / "runtime.engine", b"runtime-engine")
    _write_bytes(model_root / "unconfigured-source.pt", b"must-not-be-selected")
    fallback_dir = repo_root / "rebuilt_demo" / "data"
    fallback_dir.mkdir(parents=True)
    (fallback_dir / "a4_classifier.pkl").write_bytes(pickle.dumps(_FakeClassifier()))
    config_path = _write_config(
        model_root,
        classifier_path="assets/missing-a4.pkl",
        source_pt_path="assets/missing-source.pt",
    )

    failed_git = _FakeGitRunner(
        {
            ("rev-parse", "HEAD"): (128, "", "not a git repository"),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): (
                128,
                "",
                "not a git repository",
            ),
            ("status", "--porcelain=v1", "--untracked-files=normal"): (
                128,
                "",
                "not a git repository",
            ),
        }
    )

    def missing_torch() -> Any:
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(release_manifest, "project_root", lambda: model_root)
    monkeypatch.setattr(release_manifest, "_run_process", failed_git)
    monkeypatch.setattr(release_manifest, "_load_torch", missing_torch)
    monkeypatch.setattr(
        release_manifest,
        "_module_a_native_manifest",
        lambda: {
            "discoverable": False,
            "available": False,
            "origin": None,
            "module_file": None,
            "error": "ModuleNotFoundError: module_a_native",
            "discovery_error": None,
        },
    )

    manifest = release_manifest.build_release_manifest(
        config_path=config_path,
        repository_root=repo_root,
    )

    source = manifest["yolo"]["source_pt"]
    assert source["selection_status"] == "not_found"
    assert source["path"] is None
    assert [item["raw_path"] for item in source["candidates"]] == [
        "assets/missing-source.pt"
    ]
    classifier = manifest["a4_classifier"]
    assert classifier["configured"] is True
    assert classifier["path"] == str((model_root / "assets" / "missing-a4.pkl").resolve())
    assert classifier["exists"] is False
    assert classifier["loadable"] is False
    assert str(fallback_dir / "a4_classifier.pkl") != classifier["path"]
    assert manifest["repository"]["available"] is False
    assert len(manifest["repository"]["errors"]) == 3
    assert manifest["environment"]["torch"]["available"] is False
    assert manifest["environment"]["gpu"]["count"] == 0


def test_cli_parses_smoke_json_and_delegates_to_production(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "build_release_manifest.py"
    spec = importlib.util.spec_from_file_location("build_release_manifest_tool", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"smoke": kwargs["smoke_result"]}

    monkeypatch.setattr(tool, "build_release_manifest", fake_build)
    config_path = tmp_path / "runtime.yaml"
    repo_root = tmp_path / "repo"

    exit_code = tool.main(
        [
            "--config",
            str(config_path),
            "--repository-root",
            str(repo_root),
            "--profile",
            "release",
            "--smoke-json",
            '{"passed": true, "frames": 12}',
        ]
    )

    assert exit_code == 0
    assert captured == {
        "config_path": config_path,
        "profile": "release",
        "repository_root": repo_root,
        "smoke_result": {"passed": True, "frames": 12},
    }
    assert json.loads(capsys.readouterr().out) == {
        "smoke": {"passed": True, "frames": 12}
    }


def _write_config(
    model_root: Path,
    *,
    classifier_path: str = "assets/a4_classifier.pkl",
    source_pt_path: str = "assets/source.pt",
) -> Path:
    config_path = model_root / "configs" / "module_a_runtime.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "inference:",
                "  model_family: yolov8",
                "  backend: tensorrt",
                "  device: cpu",
                "  artifacts:",
                "    engine:",
                "      - assets/runtime.engine",
                "    pytorch:",
                f"      - {source_pt_path}",
                "module_a:",
                "  frame_size: 640",
                f"  a4_classifier_path: {classifier_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from defense.runtime.artifacts import artifact_diagnostics, resolve_artifact_candidate
from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config, project_root


SCHEMA_VERSION = 1
A4_RUNTIME_FEATURE_DIMENSION = 20
_GIT_CONFLICT_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})


def build_release_manifest(
    *,
    config_path: str | Path | None = None,
    profile: str = "default",
    repository_root: str | Path | None = None,
    smoke_result: Any = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a read-only release manifest from configured runtime assets."""
    runtime_root = Path(project_root()).resolve()
    repo_root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else runtime_root.parent.resolve()
    )
    config = Path(config_path).expanduser() if config_path is not None else DEFAULT_CONFIG_PATH
    config = config.resolve()
    runtime_config = load_runtime_config(config_path=config, profile=profile)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "repository": _git_manifest(repo_root),
        "configuration": {
            "profile": str(profile or "default"),
            "runtime_project_root": str(runtime_root),
            **_file_record(config, raw_path=str(config_path) if config_path is not None else None),
        },
        "yolo": _yolo_manifest(runtime_config, runtime_root),
        "a4_classifier": _a4_classifier_manifest(runtime_config, runtime_root),
        "raft": _raft_manifest(runtime_root),
        "module_a_native": _module_a_native_manifest(),
        "environment": _environment_manifest(),
        "smoke": smoke_result,
    }


def dumps_release_manifest(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_release_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dumps_release_manifest(manifest), encoding="utf-8")
    return output_path


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path | None, *, raw_path: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "raw_path": raw_path,
        "path": str(path) if path is not None else None,
        "exists": False,
        "is_file": False,
        "size_bytes": None,
        "sha256": None,
        "error": None,
    }
    if path is None:
        return record
    try:
        record["exists"] = path.exists()
        record["is_file"] = path.is_file()
        if record["is_file"]:
            record["size_bytes"] = int(path.stat().st_size)
            record["sha256"] = _sha256_file(path)
    except OSError as exc:
        record["error"] = _error_text(exc)
    return record


def _yolo_manifest(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    diagnostics = artifact_diagnostics(config, root=runtime_root)
    selected = Path(diagnostics["selected"]) if diagnostics.get("selected") else None
    runtime_record = _file_record(selected)
    runtime_record.update(
        {
            "selection_status": "selected" if selected is not None else "not_found",
            "candidates": [
                {
                    "kind": item.get("kind"),
                    "raw_path": item.get("path"),
                    "path": item.get("resolved_path"),
                    "exists": bool(item.get("exists")),
                    "is_file": _is_file(Path(str(item["resolved_path"]))),
                }
                for item in diagnostics.get("candidates", [])
                if item.get("resolved_path")
            ],
        }
    )
    return {
        "backend": diagnostics.get("backend"),
        "model_family": diagnostics.get("model_family"),
        "runtime_artifact": runtime_record,
        "source_pt": _source_pt_manifest(config, runtime_root),
    }


def _source_pt_manifest(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    custom_model = (
        runtime.get("custom_model", {})
        if isinstance(runtime.get("custom_model"), dict)
        else {}
    )
    artifacts = (
        inference.get("artifacts", {})
        if isinstance(inference.get("artifacts"), dict)
        else {}
    )

    selection_source = "unconfigured"
    raw_candidates: list[str] = []
    custom_source = str(custom_model.get("source_pt_path", "") or "").strip()
    inference_source = str(inference.get("source_pt_path", "") or "").strip()
    if custom_source:
        selection_source = "runtime.custom_model.source_pt_path"
        raw_candidates = [custom_source]
    elif inference_source:
        selection_source = "inference.source_pt_path"
        raw_candidates = [inference_source]
    else:
        raw_candidates = _path_values(artifacts.get("pytorch"))
        if raw_candidates:
            selection_source = "inference.artifacts.pytorch"

    candidates: list[dict[str, Any]] = []
    selected_path: Path | None = None
    selected_raw: str | None = None
    for raw in raw_candidates:
        resolved = resolve_artifact_candidate(raw, root=runtime_root)
        record = _file_record(resolved, raw_path=raw)
        candidates.append(record)
        if selected_path is None and record["is_file"]:
            selected_path = resolved
            selected_raw = raw

    selected_record = _file_record(selected_path, raw_path=selected_raw)
    selected_record.update(
        {
            "selection_source": selection_source,
            "selection_status": (
                "selected"
                if selected_path is not None
                else ("unconfigured" if not raw_candidates else "not_found")
            ),
            "candidates": candidates,
        }
    )
    return selected_record


def _path_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _a4_classifier_manifest(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    module_config = (
        config.get("module_a", {}) if isinstance(config.get("module_a"), dict) else {}
    )
    raw_path = str(module_config.get("a4_classifier_path", "") or "").strip()
    if not raw_path:
        return {
            "configured": False,
            "resolution": "unconfigured_no_fallback_selected",
            **_file_record(None),
            **_unloaded_classifier_status("classifier path is not configured"),
        }

    classifier_path = _resolve_strict_config_path(raw_path, runtime_root)
    record = _file_record(classifier_path, raw_path=raw_path)
    if not record["is_file"]:
        status = _unloaded_classifier_status(
            record["error"] or "configured classifier file does not exist"
        )
    else:
        status = _load_classifier_status(classifier_path)
    return {
        "configured": True,
        "resolution": (
            "absolute_config_path"
            if Path(raw_path).expanduser().is_absolute()
            else "config_path_relative_to_runtime_project_root"
        ),
        **record,
        **status,
    }


def _resolve_strict_config_path(raw_path: str, runtime_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _unloaded_classifier_status(error: str) -> dict[str, Any]:
    return {
        "loadable": False,
        "runtime_usable": False,
        "classifier_type": None,
        "predict_proba": False,
        "feature_dimension": None,
        "feature_dimension_source": None,
        "runtime_feature_dimension": A4_RUNTIME_FEATURE_DIMENSION,
        "feature_dimension_status": "unavailable",
        "load_error": error,
    }


def _load_classifier_status(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            classifier = pickle.load(handle)
    except Exception as exc:
        return _unloaded_classifier_status(_error_text(exc))

    predicts_probability = callable(getattr(classifier, "predict_proba", None))
    dimension, dimension_source = _classifier_feature_dimension(classifier)
    if dimension is None:
        dimension_status = "unknown"
    elif dimension == A4_RUNTIME_FEATURE_DIMENSION:
        dimension_status = "match"
    else:
        dimension_status = "runtime_pad_or_truncate"
    return {
        "loadable": True,
        "runtime_usable": predicts_probability,
        "classifier_type": (
            f"{type(classifier).__module__}.{type(classifier).__qualname__}"
        ),
        "predict_proba": predicts_probability,
        "feature_dimension": dimension,
        "feature_dimension_source": dimension_source,
        "runtime_feature_dimension": A4_RUNTIME_FEATURE_DIMENSION,
        "feature_dimension_status": dimension_status,
        "load_error": None if predicts_probability else "predict_proba is not callable",
    }


def _classifier_feature_dimension(classifier: Any) -> tuple[int | None, str | None]:
    dimension = _positive_int(getattr(classifier, "n_features_in_", None))
    if dimension is not None:
        return dimension, "n_features_in_"

    importances = getattr(classifier, "feature_importances_", None)
    if importances is not None:
        try:
            dimension = _positive_int(len(importances))
        except TypeError:
            dimension = None
        if dimension is not None:
            return dimension, "feature_importances_"

    get_booster = getattr(classifier, "get_booster", None)
    if callable(get_booster):
        try:
            dimension = _positive_int(get_booster().num_features())
        except Exception:
            dimension = None
        if dimension is not None:
            return dimension, "get_booster().num_features()"
    return None, None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _raft_manifest(runtime_root: Path) -> dict[str, Any]:
    data_dir, resolution, candidates = _resolve_rebuilt_data_dir(runtime_root)
    return {
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.exists(),
        "data_dir_resolution": resolution,
        "data_dir_candidates": candidates,
        "onnx": _file_record(data_dir / "raft_small_256.onnx"),
        "engine": _file_record(data_dir / "raft_small_fp16_256.engine"),
    }


def _resolve_rebuilt_data_dir(
    runtime_root: Path,
) -> tuple[Path, str, list[dict[str, Any]]]:
    bundled = runtime_root / "src" / "defense" / "module_a" / "rebuilt" / "data"
    candidates: list[tuple[str, Path]] = []
    env_dir = os.environ.get("MODULE_A_REBUILT_DATA_DIR")
    if env_dir:
        candidates.append(("MODULE_A_REBUILT_DATA_DIR", Path(env_dir).expanduser()))
    candidates.extend(
        [
            ("bundled_rebuilt_data", bundled),
            ("model_data", runtime_root / "data"),
            ("repository_rebuilt_demo_data", runtime_root.parent / "rebuilt_demo" / "data"),
        ]
    )

    candidate_records: list[dict[str, Any]] = []
    selected: Path | None = None
    selected_source = "bundled_default_no_existing_data_dir"
    for source, candidate in candidates:
        resolved = candidate.resolve()
        exists = resolved.exists()
        candidate_records.append(
            {"source": source, "path": str(resolved), "exists": exists}
        )
        if selected is None and exists:
            selected = resolved
            selected_source = source
    return selected or bundled.resolve(), selected_source, candidate_records


def _module_a_native_manifest() -> dict[str, Any]:
    discover_error: str | None = None
    try:
        spec = importlib.util.find_spec("module_a_native")
    except Exception as exc:
        spec = None
        discover_error = _error_text(exc)

    try:
        module = importlib.import_module("module_a_native")
    except Exception as exc:
        return {
            "discoverable": spec is not None,
            "available": False,
            "origin": getattr(spec, "origin", None),
            "module_file": None,
            "error": _error_text(exc),
            "discovery_error": discover_error,
        }
    return {
        "discoverable": spec is not None,
        "available": True,
        "origin": getattr(spec, "origin", None),
        "module_file": getattr(module, "__file__", None),
        "error": None,
        "discovery_error": discover_error,
    }


def _environment_manifest() -> dict[str, Any]:
    pixi_project_root = os.environ.get("PIXI_PROJECT_ROOT")
    pixi_environment_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    executable_hint = f"{sys.executable} {sys.prefix}".lower().replace("\\", "/")
    pixi_detected = bool(
        pixi_project_root
        or pixi_environment_name
        or os.environ.get("PIXI_IN_SHELL")
        or "/.pixi/" in executable_hint
    )

    environment: dict[str, Any] = {
        "pixi": {
            "detected": pixi_detected,
            "project_root": pixi_project_root,
            "environment_name": pixi_environment_name,
            "in_shell": os.environ.get("PIXI_IN_SHELL"),
            "environment_prefix": os.environ.get("CONDA_PREFIX") or sys.prefix,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
    }

    try:
        torch = _load_torch()
    except Exception as exc:
        error = _error_text(exc)
        environment.update(
            {
                "torch": {"available": False, "version": None, "error": error},
                "cuda": {
                    "available": False,
                    "torch_build_version": None,
                    "cudnn_version": None,
                    "error": "torch unavailable",
                },
                "gpu": {"count": 0, "devices": [], "error": "torch unavailable"},
            }
        )
        return environment

    environment["torch"] = {
        "available": True,
        "version": str(getattr(torch, "__version__", "")) or None,
        "error": None,
    }
    environment.update(_torch_accelerator_manifest(torch))
    return environment


def _load_torch() -> Any:
    return importlib.import_module("torch")


def _torch_accelerator_manifest(torch: Any) -> dict[str, Any]:
    build_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return {
            "cuda": {
                "available": False,
                "torch_build_version": build_cuda,
                "cudnn_version": _cudnn_version(torch),
                "error": _error_text(exc),
            },
            "gpu": {"count": 0, "devices": [], "error": "CUDA probe failed"},
        }

    devices: list[dict[str, Any]] = []
    gpu_error: str | None = None
    device_count = 0
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
            for index in range(device_count):
                devices.append(_gpu_device_record(torch, index))
        except Exception as exc:
            gpu_error = _error_text(exc)
    return {
        "cuda": {
            "available": cuda_available,
            "torch_build_version": build_cuda,
            "cudnn_version": _cudnn_version(torch),
            "error": None,
        },
        "gpu": {
            "count": device_count,
            "devices": devices,
            "error": gpu_error,
        },
    }


def _gpu_device_record(torch: Any, index: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": index,
        "name": str(torch.cuda.get_device_name(index)),
        "capability": None,
        "total_memory_bytes": None,
    }
    get_capability = getattr(torch.cuda, "get_device_capability", None)
    if callable(get_capability):
        record["capability"] = list(get_capability(index))
    get_properties = getattr(torch.cuda, "get_device_properties", None)
    if callable(get_properties):
        properties = get_properties(index)
        memory = getattr(properties, "total_memory", None)
        record["total_memory_bytes"] = int(memory) if memory is not None else None
    return record


def _cudnn_version(torch: Any) -> int | None:
    try:
        version = torch.backends.cudnn.version()
    except Exception:
        return None
    return int(version) if version is not None else None


def _git_manifest(repository_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    head_result = _run_git(repository_root, ["rev-parse", "HEAD"])
    head = _stdout_value(head_result)
    if head_result.returncode != 0:
        errors.append(_command_error("git rev-parse HEAD", head_result))

    branch_result = _run_git(
        repository_root, ["symbolic-ref", "--quiet", "--short", "HEAD"]
    )
    branch = _stdout_value(branch_result) if branch_result.returncode == 0 else None
    detached = bool(head and branch_result.returncode == 1)
    if branch_result.returncode not in {0, 1}:
        errors.append(_command_error("git symbolic-ref", branch_result))

    status_result = _run_git(
        repository_root, ["status", "--porcelain=v1", "--untracked-files=normal"]
    )
    if status_result.returncode == 0:
        dirty = _dirty_summary(status_result.stdout)
    else:
        dirty = {
            "available": False,
            "is_dirty": None,
            "entry_count": None,
            "staged": None,
            "unstaged": None,
            "untracked": None,
            "conflicted": None,
            "status_counts": {},
        }
        errors.append(_command_error("git status", status_result))

    return {
        "root": str(repository_root),
        "available": head_result.returncode == 0 and status_result.returncode == 0,
        "head": head,
        "branch": branch,
        "detached": detached,
        "dirty": dirty,
        "errors": errors,
    }


def _run_git(repository_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return _run_process(
        ["git", "-C", str(repository_root), *arguments],
        cwd=repository_root,
    )


def _run_process(
    command: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _stdout_value(result: subprocess.CompletedProcess[str]) -> str | None:
    value = str(result.stdout or "").strip()
    return value or None


def _dirty_summary(porcelain: str) -> dict[str, Any]:
    codes = [line[:2] for line in porcelain.splitlines() if len(line) >= 2]
    counts = Counter(codes)
    return {
        "available": True,
        "is_dirty": bool(codes),
        "entry_count": len(codes),
        "staged": sum(1 for code in codes if code[0] not in {" ", "?"}),
        "unstaged": sum(1 for code in codes if code[1] not in {" ", "?"}),
        "untracked": counts.get("??", 0),
        "conflicted": sum(counts.get(code, 0) for code in _GIT_CONFLICT_CODES),
        "status_counts": dict(sorted(counts.items())),
    }


def _command_error(label: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = str(result.stderr or result.stdout or "").strip()
    suffix = f": {detail}" if detail else ""
    return f"{label} failed with exit code {result.returncode}{suffix}"


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"

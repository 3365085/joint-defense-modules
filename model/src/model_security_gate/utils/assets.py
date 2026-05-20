from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from model_security_gate.utils.config import load_yaml_config
from model_security_gate.utils.io import IMAGE_EXTS, read_yaml


@dataclass(frozen=True)
class AssetConfig:
    suspicious_model: Path
    teacher_model: Path | None
    train_images: Path
    train_labels: Path
    data_yaml: Path
    external_replay_roots: tuple[Path, ...]
    external_eval_roots: tuple[Path, ...]
    source_materials: Path
    output_root: Path
    target_classes: tuple[str, ...]
    device: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_asset_config(path: str | Path = "configs/assets.local.yaml") -> AssetConfig:
    repo_root = _repo_root()
    data = load_yaml_config(path)
    raw = data.get("assets", data)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Asset config must contain an assets mapping: {path}")

    required = [
        "suspicious_model",
        "train_images",
        "train_labels",
        "data_yaml",
        "external_replay_roots",
        "external_eval_roots",
        "target_classes",
    ]
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError("Missing asset config keys: " + ", ".join(missing))

    return AssetConfig(
        suspicious_model=_resolve_path(raw["suspicious_model"], repo_root),
        teacher_model=_resolve_path(raw["teacher_model"], repo_root) if raw.get("teacher_model") else None,
        train_images=_resolve_path(raw["train_images"], repo_root),
        train_labels=_resolve_path(raw["train_labels"], repo_root),
        data_yaml=_resolve_path(raw["data_yaml"], repo_root),
        external_replay_roots=tuple(_resolve_path(item, repo_root) for item in _as_list(raw.get("external_replay_roots"))),
        external_eval_roots=tuple(_resolve_path(item, repo_root) for item in _as_list(raw.get("external_eval_roots"))),
        source_materials=_resolve_path(raw.get("source_materials", "source_materials"), repo_root),
        output_root=_resolve_path(raw.get("output_root", "runs/hybrid_purify_primary"), repo_root),
        target_classes=tuple(str(item) for item in _as_list(raw.get("target_classes"))),
        device=str(raw.get("device", "0")),
    )


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def _count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.txt") if item.is_file())


def validate_asset_config(config: AssetConfig) -> list[str]:
    errors: list[str] = []
    for name in ["suspicious_model", "data_yaml"]:
        path = getattr(config, name)
        if not path.is_file():
            errors.append(f"{name} not found: {path}")
    if config.teacher_model is not None and not config.teacher_model.is_file():
        errors.append(f"teacher_model not found: {config.teacher_model}")
    for name in ["train_images", "train_labels", "source_materials"]:
        path = getattr(config, name)
        if not path.is_dir():
            errors.append(f"{name} not found: {path}")
    for index, path in enumerate(config.external_replay_roots):
        if not path.is_dir():
            errors.append(f"external_replay_roots[{index}] not found: {path}")
    for index, path in enumerate(config.external_eval_roots):
        if not path.is_dir():
            errors.append(f"external_eval_roots[{index}] not found: {path}")
    if config.train_images.is_dir() and _count_images(config.train_images) == 0:
        errors.append(f"train_images contains no supported images: {config.train_images}")
    if config.train_labels.is_dir() and _count_labels(config.train_labels) == 0:
        errors.append(f"train_labels contains no YOLO txt labels: {config.train_labels}")
    if config.data_yaml.is_file():
        data = read_yaml(config.data_yaml)
        names = data.get("names")
        if not names:
            errors.append(f"data_yaml has no names mapping: {config.data_yaml}")
    if not config.target_classes:
        errors.append("target_classes is empty")
    return errors

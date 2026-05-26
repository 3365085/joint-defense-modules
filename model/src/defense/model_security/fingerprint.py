from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from defense.runtime.artifacts import artifact_diagnostics

SCANNER_VERSION = "model_security_runtime_v2"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ModelFingerprint:
    fingerprint: str
    model_hash: str | None
    model_path: str | None
    backend: str
    model_family: str | None
    image_size: Any
    confidence: Any
    nms_iou: Any
    class_names_hash: str
    ppe_mapping_hash: str
    scanner_version: str = SCANNER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _class_names_payload(config: dict[str, Any]) -> Any:
    inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    for key in ("names", "class_names", "labels"):
        if key in inference:
            return inference.get(key)
    return None


def _ppe_payload(config: dict[str, Any]) -> Any:
    payload: dict[str, Any] = {}
    for key in ("ppe", "ppe_tracking", "ppe_postprocess"):
        if isinstance(config.get(key), dict):
            payload[key] = config[key]
    module_a = config.get("module_a", {}) if isinstance(config.get("module_a"), dict) else {}
    for key in ("track_labels", "static_image_screen_labels", "static_image_target_labels"):
        if key in module_a:
            payload[f"module_a.{key}"] = module_a[key]
    return payload


def build_model_fingerprint(config: dict[str, Any], *, root: str | Path | None = None) -> ModelFingerprint:
    inference = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    diag = artifact_diagnostics(config, Path(root) if root else None)
    selected = diag.get("selected")
    model_hash = sha256_file(selected) if selected and Path(selected).exists() else None
    model_identity = {
        "model_hash": model_hash,
        "selected": str(selected) if selected else None,
        "backend": str(inference.get("backend", diag.get("backend", "unknown"))).lower(),
        "model_family": inference.get("model_family", inference.get("family", diag.get("model_family"))),
        "image_size": inference.get("image_size", inference.get("imgsz")),
        "confidence": inference.get("confidence", inference.get("conf")),
        "nms_iou": inference.get("iou", inference.get("nms_iou")),
        "class_names_hash": stable_json_hash(_class_names_payload(config)),
        "ppe_mapping_hash": stable_json_hash(_ppe_payload(config)),
        "scanner_version": SCANNER_VERSION,
    }
    return ModelFingerprint(
        fingerprint="sha256:" + stable_json_hash(model_identity),
        model_hash=("sha256:" + model_hash) if model_hash else None,
        model_path=str(selected) if selected else None,
        backend=str(model_identity["backend"]),
        model_family=model_identity["model_family"],
        image_size=model_identity["image_size"],
        confidence=model_identity["confidence"],
        nms_iou=model_identity["nms_iou"],
        class_names_hash=model_identity["class_names_hash"],
        ppe_mapping_hash=model_identity["ppe_mapping_hash"],
        scanner_version=SCANNER_VERSION,
    )

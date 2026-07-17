from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from defense.module_a.rebuilt.detector import (
    A4_FEATURE_NAMES,
    A4_FEATURE_SCHEMA_VERSION,
)

from .a4_dataset import (
    ADV_PATCH_TRAJECTORY_MODES,
    REQUIRED_ATTACK_TYPES,
    UNIQUE_YOLO_SOURCE_SHA256,
    assert_no_rebuilt_demo_path,
    load_a4_dataset_manifest,
)
from .module_a_tuning import build_effective_config, create_production_pipeline


A4_PHYSICAL_SCHEMA_VERSION = A4_FEATURE_SCHEMA_VERSION
A4_PHYSICAL_FEATURE_NAMES = tuple(A4_FEATURE_NAMES)
DEFAULT_MANIFEST_PATH: Path | None = None
ProgressCallback = Callable[[dict[str, Any]], None]
A4_CLASSIFIER_ROLE = "adv_patch_rescue"


class A4QualityGateError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        failures = self.report.get("quality_gate", {}).get("failures", [])
        super().__init__(f"a4_production_quality_gate_failed:{','.join(map(str, failures))}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_label(row: Mapping[str, Any], frame_idx: int) -> int:
    if int(row.get("label", 0) or 0) == 0:
        return 0
    attack_start = max(0, int(row.get("attack_start_frame", 0) or 0))
    ramp_frames = max(0, int(row.get("attack_ramp_frames", 0) or 0))
    if frame_idx < attack_start:
        return 0
    if frame_idx < attack_start + ramp_frames:
        return -1
    return 1


def _manifest_rows(
    manifest_path: str | Path,
    *,
    splits: Iterable[str],
) -> list[dict[str, Any]]:
    rows, _ = _manifest_rows_with_metadata(manifest_path, splits=splits)
    return rows


def _manifest_rows_with_metadata(
    manifest_path: str | Path,
    *,
    splits: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metadata = load_a4_dataset_manifest(
        manifest_path,
        verify_content_hashes=True,
    )
    requested = {str(split).strip() for split in splits if str(split).strip()}
    if requested:
        rows = [row for row in rows if str(row["split"]) in requested]
    for row in rows:
        row["path"] = str(Path(str(row["path"])).expanduser().resolve())
    return rows, metadata


def collect_production_a4_features(
    *,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    output_csv: str | Path,
    metadata_path: str | Path | None = None,
    splits: Sequence[str] = ("train", "heldout"),
    profile: str = "desktop_rtx",
    config_path: str | Path | None = None,
    max_frames_per_video: int = 300,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Collect the synchronous A1/A2/A3 feature contract from production code.

    The existing classifier is deliberately disabled while collecting so its
    decisions cannot feed back into baseline updates and contaminate features.
    """

    if manifest_path is None:
        raise ValueError(
            "A4 training requires an explicit training manifest that is "
            "separate from the authoritative final-acceptance videos."
        )
    manifest = assert_no_rebuilt_demo_path(manifest_path, field="a4_dataset_manifest")
    destination = assert_no_rebuilt_demo_path(output_csv, field="a4_features_output")
    metadata_destination = (
        assert_no_rebuilt_demo_path(metadata_path, field="a4_features_metadata")
        if metadata_path
        else destination.with_suffix(destination.suffix + ".meta.json")
    )
    rows, dataset_metadata = _manifest_rows_with_metadata(manifest, splits=splits)
    missing = [row["path"] for row in rows if not Path(str(row["path"])).is_file()]
    if missing:
        raise FileNotFoundError(
            f"A4 training manifest contains {len(missing)} missing videos; first={missing[0]}"
        )

    config, _ = build_effective_config(
        config_path=config_path,
        profile=profile,
        tuning_patch=None,
    )
    module_a = config.setdefault("module_a", {})
    disabled_classifier = destination.parent / "__a4_collection_classifier_disabled__.pkl"
    module_a["a4_classifier_path"] = str(disabled_classifier)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    fieldnames = [
        "clip_id",
        "video",
        "base_clip_id",
        "base_source_sha256",
        "content_sha256",
        "scene_id",
        "split",
        "attack_type",
        "trajectory_mode",
        "provenance_id",
        "source_manifest_sha256",
        "unique_yolo_source_sha256",
        "frame_idx",
        "source_time_s",
        "label",
        *A4_PHYSICAL_FEATURE_NAMES,
    ]

    started = time.perf_counter()
    written_rows = 0
    processed_frames = 0
    completed_videos = 0
    pipeline = create_production_pipeline(config)
    backend = getattr(pipeline, "detector_backend", None)
    backend_artifact = Path(str(getattr(backend, "artifact_path", ""))).expanduser()
    warmup_frames = int(getattr(pipeline, "warmup_frames", 0) or 0)
    try:
        if warmup_frames > 0:
            pipeline.warmup(warmup_frames)
        with temp_destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for video_index, manifest_row in enumerate(rows, start=1):
                video_path = Path(str(manifest_row["path"]))
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Unable to open A4 training video: {video_path}")
                source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                if not 0.1 <= source_fps <= 240.0:
                    source_fps = 30.0
                pipeline.reset()
                frame_idx = 0
                try:
                    while max_frames_per_video <= 0 or frame_idx < max_frames_per_video:
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            break
                        source_time_s = frame_idx / source_fps
                        _, _, info = pipeline.process_frame(
                            frame,
                            timestamp=source_time_s,
                            source_fps=source_fps,
                            source_frame_idx=frame_idx,
                        )
                        details = info.get("details", {}) if isinstance(info, Mapping) else {}
                        schema = (
                            details.get("a4_feature_schema", {})
                            if isinstance(details, Mapping)
                            else {}
                        )
                        a4 = details.get("a4", {}) if isinstance(details, Mapping) else {}
                        vector = a4.get("a4_feature_vector", []) if isinstance(a4, Mapping) else []
                        runtime_names = tuple(schema.get("names", []) or ())
                        if runtime_names != A4_PHYSICAL_FEATURE_NAMES:
                            raise RuntimeError(
                                "Production A4 feature schema mismatch while collecting: "
                                f"runtime={runtime_names!r}, expected={A4_PHYSICAL_FEATURE_NAMES!r}"
                            )
                        if len(vector) < len(A4_PHYSICAL_FEATURE_NAMES):
                            raise RuntimeError(
                                "Production A4 feature vector is too short: "
                                f"{len(vector)} < {len(A4_PHYSICAL_FEATURE_NAMES)}"
                            )
                        label = _frame_label(manifest_row, frame_idx)
                        if label in (0, 1):
                            output_row = {
                                "clip_id": str(manifest_row["clip_id"]),
                                "video": str(video_path),
                                "base_clip_id": str(manifest_row["base_clip_id"]),
                                "base_source_sha256": str(
                                    manifest_row["base_source_sha256"]
                                ).lower(),
                                "content_sha256": str(
                                    manifest_row["content_sha256"]
                                ).lower(),
                                "scene_id": str(manifest_row["scene_id"]),
                                "split": str(manifest_row["split"]),
                                "attack_type": str(manifest_row["attack_type"]),
                                "trajectory_mode": str(
                                    manifest_row["trajectory_mode"]
                                ),
                                "provenance_id": str(manifest_row["provenance_id"]),
                                "source_manifest_sha256": str(
                                    manifest_row["source_manifest_sha256"]
                                ).lower(),
                                "unique_yolo_source_sha256": str(
                                    dataset_metadata["unique_yolo_source_sha256"]
                                ).lower(),
                                "frame_idx": int(frame_idx),
                                "source_time_s": float(source_time_s),
                                "label": int(label),
                            }
                            output_row.update(
                                {
                                    name: float(vector[index])
                                    for index, name in enumerate(A4_PHYSICAL_FEATURE_NAMES)
                                }
                            )
                            writer.writerow(output_row)
                            written_rows += 1
                        frame_idx += 1
                        processed_frames += 1
                finally:
                    capture.release()
                completed_videos += 1
                if progress:
                    progress(
                        {
                            "video_index": video_index,
                            "video_count": len(rows),
                            "clip_id": str(manifest_row["clip_id"]),
                            "video": str(video_path),
                            "frames": frame_idx,
                            "written_rows": written_rows,
                        }
                    )
    finally:
        pipeline.close()

    temp_destination.replace(destination)
    detector_source = Path(__file__).resolve().parents[1] / "module_a/rebuilt/detector.py"
    config_source = (
        Path(config_path).expanduser().resolve()
        if config_path
        else Path(__file__).resolve().parents[3] / "configs/module_a_runtime.yaml"
    )
    metadata = {
        "schema_version": 1,
        "feature_schema_version": A4_PHYSICAL_SCHEMA_VERSION,
        "feature_names": list(A4_PHYSICAL_FEATURE_NAMES),
        "collector": "defense.diagnostics.a4_training.collect_production_a4_features",
        "profile": str(profile),
        "splits": list(splits),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "dataset_manifest_sha256": _sha256_file(manifest),
        "dataset_manifest_metadata_path": str(
            manifest.with_suffix(manifest.suffix + ".meta.json")
        ),
        "source_manifest_path": str(dataset_metadata["source_manifest_path"]),
        "source_manifest_sha256": str(
            dataset_metadata["source_manifest_sha256"]
        ).lower(),
        "authoritative_manifest_sha256": str(
            dataset_metadata["authoritative_manifest_sha256"]
        ).lower(),
        "authoritative_manifest_path": str(
            dataset_metadata["authoritative_manifest_path"]
        ),
        "authoritative_video_count": int(
            dataset_metadata["authoritative_video_count"]
        ),
        "unique_yolo_source_path": str(
            dataset_metadata["unique_yolo_source_path"]
        ),
        "unique_yolo_source_sha256": str(
            dataset_metadata["unique_yolo_source_sha256"]
        ).lower(),
        "dataset_generator_version": str(dataset_metadata["generator_version"]),
        "deterministic_dataset_provenance": bool(
            dataset_metadata["deterministic_provenance"]
        ),
        "features_path": str(destination),
        "features_sha256": _sha256_file(destination),
        "detector_source_path": str(detector_source),
        "detector_source_sha256": _sha256_file(detector_source),
        "config_path": str(config_source),
        "config_sha256": _sha256_file(config_source),
        "backend": str(getattr(backend, "backend", "unknown")),
        "backend_artifact_path": str(backend_artifact),
        "backend_artifact_sha256": (
            _sha256_file(backend_artifact) if backend_artifact.is_file() else ""
        ),
        "classifier_disabled_during_collection": True,
        "max_frames_per_video": int(max_frames_per_video),
        "video_count": int(completed_videos),
        "processed_frames": int(processed_frames),
        "written_rows": int(written_rows),
        "wall_seconds": float(time.perf_counter() - started),
    }
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _alarm_flags(
    probabilities: np.ndarray,
    *,
    threshold: float,
    window: int,
    required_hits: int,
) -> np.ndarray:
    hits = (np.asarray(probabilities, dtype=np.float64) >= float(threshold)).astype(np.int16)
    if hits.size == 0:
        return np.zeros(0, dtype=bool)
    prefix = np.concatenate(([0], np.cumsum(hits)))
    counts = np.empty_like(hits)
    for index in range(hits.size):
        start = max(0, index + 1 - int(window))
        counts[index] = prefix[index + 1] - prefix[start]
    return counts >= int(required_hits)


def _video_alarm_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    window: int,
    required_hits: int,
) -> dict[str, Any]:
    columns = ["clip_id", "frame_idx", "label", "attack_type"]
    if "trajectory_mode" in frame.columns:
        columns.append("trajectory_mode")
    working = frame[columns].copy()
    if "trajectory_mode" not in working.columns:
        working["trajectory_mode"] = "none"
    working["probability"] = np.asarray(probabilities, dtype=np.float64)
    clean_total = clean_false_positive = 0
    attack_total = attack_hit = 0
    per_clip: list[dict[str, Any]] = []
    for clip_id, clip in working.groupby("clip_id", sort=False):
        clip = clip.sort_values("frame_idx")
        labels = clip["label"].to_numpy(dtype=np.int16)
        probs = clip["probability"].to_numpy(dtype=np.float64)
        alarms = _alarm_flags(
            probs,
            threshold=threshold,
            window=window,
            required_hits=required_hits,
        )
        is_attack = bool(np.any(labels == 1))
        if is_attack:
            attack_total += 1
            eligible = labels == 1
            hit = bool(np.any(alarms & eligible))
            attack_hit += int(hit)
        else:
            clean_total += 1
            hit = bool(np.any(alarms))
            clean_false_positive += int(hit)
        first_alarm = int(clip.iloc[np.flatnonzero(alarms)[0]]["frame_idx"]) if np.any(alarms) else None
        per_clip.append(
            {
                "clip_id": str(clip_id),
                "attack_type": str(clip.iloc[0]["attack_type"]),
                "trajectory_mode": str(clip.iloc[0]["trajectory_mode"]),
                "is_attack": is_attack,
                "hit": hit,
                "first_alarm_frame": first_alarm,
                "max_probability": float(np.max(probs)) if probs.size else 0.0,
            }
        )
    return {
        "threshold": float(threshold),
        "window": int(window),
        "required_hits": int(required_hits),
        "clean_videos": int(clean_total),
        "clean_false_positive_videos": int(clean_false_positive),
        "clean_false_positive_rate": (
            float(clean_false_positive / clean_total) if clean_total else 0.0
        ),
        "attack_videos": int(attack_total),
        "attack_hit_videos": int(attack_hit),
        "attack_video_recall": float(attack_hit / attack_total) if attack_total else 0.0,
        "per_clip": per_clip,
    }


_FIXED_ADV_PATCH_CANDIDATE: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def _candidate_parameters(rng: np.random.RandomState, iterations: int) -> list[dict[str, Any]]:
    space = {
        "n_estimators": [250, 350, 450, 600],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 6],
        "gamma": [0.0, 0.3, 0.8],
        "reg_lambda": [1.0, 2.0, 4.0],
        "reg_alpha": [0.0, 0.1, 0.5],
    }
    target_count = max(1, int(iterations))
    candidates: list[dict[str, Any]] = [dict(_FIXED_ADV_PATCH_CANDIDATE)]
    while len(candidates) < target_count:
        candidate = {
            key: rng.choice(values).item()
            for key, values in space.items()
        }
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _candidate_selection_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    threshold_result = candidate["best_threshold"]
    metrics = threshold_result["metrics"]
    return (
        -float(metrics["clean_false_positive_videos"]),
        float(threshold_result["score"]),
        float(metrics["attack_hit_videos"]),
        float(candidate["oof_auc"]),
    )


def _equal_clip_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("clip_id")["clip_id"].transform("count").to_numpy(dtype=np.float64)
    return np.divide(
        1.0,
        counts,
        out=np.zeros_like(counts, dtype=np.float64),
        where=counts > 0,
    )


def _class_mass_preserving_clip_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = _equal_clip_weights(frame)
    labels = frame["label"].to_numpy(dtype=np.int16)
    for label in np.unique(labels):
        label_mask = labels == label
        label_weight = float(np.sum(weights[label_mask]))
        if label_weight <= 0.0:
            continue
        weights[label_mask] *= float(np.sum(label_mask)) / label_weight
    weights *= float(len(weights)) / max(1e-9, float(np.sum(weights)))
    return weights


def _unique_content_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if "content_sha256" in frame.columns:
        clip_paths = (
            frame[["clip_id", "content_sha256"]]
            .drop_duplicates("clip_id")
            .to_dict(orient="records")
        )
    else:
        clip_paths = (
            frame[["clip_id", "video"]]
            .drop_duplicates("clip_id")
            .to_dict(orient="records")
        )
    content_by_clip: dict[str, str] = {}
    first_clip_by_hash: dict[str, str] = {}
    keep_clips: set[str] = set()
    for row in clip_paths:
        clip_id = str(row["clip_id"])
        content_hash = str(row.get("content_sha256", "")).strip().lower()
        if not content_hash:
            path = assert_no_rebuilt_demo_path(row["video"], field=f"{clip_id}.video")
            content_hash = _sha256_file(path)
        content_by_clip[clip_id] = content_hash
        if content_hash not in first_clip_by_hash:
            first_clip_by_hash[content_hash] = clip_id
            keep_clips.add(clip_id)
    return (
        frame[frame["clip_id"].astype(str).isin(keep_clips)].reset_index(drop=True),
        content_by_clip,
    )


def _validated_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid_sha256:{field}:{text or '<missing>'}")
    return text


def _feature_metadata_path(source: Path) -> Path:
    candidates = (
        source.with_suffix(source.suffix + ".meta.json"),
        source.with_suffix(".meta.json"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ValueError(f"a4_feature_metadata_missing:{candidates[0]}")
    return path


def _load_feature_provenance(source: Path) -> tuple[Path, dict[str, Any]]:
    path = _feature_metadata_path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"a4_feature_metadata_invalid:{type(exc).__name__}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("a4_feature_metadata_invalid:not_an_object")
    if str(payload.get("feature_schema_version", "")) != A4_PHYSICAL_SCHEMA_VERSION:
        raise ValueError("a4_feature_schema_version_mismatch")
    if tuple(payload.get("feature_names", []) or ()) != A4_PHYSICAL_FEATURE_NAMES:
        raise ValueError("a4_feature_name_order_mismatch")
    declared_features_hash = _validated_sha256(
        payload.get("features_sha256"),
        field="features_sha256",
    )
    actual_features_hash = _sha256_file(source)
    if declared_features_hash != actual_features_hash:
        raise ValueError(
            "a4_features_sha256_mismatch:"
            f"expected={declared_features_hash},actual={actual_features_hash}"
        )
    bound_files = (
        ("source_manifest_path", "source_manifest_sha256"),
        ("manifest_path", "dataset_manifest_sha256"),
        ("authoritative_manifest_path", "authoritative_manifest_sha256"),
        ("unique_yolo_source_path", "unique_yolo_source_sha256"),
    )
    for path_field, hash_field in bound_files:
        bound_path = assert_no_rebuilt_demo_path(
            payload.get(path_field, ""),
            field=path_field,
        )
        if not bound_path.is_file():
            raise ValueError(f"a4_bound_file_missing:{path_field}:{bound_path}")
        declared_hash = _validated_sha256(
            payload.get(hash_field),
            field=hash_field,
        )
        actual_hash = _sha256_file(bound_path)
        if actual_hash != declared_hash:
            raise ValueError(
                "a4_bound_file_sha256_mismatch:"
                f"{hash_field}:expected={declared_hash},actual={actual_hash}"
            )
    source_manifest_hash = _validated_sha256(
        payload.get("source_manifest_sha256"),
        field="source_manifest_sha256",
    )
    dataset_manifest_hash = _validated_sha256(
        payload.get("dataset_manifest_sha256") or payload.get("manifest_sha256"),
        field="dataset_manifest_sha256",
    )
    authoritative_manifest_hash = _validated_sha256(
        payload.get("authoritative_manifest_sha256"),
        field="authoritative_manifest_sha256",
    )
    yolo_hash = _validated_sha256(
        payload.get("unique_yolo_source_sha256"),
        field="unique_yolo_source_sha256",
    )
    if yolo_hash != UNIQUE_YOLO_SOURCE_SHA256.lower():
        raise ValueError(
            "unique_yolo_source_sha256_mismatch:"
            f"expected={UNIQUE_YOLO_SOURCE_SHA256.lower()},actual={yolo_hash}"
        )
    if int(payload.get("authoritative_video_count", 0) or 0) != 36:
        raise ValueError("authoritative_video_count_mismatch:expected=36")
    for key, value in payload.items():
        if key.endswith("_path") and value:
            assert_no_rebuilt_demo_path(value, field=key)
    payload["source_manifest_sha256"] = source_manifest_hash
    payload["dataset_manifest_sha256"] = dataset_manifest_hash
    payload["authoritative_manifest_sha256"] = authoritative_manifest_hash
    payload["unique_yolo_source_sha256"] = yolo_hash
    return path, payload


def _validate_feature_dataset(frame: pd.DataFrame, provenance: Mapping[str, Any]) -> None:
    required = {
        *A4_PHYSICAL_FEATURE_NAMES,
        "video",
        "split",
        "scene_id",
        "clip_id",
        "base_clip_id",
        "base_source_sha256",
        "content_sha256",
        "attack_type",
        "trajectory_mode",
        "provenance_id",
        "source_manifest_sha256",
        "unique_yolo_source_sha256",
        "frame_idx",
        "label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"A4 production feature dataset is missing columns: {missing}")
    if frame.empty:
        raise ValueError("A4 production feature dataset is empty")
    expected_source_hash = str(provenance["source_manifest_sha256"])
    expected_yolo_hash = UNIQUE_YOLO_SOURCE_SHA256.lower()
    if set(frame["source_manifest_sha256"].astype(str).str.lower()) != {expected_source_hash}:
        raise ValueError("feature_rows_source_manifest_binding_mismatch")
    if set(frame["unique_yolo_source_sha256"].astype(str).str.lower()) != {expected_yolo_hash}:
        raise ValueError("feature_rows_unique_yolo_binding_mismatch")

    clip_contract = frame[
        [
            "clip_id",
            "video",
            "base_clip_id",
            "base_source_sha256",
            "content_sha256",
            "scene_id",
            "split",
            "attack_type",
            "trajectory_mode",
            "provenance_id",
        ]
    ].drop_duplicates()
    clip_counts = clip_contract.groupby("clip_id").size()
    if bool((clip_counts != 1).any()):
        raise ValueError(
            f"clip_contract_inconsistent:{clip_counts[clip_counts != 1].index.tolist()}"
        )
    for path in clip_contract["video"].astype(str):
        assert_no_rebuilt_demo_path(path, field="feature.video")
    for column in ("base_source_sha256", "content_sha256", "provenance_id"):
        for value in clip_contract[column].astype(str):
            _validated_sha256(value, field=column)
    duplicate_content = clip_contract.groupby("content_sha256")["clip_id"].nunique()
    if bool((duplicate_content > 1).any()):
        raise ValueError(
            "feature_content_not_deduplicated:"
            f"{duplicate_content[duplicate_content > 1].index.tolist()}"
        )

    base_contract = clip_contract[
        ["base_clip_id", "base_source_sha256", "scene_id", "split"]
    ].drop_duplicates()
    base_counts = base_contract.groupby("base_clip_id").size()
    if bool((base_counts != 1).any()):
        raise ValueError(
            f"base_group_split_scene_leakage:{base_counts[base_counts != 1].index.tolist()}"
        )
    hash_base_counts = base_contract.groupby("base_source_sha256")["base_clip_id"].nunique()
    if bool((hash_base_counts != 1).any()):
        raise ValueError("base_source_content_reused_across_groups")
    scene_base_counts = base_contract.groupby("scene_id")["base_clip_id"].nunique()
    if bool((scene_base_counts != 1).any()):
        raise ValueError("scene_id_reused_across_base_groups")
    train_bases = set(base_contract.loc[base_contract["split"] == "train", "base_clip_id"])
    heldout_bases = set(
        base_contract.loc[base_contract["split"] == "heldout", "base_clip_id"]
    )
    if train_bases & heldout_bases:
        raise ValueError(f"train_heldout_base_group_overlap:{sorted(train_bases & heldout_bases)}")
    train_scenes = set(base_contract.loc[base_contract["split"] == "train", "scene_id"])
    heldout_scenes = set(base_contract.loc[base_contract["split"] == "heldout", "scene_id"])
    if train_scenes & heldout_scenes:
        raise ValueError(f"train_heldout_scene_overlap:{sorted(train_scenes & heldout_scenes)}")

    expected_types = {"clean", *REQUIRED_ATTACK_TYPES}
    for base_clip_id, variants in clip_contract.groupby("base_clip_id", sort=False):
        attack_types = set(variants["attack_type"].astype(str))
        missing_types = expected_types - attack_types
        if missing_types:
            raise ValueError(
                f"base_group_attack_coverage_missing:{base_clip_id}:{sorted(missing_types)}"
            )
        trajectories = set(
            variants.loc[
                variants["attack_type"].astype(str) == "adv_patch",
                "trajectory_mode",
            ].astype(str)
        )
        if not trajectories or not trajectories.issubset(set(ADV_PATCH_TRAJECTORY_MODES)):
            raise ValueError(
                "base_group_adv_patch_trajectory_invalid:"
                f"{base_clip_id}:{sorted(trajectories)}"
            )

    for split in ("train", "heldout"):
        split_trajectories = set(
            clip_contract.loc[
                (clip_contract["split"].astype(str) == split)
                & (clip_contract["attack_type"].astype(str) == "adv_patch"),
                "trajectory_mode",
            ].astype(str)
        )
        missing_trajectories = set(ADV_PATCH_TRAJECTORY_MODES) - split_trajectories
        if missing_trajectories:
            raise ValueError(
                "split_adv_patch_trajectory_coverage_missing:"
                f"{split}:{sorted(missing_trajectories)}"
            )

    for clip_id, clip in frame.groupby("clip_id", sort=False):
        attack_type = str(clip.iloc[0]["attack_type"])
        labels = set(clip["label"].astype(int))
        if attack_type == "clean" and labels != {0}:
            raise ValueError(f"clean_clip_has_attack_label:{clip_id}:{sorted(labels)}")
        if attack_type != "clean" and 1 not in labels:
            raise ValueError(f"attack_clip_has_no_positive_frames:{clip_id}")


def _attack_type_video_metrics(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    per_type: dict[str, dict[str, Any]] = {}
    for attack_type in REQUIRED_ATTACK_TYPES:
        clips = [
            row
            for row in metrics.get("per_clip", [])
            if bool(row.get("is_attack")) and str(row.get("attack_type")) == attack_type
        ]
        hit_count = sum(bool(row.get("hit")) for row in clips)
        per_type[attack_type] = {
            "heldout_videos": int(len(clips)),
            "hit_videos": int(hit_count),
            "recall": float(hit_count / len(clips)) if clips else 0.0,
        }
    return per_type


def _adv_patch_trajectory_video_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    per_trajectory: dict[str, dict[str, Any]] = {}
    for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES:
        clips = [
            row
            for row in metrics.get("per_clip", [])
            if bool(row.get("is_attack"))
            and str(row.get("attack_type")) == "adv_patch"
            and str(row.get("trajectory_mode")) == trajectory_mode
        ]
        hit_count = sum(bool(row.get("hit")) for row in clips)
        per_trajectory[trajectory_mode] = {
            "heldout_videos": int(len(clips)),
            "hit_videos": int(hit_count),
            "recall": float(hit_count / len(clips)) if clips else 0.0,
        }
    return per_trajectory


def _evaluate_quality_gate(
    *,
    unique_heldout_metrics: Mapping[str, Any],
    unique_heldout_auc: float,
    per_attack_type: Mapping[str, Mapping[str, Any]],
    per_adv_patch_trajectory: Mapping[str, Mapping[str, Any]],
    classifier_role: str = "all_physical_attacks",
) -> dict[str, Any]:
    if classifier_role == A4_CLASSIFIER_ROLE:
        requirements = {
            "unique_heldout_clean_false_positive_videos_max": 0,
            "unique_heldout_attack_video_recall_min": 0.88,
            "unique_heldout_auc_min": 0.90,
            "per_adv_patch_trajectory_recall_min": 0.66,
            "heldout_videos_per_adv_patch_trajectory_min": 1,
        }
        observed = {
            "unique_heldout_clean_false_positive_videos": int(
                unique_heldout_metrics.get("clean_false_positive_videos", 0)
            ),
            "unique_heldout_attack_video_recall": float(
                unique_heldout_metrics.get("attack_video_recall", 0.0)
            ),
            "unique_heldout_auc": float(unique_heldout_auc),
            "per_adv_patch_trajectory": {
                trajectory_mode: dict(
                    per_adv_patch_trajectory.get(trajectory_mode, {})
                )
                for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES
            },
        }
        failures: list[str] = []
        if observed["unique_heldout_clean_false_positive_videos"] != 0:
            failures.append("unique_heldout_clean_false_positive_videos")
        if observed["unique_heldout_attack_video_recall"] < 0.88:
            failures.append("unique_heldout_attack_video_recall")
        if not math.isfinite(unique_heldout_auc) or unique_heldout_auc < 0.90:
            failures.append("unique_heldout_auc")
        for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES:
            item = observed["per_adv_patch_trajectory"][trajectory_mode]
            if int(item.get("heldout_videos", 0) or 0) < 1:
                failures.append(f"heldout_trajectory_coverage:{trajectory_mode}")
            if float(item.get("recall", 0.0) or 0.0) < 0.66:
                failures.append(f"heldout_trajectory_recall:{trajectory_mode}")
        return {
            "passed": not failures,
            "requirements": requirements,
            "observed": observed,
            "failures": failures,
        }
    requirements = {
        "unique_heldout_clean_false_positive_videos_max": 0,
        "unique_heldout_attack_video_recall_min": 0.90,
        "unique_heldout_auc_min": 0.90,
        "per_attack_type_recall_min": 0.80,
        "required_attack_types": list(REQUIRED_ATTACK_TYPES),
        "heldout_videos_per_attack_type_min": 1,
        "per_adv_patch_trajectory_recall_min": 0.80,
        "heldout_videos_per_adv_patch_trajectory_min": 1,
    }
    observed = {
        "unique_heldout_clean_false_positive_videos": int(
            unique_heldout_metrics.get("clean_false_positive_videos", 0)
        ),
        "unique_heldout_attack_video_recall": float(
            unique_heldout_metrics.get("attack_video_recall", 0.0)
        ),
        "unique_heldout_auc": float(unique_heldout_auc),
        "per_attack_type": {
            attack_type: dict(per_attack_type.get(attack_type, {}))
            for attack_type in REQUIRED_ATTACK_TYPES
        },
        "per_adv_patch_trajectory": {
            trajectory_mode: dict(
                per_adv_patch_trajectory.get(trajectory_mode, {})
            )
            for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES
        },
    }
    failures: list[str] = []
    if observed["unique_heldout_clean_false_positive_videos"] != 0:
        failures.append("unique_heldout_clean_false_positive_videos")
    if observed["unique_heldout_attack_video_recall"] < 0.90:
        failures.append("unique_heldout_attack_video_recall")
    if not math.isfinite(unique_heldout_auc) or unique_heldout_auc < 0.90:
        failures.append("unique_heldout_auc")
    for attack_type in REQUIRED_ATTACK_TYPES:
        item = observed["per_attack_type"][attack_type]
        if int(item.get("heldout_videos", 0) or 0) < 1:
            failures.append(f"heldout_coverage:{attack_type}")
        if float(item.get("recall", 0.0) or 0.0) < 0.80:
            failures.append(f"heldout_recall:{attack_type}")
    for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES:
        item = observed["per_adv_patch_trajectory"][trajectory_mode]
        if int(item.get("heldout_videos", 0) or 0) < 1:
            failures.append(f"heldout_trajectory_coverage:{trajectory_mode}")
        if float(item.get("recall", 0.0) or 0.0) < 0.80:
            failures.append(f"heldout_trajectory_recall:{trajectory_mode}")
    return {
        "passed": not failures,
        "requirements": requirements,
        "observed": observed,
        "failures": failures,
    }


def train_bound_a4_classifier(
    *,
    features_csv: str | Path,
    output_model: str | Path,
    report_path: str | Path,
    metadata_path: str | Path | None = None,
    folds: int = 5,
    iterations: int = 16,
    seed: int = 42,
    thresholds: Sequence[float] = (
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.92,
        0.94,
        0.96,
        0.98,
    ),
    alarm_window: int = 8,
    alarm_required_hits: int = 5,
) -> dict[str, Any]:
    """Train on manifest train scenes and evaluate heldout exactly once."""

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    if not thresholds:
        raise ValueError("A4 threshold search requires at least one candidate")
    normalized_thresholds = tuple(float(value) for value in thresholds)
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in normalized_thresholds):
        raise ValueError(f"A4 threshold candidates must be finite and inside (0,1): {thresholds}")
    if int(alarm_window) <= 0:
        raise ValueError("alarm_window must be positive")
    if int(alarm_required_hits) <= 0 or int(alarm_required_hits) > int(alarm_window):
        raise ValueError("alarm_required_hits must be in [1, alarm_window]")

    source = assert_no_rebuilt_demo_path(features_csv, field="a4_features_csv")
    output = assert_no_rebuilt_demo_path(output_model, field="a4_output_model")
    report_destination = assert_no_rebuilt_demo_path(
        report_path,
        field="a4_training_report",
    )
    requested_metadata_destination = (
        assert_no_rebuilt_demo_path(metadata_path, field="a4_output_metadata")
        if metadata_path
        else output.with_suffix(output.suffix + ".meta.json")
    )
    canonical_metadata_destination = output.with_suffix(output.suffix + ".meta.json")
    metadata_destinations = [canonical_metadata_destination]
    if requested_metadata_destination != canonical_metadata_destination:
        metadata_destinations.append(requested_metadata_destination)
    feature_metadata_path, training_provenance = _load_feature_provenance(source)
    frame = pd.read_csv(source)
    if "label" not in frame.columns:
        raise ValueError("A4 production feature dataset is missing columns: ['label']")
    frame = frame[frame["label"].isin([0, 1])].reset_index(drop=True)
    _validate_feature_dataset(frame, training_provenance)
    frame = frame[
        frame["attack_type"].astype(str).isin(("clean", "adv_patch"))
    ].reset_index(drop=True)
    train = frame[frame["split"].astype(str) == "train"].reset_index(drop=True)
    heldout = frame[frame["split"].astype(str) == "heldout"].reset_index(drop=True)
    if train.empty or heldout.empty:
        raise ValueError("A4 training requires both train and heldout rows")
    if len(np.unique(train["label"].to_numpy(dtype=np.int16))) != 2:
        raise ValueError("A4 training split requires both clean and attack labels")
    if len(np.unique(heldout["label"].to_numpy(dtype=np.int16))) != 2:
        raise ValueError("A4 heldout split requires both clean and attack labels")

    x_train = train[list(A4_PHYSICAL_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int16)
    clip_weights = _class_mass_preserving_clip_weights(train)
    groups = train["scene_id"].astype(str).to_numpy()
    group_count = int(train["scene_id"].nunique())
    effective_folds = min(max(2, int(folds)), group_count)
    if effective_folds < 2:
        raise ValueError("A4 training requires at least two train scene groups")
    splitter = StratifiedGroupKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=int(seed),
    )
    split_indices = list(splitter.split(x_train, y_train, groups))
    rng = np.random.RandomState(int(seed))
    candidates = _candidate_parameters(rng, iterations)
    results: list[dict[str, Any]] = []

    for params in candidates:
        oof = np.zeros(len(train), dtype=np.float64)
        for train_indices, validation_indices in split_indices:
            y_fold = y_train[train_indices]
            fold_weights = clip_weights[train_indices]
            classifier = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(seed),
                tree_method="hist",
                n_jobs=4,
                scale_pos_weight=1.0,
                **params,
            )
            classifier.fit(
                x_train[train_indices],
                y_fold,
                sample_weight=fold_weights,
            )
            oof[validation_indices] = classifier.predict_proba(
                x_train[validation_indices]
            )[:, 1]
        auc = float(roc_auc_score(y_train, oof))
        threshold_results: list[dict[str, Any]] = []
        for threshold in normalized_thresholds:
            metrics = _video_alarm_metrics(
                train,
                oof,
                threshold=float(threshold),
                window=alarm_window,
                required_hits=alarm_required_hits,
            )
            per_attack_type = _attack_type_video_metrics(metrics)
            per_adv_patch_trajectory = _adv_patch_trajectory_video_metrics(
                metrics
            )
            trajectory_recalls = [
                float(per_adv_patch_trajectory[trajectory_mode]["recall"])
                for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES
            ]
            trajectory_macro_recall = float(np.mean(trajectory_recalls))
            score = (
                float(metrics["attack_video_recall"])
                - 3.0 * float(metrics["clean_false_positive_rate"])
                + 0.10 * auc
                + 0.45 * trajectory_macro_recall
                + 0.30 * min(trajectory_recalls)
            )
            threshold_results.append(
                {
                    "threshold": float(threshold),
                    "score": float(score),
                    "metrics": metrics,
                    "per_attack_type": per_attack_type,
                    "per_adv_patch_trajectory": per_adv_patch_trajectory,
                }
            )
        best_threshold = max(
            threshold_results,
            key=lambda item: (
                item["score"],
                -item["metrics"]["clean_false_positive_videos"],
                item["metrics"]["attack_hit_videos"],
            ),
        )
        results.append(
            {
                "params": params,
                "oof_auc": auc,
                "best_threshold": best_threshold,
            }
        )

    best = max(
        results,
        key=_candidate_selection_key,
    )
    final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        tree_method="hist",
        n_jobs=4,
        scale_pos_weight=1.0,
        **best["params"],
    )
    final.fit(x_train, y_train, sample_weight=clip_weights)

    x_heldout = heldout[list(A4_PHYSICAL_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y_heldout = heldout["label"].to_numpy(dtype=np.int16)
    heldout_probabilities = final.predict_proba(x_heldout)[:, 1]
    heldout_auc = (
        float(roc_auc_score(y_heldout, heldout_probabilities))
        if len(np.unique(y_heldout)) > 1
        else math.nan
    )
    selected_threshold = float(best["best_threshold"]["threshold"])
    heldout_metrics = _video_alarm_metrics(
        heldout,
        heldout_probabilities,
        threshold=selected_threshold,
        window=alarm_window,
        required_hits=alarm_required_hits,
    )
    unique_heldout, heldout_content_hashes = _unique_content_frame(heldout)
    unique_heldout_probabilities = final.predict_proba(
        unique_heldout[list(A4_PHYSICAL_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    )[:, 1]
    unique_heldout_metrics = _video_alarm_metrics(
        unique_heldout,
        unique_heldout_probabilities,
        threshold=selected_threshold,
        window=alarm_window,
        required_hits=alarm_required_hits,
    )
    unique_y_heldout = unique_heldout["label"].to_numpy(dtype=np.int16)
    unique_heldout_auc = (
        float(roc_auc_score(unique_y_heldout, unique_heldout_probabilities))
        if len(np.unique(unique_y_heldout)) > 1
        else math.nan
    )
    per_attack_type = _attack_type_video_metrics(unique_heldout_metrics)
    per_adv_patch_trajectory = _adv_patch_trajectory_video_metrics(
        unique_heldout_metrics
    )
    quality_gate = _evaluate_quality_gate(
        unique_heldout_metrics=unique_heldout_metrics,
        unique_heldout_auc=unique_heldout_auc,
        per_attack_type=per_attack_type,
        per_adv_patch_trajectory=per_adv_patch_trajectory,
        classifier_role=A4_CLASSIFIER_ROLE,
    )

    metadata_common = {
        "artifact_contract_version": 2,
        "schema_version": 2,
        "classifier_role": A4_CLASSIFIER_ROLE,
        "runtime_fusion": "max_rule_and_classifier_with_temporal_confirmation",
        "production_candidate_eligible": bool(quality_gate["passed"]),
        "feature_schema_version": A4_PHYSICAL_SCHEMA_VERSION,
        "feature_names": list(A4_PHYSICAL_FEATURE_NAMES),
        "feature_count": len(A4_PHYSICAL_FEATURE_NAMES),
        "preprocessing": "raw_float32_no_scaler",
        "model_type": f"{type(final).__module__}.{type(final).__name__}",
        "source_manifest_sha256": str(
            training_provenance["source_manifest_sha256"]
        ),
        "dataset_manifest_sha256": str(
            training_provenance["dataset_manifest_sha256"]
        ),
        "authoritative_manifest_sha256": str(
            training_provenance["authoritative_manifest_sha256"]
        ),
        "unique_yolo_source_sha256": UNIQUE_YOLO_SOURCE_SHA256.lower(),
        "training_features_path": str(source),
        "training_features_sha256": _sha256_file(source),
        "training_features_metadata_path": str(feature_metadata_path),
        "training_provenance": training_provenance,
        "training_rows": int(len(train)),
        "training_scenes": int(train["scene_id"].nunique()),
        "training_clips": int(train["clip_id"].nunique()),
        "per_class_clip_equal_weighting": True,
        "sample_weight_preserves_class_row_mass": True,
        "sample_weight_mean_normalized": True,
        "scale_pos_weight": 1.0,
        "candidate_selection_policy": (
            "min_oof_clean_false_positive_videos_then_score"
        ),
        "heldout_rows": int(len(heldout)),
        "heldout_scenes": int(heldout["scene_id"].nunique()),
        "heldout_clips": int(heldout["clip_id"].nunique()),
        "selected_threshold": selected_threshold,
        "alarm_window": int(alarm_window),
        "alarm_required_hits": int(alarm_required_hits),
        "quality_gate": quality_gate,
        "heldout_adv_patch_trajectory_metrics": per_adv_patch_trajectory,
        "seed": int(seed),
        "folds": int(effective_folds),
    }
    report = {
        "schema_version": 2,
        "production_candidate_eligible": bool(quality_gate["passed"]),
        "feature_schema_version": A4_PHYSICAL_SCHEMA_VERSION,
        "feature_names": list(A4_PHYSICAL_FEATURE_NAMES),
        "best": best,
        "selected_threshold": selected_threshold,
        "alarm_window": int(alarm_window),
        "alarm_required_hits": int(alarm_required_hits),
        "heldout_auc": heldout_auc,
        "heldout_metrics": heldout_metrics,
        "heldout_unique_content_auc": unique_heldout_auc,
        "heldout_unique_content_metrics": unique_heldout_metrics,
        "heldout_unique_content_per_attack_type": per_attack_type,
        "heldout_unique_content_adv_patch_trajectory": per_adv_patch_trajectory,
        "heldout_content_hashes": heldout_content_hashes,
        "heldout_role": "regression_dev_gate_not_independent_test",
        "quality_gate": quality_gate,
        "artifact": None,
        "artifact_candidate_metadata": metadata_common,
        "candidates": sorted(
            results,
            key=_candidate_selection_key,
            reverse=True,
        ),
    }
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report_destination.with_suffix(
        report_destination.suffix + ".tmp"
    )
    report_temporary.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_temporary.replace(report_destination)
    if not bool(quality_gate["passed"]):
        raise A4QualityGateError(report)

    output.parent.mkdir(parents=True, exist_ok=True)
    for destination in metadata_destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output.with_suffix(output.suffix + ".tmp")
    metadata_temporaries = {
        destination: destination.with_suffix(destination.suffix + ".tmp")
        for destination in metadata_destinations
    }
    try:
        with output_temporary.open("wb") as handle:
            pickle.dump(final, handle)
        metadata = {
            **metadata_common,
            "production_candidate_eligible": True,
            "model_sha256": _sha256_file(output_temporary),
        }
        metadata_text = json.dumps(
            _jsonable(metadata),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        for temporary in metadata_temporaries.values():
            temporary.write_text(metadata_text, encoding="utf-8")
        output_temporary.replace(output)
        for destination, temporary in metadata_temporaries.items():
            temporary.replace(destination)
    except Exception:
        output_temporary.unlink(missing_ok=True)
        for temporary in metadata_temporaries.values():
            temporary.unlink(missing_ok=True)
        raise

    report["artifact"] = metadata
    report["artifact_candidate_metadata"] = metadata
    report_temporary.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_temporary.replace(report_destination)
    return report

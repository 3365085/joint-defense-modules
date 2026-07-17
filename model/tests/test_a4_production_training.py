from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defense.diagnostics.a4_dataset import (
    ADV_PATCH_TRAJECTORY_MODES,
    UNIQUE_YOLO_SOURCE_SHA256,
)
from defense.diagnostics.a4_training import (
    A4_PHYSICAL_FEATURE_NAMES,
    A4_PHYSICAL_SCHEMA_VERSION,
    A4QualityGateError,
    _FIXED_ADV_PATCH_CANDIDATE,
    _alarm_flags,
    _candidate_parameters,
    _candidate_selection_key,
    _class_mass_preserving_clip_weights,
    _equal_clip_weights,
    _evaluate_quality_gate,
    _frame_label,
    _validate_feature_dataset,
    _video_alarm_metrics,
    train_bound_a4_classifier,
)
from defense.module_a.rebuilt.a4_artifact import (
    load_a4_artifact_metadata,
    validate_a4_artifact_metadata,
)


def test_physical_a4_schema_excludes_async_a3b_features() -> None:
    assert len(A4_PHYSICAL_FEATURE_NAMES) == 96
    assert all(not name.startswith("a3b.") for name in A4_PHYSICAL_FEATURE_NAMES)
    assert sum(name.startswith("a4_patch.") for name in A4_PHYSICAL_FEATURE_NAMES) == 40
    assert sum(
        name.startswith("a4_patch_delta.")
        for name in A4_PHYSICAL_FEATURE_NAMES
    ) == 40


def test_manifest_attack_ramp_is_excluded_from_training_labels() -> None:
    row = {
        "label": 1,
        "attack_start_frame": 4,
        "attack_ramp_frames": 6,
    }
    assert _frame_label(row, 3) == 0
    assert _frame_label(row, 4) == -1
    assert _frame_label(row, 9) == -1
    assert _frame_label(row, 10) == 1


def test_alarm_flags_use_n_of_m_confirmation() -> None:
    probabilities = np.asarray([0.1, 0.9, 0.9, 0.2, 0.9, 0.9, 0.9])
    flags = _alarm_flags(
        probabilities,
        threshold=0.8,
        window=5,
        required_hits=4,
    )
    assert flags.tolist() == [False, False, False, False, False, True, True]


def test_video_metrics_are_clip_level_not_frame_level() -> None:
    frame = pd.DataFrame(
        [
            {"clip_id": "clean", "frame_idx": i, "label": 0, "attack_type": "clean"}
            for i in range(6)
        ]
        + [
            {"clip_id": "attack", "frame_idx": i, "label": 1, "attack_type": "visibility"}
            for i in range(6)
        ]
    )
    probabilities = np.asarray([0.1] * 6 + [0.9] * 6)
    metrics = _video_alarm_metrics(
        frame,
        probabilities,
        threshold=0.8,
        window=5,
        required_hits=3,
    )
    assert metrics["clean_false_positive_videos"] == 0
    assert metrics["attack_hit_videos"] == 1


def test_equal_clip_weights_prevent_long_clips_from_dominating() -> None:
    frame = pd.DataFrame(
        {
            "clip_id": ["short", "long", "long", "long"],
        }
    )
    weights = _equal_clip_weights(frame)

    assert weights[0] == 1.0
    assert weights[1:].sum() == 1.0


def test_training_weights_preserve_class_mass_and_equalize_clips_within_class() -> None:
    frame = pd.DataFrame(
        {
            "clip_id": ["clean_short", "clean_long", "clean_long", "attack"],
            "label": [0, 0, 0, 1],
        }
    )

    weights = _class_mass_preserving_clip_weights(frame)

    assert weights.mean() == pytest.approx(1.0)
    assert weights[frame["label"] == 0].sum() == pytest.approx(3.0)
    assert weights[frame["label"] == 1].sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(weights[1:3].sum())


def test_candidate_search_always_includes_verified_adv_patch_parameters() -> None:
    candidates = _candidate_parameters(np.random.RandomState(42), iterations=16)

    assert len(candidates) == 16
    assert candidates[0] == _FIXED_ADV_PATCH_CANDIDATE


def test_candidate_selection_prioritizes_clean_false_positive_safety() -> None:
    lower_false_positive = {
        "oof_auc": 0.84,
        "best_threshold": {
            "score": 0.90,
            "metrics": {
                "clean_false_positive_videos": 1,
                "attack_hit_videos": 43,
            },
        },
    }
    higher_score = {
        "oof_auc": 0.86,
        "best_threshold": {
            "score": 0.96,
            "metrics": {
                "clean_false_positive_videos": 3,
                "attack_hit_videos": 47,
            },
        },
    }

    assert max(
        [higher_score, lower_false_positive],
        key=_candidate_selection_key,
    ) is lower_false_positive


def test_quality_gate_requires_all_five_heldout_attack_types() -> None:
    metrics = {
        "clean_false_positive_videos": 0,
        "attack_video_recall": 1.0,
    }
    per_type = {
        attack_type: {"heldout_videos": 1, "hit_videos": 1, "recall": 1.0}
        for attack_type in (
            "adv_patch",
            "glare",
            "motion_blur",
            "occlusion",
            "visibility_degradation",
        )
    }
    per_type["adv_patch"] = {"heldout_videos": 0, "hit_videos": 0, "recall": 0.0}
    per_trajectory = {
        trajectory_mode: {"heldout_videos": 1, "hit_videos": 1, "recall": 1.0}
        for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES
    }

    gate = _evaluate_quality_gate(
        unique_heldout_metrics=metrics,
        unique_heldout_auc=0.99,
        per_attack_type=per_type,
        per_adv_patch_trajectory=per_trajectory,
    )

    assert gate["passed"] is False
    assert "heldout_coverage:adv_patch" in gate["failures"]


def _synthetic_feature_dataset(tmp_path: Path, *, heldout_attack_value: float) -> Path:
    variants = [("clean", "none", 0)]
    variants.extend(("adv_patch", mode, 1) for mode in ADV_PATCH_TRAJECTORY_MODES)
    variants.extend(
        (
            ("glare", "none", 1),
            ("motion_blur", "none", 1),
            ("occlusion", "none", 1),
            ("visibility_degradation", "none", 1),
        )
    )
    rows: list[dict] = []
    source_manifest_hash = hashlib.sha256(b"source\n").hexdigest()
    base_splits = [(index, "train") for index in range(1, 11)] + [(11, "heldout")]
    for base_index, split in base_splits:
        base_id = f"base_{base_index}"
        base_hash = hashlib.sha256(base_id.encode()).hexdigest()
        for variant_index, (attack_type, trajectory_mode, label) in enumerate(variants):
            clip_id = f"{base_id}_{variant_index:02d}"
            content_hash = hashlib.sha256(f"content:{clip_id}".encode()).hexdigest()
            provenance_id = hashlib.sha256(f"provenance:{clip_id}".encode()).hexdigest()
            if label == 0:
                feature_value = 0.0
            elif split == "heldout":
                feature_value = float(heldout_attack_value)
            else:
                feature_value = 4.0
            for frame_idx in range(6):
                row = {
                    "clip_id": clip_id,
                    "video": str(tmp_path / f"{clip_id}.mp4"),
                    "base_clip_id": base_id,
                    "base_source_sha256": base_hash,
                    "content_sha256": content_hash,
                    "scene_id": f"scene_{base_index}",
                    "split": split,
                    "attack_type": attack_type,
                    "trajectory_mode": trajectory_mode,
                    "provenance_id": provenance_id,
                    "source_manifest_sha256": source_manifest_hash,
                    "unique_yolo_source_sha256": UNIQUE_YOLO_SOURCE_SHA256.lower(),
                    "frame_idx": frame_idx,
                    "source_time_s": frame_idx / 10.0,
                    "label": label,
                }
                row.update(
                    {
                        name: feature_value + feature_index * 0.001
                        for feature_index, name in enumerate(A4_PHYSICAL_FEATURE_NAMES)
                    }
                )
                rows.append(row)
    source = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    source_manifest = tmp_path / "source_manifest.csv"
    dataset_manifest = tmp_path / "dataset_manifest.csv"
    authoritative_manifest = tmp_path / "authoritative_manifest.json"
    source_manifest.write_bytes(b"source\n")
    dataset_manifest.write_bytes(b"dataset\n")
    authoritative_manifest.write_bytes(b"{}\n")
    unique_yolo = (
        Path(__file__).resolve().parents[2]
        / "素材/model/yolov8/mask_bd_v4_clean_baseline.pt"
    )
    metadata = {
        "feature_schema_version": A4_PHYSICAL_SCHEMA_VERSION,
        "feature_names": list(A4_PHYSICAL_FEATURE_NAMES),
        "features_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
        "authoritative_manifest_path": str(authoritative_manifest),
        "authoritative_manifest_sha256": hashlib.sha256(
            authoritative_manifest.read_bytes()
        ).hexdigest(),
        "authoritative_video_count": 36,
        "unique_yolo_source_path": str(unique_yolo),
        "unique_yolo_source_sha256": UNIQUE_YOLO_SOURCE_SHA256.lower(),
    }
    source.with_suffix(source.suffix + ".meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return source


def test_training_emits_eligible_schema_bound_artifact_only_after_gates(
    tmp_path: Path,
) -> None:
    source = _synthetic_feature_dataset(tmp_path, heldout_attack_value=4.0)
    model = tmp_path / "a4.pkl"
    report_path = tmp_path / "report.json"

    report = train_bound_a4_classifier(
        features_csv=source,
        output_model=model,
        report_path=report_path,
        folds=2,
        iterations=1,
        thresholds=(0.5,),
        alarm_window=2,
        alarm_required_hits=1,
        seed=7,
    )

    assert report["production_candidate_eligible"] is True
    assert report["quality_gate"]["passed"] is True
    metadata = load_a4_artifact_metadata(model)
    assert metadata["selected_threshold"] == 0.5
    assert metadata["alarm_window"] == 2
    assert metadata["alarm_required_hits"] == 1
    assert metadata["unique_yolo_source_sha256"] == UNIQUE_YOLO_SOURCE_SHA256.lower()
    assert metadata["heldout_adv_patch_trajectory_metrics"]["discrete_jump/jitter"][
        "heldout_videos"
    ] >= 1
    validate_a4_artifact_metadata(
        metadata,
        model_path=model,
        expected_schema_version=A4_PHYSICAL_SCHEMA_VERSION,
        expected_feature_names=A4_PHYSICAL_FEATURE_NAMES,
    )


def test_training_gate_failure_writes_report_but_no_artifact(tmp_path: Path) -> None:
    source = _synthetic_feature_dataset(tmp_path, heldout_attack_value=0.0)
    model = tmp_path / "a4.pkl"
    report_path = tmp_path / "report.json"

    with pytest.raises(A4QualityGateError, match="a4_production_quality_gate_failed"):
        train_bound_a4_classifier(
            features_csv=source,
            output_model=model,
            report_path=report_path,
            folds=2,
            iterations=1,
            thresholds=(0.5,),
            alarm_window=2,
            alarm_required_hits=1,
            seed=7,
        )

    assert not model.exists()
    assert not model.with_suffix(model.suffix + ".meta.json").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["production_candidate_eligible"] is False
    assert report["quality_gate"]["passed"] is False
    assert report["artifact"] is None


def test_feature_contract_rejects_base_group_split_leakage(tmp_path: Path) -> None:
    source = _synthetic_feature_dataset(tmp_path, heldout_attack_value=4.0)
    frame = pd.read_csv(source)
    metadata = json.loads(
        source.with_suffix(source.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    leaked_clip = frame.loc[frame["base_clip_id"] == "base_1", "clip_id"].iloc[0]
    frame.loc[frame["clip_id"] == leaked_clip, "split"] = "heldout"

    with pytest.raises(ValueError, match="base_group_split_scene_leakage"):
        _validate_feature_dataset(frame, metadata)


def test_feature_contract_rejects_train_heldout_content_duplicate(
    tmp_path: Path,
) -> None:
    source = _synthetic_feature_dataset(tmp_path, heldout_attack_value=4.0)
    frame = pd.read_csv(source)
    metadata = json.loads(
        source.with_suffix(source.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    train_clip = frame.loc[frame["split"] == "train", "clip_id"].iloc[0]
    heldout_hash = frame.loc[frame["split"] == "heldout", "content_sha256"].iloc[0]
    frame.loc[frame["clip_id"] == train_clip, "content_sha256"] = heldout_hash

    with pytest.raises(ValueError, match="feature_content_not_deduplicated"):
        _validate_feature_dataset(frame, metadata)

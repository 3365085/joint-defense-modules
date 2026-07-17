from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from defense.module_a.rebuilt.a4_artifact import (
    ADV_PATCH_TRAJECTORY_MODES,
    A4ArtifactValidationError,
    UNIQUE_YOLO_SOURCE_SHA256,
    load_a4_artifact_metadata,
    metadata_path_for_model,
    validate_a4_artifact_metadata,
)


FEATURES = ("a1", "a2")


def _quality_gate() -> dict:
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
    per_trajectory = {
        trajectory_mode: {"heldout_videos": 1, "hit_videos": 1, "recall": 1.0}
        for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES
    }
    return {
        "passed": True,
        "failures": [],
        "requirements": {
            "unique_heldout_clean_false_positive_videos_max": 0,
            "unique_heldout_attack_video_recall_min": 0.90,
            "unique_heldout_auc_min": 0.90,
            "per_attack_type_recall_min": 0.80,
            "required_attack_types": list(per_type),
            "heldout_videos_per_attack_type_min": 1,
            "per_adv_patch_trajectory_recall_min": 0.80,
            "heldout_videos_per_adv_patch_trajectory_min": 1,
        },
        "observed": {
            "unique_heldout_clean_false_positive_videos": 0,
            "unique_heldout_attack_video_recall": 1.0,
            "unique_heldout_auc": 1.0,
            "per_attack_type": per_type,
            "per_adv_patch_trajectory": per_trajectory,
        },
    }


def _write_artifact(tmp_path: Path) -> tuple[Path, dict]:
    model_path = tmp_path / "a4.pkl"
    model_path.write_bytes(b"classifier")
    metadata = {
        "artifact_contract_version": 2,
        "production_candidate_eligible": True,
        "feature_schema_version": "schema-v2",
        "feature_names": list(FEATURES),
        "feature_count": len(FEATURES),
        "preprocessing": "raw_float32_no_scaler",
        "source_manifest_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "authoritative_manifest_sha256": "c" * 64,
        "unique_yolo_source_sha256": UNIQUE_YOLO_SOURCE_SHA256.lower(),
        "selected_threshold": 0.73,
        "alarm_window": 8,
        "alarm_required_hits": 5,
        "quality_gate": _quality_gate(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    metadata_path_for_model(model_path).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return model_path, metadata


def test_bound_a4_metadata_validates_exact_schema_and_model_hash(tmp_path: Path) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    loaded = load_a4_artifact_metadata(model_path)

    validated = validate_a4_artifact_metadata(
        loaded,
        model_path=model_path,
        expected_schema_version="schema-v2",
        expected_feature_names=FEATURES,
    )

    assert validated == metadata


def test_bound_a4_metadata_rejects_missing_sidecar(tmp_path: Path) -> None:
    model_path = tmp_path / "a4.pkl"
    model_path.write_bytes(b"classifier")

    with pytest.raises(A4ArtifactValidationError, match="schema_metadata_missing"):
        load_a4_artifact_metadata(model_path)


def test_bound_a4_metadata_rejects_feature_order_mismatch(tmp_path: Path) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    metadata["feature_names"] = list(reversed(FEATURES))

    with pytest.raises(A4ArtifactValidationError, match="feature_name_order_mismatch"):
        validate_a4_artifact_metadata(
            metadata,
            model_path=model_path,
            expected_schema_version="schema-v2",
            expected_feature_names=FEATURES,
        )


def test_bound_a4_metadata_rejects_ineligible_or_missing_candidate_flag(
    tmp_path: Path,
) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    metadata.pop("production_candidate_eligible")

    with pytest.raises(A4ArtifactValidationError, match="production_candidate_not_eligible"):
        validate_a4_artifact_metadata(
            metadata,
            model_path=model_path,
            expected_schema_version="schema-v2",
            expected_feature_names=FEATURES,
        )


@pytest.mark.parametrize("threshold", [None, 0.0, 1.0, -0.1, 1.1, float("nan")])
def test_bound_a4_metadata_rejects_invalid_selected_threshold(
    tmp_path: Path,
    threshold: float | None,
) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    metadata["selected_threshold"] = threshold

    with pytest.raises(A4ArtifactValidationError, match="selected_threshold"):
        validate_a4_artifact_metadata(
            metadata,
            model_path=model_path,
            expected_schema_version="schema-v2",
            expected_feature_names=FEATURES,
        )


def test_bound_a4_metadata_rejects_claimed_pass_with_failed_per_type_recall(
    tmp_path: Path,
) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    metadata["quality_gate"]["observed"]["per_attack_type"]["adv_patch"][
        "recall"
    ] = 0.79

    with pytest.raises(
        A4ArtifactValidationError,
        match="quality_gate_attack_type_recall_failed:adv_patch",
    ):
        validate_a4_artifact_metadata(
            metadata,
            model_path=model_path,
            expected_schema_version="schema-v2",
            expected_feature_names=FEATURES,
        )


def test_bound_a4_metadata_rejects_failed_adv_patch_trajectory_recall(
    tmp_path: Path,
) -> None:
    model_path, metadata = _write_artifact(tmp_path)
    metadata["quality_gate"]["observed"]["per_adv_patch_trajectory"][
        "discrete_jump/jitter"
    ]["recall"] = 0.79

    with pytest.raises(
        A4ArtifactValidationError,
        match="quality_gate_adv_patch_trajectory_recall_failed:discrete_jump/jitter",
    ):
        validate_a4_artifact_metadata(
            metadata,
            model_path=model_path,
            expected_schema_version="schema-v2",
            expected_feature_names=FEATURES,
        )

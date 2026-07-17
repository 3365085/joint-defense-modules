from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


UNIQUE_YOLO_SOURCE_SHA256 = (
    "4D7A23D3866AC2D9DB6E59AE537DA1274D988BD53CA6C7D519297FCBB96626F8"
)
REQUIRED_ATTACK_TYPES: tuple[str, ...] = (
    "adv_patch",
    "glare",
    "motion_blur",
    "occlusion",
    "visibility_degradation",
)
ADV_PATCH_TRAJECTORY_MODES: tuple[str, ...] = (
    "target_anchored_static",
    "smooth_drift",
    "discrete_jump/jitter",
    "scale_rotation",
    "partial_outside_roi/occlusion",
)


class A4ArtifactValidationError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    artifact = Path(path).expanduser()
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path_for_model(path: str | Path) -> Path:
    model_path = Path(path).expanduser()
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def load_a4_artifact_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = metadata_path_for_model(path)
    if not metadata_path.is_file():
        raise A4ArtifactValidationError(
            f"schema_metadata_missing:{metadata_path}"
        )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise A4ArtifactValidationError(
            f"schema_metadata_invalid:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise A4ArtifactValidationError("schema_metadata_invalid:not_an_object")
    return payload


def _sha256_field(metadata: Mapping[str, Any], field: str) -> str:
    value = str(metadata.get(field, "") or "").strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise A4ArtifactValidationError(
            f"{field}_invalid:{value or '<missing>'}"
        )
    return value


def _finite_float(metadata: Mapping[str, Any], field: str) -> float:
    value = metadata.get(field)
    if isinstance(value, bool):
        raise A4ArtifactValidationError(f"{field}_invalid:{value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise A4ArtifactValidationError(f"{field}_invalid:{value!r}") from exc
    if not math.isfinite(parsed):
        raise A4ArtifactValidationError(f"{field}_invalid:{parsed!r}")
    return parsed


def _positive_int(metadata: Mapping[str, Any], field: str) -> int:
    value = metadata.get(field)
    if isinstance(value, bool):
        raise A4ArtifactValidationError(f"{field}_invalid:{value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise A4ArtifactValidationError(f"{field}_invalid:{value!r}") from exc
    if parsed <= 0 or parsed != value:
        raise A4ArtifactValidationError(f"{field}_invalid:{value!r}")
    return parsed


def _validate_quality_gate(metadata: Mapping[str, Any]) -> dict[str, Any]:
    def numeric(mapping: Mapping[str, Any], field: str) -> float:
        value = mapping.get(field)
        if isinstance(value, bool):
            raise A4ArtifactValidationError(f"quality_gate_value_invalid:{field}:{value!r}")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise A4ArtifactValidationError(
                f"quality_gate_value_invalid:{field}:{value!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise A4ArtifactValidationError(
                f"quality_gate_value_invalid:{field}:{parsed!r}"
            )
        return parsed

    def integer(mapping: Mapping[str, Any], field: str) -> int:
        parsed = numeric(mapping, field)
        if not parsed.is_integer():
            raise A4ArtifactValidationError(
                f"quality_gate_value_invalid:{field}:{parsed!r}"
            )
        return int(parsed)

    quality_gate = metadata.get("quality_gate")
    if not isinstance(quality_gate, Mapping):
        raise A4ArtifactValidationError("quality_gate_missing_or_invalid")
    if quality_gate.get("passed") is not True:
        raise A4ArtifactValidationError("quality_gate_not_passed")
    failures = quality_gate.get("failures", [])
    if not isinstance(failures, list) or failures:
        raise A4ArtifactValidationError(f"quality_gate_failures_present:{failures!r}")
    requirements = quality_gate.get("requirements")
    observed = quality_gate.get("observed")
    if not isinstance(requirements, Mapping) or not isinstance(observed, Mapping):
        raise A4ArtifactValidationError("quality_gate_summary_missing")
    classifier_role = str(
        metadata.get("classifier_role", "all_physical_attacks") or ""
    )
    if classifier_role == "adv_patch_rescue":
        if integer(requirements, "unique_heldout_clean_false_positive_videos_max") != 0:
            raise A4ArtifactValidationError(
                "quality_gate_clean_fp_requirement_mismatch"
            )
        if numeric(requirements, "unique_heldout_attack_video_recall_min") < 0.88:
            raise A4ArtifactValidationError(
                "quality_gate_attack_recall_requirement_too_weak"
            )
        if numeric(requirements, "unique_heldout_auc_min") < 0.90:
            raise A4ArtifactValidationError(
                "quality_gate_auc_requirement_too_weak"
            )
        if numeric(requirements, "per_adv_patch_trajectory_recall_min") < 0.66:
            raise A4ArtifactValidationError(
                "quality_gate_per_adv_patch_trajectory_requirement_too_weak"
            )
        if integer(requirements, "heldout_videos_per_adv_patch_trajectory_min") < 1:
            raise A4ArtifactValidationError(
                "quality_gate_heldout_trajectory_coverage_requirement_too_weak"
            )
        clean_fp = integer(
            observed,
            "unique_heldout_clean_false_positive_videos",
        )
        attack_recall = numeric(
            observed,
            "unique_heldout_attack_video_recall",
        )
        auc = numeric(observed, "unique_heldout_auc")
        if clean_fp != 0:
            raise A4ArtifactValidationError(
                f"quality_gate_clean_fp_failed:{clean_fp}"
            )
        if attack_recall < 0.88:
            raise A4ArtifactValidationError(
                f"quality_gate_attack_recall_failed:{attack_recall}"
            )
        if auc < 0.90:
            raise A4ArtifactValidationError(
                f"quality_gate_auc_failed:{auc}"
            )
        per_trajectory = observed.get("per_adv_patch_trajectory")
        if not isinstance(per_trajectory, Mapping):
            raise A4ArtifactValidationError(
                "quality_gate_per_adv_patch_trajectory_missing"
            )
        for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES:
            item = per_trajectory.get(trajectory_mode)
            if not isinstance(item, Mapping):
                raise A4ArtifactValidationError(
                    f"quality_gate_adv_patch_trajectory_missing:{trajectory_mode}"
                )
            heldout_videos = integer(item, "heldout_videos")
            recall = numeric(item, "recall")
            if heldout_videos < 1:
                raise A4ArtifactValidationError(
                    f"quality_gate_adv_patch_trajectory_no_heldout:{trajectory_mode}"
                )
            if recall < 0.66:
                raise A4ArtifactValidationError(
                    "quality_gate_adv_patch_trajectory_recall_failed:"
                    f"{trajectory_mode}:{recall}"
                )
        return dict(quality_gate)
    if integer(requirements, "unique_heldout_clean_false_positive_videos_max") != 0:
        raise A4ArtifactValidationError("quality_gate_clean_fp_requirement_mismatch")
    if numeric(requirements, "unique_heldout_attack_video_recall_min") < 0.90:
        raise A4ArtifactValidationError("quality_gate_attack_recall_requirement_too_weak")
    if numeric(requirements, "unique_heldout_auc_min") < 0.90:
        raise A4ArtifactValidationError("quality_gate_auc_requirement_too_weak")
    if numeric(requirements, "per_attack_type_recall_min") < 0.80:
        raise A4ArtifactValidationError("quality_gate_per_type_requirement_too_weak")
    required_types = tuple(str(value) for value in requirements.get("required_attack_types", []) or [])
    if set(required_types) != set(REQUIRED_ATTACK_TYPES):
        raise A4ArtifactValidationError(
            f"quality_gate_required_attack_types_mismatch:{required_types!r}"
        )
    if integer(requirements, "heldout_videos_per_attack_type_min") < 1:
        raise A4ArtifactValidationError("quality_gate_heldout_coverage_requirement_too_weak")
    if numeric(requirements, "per_adv_patch_trajectory_recall_min") < 0.80:
        raise A4ArtifactValidationError(
            "quality_gate_per_adv_patch_trajectory_requirement_too_weak"
        )
    if integer(requirements, "heldout_videos_per_adv_patch_trajectory_min") < 1:
        raise A4ArtifactValidationError(
            "quality_gate_heldout_trajectory_coverage_requirement_too_weak"
        )

    clean_fp = integer(observed, "unique_heldout_clean_false_positive_videos")
    if clean_fp != 0:
        raise A4ArtifactValidationError(f"quality_gate_clean_fp_failed:{clean_fp}")
    attack_recall = numeric(observed, "unique_heldout_attack_video_recall")
    if not math.isfinite(attack_recall) or attack_recall < 0.90:
        raise A4ArtifactValidationError(
            f"quality_gate_attack_recall_failed:{attack_recall}"
        )
    auc = numeric(observed, "unique_heldout_auc")
    if not math.isfinite(auc) or auc < 0.90:
        raise A4ArtifactValidationError(f"quality_gate_auc_failed:{auc}")
    per_attack_type = observed.get("per_attack_type")
    if not isinstance(per_attack_type, Mapping):
        raise A4ArtifactValidationError("quality_gate_per_attack_type_missing")
    for attack_type in REQUIRED_ATTACK_TYPES:
        item = per_attack_type.get(attack_type)
        if not isinstance(item, Mapping):
            raise A4ArtifactValidationError(
                f"quality_gate_attack_type_missing:{attack_type}"
            )
        heldout_videos = integer(item, "heldout_videos")
        recall = numeric(item, "recall")
        if heldout_videos < 1:
            raise A4ArtifactValidationError(
                f"quality_gate_attack_type_no_heldout:{attack_type}"
            )
        if not math.isfinite(recall) or recall < 0.80:
            raise A4ArtifactValidationError(
                f"quality_gate_attack_type_recall_failed:{attack_type}:{recall}"
            )
    per_trajectory = observed.get("per_adv_patch_trajectory")
    if not isinstance(per_trajectory, Mapping):
        raise A4ArtifactValidationError(
            "quality_gate_per_adv_patch_trajectory_missing"
        )
    for trajectory_mode in ADV_PATCH_TRAJECTORY_MODES:
        item = per_trajectory.get(trajectory_mode)
        if not isinstance(item, Mapping):
            raise A4ArtifactValidationError(
                f"quality_gate_adv_patch_trajectory_missing:{trajectory_mode}"
            )
        heldout_videos = integer(item, "heldout_videos")
        recall = numeric(item, "recall")
        if heldout_videos < 1:
            raise A4ArtifactValidationError(
                f"quality_gate_adv_patch_trajectory_no_heldout:{trajectory_mode}"
            )
        if not math.isfinite(recall) or recall < 0.80:
            raise A4ArtifactValidationError(
                "quality_gate_adv_patch_trajectory_recall_failed:"
                f"{trajectory_mode}:{recall}"
            )
    return dict(quality_gate)


def validate_a4_artifact_metadata(
    metadata: Mapping[str, Any],
    *,
    model_path: str | Path,
    expected_schema_version: str,
    expected_feature_names: Sequence[str],
) -> dict[str, Any]:
    if int(metadata.get("artifact_contract_version", 0) or 0) != 2:
        raise A4ArtifactValidationError("artifact_contract_version_mismatch:expected=2")
    if metadata.get("production_candidate_eligible") is not True:
        raise A4ArtifactValidationError("production_candidate_not_eligible")
    classifier_role = str(
        metadata.get("classifier_role", "all_physical_attacks") or ""
    )
    if classifier_role not in {"all_physical_attacks", "adv_patch_rescue"}:
        raise A4ArtifactValidationError(
            f"classifier_role_invalid:{classifier_role or '<missing>'}"
        )
    if classifier_role == "adv_patch_rescue" and str(
        metadata.get("runtime_fusion", "") or ""
    ) != "max_rule_and_classifier_with_temporal_confirmation":
        raise A4ArtifactValidationError("runtime_fusion_mismatch")
    actual_schema = str(metadata.get("feature_schema_version", "") or "")
    if actual_schema != str(expected_schema_version):
        raise A4ArtifactValidationError(
            "feature_schema_version_mismatch:"
            f"expected={expected_schema_version},actual={actual_schema or '<missing>'}"
        )
    actual_names = tuple(str(name) for name in metadata.get("feature_names", []) or [])
    expected_names = tuple(str(name) for name in expected_feature_names)
    if actual_names != expected_names:
        raise A4ArtifactValidationError(
            "feature_name_order_mismatch:"
            f"expected={expected_names!r},actual={actual_names!r}"
        )
    actual_count = int(metadata.get("feature_count", len(actual_names)) or 0)
    if actual_count != len(expected_names):
        raise A4ArtifactValidationError(
            "feature_count_mismatch:"
            f"expected={len(expected_names)},actual={actual_count}"
        )
    preprocessing = str(metadata.get("preprocessing", "") or "")
    if preprocessing != "raw_float32_no_scaler":
        raise A4ArtifactValidationError(
            "preprocessing_mismatch:"
            f"expected=raw_float32_no_scaler,actual={preprocessing or '<missing>'}"
        )
    source_manifest_hash = _sha256_field(metadata, "source_manifest_sha256")
    dataset_manifest_hash = _sha256_field(metadata, "dataset_manifest_sha256")
    authoritative_manifest_hash = _sha256_field(
        metadata,
        "authoritative_manifest_sha256",
    )
    del source_manifest_hash, dataset_manifest_hash, authoritative_manifest_hash
    yolo_hash = _sha256_field(metadata, "unique_yolo_source_sha256")
    if yolo_hash != UNIQUE_YOLO_SOURCE_SHA256.lower():
        raise A4ArtifactValidationError(
            "unique_yolo_source_sha256_mismatch:"
            f"expected={UNIQUE_YOLO_SOURCE_SHA256.lower()},actual={yolo_hash}"
        )
    selected_threshold = _finite_float(metadata, "selected_threshold")
    if not 0.0 < selected_threshold < 1.0:
        raise A4ArtifactValidationError(
            f"selected_threshold_out_of_range:{selected_threshold}"
        )
    alarm_window = _positive_int(metadata, "alarm_window")
    alarm_required_hits = _positive_int(metadata, "alarm_required_hits")
    if alarm_required_hits > alarm_window:
        raise A4ArtifactValidationError(
            "alarm_required_hits_exceeds_window:"
            f"required={alarm_required_hits},window={alarm_window}"
        )
    _validate_quality_gate(metadata)
    expected_hash = str(metadata.get("model_sha256", "") or "").lower()
    actual_hash = sha256_file(model_path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise A4ArtifactValidationError(
            "model_sha256_mismatch:"
            f"expected={expected_hash or '<missing>'},actual={actual_hash}"
        )
    return dict(metadata)

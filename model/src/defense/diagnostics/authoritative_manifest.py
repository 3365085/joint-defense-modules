from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
STRICT_COUNTS = {
    "model": 1,
    "a3b": 1,
    "physical": 5,
    "normal": 30,
    "videos": 36,
    "records": 37,
}
BASE_ASSET_FIELDS = (
    "asset_id",
    "relative_path",
    "canonical_path",
    "size_bytes",
    "sha256",
    "role",
    "label",
    "purpose",
)
VIDEO_ASSET_FIELDS = (
    "attack_type",
    "expected_module_a_alert",
    "expected_a3b_trigger",
    "expected_module_a_evidence_events",
    "acceptance_order",
)
PHYSICAL_ATTACK_TYPES = {
    "adv_patch",
    "glare",
    "motion_blur",
    "occlusion",
    "visibility_degradation",
}

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ROLE_CATEGORY_HINTS = {
    "a3b": "a3b",
    "a3b_attack": "a3b",
    "a3b_target": "a3b",
    "a3b_video": "a3b",
    "physical": "physical",
    "physical_attack": "physical",
    "module_a_physical": "physical",
    "module_a_physical_attack": "physical",
    "normal": "normal",
    "normal_video": "normal",
    "module_a_normal": "normal",
    "model": "model",
    "unique_model": "model",
    "authoritative_model": "model",
}


class ManifestValidationError(ValueError):
    """Raised when an authoritative manifest fails a strict validation gate."""

    def __init__(self, result: "ManifestValidationResult") -> None:
        self.result = result
        messages = "; ".join(error["message"] for error in result.errors[:5])
        if len(result.errors) > 5:
            messages += f"; ... ({len(result.errors)} errors total)"
        super().__init__(messages or "authoritative manifest validation failed")


@dataclass(frozen=True)
class AuthoritativeAsset:
    asset_id: str
    relative_path: str
    canonical_path: str
    size_bytes: int
    sha256: str
    role: str
    label: str
    purpose: str
    category: str
    attack_type: str | None = None
    expected_module_a_alert: bool | None = None
    expected_a3b_trigger: bool | None = None
    expected_module_a_evidence_events: int | str | None = None
    acceptance_order: int | None = None

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        """Identity is path + expected label/role + SHA, never SHA alone."""

        return (
            _path_key(Path(self.canonical_path)),
            self.label,
            self.role,
            self.sha256,
        )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "canonical_path": self.canonical_path,
            "label": self.label,
            "role": self.role,
            "sha256": self.sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "canonical_path": self.canonical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
            "label": self.label,
            "purpose": self.purpose,
            "category": self.category,
            "identity": self.identity,
        }
        if self.category != "model":
            payload.update(
                {
                    "attack_type": self.attack_type,
                    "expected_module_a_alert": self.expected_module_a_alert,
                    "expected_a3b_trigger": self.expected_a3b_trigger,
                    "expected_module_a_evidence_events": (
                        self.expected_module_a_evidence_events
                    ),
                    "acceptance_order": self.acceptance_order,
                }
            )
        return payload


@dataclass(frozen=True)
class AuthoritativeManifest:
    schema_version: int
    snapshot_date: str
    material_root: str
    unique_model: AuthoritativeAsset
    videos: tuple[AuthoritativeAsset, ...]
    manifest_path: str | None = None

    @property
    def records(self) -> tuple[AuthoritativeAsset, ...]:
        return (self.unique_model, *self.videos)

    @property
    def ordered_videos(self) -> tuple[AuthoritativeAsset, ...]:
        return tuple(
            sorted(
                self.videos,
                key=lambda asset: (
                    asset.acceptance_order
                    if asset.acceptance_order is not None
                    else 1_000_000,
                    asset.asset_id,
                ),
            )
        )

    def asset_by_id(self, asset_id: str) -> AuthoritativeAsset:
        for asset in self.records:
            if asset.asset_id == asset_id:
                return asset
        raise KeyError(asset_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_date": self.snapshot_date,
            "material_root": self.material_root,
            "manifest_path": self.manifest_path,
            "unique_model": self.unique_model.to_dict(),
            "videos": [asset.to_dict() for asset in self.videos],
        }


@dataclass(frozen=True)
class ManifestValidationResult:
    valid: bool
    manifest_path: str | None
    manifest: AuthoritativeManifest | None
    counts: dict[str, int]
    strict_gate: dict[str, Any]
    errors: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    record_checks: tuple[dict[str, Any], ...]
    duplicate_hash_groups: tuple[dict[str, Any], ...]

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.valid,
            "valid": self.valid,
            "manifest_path": self.manifest_path,
            "schema_version": (
                self.manifest.schema_version if self.manifest is not None else None
            ),
            "snapshot_date": (
                self.manifest.snapshot_date if self.manifest is not None else None
            ),
            "material_root": (
                self.manifest.material_root if self.manifest is not None else None
            ),
            "counts": dict(self.counts),
            "strict_gate": dict(self.strict_gate),
            "errors": [dict(item) for item in self.errors],
            "warnings": [dict(item) for item in self.warnings],
            "duplicate_hash_groups": [
                dict(item) for item in self.duplicate_hash_groups
            ],
        }
        if include_records:
            payload["record_checks"] = [dict(item) for item in self.record_checks]
        return payload


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_authoritative_manifest(
    source: str | Path | Mapping[str, Any],
    *,
    verify_files: bool = True,
    strict_counts: bool = True,
) -> ManifestValidationResult:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    record_checks: list[dict[str, Any]] = []
    manifest_path: str | None = None

    if isinstance(source, Mapping):
        payload: Any = dict(source)
    else:
        path = Path(source).expanduser()
        manifest_path = str(path.resolve(strict=False))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _append_error(
                errors,
                "manifest_missing",
                f"manifest file does not exist: {path}",
                field="manifest",
            )
            return _empty_result(manifest_path, errors)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _append_error(
                errors,
                "manifest_unreadable",
                f"failed to read manifest JSON: {exc}",
                field="manifest",
            )
            return _empty_result(manifest_path, errors)

    if not isinstance(payload, dict):
        _append_error(
            errors,
            "root_type",
            "manifest root must be a JSON object",
            field="$",
        )
        return _empty_result(manifest_path, errors)

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        _append_error(
            errors,
            "schema_version",
            f"schema_version must equal {SCHEMA_VERSION}",
            field="schema_version",
            actual=schema_version,
        )

    snapshot_date = payload.get("snapshot_date")
    if not isinstance(snapshot_date, str) or not snapshot_date.strip():
        _append_error(
            errors,
            "snapshot_date",
            "snapshot_date must be a non-empty ISO date string",
            field="snapshot_date",
            actual=snapshot_date,
        )
        snapshot_date_text = ""
    else:
        snapshot_date_text = snapshot_date.strip()
        try:
            date.fromisoformat(snapshot_date_text)
        except ValueError:
            _append_error(
                errors,
                "snapshot_date",
                "snapshot_date must use ISO YYYY-MM-DD form",
                field="snapshot_date",
                actual=snapshot_date,
            )

    material_root_value = payload.get("material_root")
    material_root: Path | None = None
    material_root_text = ""
    if not isinstance(material_root_value, str) or not material_root_value.strip():
        _append_error(
            errors,
            "material_root",
            "material_root must be a non-empty absolute path",
            field="material_root",
            actual=material_root_value,
        )
    else:
        material_root = Path(material_root_value).expanduser()
        material_root_text = str(material_root.resolve(strict=False))
        if not material_root.is_absolute():
            _append_error(
                errors,
                "material_root_not_absolute",
                "material_root must be absolute",
                field="material_root",
                actual=material_root_value,
            )
        if verify_files and (
            not material_root.exists() or not material_root.is_dir()
        ):
            _append_error(
                errors,
                "material_root_missing",
                f"material_root does not exist or is not a directory: {material_root}",
                field="material_root",
            )

    unique_model_raw = payload.get("unique_model")
    unique_model: AuthoritativeAsset | None = None
    if not isinstance(unique_model_raw, dict):
        _append_error(
            errors,
            "unique_model_type",
            "unique_model must be one asset object",
            field="unique_model",
        )
    elif material_root is not None:
        unique_model = _validate_asset(
            unique_model_raw,
            material_root=material_root,
            field_prefix="unique_model",
            is_video=False,
            verify_files=verify_files,
            errors=errors,
            record_checks=record_checks,
        )

    videos_raw = payload.get("videos")
    videos: list[AuthoritativeAsset] = []
    if not isinstance(videos_raw, list):
        _append_error(
            errors,
            "videos_type",
            "videos must be an array",
            field="videos",
        )
        videos_raw = []
    if material_root is not None:
        for index, raw in enumerate(videos_raw):
            field_prefix = f"videos[{index}]"
            if not isinstance(raw, dict):
                _append_error(
                    errors,
                    "asset_type",
                    f"{field_prefix} must be an object",
                    field=field_prefix,
                )
                continue
            asset = _validate_asset(
                raw,
                material_root=material_root,
                field_prefix=field_prefix,
                is_video=True,
                verify_files=verify_files,
                errors=errors,
                record_checks=record_checks,
            )
            if asset is not None:
                videos.append(asset)

    records = ([unique_model] if unique_model is not None else []) + videos
    _validate_record_uniqueness(records, errors)
    duplicate_hash_groups = _duplicate_hash_groups(records)

    counts = {
        "model": 1 if unique_model is not None else 0,
        "a3b": sum(asset.category == "a3b" for asset in videos),
        "physical": sum(asset.category == "physical" for asset in videos),
        "normal": sum(asset.category == "normal" for asset in videos),
        "videos": len(videos),
        "records": len(records),
    }
    strict_mismatches = {
        name: {"expected": expected, "actual": counts.get(name, 0)}
        for name, expected in STRICT_COUNTS.items()
        if counts.get(name, 0) != expected
    }
    strict_gate = {
        "enabled": bool(strict_counts),
        "passed": not strict_mismatches if strict_counts else True,
        "expected": dict(STRICT_COUNTS),
        "actual": dict(counts),
        "mismatches": strict_mismatches,
    }
    if strict_counts:
        for name, mismatch in strict_mismatches.items():
            _append_error(
                errors,
                "strict_count_mismatch",
                (
                    f"strict count {name} must be {mismatch['expected']}, "
                    f"got {mismatch['actual']}"
                ),
                field=name,
                expected=mismatch["expected"],
                actual=mismatch["actual"],
            )
        orders = [
            asset.acceptance_order
            for asset in videos
            if asset.acceptance_order is not None
        ]
        expected_orders = set(range(1, STRICT_COUNTS["videos"] + 1))
        actual_orders = set(orders)
        if (
            len(orders) != STRICT_COUNTS["videos"]
            or len(actual_orders) != len(orders)
            or actual_orders != expected_orders
        ):
            _append_error(
                errors,
                "acceptance_order_gate",
                "acceptance_order values must be unique and exactly 1..36",
                field="videos.acceptance_order",
                expected=list(range(1, STRICT_COUNTS["videos"] + 1)),
                actual=sorted(value for value in actual_orders if value is not None),
            )

    manifest: AuthoritativeManifest | None = None
    if unique_model is not None and material_root is not None:
        manifest = AuthoritativeManifest(
            schema_version=(
                int(schema_version)
                if isinstance(schema_version, int) and not isinstance(schema_version, bool)
                else -1
            ),
            snapshot_date=snapshot_date_text,
            material_root=material_root_text,
            unique_model=unique_model,
            videos=tuple(videos),
            manifest_path=manifest_path,
        )

    return ManifestValidationResult(
        valid=not errors,
        manifest_path=manifest_path,
        manifest=manifest,
        counts=counts,
        strict_gate=strict_gate,
        errors=tuple(errors),
        warnings=tuple(warnings),
        record_checks=tuple(record_checks),
        duplicate_hash_groups=tuple(duplicate_hash_groups),
    )


def load_authoritative_manifest(
    source: str | Path | Mapping[str, Any],
    *,
    verify_files: bool = True,
    strict_counts: bool = True,
) -> AuthoritativeManifest:
    result = validate_authoritative_manifest(
        source,
        verify_files=verify_files,
        strict_counts=strict_counts,
    )
    if not result.valid or result.manifest is None:
        raise ManifestValidationError(result)
    return result.manifest


def _validate_asset(
    raw: Mapping[str, Any],
    *,
    material_root: Path,
    field_prefix: str,
    is_video: bool,
    verify_files: bool,
    errors: list[dict[str, Any]],
    record_checks: list[dict[str, Any]],
) -> AuthoritativeAsset | None:
    required = BASE_ASSET_FIELDS + (VIDEO_ASSET_FIELDS if is_video else ())
    missing = [field for field in required if field not in raw]
    if missing:
        _append_error(
            errors,
            "missing_fields",
            f"{field_prefix} missing required fields: {missing}",
            field=field_prefix,
            missing=missing,
        )

    strings: dict[str, str] = {}
    for field in (
        "asset_id",
        "relative_path",
        "canonical_path",
        "sha256",
        "role",
        "label",
        "purpose",
    ):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            _append_error(
                errors,
                "field_type",
                f"{field_prefix}.{field} must be a non-empty string",
                field=f"{field_prefix}.{field}",
                actual=value,
            )
            strings[field] = ""
        else:
            strings[field] = value.strip()

    size_value = raw.get("size_bytes")
    if (
        isinstance(size_value, bool)
        or not isinstance(size_value, int)
        or size_value <= 0
    ):
        _append_error(
            errors,
            "size_bytes",
            f"{field_prefix}.size_bytes must be a positive integer",
            field=f"{field_prefix}.size_bytes",
            actual=size_value,
        )
        size_bytes = -1
    else:
        size_bytes = size_value

    sha256 = strings["sha256"].lower()
    if sha256 and not _SHA256_RE.fullmatch(sha256):
        _append_error(
            errors,
            "sha256_format",
            f"{field_prefix}.sha256 must contain exactly 64 hexadecimal characters",
            field=f"{field_prefix}.sha256",
            actual=strings["sha256"],
        )

    relative_path_text = strings["relative_path"]
    relative_path = Path(relative_path_text) if relative_path_text else Path()
    if relative_path_text and (
        relative_path.is_absolute() or ".." in relative_path.parts
    ):
        _append_error(
            errors,
            "relative_path",
            f"{field_prefix}.relative_path must stay under material_root",
            field=f"{field_prefix}.relative_path",
            actual=relative_path_text,
        )

    canonical_path_text = strings["canonical_path"]
    canonical_path = (
        Path(canonical_path_text).expanduser()
        if canonical_path_text
        else material_root / "__invalid__"
    )
    expected_path = (material_root / relative_path).resolve(strict=False)
    if canonical_path_text and not canonical_path.is_absolute():
        _append_error(
            errors,
            "canonical_path_not_absolute",
            f"{field_prefix}.canonical_path must be absolute",
            field=f"{field_prefix}.canonical_path",
            actual=canonical_path_text,
        )
    if canonical_path_text and _path_key(canonical_path) != _path_key(expected_path):
        _append_error(
            errors,
            "canonical_path_mismatch",
            (
                f"{field_prefix}.canonical_path must equal "
                "material_root/relative_path"
            ),
            field=f"{field_prefix}.canonical_path",
            expected=str(expected_path),
            actual=canonical_path_text,
        )

    attack_type: str | None = None
    expected_alert: bool | None = None
    expected_a3b: bool | None = None
    evidence_expectation: int | str | None = None
    acceptance_order: int | None = None
    category = "model"
    if is_video:
        attack_type_value = raw.get("attack_type")
        if attack_type_value is None:
            attack_type = None
        elif isinstance(attack_type_value, str) and attack_type_value.strip():
            attack_type = attack_type_value.strip()
        else:
            _append_error(
                errors,
                "attack_type",
                f"{field_prefix}.attack_type must be null or a non-empty string",
                field=f"{field_prefix}.attack_type",
                actual=attack_type_value,
            )

        expected_alert = _strict_bool(
            raw.get("expected_module_a_alert"),
            field=f"{field_prefix}.expected_module_a_alert",
            errors=errors,
        )
        expected_a3b = _strict_bool(
            raw.get("expected_a3b_trigger"),
            field=f"{field_prefix}.expected_a3b_trigger",
            errors=errors,
        )
        evidence_value = raw.get("expected_module_a_evidence_events")
        if (
            isinstance(evidence_value, int)
            and not isinstance(evidence_value, bool)
            and evidence_value == 0
        ):
            evidence_expectation = 0
        elif isinstance(evidence_value, str) and evidence_value.strip() == ">=1":
            evidence_expectation = ">=1"
        else:
            _append_error(
                errors,
                "evidence_expectation",
                (
                    f"{field_prefix}.expected_module_a_evidence_events "
                    'must be 0 or ">=1"'
                ),
                field=f"{field_prefix}.expected_module_a_evidence_events",
                actual=evidence_value,
            )

        order_value = raw.get("acceptance_order")
        if (
            isinstance(order_value, bool)
            or not isinstance(order_value, int)
            or order_value <= 0
        ):
            _append_error(
                errors,
                "acceptance_order",
                f"{field_prefix}.acceptance_order must be a positive integer",
                field=f"{field_prefix}.acceptance_order",
                actual=order_value,
            )
        else:
            acceptance_order = order_value

        category = _video_category(
            expected_a3b=bool(expected_a3b),
            attack_type=attack_type,
            expected_alert=bool(expected_alert),
        )
        _validate_video_expectations(
            field_prefix=field_prefix,
            category=category,
            role=strings["role"],
            attack_type=attack_type,
            expected_alert=expected_alert,
            expected_a3b=expected_a3b,
            evidence_expectation=evidence_expectation,
            errors=errors,
        )
    else:
        hinted = _role_category(strings["role"])
        if hinted is not None and hinted != "model":
            _append_error(
                errors,
                "role_category_mismatch",
                f"{field_prefix}.role identifies a non-model asset",
                field=f"{field_prefix}.role",
                actual=strings["role"],
            )

    actual_size: int | None = None
    actual_sha256: str | None = None
    path_exists = canonical_path.exists() and canonical_path.is_file()
    if verify_files:
        if not path_exists:
            _append_error(
                errors,
                "asset_missing",
                f"asset file does not exist: {canonical_path}",
                field=f"{field_prefix}.canonical_path",
                asset_id=strings["asset_id"] or None,
            )
        else:
            try:
                actual_size = canonical_path.stat().st_size
                if size_bytes > 0 and actual_size != size_bytes:
                    _append_error(
                        errors,
                        "size_mismatch",
                        (
                            f"{field_prefix} size mismatch: expected "
                            f"{size_bytes}, got {actual_size}"
                        ),
                        field=f"{field_prefix}.size_bytes",
                        asset_id=strings["asset_id"] or None,
                        expected=size_bytes,
                        actual=actual_size,
                    )
                actual_sha256 = sha256_file(canonical_path)
                if _SHA256_RE.fullmatch(sha256) and actual_sha256 != sha256:
                    _append_error(
                        errors,
                        "hash_mismatch",
                        (
                            f"{field_prefix} SHA-256 mismatch: expected "
                            f"{sha256}, got {actual_sha256}"
                        ),
                        field=f"{field_prefix}.sha256",
                        asset_id=strings["asset_id"] or None,
                        expected=sha256,
                        actual=actual_sha256,
                    )
            except OSError as exc:
                _append_error(
                    errors,
                    "asset_unreadable",
                    f"failed to inspect asset file: {exc}",
                    field=f"{field_prefix}.canonical_path",
                    asset_id=strings["asset_id"] or None,
                )

    record_checks.append(
        {
            "asset_id": strings["asset_id"] or None,
            "category": category,
            "canonical_path": canonical_path_text or None,
            "identity": {
                "canonical_path": canonical_path_text or None,
                "label": strings["label"] or None,
                "role": strings["role"] or None,
                "sha256": sha256 or None,
            },
            "path_exists": path_exists,
            "expected_size_bytes": size_bytes if size_bytes > 0 else None,
            "actual_size_bytes": actual_size,
            "expected_sha256": sha256 or None,
            "actual_sha256": actual_sha256,
            "file_check_performed": bool(verify_files),
            "size_matches": (
                None
                if not verify_files or actual_size is None or size_bytes <= 0
                else actual_size == size_bytes
            ),
            "hash_matches": (
                None
                if not verify_files or actual_sha256 is None or not sha256
                else actual_sha256 == sha256
            ),
        }
    )

    essential_valid = (
        all(strings[field] for field in strings)
        and size_bytes > 0
        and bool(_SHA256_RE.fullmatch(sha256))
    )
    if not essential_valid:
        return None
    return AuthoritativeAsset(
        asset_id=strings["asset_id"],
        relative_path=relative_path_text,
        canonical_path=str(canonical_path.resolve(strict=False)),
        size_bytes=size_bytes,
        sha256=sha256,
        role=strings["role"],
        label=strings["label"],
        purpose=strings["purpose"],
        category=category,
        attack_type=attack_type,
        expected_module_a_alert=expected_alert,
        expected_a3b_trigger=expected_a3b,
        expected_module_a_evidence_events=evidence_expectation,
        acceptance_order=acceptance_order,
    )


def _validate_video_expectations(
    *,
    field_prefix: str,
    category: str,
    role: str,
    attack_type: str | None,
    expected_alert: bool | None,
    expected_a3b: bool | None,
    evidence_expectation: int | str | None,
    errors: list[dict[str, Any]],
) -> None:
    role_category = _role_category(role)
    if role_category is not None and role_category != category:
        _append_error(
            errors,
            "role_category_mismatch",
            (
                f"{field_prefix}.role implies {role_category}, "
                f"but expectation fields imply {category}"
            ),
            field=f"{field_prefix}.role",
            actual=role,
        )

    if category == "a3b":
        if expected_a3b is not True:
            _append_error(
                errors,
                "a3b_expectation",
                f"{field_prefix} A3b record must expect an A3b trigger",
                field=f"{field_prefix}.expected_a3b_trigger",
            )
        if evidence_expectation != ">=1":
            _append_error(
                errors,
                "a3b_evidence_expectation",
                f"{field_prefix} A3b record must require at least one evidence event",
                field=f"{field_prefix}.expected_module_a_evidence_events",
            )
    elif category == "physical":
        if attack_type not in PHYSICAL_ATTACK_TYPES:
            _append_error(
                errors,
                "physical_attack_type",
                (
                    f"{field_prefix} physical record attack_type must be one of "
                    f"{sorted(PHYSICAL_ATTACK_TYPES)}"
                ),
                field=f"{field_prefix}.attack_type",
                actual=attack_type,
            )
        if expected_alert is not True:
            _append_error(
                errors,
                "physical_alert_expectation",
                f"{field_prefix} physical record must expect Module A alert",
                field=f"{field_prefix}.expected_module_a_alert",
            )
        if expected_a3b is not False:
            _append_error(
                errors,
                "physical_a3b_expectation",
                f"{field_prefix} physical record must not expect A3b trigger",
                field=f"{field_prefix}.expected_a3b_trigger",
            )
        if evidence_expectation != ">=1":
            _append_error(
                errors,
                "physical_evidence_expectation",
                (
                    f"{field_prefix} physical record must require at least "
                    "one evidence event"
                ),
                field=f"{field_prefix}.expected_module_a_evidence_events",
            )
    else:
        if attack_type is not None:
            _append_error(
                errors,
                "normal_attack_type",
                f"{field_prefix} normal record attack_type must be null",
                field=f"{field_prefix}.attack_type",
                actual=attack_type,
            )
        if expected_alert is not False:
            _append_error(
                errors,
                "normal_alert_expectation",
                f"{field_prefix} normal record must expect no Module A alert",
                field=f"{field_prefix}.expected_module_a_alert",
            )
        if expected_a3b is not False:
            _append_error(
                errors,
                "normal_a3b_expectation",
                f"{field_prefix} normal record must expect no A3b trigger",
                field=f"{field_prefix}.expected_a3b_trigger",
            )
        if evidence_expectation != 0:
            _append_error(
                errors,
                "normal_evidence_expectation",
                f"{field_prefix} normal record must require zero evidence events",
                field=f"{field_prefix}.expected_module_a_evidence_events",
            )


def _validate_record_uniqueness(
    records: list[AuthoritativeAsset],
    errors: list[dict[str, Any]],
) -> None:
    ids: dict[str, list[AuthoritativeAsset]] = {}
    identities: dict[tuple[str, str, str, str], list[AuthoritativeAsset]] = {}
    for asset in records:
        ids.setdefault(asset.asset_id, []).append(asset)
        identities.setdefault(asset.identity_key, []).append(asset)
    for asset_id, group in ids.items():
        if len(group) > 1:
            _append_error(
                errors,
                "duplicate_asset_id",
                f"asset_id must be unique: {asset_id}",
                field="asset_id",
                asset_id=asset_id,
                count=len(group),
            )
    for identity, group in identities.items():
        if len(group) > 1:
            _append_error(
                errors,
                "duplicate_asset_identity",
                (
                    "asset identity must be unique by canonical_path + "
                    "label + role + sha256"
                ),
                field="identity",
                identity={
                    "canonical_path": group[0].canonical_path,
                    "label": identity[1],
                    "role": identity[2],
                    "sha256": identity[3],
                },
                asset_ids=[asset.asset_id for asset in group],
            )


def _duplicate_hash_groups(
    records: list[AuthoritativeAsset],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[AuthoritativeAsset]] = {}
    for asset in records:
        by_hash.setdefault(asset.sha256, []).append(asset)
    return [
        {
            "sha256": sha256,
            "record_count": len(group),
            "allowed": True,
            "asset_ids": [asset.asset_id for asset in group],
            "identities": [asset.identity for asset in group],
        }
        for sha256, group in sorted(by_hash.items())
        if len(group) > 1
    ]


def _video_category(
    *,
    expected_a3b: bool,
    attack_type: str | None,
    expected_alert: bool,
) -> str:
    if expected_a3b:
        return "a3b"
    if attack_type is not None or expected_alert:
        return "physical"
    return "normal"


def _role_category(role: str) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "_", role.strip().lower()).strip("_")
    return _ROLE_CATEGORY_HINTS.get(token)


def _strict_bool(
    value: Any,
    *,
    field: str,
    errors: list[dict[str, Any]],
) -> bool | None:
    if isinstance(value, bool):
        return value
    _append_error(
        errors,
        "field_type",
        f"{field} must be a JSON boolean",
        field=field,
        actual=value,
    )
    return None


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def _append_error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    **details: Any,
) -> None:
    errors.append({"code": code, "message": message, **details})


def _empty_result(
    manifest_path: str | None,
    errors: list[dict[str, Any]],
) -> ManifestValidationResult:
    counts = {key: 0 for key in STRICT_COUNTS}
    strict_gate = {
        "enabled": True,
        "passed": False,
        "expected": dict(STRICT_COUNTS),
        "actual": dict(counts),
        "mismatches": {
            name: {"expected": expected, "actual": 0}
            for name, expected in STRICT_COUNTS.items()
        },
    }
    return ManifestValidationResult(
        valid=False,
        manifest_path=manifest_path,
        manifest=None,
        counts=counts,
        strict_gate=strict_gate,
        errors=tuple(errors),
        warnings=(),
        record_checks=(),
        duplicate_hash_groups=(),
    )


__all__ = [
    "AuthoritativeAsset",
    "AuthoritativeManifest",
    "BASE_ASSET_FIELDS",
    "ManifestValidationError",
    "ManifestValidationResult",
    "PHYSICAL_ATTACK_TYPES",
    "SCHEMA_VERSION",
    "STRICT_COUNTS",
    "VIDEO_ASSET_FIELDS",
    "load_authoritative_manifest",
    "sha256_file",
    "validate_authoritative_manifest",
]

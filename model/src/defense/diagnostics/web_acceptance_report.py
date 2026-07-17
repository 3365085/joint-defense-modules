from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlencode

from .authoritative_manifest import (
    AuthoritativeAsset,
    AuthoritativeManifest,
    STRICT_COUNTS,
    sha256_file,
)


WEB_ACCEPTANCE_SCHEMA_VERSION = 1
REPORT_TYPE = "module_a_authoritative_web_acceptance"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MIN_DETECTOR_FPS = 25.0
MIN_DETECTION_SOURCE_COVERAGE = 0.90
A3B_FIRST_TRIGGER_MIN_S = 0.50
A3B_FIRST_TRIGGER_MAX_S = 1.50
_A3B_CONTINUITY_HARD_SUPPRESSION_GATES = frozenset(
    {
        "border_suppressed",
        "camera_motion_suppressed",
        "physical_motion_suppressed",
        "rebuilt_result_stale",
        "rebuilt_candidate_disallowed",
        "rebuilt_policy_suppressed",
    }
)


class JsonClient(Protocol):
    """Small common surface shared by TestClient, httpx and fake clients."""

    def get(self, path: str, **kwargs: Any) -> Any: ...

    def post(self, path: str, **kwargs: Any) -> Any: ...


class WebAcceptanceContractError(AssertionError):
    def __init__(self, errors: Sequence[Mapping[str, Any]]) -> None:
        self.errors = tuple(dict(error) for error in errors)
        message = "; ".join(str(error.get("message") or error) for error in self.errors)
        super().__init__(message or "web acceptance report contract failed")


def run_web_preflight(
    client: JsonClient,
    manifest: AuthoritativeManifest,
    *,
    verify_status_artifact_files: bool = True,
) -> dict[str, Any]:
    """Inspect the live FastAPI contract without claiming any video result."""

    blockers: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    status_snapshot: dict[str, Any] = {}
    overlay_snapshot: dict[str, Any] = {}
    evidence_snapshot: list[Any] = []

    try:
        status_response = _request_json(client, "get", "/api/status")
        status_payload = status_response["payload"]
        status_ok = (
            status_response["status_code"] < 400
            and isinstance(status_payload, dict)
            and status_payload.get("ok") is True
            and isinstance(status_payload.get("status"), dict)
        )
        _record_check(
            checks,
            blockers,
            name="status_contract",
            passed=status_ok,
            message="GET /api/status must return {ok: true, status: object}",
            actual={
                "status_code": status_response["status_code"],
                "ok": (
                    status_payload.get("ok")
                    if isinstance(status_payload, dict)
                    else None
                ),
                "status_is_object": (
                    isinstance(status_payload, dict)
                    and isinstance(status_payload.get("status"), dict)
                ),
            },
        )
        if status_ok:
            status_snapshot = dict(status_payload["status"])
    except Exception as exc:
        _record_check(
            checks,
            blockers,
            name="status_contract",
            passed=False,
            message=f"GET /api/status failed: {exc}",
        )

    if status_snapshot:
        _evaluate_runtime_status(
            status_snapshot,
            manifest=manifest,
            checks=checks,
            blockers=blockers,
            verify_status_artifact_files=verify_status_artifact_files,
        )

    try:
        overlay_response = _request_json(
            client,
            "get",
            "/api/overlay",
            params={"since_seq": 0},
        )
        overlay_payload = overlay_response["payload"]
        overlay = (
            overlay_payload.get("overlay")
            if isinstance(overlay_payload, dict)
            else None
        )
        overlay_ok = (
            overlay_response["status_code"] < 400
            and isinstance(overlay_payload, dict)
            and overlay_payload.get("ok") is True
            and isinstance(overlay, dict)
            and isinstance(overlay.get("records"), list)
            and _is_int(overlay.get("latest_seq"))
        )
        _record_check(
            checks,
            blockers,
            name="overlay_contract",
            passed=overlay_ok,
            message=(
                "GET /api/overlay must return overlay.records array and "
                "integer latest_seq"
            ),
            actual={
                "status_code": overlay_response["status_code"],
                "ok": (
                    overlay_payload.get("ok")
                    if isinstance(overlay_payload, dict)
                    else None
                ),
                "records_is_array": (
                    isinstance(overlay, dict)
                    and isinstance(overlay.get("records"), list)
                ),
                "latest_seq": (
                    overlay.get("latest_seq")
                    if isinstance(overlay, dict)
                    else None
                ),
            },
        )
        if overlay_ok:
            overlay_snapshot = dict(overlay)
    except Exception as exc:
        _record_check(
            checks,
            blockers,
            name="overlay_contract",
            passed=False,
            message=f"GET /api/overlay failed: {exc}",
        )

    try:
        evidence_response = _request_json(
            client,
            "get",
            "/api/evidence/events",
            params={"limit": 1},
        )
        evidence_payload = evidence_response["payload"]
        evidence = _extract_evidence_events(evidence_payload)
        evidence_ok = (
            evidence_response["status_code"] < 400
            and isinstance(evidence_payload, dict)
            and evidence_payload.get("ok") is True
            and isinstance(evidence, list)
        )
        _record_check(
            checks,
            blockers,
            name="evidence_contract",
            passed=evidence_ok,
            message=(
                "GET /api/evidence/events must return an evidence array or "
                "evidence.events array"
            ),
            actual={
                "status_code": evidence_response["status_code"],
                "ok": (
                    evidence_payload.get("ok")
                    if isinstance(evidence_payload, dict)
                    else None
                ),
                "event_count": len(evidence) if isinstance(evidence, list) else None,
                "shape": (
                    "evidence.events"
                    if isinstance(evidence_payload, dict)
                    and isinstance(evidence_payload.get("evidence"), Mapping)
                    else "evidence"
                ),
            },
        )
        if evidence_ok:
            evidence_snapshot = list(evidence)
    except Exception as exc:
        _record_check(
            checks,
            blockers,
            name="evidence_contract",
            passed=False,
            message=f"GET /api/evidence/events failed: {exc}",
        )

    return {
        "schema_version": WEB_ACCEPTANCE_SCHEMA_VERSION,
        "checked_at": _utc_now(),
        "passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "status_snapshot": status_snapshot,
        "overlay_snapshot": overlay_snapshot,
        "evidence_snapshot_count": len(evidence_snapshot),
        "contract": {
            "transport": "FastAPI/TestClient-compatible HTTP JSON",
            "status_endpoint": "/api/status",
            "overlay_endpoint": "/api/overlay",
            "evidence_endpoint": "/api/evidence/events",
        },
    }


def run_authoritative_web_acceptance(
    client: JsonClient,
    manifest: AuthoritativeManifest,
    *,
    selected_asset_ids: Sequence[str] | None = None,
    profile: str = "default",
    ready_timeout_s: float = 45.0,
    asset_timeout_s: float = 600.0,
    poll_interval_s: float = 0.25,
    evidence_limit: int = 5000,
    verify_status_artifact_files: bool = True,
    require_preflight_pass: bool = True,
) -> dict[str, Any]:
    """Run selected authoritative videos through the real Web/latest-only API."""

    selected = _select_videos(manifest, selected_asset_ids)
    preflight = run_web_preflight(
        client,
        manifest,
        verify_status_artifact_files=verify_status_artifact_files,
    )
    asset_reports: list[dict[str, Any]] = []
    run_started_at = _utc_now()
    if preflight["passed"] or not require_preflight_pass:
        for asset in selected:
            asset_reports.append(
                _run_one_asset(
                    client,
                    asset,
                    manifest=manifest,
                    profile=profile,
                    ready_timeout_s=ready_timeout_s,
                    asset_timeout_s=asset_timeout_s,
                    poll_interval_s=poll_interval_s,
                    evidence_limit=evidence_limit,
                )
            )

    report = build_web_acceptance_report(
        manifest=manifest,
        preflight=preflight,
        asset_reports=asset_reports,
        selected_asset_ids=[asset.asset_id for asset in selected],
        run_started_at=run_started_at,
        run_finished_at=_utc_now(),
    )
    contract_errors = validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=(
            len(selected) == STRICT_COUNTS["videos"]
            and {asset.asset_id for asset in selected}
            == {asset.asset_id for asset in manifest.videos}
        ),
    )
    report["contract_validation"] = {
        "passed": not contract_errors,
        "errors": contract_errors,
    }
    if contract_errors:
        report["summary"]["passed"] = False
        report["summary"]["final_acceptance_eligible"] = False
        report["summary"]["blockers"].append(
            {
                "code": "report_contract_invalid",
                "message": "generated report failed its own schema/aggregation checks",
                "errors": contract_errors,
            }
        )
    return report


def build_web_acceptance_report(
    *,
    manifest: AuthoritativeManifest,
    preflight: Mapping[str, Any],
    asset_reports: Sequence[Mapping[str, Any]],
    selected_asset_ids: Sequence[str],
    run_started_at: str | None = None,
    run_finished_at: str | None = None,
) -> dict[str, Any]:
    reports = [dict(report) for report in asset_reports]
    summary = aggregate_web_acceptance_report(
        manifest=manifest,
        preflight=preflight,
        asset_reports=reports,
        selected_asset_ids=selected_asset_ids,
    )
    manifest_identity = {
        "manifest_path": manifest.manifest_path,
        "schema_version": manifest.schema_version,
        "snapshot_date": manifest.snapshot_date,
        "material_root": manifest.material_root,
        "unique_model": {
            "asset_id": manifest.unique_model.asset_id,
            "canonical_path": manifest.unique_model.canonical_path,
            "sha256": manifest.unique_model.sha256,
            "label": manifest.unique_model.label,
            "role": manifest.unique_model.role,
        },
        "record_count": len(manifest.records),
        "video_count": len(manifest.videos),
    }
    return {
        "schema_version": WEB_ACCEPTANCE_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "generated_at": _utc_now(),
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "manifest": manifest_identity,
        "preflight": dict(preflight),
        "selected_asset_ids": list(selected_asset_ids),
        "assets": reports,
        "summary": summary,
        "execution_contract": {
            "runtime_surface": "FastAPI HTTP JSON",
            "source_type": "file",
            "realtime": True,
            "detector_queue_policy_required": "latest_only",
            "successful_completion_requires_source_ended": True,
            "timeout_is_success": False,
            "offline_frame_replay_is_final_acceptance": False,
        },
    }


def aggregate_web_acceptance_report(
    *,
    manifest: AuthoritativeManifest,
    preflight: Mapping[str, Any],
    asset_reports: Sequence[Mapping[str, Any]],
    selected_asset_ids: Sequence[str],
) -> dict[str, Any]:
    reports = [dict(report) for report in asset_reports]
    selected_set = set(selected_asset_ids)
    report_ids = [
        str(report.get("asset_id") or "")
        for report in reports
    ]
    report_id_set = set(report_ids)
    manifest_ids = {asset.asset_id for asset in manifest.videos}
    complete_selection = (
        len(selected_asset_ids) == len(manifest.videos)
        and len(selected_set) == len(selected_asset_ids)
        and selected_set == manifest_ids
    )
    category_totals = {
        category: sum(
            1 for report in reports if report.get("category") == category
        )
        for category in ("a3b", "physical", "normal")
    }
    category_passed = {
        category: sum(
            1
            for report in reports
            if report.get("category") == category and report.get("passed") is True
        )
        for category in ("a3b", "physical", "normal")
    }
    source_ended_count = sum(
        report.get("execution", {}).get("source_ended") is True
        for report in reports
    )
    timeout_count = sum(
        report.get("execution", {}).get("timed_out") is True
        for report in reports
    )
    passed_count = sum(report.get("passed") is True for report in reports)
    normal_false_positive_videos = sum(
        report.get("category") == "normal"
        and (
            report.get("observations", {}).get("alert_confirmed_observed") is True
            or int(
                report.get("observations", {}).get(
                    "module_a_evidence_event_count", 0
                )
                or 0
            )
            > 0
        )
        for report in reports
    )
    physical_alert_hit_videos = sum(
        report.get("category") == "physical"
        and report.get("observations", {}).get(
            "physical_alert_confirmed_observed"
        )
        is True
        for report in reports
    )
    a3b_trigger_hit_videos = sum(
        report.get("category") == "a3b"
        and report.get("observations", {}).get("a3b_confirmed_observed") is True
        for report in reports
    )
    total_evidence_events = sum(
        int(
            report.get("observations", {}).get(
                "module_a_evidence_event_count", 0
            )
            or 0
        )
        for report in reports
    )

    blockers: list[dict[str, Any]] = []
    if preflight.get("passed") is not True:
        blockers.append(
            {
                "code": "preflight_failed",
                "message": "Web preflight has blockers",
                "preflight_blockers": list(preflight.get("blockers") or []),
            }
        )
    if not complete_selection:
        blockers.append(
            {
                "code": "incomplete_manifest_selection",
                "message": (
                    f"final acceptance requires all {len(manifest.videos)} "
                    "authoritative videos exactly once"
                ),
                "selected_count": len(selected_asset_ids),
                "expected_count": len(manifest.videos),
            }
        )
    if len(reports) != len(selected_asset_ids):
        blockers.append(
            {
                "code": "missing_asset_reports",
                "message": "not every selected asset has a runtime report",
                "selected_count": len(selected_asset_ids),
                "report_count": len(reports),
            }
        )
    if (
        len(report_ids) != len(report_id_set)
        or report_id_set != selected_set
        or report_ids != list(selected_asset_ids)
    ):
        blockers.append(
            {
                "code": "asset_report_identity_mismatch",
                "message": (
                    "asset reports must match selected_asset_ids exactly, "
                    "once each and in acceptance order"
                ),
                "selected_asset_ids": list(selected_asset_ids),
                "reported_asset_ids": report_ids,
            }
        )
    if source_ended_count != len(reports):
        blockers.append(
            {
                "code": "source_not_ended",
                "message": (
                    "every completed asset report must observe source_ended=true"
                ),
                "source_ended_count": source_ended_count,
                "report_count": len(reports),
            }
        )
    if timeout_count:
        blockers.append(
            {
                "code": "asset_timeouts",
                "message": "timeouts are failures and never successful completion",
                "timeout_count": timeout_count,
            }
        )
    failed_ids = [
        str(report.get("asset_id"))
        for report in reports
        if report.get("passed") is not True
    ]
    if failed_ids:
        blockers.append(
            {
                "code": "asset_gate_failures",
                "message": "one or more per-asset acceptance gates failed",
                "asset_ids": failed_ids,
            }
        )

    passed = (
        not blockers
        and complete_selection
        and len(reports) == len(manifest.videos)
        and passed_count == len(reports)
    )
    return {
        "passed": passed,
        "final_acceptance_eligible": passed,
        "preflight_passed": preflight.get("passed") is True,
        "complete_manifest_selection": complete_selection,
        "expected_video_count": len(manifest.videos),
        "selected_asset_count": len(selected_asset_ids),
        "reported_asset_count": len(reports),
        "source_ended_count": source_ended_count,
        "timeout_count": timeout_count,
        "passed_asset_count": passed_count,
        "failed_asset_count": len(reports) - passed_count,
        "categories": {
            category: {
                "reported": category_totals[category],
                "passed": category_passed[category],
            }
            for category in ("a3b", "physical", "normal")
        },
        "physical_alert_hit_videos": physical_alert_hit_videos,
        "a3b_trigger_hit_videos": a3b_trigger_hit_videos,
        "normal_false_positive_videos": normal_false_positive_videos,
        "module_a_evidence_event_count": total_evidence_events,
        "blockers": blockers,
    }


def validate_web_acceptance_report(
    report: Mapping[str, Any],
    *,
    manifest: AuthoritativeManifest | None = None,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if report.get("schema_version") != WEB_ACCEPTANCE_SCHEMA_VERSION:
        _contract_error(
            errors,
            "schema_version",
            f"schema_version must equal {WEB_ACCEPTANCE_SCHEMA_VERSION}",
        )
    if report.get("report_type") != REPORT_TYPE:
        _contract_error(errors, "report_type", f"report_type must equal {REPORT_TYPE}")
    for field in (
        "manifest",
        "preflight",
        "selected_asset_ids",
        "assets",
        "summary",
        "execution_contract",
    ):
        if field not in report:
            _contract_error(errors, "missing_field", f"report missing {field}", field=field)

    selected = report.get("selected_asset_ids")
    assets = report.get("assets")
    summary = report.get("summary")
    if not isinstance(selected, list):
        _contract_error(
            errors,
            "selected_asset_ids_type",
            "selected_asset_ids must be an array",
        )
        selected = []
    if not isinstance(assets, list):
        _contract_error(errors, "assets_type", "assets must be an array")
        assets = []
    if not isinstance(summary, dict):
        _contract_error(errors, "summary_type", "summary must be an object")
        summary = {}

    seen_ids: set[str] = set()
    for index, asset in enumerate(assets):
        prefix = f"assets[{index}]"
        if not isinstance(asset, dict):
            _contract_error(
                errors,
                "asset_type",
                f"{prefix} must be an object",
                field=prefix,
            )
            continue
        for field in (
            "asset_id",
            "identity",
            "category",
            "expectations",
            "execution",
            "source_identity",
            "lineage",
            "observations",
            "runtime_contract",
            "gates",
            "passed",
            "blockers",
        ):
            if field not in asset:
                _contract_error(
                    errors,
                    "asset_missing_field",
                    f"{prefix} missing {field}",
                    field=f"{prefix}.{field}",
                )
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            _contract_error(
                errors,
                "asset_id",
                f"{prefix}.asset_id must be a non-empty string",
            )
        elif asset_id in seen_ids:
            _contract_error(
                errors,
                "duplicate_asset_report",
                f"duplicate asset report: {asset_id}",
            )
        else:
            seen_ids.add(asset_id)
        execution = asset.get("execution")
        if isinstance(execution, dict):
            if asset.get("passed") is True and execution.get("source_ended") is not True:
                _contract_error(
                    errors,
                    "passed_without_source_end",
                    f"{prefix} cannot pass without source_ended=true",
                )
            if asset.get("passed") is True and execution.get("timed_out") is True:
                _contract_error(
                    errors,
                    "timeout_marked_success",
                    f"{prefix} timeout cannot be marked passed",
                )
        runtime_contract = asset.get("runtime_contract")
        if (
            asset.get("passed") is True
            and (
                not isinstance(runtime_contract, dict)
                or runtime_contract.get("passed") is not True
            )
        ):
            _contract_error(
                errors,
                "passed_without_runtime_contract",
                f"{prefix} cannot pass without production runtime contract",
            )
        lineage = asset.get("lineage")
        if (
            asset.get("passed") is True
            and (
                not isinstance(lineage, dict)
                or lineage.get("passed") is not True
            )
        ):
            _contract_error(
                errors,
                "passed_without_runtime_lineage",
                f"{prefix} cannot pass without run/source/evidence lineage",
            )
        gates = asset.get("gates")
        if asset.get("passed") is True:
            failed_gate_names = [
                str(name)
                for name, gate in (
                    gates.items() if isinstance(gates, Mapping) else ()
                )
                if not isinstance(gate, Mapping)
                or gate.get("passed") is not True
            ]
            if not isinstance(gates, Mapping) or failed_gate_names:
                _contract_error(
                    errors,
                    "passed_with_failed_gates",
                    f"{prefix} cannot pass with failed or missing gates",
                    failed_gates=failed_gate_names,
                )

    reported_ids = [
        str(asset.get("asset_id") or "")
        for asset in assets
        if isinstance(asset, dict)
    ]
    if reported_ids != list(selected):
        _contract_error(
            errors,
            "selected_report_identity_mismatch",
            (
                "asset report IDs must equal selected_asset_ids exactly, "
                "once each and in the same order"
            ),
            expected=list(selected),
            actual=reported_ids,
        )

    if manifest is not None:
        manifest_by_id = {
            asset.asset_id: asset
            for asset in manifest.videos
        }
        expected_selected = [
            asset.asset_id
            for asset in manifest.ordered_videos
            if asset.asset_id in set(selected)
        ]
        if list(selected) != expected_selected:
            _contract_error(
                errors,
                "manifest_selection_order_mismatch",
                "selected_asset_ids must follow authoritative acceptance order",
                expected=expected_selected,
                actual=list(selected),
            )
        for index, asset_report in enumerate(assets):
            if not isinstance(asset_report, dict):
                continue
            asset_id = str(asset_report.get("asset_id") or "")
            expected_asset = manifest_by_id.get(asset_id)
            if expected_asset is None:
                _contract_error(
                    errors,
                    "unknown_manifest_asset",
                    f"assets[{index}] is not present in the authoritative manifest",
                    asset_id=asset_id,
                )
                continue
            expected_values = {
                "identity": expected_asset.identity,
                "relative_path": expected_asset.relative_path,
                "category": expected_asset.category,
                "attack_type": expected_asset.attack_type,
                "acceptance_order": expected_asset.acceptance_order,
                "expectations": {
                    "module_a_alert": expected_asset.expected_module_a_alert,
                    "a3b_trigger": expected_asset.expected_a3b_trigger,
                    "module_a_evidence_events": (
                        expected_asset.expected_module_a_evidence_events
                    ),
                },
            }
            for field, expected in expected_values.items():
                if asset_report.get(field) != expected:
                    _contract_error(
                        errors,
                        "manifest_asset_mismatch",
                        (
                            f"assets[{index}].{field} must match the "
                            "authoritative manifest"
                        ),
                        asset_id=asset_id,
                        field=f"assets[{index}].{field}",
                        expected=expected,
                        actual=asset_report.get(field),
                    )

    if summary:
        expected_values = {
            "selected_asset_count": len(selected),
            "reported_asset_count": len(assets),
            "source_ended_count": sum(
                isinstance(asset, dict)
                and isinstance(asset.get("execution"), dict)
                and asset["execution"].get("source_ended") is True
                for asset in assets
            ),
            "timeout_count": sum(
                isinstance(asset, dict)
                and isinstance(asset.get("execution"), dict)
                and asset["execution"].get("timed_out") is True
                for asset in assets
            ),
            "passed_asset_count": sum(
                isinstance(asset, dict) and asset.get("passed") is True
                for asset in assets
            ),
        }
        for field, expected in expected_values.items():
            if summary.get(field) != expected:
                _contract_error(
                    errors,
                    "summary_mismatch",
                    f"summary.{field} must equal {expected}",
                    field=f"summary.{field}",
                    expected=expected,
                    actual=summary.get(field),
                )
        if summary.get("passed") is True:
            if summary.get("preflight_passed") is not True:
                _contract_error(
                    errors,
                    "summary_passed_without_preflight",
                    "summary cannot pass when preflight failed",
                )
            if summary.get("complete_manifest_selection") is not True:
                _contract_error(
                    errors,
                    "summary_passed_incomplete",
                    "summary cannot pass an incomplete manifest selection",
                )
            if summary.get("source_ended_count") != len(assets):
                _contract_error(
                    errors,
                    "summary_passed_without_source_end",
                    "summary cannot pass unless every source ended",
                )
            if summary.get("timeout_count") != 0:
                _contract_error(
                    errors,
                    "summary_timeout_marked_success",
                    "summary cannot pass with timeouts",
                )
    if require_complete:
        if len(selected) != STRICT_COUNTS["videos"]:
            _contract_error(
                errors,
                "complete_selection_count",
                "complete report must select exactly 37 videos",
            )
        if len(assets) != STRICT_COUNTS["videos"]:
            _contract_error(
                errors,
                "complete_report_count",
                "complete report must contain exactly 37 asset reports",
            )
        category_counts = {
            category: sum(
                isinstance(asset, dict) and asset.get("category") == category
                for asset in assets
            )
            for category in ("a3b", "physical", "normal")
        }
        for category in ("a3b", "physical", "normal"):
            if category_counts[category] != STRICT_COUNTS[category]:
                _contract_error(
                    errors,
                    "complete_category_count",
                    (
                        f"complete report category {category} must contain "
                        f"{STRICT_COUNTS[category]} assets"
                    ),
                    category=category,
                    expected=STRICT_COUNTS[category],
                    actual=category_counts[category],
                )
    return errors


def assert_web_acceptance_report(
    report: Mapping[str, Any],
    *,
    manifest: AuthoritativeManifest | None = None,
    require_complete: bool = False,
) -> None:
    errors = validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=require_complete,
    )
    if errors:
        raise WebAcceptanceContractError(errors)


def _evaluate_runtime_status(
    status: Mapping[str, Any],
    *,
    manifest: AuthoritativeManifest,
    checks: dict[str, dict[str, Any]],
    blockers: list[dict[str, Any]],
    verify_status_artifact_files: bool,
) -> None:
    authoritative = status.get("authoritative_model")
    if not isinstance(authoritative, Mapping):
        authoritative = {}

    backend = _first_nonempty(
        status.get("backend"),
        authoritative.get("effective_backend"),
        authoritative.get("backend"),
        status.get("detector_backend"),
    )
    backend_ok = _is_tensorrt(backend)
    _record_check(
        checks,
        blockers,
        name="production_tensorrt",
        passed=backend_ok,
        message="effective production detector backend must be TensorRT",
        actual=backend,
    )

    queue_policy = status.get("detector_queue_policy")
    _record_check(
        checks,
        blockers,
        name="latest_only",
        passed=queue_policy == "latest_only",
        message="detector_queue_policy must equal latest_only",
        actual=queue_policy,
    )

    source_path = _first_nonempty(
        _node_value(authoritative.get("source"), "canonical_path", "path", "artifact"),
        authoritative.get("source_path"),
        authoritative.get("source_model_path"),
        status.get("authoritative_source_path"),
    )
    source_hash = _normalize_hash(
        _first_nonempty(
            _node_value(authoritative.get("source"), "sha256", "hash"),
            authoritative.get("source_sha256"),
            authoritative.get("source_model_sha256"),
            status.get("authoritative_source_sha256"),
        )
    )
    source_path_matches = (
        bool(source_path)
        and _path_key(Path(str(source_path)))
        == _path_key(Path(manifest.unique_model.canonical_path))
    )
    source_hash_matches = source_hash == manifest.unique_model.sha256
    source_file_ok: bool | None = None
    source_file_actual_hash: str | None = None
    if verify_status_artifact_files and source_path:
        source_file = Path(str(source_path))
        if source_file.exists() and source_file.is_file():
            try:
                source_file_actual_hash = sha256_file(source_file)
                source_file_ok = source_file_actual_hash == source_hash
            except OSError:
                source_file_ok = False
        else:
            source_file_ok = False
    source_binding_ok = (
        source_path_matches
        and source_hash_matches
        and (source_file_ok is not False)
    )
    _record_check(
        checks,
        blockers,
        name="authoritative_source_binding",
        passed=source_binding_ok,
        message=(
            "status authoritative source path/hash must match manifest unique_model"
        ),
        actual={
            "path": source_path,
            "sha256": source_hash,
            "file_sha256": source_file_actual_hash,
        },
        expected={
            "path": manifest.unique_model.canonical_path,
            "sha256": manifest.unique_model.sha256,
        },
    )

    engine_node = authoritative.get("engine")
    engine_path = _first_nonempty(
        _node_value(engine_node, "canonical_path", "path", "artifact"),
        authoritative.get("engine_path"),
        status.get("authoritative_engine_path"),
        status.get("artifact") if _is_tensorrt(backend) else None,
    )
    engine_hash = _normalize_hash(
        _first_nonempty(
            _node_value(engine_node, "sha256", "hash", "artifact_sha256"),
            authoritative.get("engine_sha256"),
            status.get("authoritative_engine_sha256"),
            status.get("artifact_sha256"),
        )
    )
    bound_source_hash = _normalize_hash(
        _first_nonempty(
            _node_value(
                engine_node,
                "source_sha256",
                "source_model_sha256",
                "derived_from_sha256",
                "parent_sha256",
            ),
            _nested_value(engine_node, "metadata", "source_sha256"),
            _nested_value(engine_node, "metadata", "source_model_sha256"),
            authoritative.get("engine_source_sha256"),
            _nested_value(authoritative, "binding", "source_sha256"),
            _nested_value(authoritative, "binding", "source_model_sha256"),
        )
    )
    engine_hash_format_ok = bool(
        engine_hash and _SHA256_RE.fullmatch(engine_hash)
    )
    engine_file_ok: bool | None = None
    engine_file_actual_hash: str | None = None
    if verify_status_artifact_files and engine_path:
        engine_file = Path(str(engine_path))
        if engine_file.exists() and engine_file.is_file():
            try:
                engine_file_actual_hash = sha256_file(engine_file)
                engine_file_ok = engine_file_actual_hash == engine_hash
            except OSError:
                engine_file_ok = False
        else:
            engine_file_ok = False
    metadata_valid = authoritative.get("metadata_valid")
    engine_binding_ok = (
        bool(engine_path)
        and engine_hash_format_ok
        and bound_source_hash == manifest.unique_model.sha256
        and metadata_valid is True
        and (engine_file_ok is not False)
    )
    _record_check(
        checks,
        blockers,
        name="authoritative_engine_binding",
        passed=engine_binding_ok,
        message=(
            "TensorRT engine must expose its own SHA-256 and metadata binding "
            "to the authoritative source SHA-256"
        ),
        actual={
            "path": engine_path,
            "sha256": engine_hash,
            "file_sha256": engine_file_actual_hash,
            "bound_source_sha256": bound_source_hash,
            "metadata_valid": metadata_valid,
        },
        expected_source_sha256=manifest.unique_model.sha256,
    )

    decoder_presence = _required_status_fields(
        status,
        {
            "backend": (
                ("decoder", "backend"),
                ("video_decoder", "backend"),
                ("decoder_backend",),
                ("source_decoder_backend",),
            ),
            "codec": (
                ("decoder", "codec"),
                ("video_decoder", "codec"),
                ("decoder_codec",),
                ("source_decoder_codec",),
            ),
            "gpu_device": (
                ("decoder", "gpu_device"),
                ("video_decoder", "gpu_device"),
                ("decoder_gpu_device",),
                ("source_decoder_gpu_device",),
            ),
            "decode_p50_ms": (
                ("decoder", "decode_ms", "p50"),
                ("decoder", "decode_p50_ms"),
                ("decoder_decode_p50_ms",),
                ("decode_p50_ms",),
            ),
            "decode_p95_ms": (
                ("decoder", "decode_ms", "p95"),
                ("decoder", "decode_p95_ms"),
                ("decoder_decode_p95_ms",),
                ("decode_p95_ms",),
            ),
            "gpu_to_cpu_copy_p50_ms": (
                ("decoder", "gpu_to_cpu_copy_ms", "p50"),
                ("decoder", "gpu_to_cpu_copy_p50_ms"),
                ("decoder_gpu_to_cpu_copy_p50_ms",),
                ("gpu_to_cpu_copy_p50_ms",),
            ),
            "gpu_to_cpu_copy_p95_ms": (
                ("decoder", "gpu_to_cpu_copy_ms", "p95"),
                ("decoder", "gpu_to_cpu_copy_p95_ms"),
                ("decoder_gpu_to_cpu_copy_p95_ms",),
                ("gpu_to_cpu_copy_p95_ms",),
            ),
            "fallback_reason": (
                ("decoder", "fallback_reason"),
                ("video_decoder", "fallback_reason"),
                ("decoder_fallback_reason",),
                ("source_decoder_fallback_reason",),
            ),
        },
    )
    _record_check(
        checks,
        blockers,
        name="decoder_status_fields",
        passed=not decoder_presence["missing"],
        message=(
            "status must expose decoder backend/codec/device/timing/copy/"
            "fallback fields"
        ),
        actual=decoder_presence,
    )

    native_presence = _required_status_fields(
        status,
        {
            "available": (
                ("native", "available"),
                ("module_a_native", "available"),
                ("native_available",),
            ),
            "version": (
                ("native", "version"),
                ("module_a_native", "version"),
                ("native_version",),
            ),
            "binary_sha256": (
                ("native", "binary_sha256"),
                ("native", "sha256"),
                ("module_a_native", "binary_sha256"),
                ("native_binary_sha256",),
            ),
            "enabled_stages": (
                ("native", "enabled_stages"),
                ("native", "hit_stages"),
                ("module_a_native", "enabled_stages"),
                ("native_enabled_stages",),
                ("native_hit_stages",),
            ),
            "fallback_reason": (
                ("native", "fallback_reason"),
                ("module_a_native", "fallback_reason"),
                ("native_fallback_reason",),
            ),
        },
    )
    _record_check(
        checks,
        blockers,
        name="native_status_fields",
        passed=not native_presence["missing"],
        message=(
            "status must expose native availability/version/hash/stages/"
            "fallback fields"
        ),
        actual=native_presence,
    )

    fallback_paths = [
        path
        for path, _value in _flatten_mapping(status)
        if path[-1].lower().endswith("fallback_reason")
    ]
    _record_check(
        checks,
        blockers,
        name="fallback_visibility",
        passed=bool(fallback_paths),
        message="status must visibly expose fallback_reason fields",
        actual=[".".join(path) for path in fallback_paths],
    )

    if status.get("source_ended") is True:
        completion_barrier = {
            "source_eof_reached": status.get("source_eof_reached"),
            "process_done": status.get("process_done"),
            "detector_drain_completed": status.get(
                "detector_drain_completed"
            ),
            "detector_drain_timed_out": status.get(
                "detector_drain_timed_out"
            ),
            "evidence_drain_completed": status.get(
                "evidence_drain_completed"
            ),
            "evidence_drain_failed": status.get(
                "evidence_drain_failed"
            ),
            "evidence_writer_pending": status.get(
                "evidence_writer_pending"
            ),
            "evidence_writer_failed": status.get(
                "evidence_writer_failed"
            ),
            "evidence_writer_queue_full": status.get(
                "evidence_writer_queue_full"
            ),
            "evidence_writer_last_error": status.get(
                "evidence_writer_last_error"
            ),
        }
        completion_barrier_ok = bool(
            completion_barrier["source_eof_reached"] is True
            and completion_barrier["process_done"] is True
            and completion_barrier["detector_drain_completed"] is True
            and completion_barrier["detector_drain_timed_out"] is False
            and completion_barrier["evidence_drain_completed"] is True
            and completion_barrier["evidence_drain_failed"] is False
            and completion_barrier["evidence_writer_pending"] is not None
            and int(completion_barrier["evidence_writer_pending"] or 0) == 0
            and completion_barrier["evidence_writer_failed"] is not None
            and int(completion_barrier["evidence_writer_failed"] or 0) == 0
            and completion_barrier["evidence_writer_queue_full"] is not None
            and int(completion_barrier["evidence_writer_queue_full"] or 0)
            == 0
            and completion_barrier["evidence_writer_last_error"] is not None
            and not str(
                completion_barrier["evidence_writer_last_error"] or ""
            ).strip()
        )
        _record_check(
            checks,
            blockers,
            name="source_completion_barrier",
            passed=completion_barrier_ok,
            message=(
                "source_ended=true requires decoder EOF, detector completion "
                "and successful evidence drain"
            ),
            actual=completion_barrier,
            expected={
                "source_eof_reached": True,
                "process_done": True,
                "detector_drain_completed": True,
                "detector_drain_timed_out": False,
                "evidence_drain_completed": True,
                "evidence_drain_failed": False,
                "evidence_writer_pending": 0,
                "evidence_writer_failed": 0,
                "evidence_writer_queue_full": 0,
                "evidence_writer_last_error": "",
            },
        )


def _run_one_asset(
    client: JsonClient,
    asset: AuthoritativeAsset,
    *,
    manifest: AuthoritativeManifest,
    profile: str,
    ready_timeout_s: float,
    asset_timeout_s: float,
    poll_interval_s: float,
    evidence_limit: int,
) -> dict[str, Any]:
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    api_errors: list[dict[str, Any]] = []
    lineage_errors: list[dict[str, Any]] = []
    source_identity_before = _asset_source_identity(asset)
    if source_identity_before.get("matches_manifest") is not True:
        lineage_errors.append(
            {
                "stage": "source_identity_before",
                "message": "source path/size/hash did not match manifest before start",
                "identity": source_identity_before,
            }
        )
    try:
        initial_evidence = _module_a_evidence_events(
            _read_evidence_events(client, evidence_limit)
        )
    except Exception as exc:
        initial_evidence = []
        api_errors.append(
            {"stage": "evidence_before", "message": str(exc)}
        )
    initial_evidence_ids = _evidence_ids(initial_evidence)
    run_id: int | None = None
    completion_reason = "start_failed"
    timed_out = False
    source_ended = False
    status_samples: list[dict[str, Any]] = []
    overlay_records: list[dict[str, Any]] = []
    latest_overlay_seq = 0
    start_payload: dict[str, Any] = {}
    final_status: dict[str, Any] = {}
    observed_source_epochs: set[int] = set()

    try:
        start_response = _request_json(
            client,
            "post",
            "/api/runs/start",
            json={
                "source_type": "file",
                "source": asset.canonical_path,
                "profile": profile,
                "realtime": True,
                "ready_timeout_s": ready_timeout_s,
            },
        )
        start_payload = (
            dict(start_response["payload"])
            if isinstance(start_response["payload"], dict)
            else {"raw": start_response["payload"]}
        )
        if (
            start_response["status_code"] >= 400
            or start_payload.get("ok") is not True
        ):
            api_errors.append(
                {
                    "stage": "start",
                    "status_code": start_response["status_code"],
                    "payload": start_payload,
                }
            )
        else:
            run_id_value = start_payload.get("run_id")
            if not _is_int(run_id_value):
                api_errors.append(
                    {
                        "stage": "start",
                        "message": "start response did not contain integer run_id",
                        "payload": start_payload,
                    }
                )
            else:
                run_id = int(run_id_value)
                if isinstance(start_payload.get("status"), dict):
                    start_status = dict(start_payload["status"])
                    status_samples.append(start_status)
                    lineage_errors.extend(
                        _runtime_status_lineage_errors(
                            start_status,
                            run_id=run_id,
                            asset=asset,
                            stage="start_status",
                        )
                    )
                    if _is_int(start_status.get("source_epoch")):
                        observed_source_epochs.add(
                            int(start_status["source_epoch"])
                        )
                deadline = time.monotonic() + max(0.01, asset_timeout_s)
                completion_reason = "running"
                while True:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        completion_reason = "timeout"
                        break
                    status_response = _request_json(client, "get", "/api/status")
                    payload = status_response["payload"]
                    if (
                        status_response["status_code"] >= 400
                        or not isinstance(payload, dict)
                        or payload.get("ok") is not True
                        or not isinstance(payload.get("status"), dict)
                    ):
                        api_errors.append(
                            {
                                "stage": "status",
                                "status_code": status_response["status_code"],
                                "payload": payload,
                            }
                        )
                        completion_reason = "status_contract_failed"
                        break
                    status = dict(payload["status"])
                    final_status = status
                    status_samples.append(status)
                    status_lineage_errors = _runtime_status_lineage_errors(
                        status,
                        run_id=run_id,
                        asset=asset,
                        stage="status",
                    )
                    lineage_errors.extend(status_lineage_errors)
                    if _is_int(status.get("source_epoch")):
                        observed_source_epochs.add(int(status["source_epoch"]))
                    if status_lineage_errors:
                        completion_reason = "runtime_lineage_failed"
                        break

                    overlay_response = _request_json(
                        client,
                        "get",
                        f"/api/runs/{run_id}/overlay",
                        params={"since_seq": latest_overlay_seq},
                    )
                    overlay_payload = overlay_response["payload"]
                    overlay = (
                        overlay_payload.get("overlay")
                        if isinstance(overlay_payload, dict)
                        else None
                    )
                    if (
                        overlay_response["status_code"] >= 400
                        or not isinstance(overlay_payload, dict)
                        or overlay_payload.get("ok") is not True
                        or not isinstance(overlay, dict)
                        or not isinstance(overlay.get("records"), list)
                    ):
                        api_errors.append(
                            {
                                "stage": "overlay",
                                "status_code": overlay_response["status_code"],
                                "payload": overlay_payload,
                            }
                        )
                        completion_reason = "overlay_contract_failed"
                        break
                    overlay_lineage_errors = _overlay_lineage_errors(
                        overlay,
                        run_id=run_id,
                        observed_source_epochs=observed_source_epochs,
                    )
                    lineage_errors.extend(overlay_lineage_errors)
                    if overlay_lineage_errors:
                        completion_reason = "overlay_lineage_failed"
                        break
                    overlay_records.extend(
                        dict(record)
                        for record in overlay["records"]
                        if isinstance(record, dict)
                    )
                    if _is_int(overlay.get("latest_seq")):
                        latest_overlay_seq = max(
                            latest_overlay_seq, int(overlay["latest_seq"])
                        )

                    source_ended = status.get("source_ended") is True
                    if source_ended:
                        completion_reason = "source_ended"
                        break
                    runtime_error = str(status.get("error") or "").strip()
                    if runtime_error:
                        completion_reason = "runtime_error"
                        api_errors.append(
                            {
                                "stage": "runtime",
                                "message": runtime_error,
                            }
                        )
                        break
                    if status.get("running") is not True:
                        completion_reason = "stopped_before_source_end"
                        break
                    time.sleep(max(0.0, poll_interval_s))
    except Exception as exc:
        completion_reason = "client_exception"
        api_errors.append({"stage": "client", "message": str(exc)})
    finally:
        if run_id is not None:
            try:
                stop_response = _request_json(
                    client,
                    "post",
                    f"/api/runs/{run_id}/stop",
                    json={},
                )
                if (
                    stop_response["status_code"] >= 400
                    or not isinstance(stop_response["payload"], dict)
                    or stop_response["payload"].get("ok") is not True
                ):
                    api_errors.append(
                        {
                            "stage": "stop",
                            "status_code": stop_response["status_code"],
                            "payload": stop_response["payload"],
                        }
                    )
            except Exception as exc:
                api_errors.append({"stage": "stop", "message": str(exc)})

    try:
        final_evidence = _module_a_evidence_events(
            _read_evidence_events(client, evidence_limit)
        )
    except Exception as exc:
        final_evidence = []
        api_errors.append({"stage": "evidence_after", "message": str(exc)})
    new_evidence_events = _new_evidence_events(
        final_evidence,
        initial_evidence_ids,
    )
    bound_evidence_events, unbound_evidence_events = _partition_evidence_events(
        new_evidence_events,
        run_id=run_id,
        asset=asset,
        observed_source_epochs=observed_source_epochs,
    )
    if unbound_evidence_events:
        lineage_errors.append(
            {
                "stage": "evidence_lineage",
                "message": (
                    "new Module A evidence was not bound to the current "
                    "run/source/source_epoch"
                ),
                "event_ids": sorted(_evidence_ids(unbound_evidence_events)),
            }
        )
    source_identity_after = _asset_source_identity(asset)
    if source_identity_after.get("matches_manifest") is not True:
        lineage_errors.append(
            {
                "stage": "source_identity_after",
                "message": "source path/size/hash did not match manifest after run",
                "identity": source_identity_after,
            }
        )
    observations = _summarize_observations(
        status_samples=status_samples,
        overlay_records=overlay_records,
        bound_evidence_events=bound_evidence_events,
        unbound_evidence_events=unbound_evidence_events,
    )
    runtime_contract_checks: dict[str, dict[str, Any]] = {}
    runtime_contract_blockers: list[dict[str, Any]] = []
    if final_status:
        _evaluate_runtime_status(
            final_status,
            manifest=manifest,
            checks=runtime_contract_checks,
            blockers=runtime_contract_blockers,
            verify_status_artifact_files=False,
        )
    else:
        runtime_contract_blockers.append(
            {
                "code": "runtime_status_missing",
                "message": "no final runtime status was available for contract checks",
            }
        )
    gates = _asset_gates(
        asset=asset,
        source_ended=source_ended,
        timed_out=timed_out,
        completion_reason=completion_reason,
        observations=observations,
        api_errors=api_errors,
        lineage_errors=lineage_errors,
        runtime_contract_blockers=runtime_contract_blockers,
    )
    blockers = [
        {
            "code": name,
            "message": gate["message"],
            "expected": gate.get("expected"),
            "actual": gate.get("actual"),
        }
        for name, gate in gates.items()
        if gate.get("passed") is not True
    ]
    return {
        "asset_id": asset.asset_id,
        "identity": asset.identity,
        "relative_path": asset.relative_path,
        "category": asset.category,
        "attack_type": asset.attack_type,
        "acceptance_order": asset.acceptance_order,
        "expectations": {
            "module_a_alert": asset.expected_module_a_alert,
            "a3b_trigger": asset.expected_a3b_trigger,
            "module_a_evidence_events": (
                asset.expected_module_a_evidence_events
            ),
        },
        "execution": {
            "profile": profile,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_time_s": max(0.0, time.monotonic() - started_monotonic),
            "completion_reason": completion_reason,
            "source_ended": source_ended,
            "timed_out": timed_out,
            "api_error_count": len(api_errors),
            "api_errors": api_errors,
            "start_response": start_payload,
        },
        "source_identity": {
            "expected": {
                "canonical_path": asset.canonical_path,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
            },
            "before": source_identity_before,
            "after": source_identity_after,
        },
        "lineage": {
            "passed": not lineage_errors,
            "run_id": run_id,
            "observed_source_epochs": sorted(observed_source_epochs),
            "errors": lineage_errors,
        },
        "observations": observations,
        "final_status": final_status,
        "overlay": {
            "record_count": len(overlay_records),
            "latest_seq": latest_overlay_seq,
        },
        "evidence": {
            "new_event_count": len(new_evidence_events),
            "new_event_ids": sorted(_evidence_ids(new_evidence_events)),
            "bound_event_count": len(bound_evidence_events),
            "bound_event_ids": sorted(_evidence_ids(bound_evidence_events)),
            "unbound_event_count": len(unbound_evidence_events),
            "unbound_event_ids": sorted(
                _evidence_ids(unbound_evidence_events)
            ),
        },
        "runtime_contract": {
            "passed": not runtime_contract_blockers,
            "checks": runtime_contract_checks,
            "blockers": runtime_contract_blockers,
        },
        "gates": gates,
        "passed": not blockers,
        "blockers": blockers,
    }


def _asset_gates(
    *,
    asset: AuthoritativeAsset,
    source_ended: bool,
    timed_out: bool,
    completion_reason: str,
    observations: Mapping[str, Any],
    api_errors: Sequence[Mapping[str, Any]],
    lineage_errors: Sequence[Mapping[str, Any]],
    runtime_contract_blockers: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    observed_physical_alert = (
        observations.get("physical_alert_confirmed_observed") is True
    )
    observed_module_a_alert = (
        observations.get("module_a_alert_confirmed_observed") is True
    )
    observed_a3b = observations.get("a3b_confirmed_observed") is True
    if asset.category == "physical":
        observed_alert = observed_physical_alert
        evidence_count = int(
            observations.get("physical_evidence_event_count", 0) or 0
        )
        alert_evidence_correlated = bool(
            observations.get("physical_alert_evidence_correlated")
        )
    elif asset.category == "a3b":
        observed_alert = observed_module_a_alert and observed_a3b
        evidence_count = int(
            observations.get("a3b_evidence_event_count", 0) or 0
        )
        alert_evidence_correlated = bool(
            observations.get("a3b_alert_evidence_correlated")
        )
    else:
        # Normal assets must remain negative in every Module A branch.
        observed_alert = observed_module_a_alert
        evidence_count = int(
            observations.get("module_a_evidence_event_count", 0) or 0
        )
        alert_evidence_correlated = bool(
            observations.get("alert_evidence_correlated")
        )
    expected_evidence = asset.expected_module_a_evidence_events
    evidence_ok = (
        evidence_count == 0
        if expected_evidence == 0
        else evidence_count >= 1
        if expected_evidence == ">=1"
        else False
    )
    performance = (
        observations.get("performance", {})
        if isinstance(observations.get("performance"), Mapping)
        else {}
    )
    decoder = (
        performance.get("decoder", {})
        if isinstance(performance.get("decoder"), Mapping)
        else {}
    )
    native = (
        performance.get("native", {})
        if isinstance(performance.get("native"), Mapping)
        else {}
    )
    detector_fps = _number_or_none(performance.get("detector_fps"))
    detector_compute_fps = _number_or_none(
        performance.get("detector_compute_fps")
    )
    detector_throughput_fps = (
        detector_compute_fps
        if detector_compute_fps is not None
        and detector_compute_fps > 0.0
        else detector_fps
    )
    coverage = _number_or_none(
        performance.get("detection_source_coverage_ratio")
    )
    decoder_backend = str(
        decoder.get("effective_backend")
        or decoder.get("backend")
        or ""
    ).strip().lower()
    decoder_fallback_count = int(
        _number_or_none(decoder.get("fallback_count")) or 0
    )
    decoder_fallback_reason = str(
        decoder.get("fallback_reason") or ""
    ).strip().lower()
    decoder_no_fallback = (
        decoder_fallback_count == 0
        and decoder_fallback_reason in {"", "none"}
    )
    derived_cache_used = bool(decoder.get("derived_cache_used", False))
    derived_source_sha = _normalize_hash(decoder.get("source_sha256"))
    derived_decode_sha = _normalize_hash(
        decoder.get("decode_source_sha256")
    )
    derived_metadata_sha = _normalize_hash(
        decoder.get("derived_metadata_sha256")
    )
    derived_expected_frames = int(
        _number_or_none(decoder.get("derived_expected_frame_count")) or 0
    )
    derived_decoded_frames = int(
        _number_or_none(decoder.get("frames_decoded")) or 0
    )
    derived_provenance_ok = (
        not derived_cache_used
        or (
            str(decoder.get("derived_cache_validation") or "")
            .strip()
            .lower()
            == "verified"
            and derived_source_sha == _normalize_hash(asset.sha256)
            and bool(derived_decode_sha)
            and bool(decoder.get("derived_metadata_path"))
            and bool(derived_metadata_sha)
            and str(decoder.get("source_asset_id") or "") == asset.asset_id
            and str(decoder.get("source_role") or "") == asset.role
            and str(decoder.get("source_label") or "") == asset.label
            and (
                decoder.get("source_attack_type") == asset.attack_type
            )
            and bool(str(decoder.get("derived_profile_id") or "").strip())
            and bool(
                _normalize_hash(decoder.get("derived_profile_sha256"))
            )
            and str(decoder.get("transcode_encode_backend") or "")
            .strip()
            .lower()
            in {"h264_nvenc", "hevc_nvenc"}
            and bool(
                str(decoder.get("transcode_decode_backend") or "").strip()
            )
            and decoder.get("derived_frame_parity") is True
            and decoder.get("derived_frame_count_match") is True
            and decoder.get("derived_fps_match") is True
            and derived_expected_frames > 0
            and derived_decoded_frames == derived_expected_frames
            and decoder.get("eof") is True
        )
    )
    source_frames_skipped = int(
        _number_or_none(
            performance.get("source_frames_skipped_for_realtime")
        )
        or 0
    )
    native_hit_count = _mapping_int_total(native.get("hit_counts"))
    native_fallback_count = _mapping_int_total(
        native.get("fallback_counts")
    )
    native_binary_sha = _normalize_hash(native.get("binary_sha256"))
    native_ok = (
        native.get("available") is True
        and bool(native_binary_sha)
        and bool(native.get("enabled_stages"))
        and native_hit_count > 0
        and native_fallback_count == 0
        and str(native.get("fallback_reason") or "").strip().lower()
        in {"", "none"}
    )
    first_a3b_time = _number_or_none(
        observations.get("first_a3b_source_time_s")
    )
    a3b_timing_ok = (
        asset.category != "a3b"
        or (
            first_a3b_time is not None
            and A3B_FIRST_TRIGGER_MIN_S
            <= first_a3b_time
            <= A3B_FIRST_TRIGGER_MAX_S
        )
    )
    a3b_continuity_ok = (
        asset.category != "a3b"
        or int(observations.get("a3b_internal_false_count") or 0) == 0
    )
    alert_evidence_correlation_required = bool(
        asset.expected_module_a_alert
        and asset.expected_module_a_evidence_events == ">=1"
    )
    return {
        "source_ended": {
            "passed": source_ended is True,
            "message": "successful completion requires source_ended=true",
            "expected": True,
            "actual": source_ended,
        },
        "no_timeout": {
            "passed": timed_out is False,
            "message": "timeout is a failure, never success",
            "expected": False,
            "actual": timed_out,
        },
        "completion_reason": {
            "passed": completion_reason == "source_ended",
            "message": "completion_reason must be source_ended",
            "expected": "source_ended",
            "actual": completion_reason,
        },
        "api_contract": {
            "passed": not api_errors,
            "message": (
                "start/status/overlay/evidence/stop HTTP contract must not fail"
            ),
            "expected": 0,
            "actual": len(api_errors),
        },
        "runtime_lineage": {
            "passed": not lineage_errors,
            "message": (
                "status/overlay/evidence must remain bound to the current "
                "run_id, source path/hash and source_epoch"
            ),
            "expected": 0,
            "actual": len(lineage_errors),
        },
        "production_runtime_contract": {
            "passed": not runtime_contract_blockers,
            "message": (
                "loaded per-asset status must retain TensorRT/hash/latest-only/"
                "decoder/native production contracts"
            ),
            "expected": 0,
            "actual": len(runtime_contract_blockers),
        },
        "module_a_alert": {
            "passed": observed_alert is asset.expected_module_a_alert,
            "message": "observed Module A alert must match manifest expectation",
            "expected": asset.expected_module_a_alert,
            "actual": observed_alert,
        },
        "a3b_trigger": {
            "passed": observed_a3b is asset.expected_a3b_trigger,
            "message": "observed A3b trigger must match manifest expectation",
            "expected": asset.expected_a3b_trigger,
            "actual": observed_a3b,
        },
        "module_a_evidence_events": {
            "passed": evidence_ok,
            "message": (
                "observed Module A evidence count must satisfy manifest "
                "expectation"
            ),
            "expected": expected_evidence,
            "actual": evidence_count,
        },
        "detector_fps": {
            "passed": (
                detector_throughput_fps is not None
                and detector_throughput_fps >= MIN_DETECTOR_FPS
            ),
            "message": (
                "production Module A detector compute throughput must be "
                ">=25 FPS; source-paced completion FPS remains separately "
                "observable"
            ),
            "expected": f">={MIN_DETECTOR_FPS}",
            "actual": detector_throughput_fps,
        },
        "detection_source_coverage": {
            "passed": (
                coverage is not None
                and coverage >= MIN_DETECTION_SOURCE_COVERAGE
            ),
            "message": "processed/source detection coverage must be >=90%",
            "expected": f">={MIN_DETECTION_SOURCE_COVERAGE}",
            "actual": coverage,
        },
        "nvdec_effective_backend": {
            "passed": decoder_backend == "nvdec",
            "message": "file acceptance must actually use NVDEC",
            "expected": "nvdec",
            "actual": decoder_backend or None,
        },
        "decoder_no_fallback": {
            "passed": decoder_no_fallback,
            "message": "authoritative file acceptance forbids decoder fallback",
            "expected": {"fallback_count": 0, "fallback_reason": "none"},
            "actual": {
                "fallback_count": decoder_fallback_count,
                "fallback_reason": decoder_fallback_reason,
            },
        },
        "derived_video_provenance": {
            "passed": derived_provenance_ok,
            "message": (
                "a derived decode source, when used, must remain bound to the "
                "authoritative source SHA and verified lossless/NVENC metadata"
            ),
            "expected": (
                "not_used_or_verified_source_sha_bound_nvenc_derivative"
            ),
            "actual": {
                "used": derived_cache_used,
                "validation": decoder.get("derived_cache_validation"),
                "source_sha256": derived_source_sha,
                "decode_source_sha256": derived_decode_sha,
                "metadata_path": decoder.get("derived_metadata_path"),
                "metadata_sha256": derived_metadata_sha,
                "source_asset_id": decoder.get("source_asset_id"),
                "source_role": decoder.get("source_role"),
                "source_label": decoder.get("source_label"),
                "source_attack_type": decoder.get("source_attack_type"),
                "profile_id": decoder.get("derived_profile_id"),
                "profile_sha256": decoder.get(
                    "derived_profile_sha256"
                ),
                "transcode_decode_backend": decoder.get(
                    "transcode_decode_backend"
                ),
                "transcode_encode_backend": decoder.get(
                    "transcode_encode_backend"
                ),
                "frame_parity": decoder.get("derived_frame_parity"),
                "frame_count_match": decoder.get(
                    "derived_frame_count_match"
                ),
                "fps_match": decoder.get("derived_fps_match"),
                "expected_frame_count": derived_expected_frames,
                "decoded_frame_count": derived_decoded_frames,
                "eof": decoder.get("eof"),
            },
        },
        "capture_no_source_skip": {
            "passed": source_frames_skipped == 0,
            "message": (
                "capture layer must not discard source frames to catch wall clock"
            ),
            "expected": 0,
            "actual": source_frames_skipped,
        },
        "native_runtime_hit": {
            "passed": native_ok,
            "message": (
                "native binary must be available, hash-bound, hit in production "
                "and have zero fallback calls"
            ),
            "expected": {
                "available": True,
                "hit_count": ">0",
                "fallback_count": 0,
            },
            "actual": {
                "available": native.get("available"),
                "binary_sha256": native_binary_sha,
                "enabled_stages": list(native.get("enabled_stages") or []),
                "hit_count": native_hit_count,
                "fallback_count": native_fallback_count,
                "fallback_reason": native.get("fallback_reason"),
            },
        },
        "a3b_first_trigger_time": {
            "passed": a3b_timing_ok,
            "message": "A3b must first trigger near frame 30 / 1.0 second",
            "expected": (
                {
                    "min_s": A3B_FIRST_TRIGGER_MIN_S,
                    "max_s": A3B_FIRST_TRIGGER_MAX_S,
                }
                if asset.category == "a3b"
                else "not_applicable"
            ),
            "actual": first_a3b_time,
        },
        "a3b_continuity": {
            "passed": a3b_continuity_ok,
            "message": (
                "A3b confirmation must not flash off between true samples "
                "within an uninterrupted continuity segment"
            ),
            "expected": 0,
            "actual": int(
                observations.get("a3b_internal_false_count") or 0
            ),
        },
        "alert_event_evidence_correlation": {
            "passed": (
                alert_evidence_correlated
                if alert_evidence_correlation_required
                else True
            ),
            "message": (
                "red alert confirmation and Module A evidence must refer to "
                "the same source frame/time interval"
            ),
            "expected": (
                True if alert_evidence_correlation_required else "not_applicable"
            ),
            "actual": alert_evidence_correlated,
        },
    }


def _summarize_observations(
    *,
    status_samples: Sequence[Mapping[str, Any]],
    overlay_records: Sequence[Mapping[str, Any]],
    bound_evidence_events: Sequence[Mapping[str, Any]],
    unbound_evidence_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [dict(sample) for sample in status_samples] + [
        dict(record) for record in overlay_records
    ]
    physical_alert_samples = []
    for sample in samples:
        if "physical_alert_confirmed" in sample:
            physical_confirmed = (
                sample.get("physical_alert_confirmed") is True
            )
        else:
            # Compatibility for old captured reports that predate the
            # explicit physical/module-A channel split.
            physical_confirmed = sample.get("alert_confirmed") is True
        if physical_confirmed:
            physical_alert_samples.append(sample)
    a3b_trigger_samples = [
        sample for sample in samples if sample.get("a3b_triggered") is True
    ]
    a3b_confirmed_samples = [
        sample
        for sample in a3b_trigger_samples
        if sample.get("a3b_confirmed_alert") is True
        or str(sample.get("a3b_state") or "").strip().lower() == "confirmed"
    ]
    module_a_alert_samples = [
        sample
        for sample in samples
        if sample.get("module_a_alert_confirmed") is True
        or sample in physical_alert_samples
        or sample in a3b_confirmed_samples
    ]
    status_evidence_count = max(
        (
            int(sample.get("evidence_saved_event_count") or 0)
            for sample in status_samples
        ),
        default=0,
    )
    physical_evidence_events = [
        event
        for event in bound_evidence_events
        if str(event.get("channel") or "").strip().lower() == "module_a"
    ]
    a3b_evidence_events = [
        event
        for event in bound_evidence_events
        if str(event.get("channel") or "").strip().lower() == "a3b"
    ]
    evidence_count = len(physical_evidence_events) + len(a3b_evidence_events)
    final_status = dict(status_samples[-1]) if status_samples else {}
    ordered_overlay = sorted(
        (dict(record) for record in overlay_records),
        key=lambda record: (
            _number_or_none(record.get("source_time_s")) or 0.0,
            _int_or_none(record.get("frame_idx")) or 0,
            _int_or_none(record.get("seq")) or 0,
        ),
    )
    (
        a3b_internal_false_count,
        a3b_continuity_break_count,
        a3b_continuity_segment_count,
    ) = _summarize_a3b_continuity(
        record
        for record in ordered_overlay
        if "a3b_triggered" in record
    )
    alert_evidence_correlated = any(
        _event_correlates_with_sample(event, sample)
        for event in bound_evidence_events
        for sample in module_a_alert_samples
    )
    physical_alert_evidence_correlated = any(
        _event_correlates_with_sample(event, sample)
        for event in physical_evidence_events
        for sample in physical_alert_samples
    )
    a3b_alert_evidence_correlated = any(
        _event_correlates_with_sample(event, sample)
        for event in a3b_evidence_events
        for sample in a3b_confirmed_samples
    )
    max_p_adv = max(
        (
            float(sample.get("p_adv") or sample.get("p_adv_display") or 0.0)
            for sample in samples
        ),
        default=0.0,
    )
    max_a3b = max(
        (
            float(
                sample.get("a3b_score")
                or sample.get("a3b_display_score")
                or 0.0
            )
            for sample in samples
        ),
        default=0.0,
    )
    processing_samples = _numeric_series(ordered_overlay, "processing_ms")
    module_a_samples = _numeric_series(
        ordered_overlay,
        "module_a_timing_ms",
    )
    inference_samples = _numeric_series(
        ordered_overlay,
        "detector_inference_ms",
    )
    detector_cycle_samples = _numeric_series(
        ordered_overlay,
        "detector_cycle_ms",
    )
    evidence_update_samples = _numeric_series(
        ordered_overlay,
        "evidence_update_ms",
    )
    overlay_publish_samples = _numeric_series(
        ordered_overlay,
        "overlay_status_publish_ms",
    )
    a3b_trace = _compact_a3b_trace(ordered_overlay)
    return {
        # Keep the legacy keys for report readers while making the acceptance
        # semantics explicit and channel-safe.
        "alert_confirmed_observed": bool(module_a_alert_samples),
        "physical_alert_confirmed_observed": bool(physical_alert_samples),
        "module_a_alert_confirmed_observed": bool(module_a_alert_samples),
        "a3b_triggered_observed": bool(a3b_trigger_samples),
        "a3b_confirmed_observed": bool(a3b_confirmed_samples),
        "first_alert_source_time_s": _first_source_time(
            module_a_alert_samples
        ),
        "first_physical_alert_source_time_s": _first_source_time(
            physical_alert_samples
        ),
        "first_a3b_source_time_s": _first_source_time(
            a3b_confirmed_samples
        ),
        "module_a_evidence_event_count": evidence_count,
        "physical_evidence_event_count": len(physical_evidence_events),
        "a3b_evidence_event_count": len(a3b_evidence_events),
        "unbound_module_a_evidence_event_count": len(
            unbound_evidence_events
        ),
        "alert_evidence_correlated": alert_evidence_correlated,
        "physical_alert_evidence_correlated": (
            physical_alert_evidence_correlated
        ),
        "a3b_alert_evidence_correlated": a3b_alert_evidence_correlated,
        "a3b_internal_false_count": a3b_internal_false_count,
        "a3b_continuity_break_count": a3b_continuity_break_count,
        "a3b_continuity_segment_count": a3b_continuity_segment_count,
        "status_evidence_saved_event_count": status_evidence_count,
        "new_evidence_api_event_count": (
            len(bound_evidence_events) + len(unbound_evidence_events)
        ),
        "overlay_record_count": len(overlay_records),
        "status_sample_count": len(status_samples),
        "max_p_adv": max_p_adv,
        "max_a3b_score": max_a3b,
        "a3b_transition_trace": a3b_trace,
        "performance": {
            "detector_fps": _number_or_none(final_status.get("fps")),
            "detector_compute_fps": _number_or_none(
                final_status.get("detector_compute_fps")
            ),
            "preview_fps": _number_or_none(final_status.get("preview_fps")),
            "processed_detection_frames": _int_or_none(
                final_status.get("processed_detection_frames")
            ),
            "capture_frames_published": _int_or_none(
                final_status.get("capture_frames_published")
            ),
            "detection_source_coverage_ratio": _number_or_none(
                final_status.get("detection_source_coverage_ratio")
            ),
            "source_frames_skipped_for_realtime": _int_or_none(
                final_status.get("source_frames_skipped_for_realtime")
            ),
            "processing_ms": _number_or_none(final_status.get("processing_ms")),
            "detector_inference_ms": _number_or_none(
                final_status.get("detector_inference_ms")
            ),
            "module_a_timing_ms": _number_or_none(
                final_status.get("module_a_timing_ms")
            ),
            "detector_cycle_ms": _number_or_none(
                final_status.get("detector_cycle_ms")
            ),
            "evidence_update_ms": _number_or_none(
                final_status.get("evidence_update_ms")
            ),
            "overlay_status_publish_ms": _number_or_none(
                final_status.get("overlay_status_publish_ms")
            ),
            "processing_ms_distribution": _numeric_distribution(
                processing_samples
            ),
            "module_a_timing_ms_distribution": _numeric_distribution(
                module_a_samples
            ),
            "detector_inference_ms_distribution": _numeric_distribution(
                inference_samples
            ),
            "detector_cycle_ms_distribution": _numeric_distribution(
                detector_cycle_samples
            ),
            "evidence_update_ms_distribution": _numeric_distribution(
                evidence_update_samples
            ),
            "overlay_status_publish_ms_distribution": (
                _numeric_distribution(overlay_publish_samples)
            ),
            "detector_drain": {
                "source_eof_reached": bool(
                    final_status.get("source_eof_reached")
                ),
                "process_done": bool(final_status.get("process_done")),
                "completed": bool(
                    final_status.get("detector_drain_completed")
                ),
                "timed_out": bool(
                    final_status.get("detector_drain_timed_out")
                ),
                "drain_ms": _number_or_none(
                    final_status.get("detector_drain_ms")
                ),
                "failed_reason": str(
                    final_status.get("detector_drain_failed_reason") or ""
                ),
            },
            "evidence_drain": {
                "completed": bool(
                    final_status.get("evidence_drain_completed")
                ),
                "failed": bool(
                    final_status.get("evidence_drain_failed")
                ),
                "drain_ms": _number_or_none(
                    final_status.get("evidence_drain_ms")
                ),
                "error": str(
                    final_status.get("evidence_drain_error") or ""
                ),
            },
            "evidence_writer": {
                "enabled": bool(
                    final_status.get("evidence_writer_enabled")
                ),
                "alive": bool(
                    final_status.get("evidence_writer_alive")
                ),
                "queue_capacity": int(
                    final_status.get(
                        "evidence_writer_queue_capacity"
                    )
                    or 0
                ),
                "pending": int(
                    final_status.get("evidence_writer_pending") or 0
                ),
                "completed": int(
                    final_status.get("evidence_writer_completed") or 0
                ),
                "failed": int(
                    final_status.get("evidence_writer_failed") or 0
                ),
                "queue_full": int(
                    final_status.get("evidence_writer_queue_full") or 0
                ),
                "drain_ms": _number_or_none(
                    final_status.get("evidence_writer_drain_ms")
                ),
                "last_error": str(
                    final_status.get("evidence_writer_last_error") or ""
                ),
            },
            "decoder": _status_group_snapshot(final_status, "decoder"),
            "native": _status_group_snapshot(final_status, "native"),
        },
    }


def _numeric_series(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _number_or_none(record.get(key))
        if value is not None and value >= 0.0:
            values.append(float(value))
    return values


def _numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "sample_count": 0,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    def percentile(ratio: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {
        "sample_count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _summarize_a3b_continuity(
    records: Iterable[Mapping[str, Any]],
) -> tuple[int, int, int]:
    segments: list[list[bool]] = []
    current_segment: list[bool] = []
    break_count = 0
    in_hard_break = False

    for record in records:
        debug = (
            record.get("a3b_debug")
            if isinstance(record.get("a3b_debug"), Mapping)
            else {}
        )
        failed_gates = debug.get("failed_gates")
        hard_break = bool(
            isinstance(failed_gates, (list, tuple))
            and _A3B_CONTINUITY_HARD_SUPPRESSION_GATES.intersection(
                str(gate).strip().lower() for gate in failed_gates
            )
        )
        if hard_break:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            if not in_hard_break:
                break_count += 1
            in_hard_break = True
            continue

        in_hard_break = False
        current_segment.append(
            bool(
                record.get("a3b_confirmed_alert")
                or (
                    record.get("a3b_triggered")
                    and str(record.get("a3b_state") or "").strip().lower()
                    == "confirmed"
                )
            )
        )

    if current_segment:
        segments.append(current_segment)

    continuity_segments = [segment for segment in segments if any(segment)]
    internal_false_count = 0
    for segment in continuity_segments:
        true_indices = [
            index for index, confirmed in enumerate(segment) if confirmed
        ]
        if len(true_indices) >= 2:
            internal_false_count += sum(
                not confirmed
                for confirmed in segment[
                    true_indices[0] : true_indices[-1] + 1
                ]
            )

    return internal_false_count, break_count, len(continuity_segments)


def _compact_a3b_trace(
    ordered_overlay: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    previous_signature: tuple[Any, ...] | None = None
    for record in ordered_overlay:
        debug = (
            record.get("a3b_debug")
            if isinstance(record.get("a3b_debug"), Mapping)
            else {}
        )
        failed_gates = tuple(
            str(item)
            for item in (
                debug.get("failed_gates")
                if isinstance(debug.get("failed_gates"), list)
                else []
            )
        )
        result_seq = _int_or_none(record.get("a3b_result_seq"))
        signature = (
            "result",
            result_seq,
        ) if result_seq and result_seq > 0 else (
            "state",
            bool(record.get("a3b_confirmed_alert")),
            bool(record.get("a3b_triggered")),
            str(record.get("a3b_state") or "normal"),
            str(record.get("a3b_triggered_source") or "none"),
            str(record.get("a3b_reason") or ""),
            failed_gates,
        )
        if signature == previous_signature:
            continue
        previous_signature = signature
        trace.append(
            {
                "frame_idx": _int_or_none(record.get("frame_idx")),
                "module_a_processed_frame_idx": _int_or_none(
                    record.get("module_a_processed_frame_idx")
                ),
                "module_a_source_frame_idx": _int_or_none(
                    record.get("module_a_source_frame_idx")
                ),
                "module_a_input_frame_idx": _int_or_none(
                    record.get("module_a_input_frame_idx")
                ),
                "source_time_s": _number_or_none(
                    record.get("source_time_s")
                ),
                "a3b_result_seq": _int_or_none(
                    record.get("a3b_result_seq")
                ),
                "a3b_source_frame_idx": _int_or_none(
                    record.get("a3b_source_frame_idx")
                ),
                "a3b_source_timestamp": _number_or_none(
                    record.get("a3b_source_timestamp")
                ),
                "a3b_source_fps": _number_or_none(
                    record.get("a3b_source_fps")
                ),
                "a3b_source_interval_frames": _int_or_none(
                    record.get("a3b_source_interval_frames")
                ),
                "a3b_result_fresh": bool(
                    record.get("a3b_result_fresh", False)
                ),
                "confirmed": bool(record.get("a3b_confirmed_alert")),
                "triggered": bool(record.get("a3b_triggered")),
                "state": str(record.get("a3b_state") or "normal"),
                "source": str(
                    record.get("a3b_triggered_source") or "none"
                ),
                "reason": str(record.get("a3b_reason") or ""),
                "failed_gates": list(failed_gates),
                "candidate_score": _number_or_none(
                    debug.get("rebuilt_candidate_score")
                ),
                "edge_score": _number_or_none(
                    debug.get("rebuilt_edge_score")
                ),
                "border_contrast": _number_or_none(
                    debug.get("rebuilt_border_contrast")
                ),
                "media_source_frame_units": _int_or_none(
                    debug.get("media_source_frame_units")
                ),
                "quality_window_result_hits": _int_or_none(
                    debug.get("quality_window_result_hits")
                ),
                "observed_only_window_result_hits": _int_or_none(
                    debug.get("observed_only_window_result_hits")
                ),
                "stable_result_hits": _int_or_none(
                    debug.get("stable_result_hits")
                ),
                "media_tighten_aspect_ratio": _number_or_none(
                    debug.get("media_tighten_aspect_ratio")
                ),
                "media_tighten_aspect_pass": debug.get(
                    "media_tighten_aspect_pass"
                ),
                "processing_ms": _number_or_none(
                    record.get("processing_ms")
                ),
                "module_a_timing_ms": _number_or_none(
                    record.get("module_a_timing_ms")
                ),
                "detector_cycle_ms": _number_or_none(
                    record.get("detector_cycle_ms")
                ),
                "evidence_update_ms": _number_or_none(
                    record.get("evidence_update_ms")
                ),
                "overlay_status_publish_ms": _number_or_none(
                    record.get("overlay_status_publish_ms")
                ),
            }
        )
    return trace


def _select_videos(
    manifest: AuthoritativeManifest,
    selected_asset_ids: Sequence[str] | None,
) -> list[AuthoritativeAsset]:
    if selected_asset_ids is None:
        return list(manifest.ordered_videos)
    selected: list[AuthoritativeAsset] = []
    seen: set[str] = set()
    for asset_id in selected_asset_ids:
        if asset_id in seen:
            raise ValueError(f"duplicate selected asset_id: {asset_id}")
        seen.add(asset_id)
        asset = manifest.asset_by_id(asset_id)
        if asset.category == "model":
            raise ValueError(f"model asset cannot be run as video: {asset_id}")
        selected.append(asset)
    return sorted(
        selected,
        key=lambda asset: (
            asset.acceptance_order
            if asset.acceptance_order is not None
            else 1_000_000,
            asset.asset_id,
        ),
    )


def _request_json(
    client: JsonClient,
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    json: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    call = getattr(client, method)
    kwargs: dict[str, Any] = {}
    if params:
        kwargs["params"] = dict(params)
    if json is not None:
        kwargs["json"] = dict(json)
    try:
        response = call(path, **kwargs)
    except TypeError:
        if params:
            query = urlencode(params)
            path = f"{path}?{query}"
        kwargs.pop("params", None)
        response = call(path, **kwargs)

    if isinstance(response, Mapping):
        return {"status_code": 200, "payload": dict(response)}
    status_code = int(getattr(response, "status_code", 200))
    json_method = getattr(response, "json", None)
    if callable(json_method):
        payload = json_method()
    elif isinstance(getattr(response, "data", None), Mapping):
        payload = dict(response.data)
    else:
        text = getattr(response, "text", "")
        payload = json_module_loads(text)
    return {"status_code": status_code, "payload": payload}


def json_module_loads(text: Any) -> Any:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise TypeError("HTTP response does not expose JSON content")
    return json.loads(text)


def _read_evidence_events(client: JsonClient, limit: int) -> list[Any]:
    response = _request_json(
        client,
        "get",
        "/api/evidence/events",
        params={"limit": max(1, int(limit))},
    )
    payload = response["payload"]
    if (
        response["status_code"] >= 400
        or not isinstance(payload, dict)
        or payload.get("ok") is not True
    ):
        raise RuntimeError(
            "evidence endpoint returned a non-success contract: "
            f"status={response['status_code']} payload={payload!r}"
        )
    evidence = _extract_evidence_events(payload)
    if not isinstance(evidence, list):
        raise RuntimeError(
            "evidence endpoint missing evidence array/evidence.events array"
        )
    return list(evidence)


def _extract_evidence_events(payload: Any) -> list[Any] | None:
    if not isinstance(payload, Mapping):
        return None
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        return evidence
    if isinstance(evidence, Mapping) and isinstance(evidence.get("events"), list):
        return list(evidence["events"])
    return None


def _module_a_evidence_events(events: Iterable[Any]) -> list[Any]:
    selected: list[Any] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        channel = str(event.get("channel") or "").strip().lower()
        if channel in {"module_a", "a3b"}:
            selected.append(event)
    return selected


def _evidence_identity(event: Any, *, fallback_index: int = 0) -> str:
    if isinstance(event, Mapping):
        identity = _first_nonempty(
            event.get("event_key"),
            event.get("evidence_event_key"),
            event.get("id"),
            event.get("manifest_path"),
            event.get("path"),
        )
        if identity:
            return str(identity)
        return hashlib.sha256(
            json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
    return f"{fallback_index}:{event!r}"


def _evidence_ids(events: Iterable[Any]) -> set[str]:
    return {
        _evidence_identity(event, fallback_index=index)
        for index, event in enumerate(events)
    }


def _new_evidence_events(
    events: Sequence[Any],
    initial_ids: set[str],
) -> list[Mapping[str, Any]]:
    return [
        dict(event)
        for index, event in enumerate(events)
        if isinstance(event, Mapping)
        and _evidence_identity(event, fallback_index=index) not in initial_ids
    ]


def _asset_source_identity(asset: AuthoritativeAsset) -> dict[str, Any]:
    path = Path(asset.canonical_path).resolve(strict=False)
    result: dict[str, Any] = {
        "canonical_path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
        "matches_manifest": False,
    }
    if not path.is_file():
        return result
    try:
        result["size_bytes"] = int(path.stat().st_size)
        result["sha256"] = sha256_file(path)
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result
    result["matches_manifest"] = bool(
        _path_key(path) == _path_key(Path(asset.canonical_path))
        and int(result["size_bytes"]) == int(asset.size_bytes)
        and _normalize_hash(result["sha256"])
        == _normalize_hash(asset.sha256)
    )
    return result


def _runtime_status_lineage_errors(
    status: Mapping[str, Any],
    *,
    run_id: int,
    asset: AuthoritativeAsset,
    stage: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    status_run_id = status.get("run_id")
    if not _is_int(status_run_id) or int(status_run_id) != int(run_id):
        errors.append(
            {
                "stage": stage,
                "message": "status run_id does not match started run",
                "expected": run_id,
                "actual": status_run_id,
            }
        )
    source_type = str(status.get("source_type") or "").strip().lower()
    if source_type != "file":
        errors.append(
            {
                "stage": stage,
                "message": "status source_type must remain file",
                "expected": "file",
                "actual": source_type,
            }
        )
    source = status.get("source")
    source_matches = bool(
        source
        and _path_key(Path(str(source)))
        == _path_key(Path(asset.canonical_path))
    )
    if not source_matches:
        errors.append(
            {
                "stage": stage,
                "message": "status source path does not match manifest asset",
                "expected": asset.canonical_path,
                "actual": source,
            }
        )
    return errors


def _overlay_lineage_errors(
    overlay: Mapping[str, Any],
    *,
    run_id: int,
    observed_source_epochs: set[int],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    overlay_run_id = overlay.get("run_id")
    if not _is_int(overlay_run_id) or int(overlay_run_id) != int(run_id):
        errors.append(
            {
                "stage": "overlay",
                "message": "overlay payload run_id does not match started run",
                "expected": run_id,
                "actual": overlay_run_id,
            }
        )
    records = overlay.get("records")
    if not isinstance(records, list):
        return errors
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        record_run_id = record.get("run_id")
        if not _is_int(record_run_id) or int(record_run_id) != int(run_id):
            errors.append(
                {
                    "stage": "overlay",
                    "message": "overlay record run_id mismatch",
                    "record_index": index,
                    "expected": run_id,
                    "actual": record_run_id,
                }
            )
        source_epoch = record.get("source_epoch")
        if (
            not _is_int(source_epoch)
            or int(source_epoch) not in observed_source_epochs
        ):
            errors.append(
                {
                    "stage": "overlay",
                    "message": "overlay record source_epoch was not observed",
                    "record_index": index,
                    "expected": sorted(observed_source_epochs),
                    "actual": source_epoch,
                }
            )
    return errors


def _partition_evidence_events(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: int | None,
    asset: AuthoritativeAsset,
    observed_source_epochs: set[int],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    bound: list[Mapping[str, Any]] = []
    unbound: list[Mapping[str, Any]] = []
    for event in events:
        event_run_id = event.get("run_id")
        event_epoch = event.get("source_epoch")
        event_source = event.get("source")
        event_source_type = str(
            event.get("source_type") or ""
        ).strip().lower()
        matches = bool(
            run_id is not None
            and _is_int(event_run_id)
            and int(event_run_id) == int(run_id)
            and _is_int(event_epoch)
            and int(event_epoch) in observed_source_epochs
            and event_source_type == "file"
            and event_source
            and _path_key(Path(str(event_source)))
            == _path_key(Path(asset.canonical_path))
            and not event.get("lineage_conflict")
        )
        (bound if matches else unbound).append(dict(event))
    return bound, unbound


def _event_correlates_with_sample(
    event: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> bool:
    sample_frame = _int_or_none(sample.get("frame_idx"))
    event_frame_start = _int_or_none(event.get("source_frame_start"))
    event_frame_end = _int_or_none(event.get("source_frame_end"))
    if (
        sample_frame is not None
        and event_frame_start is not None
        and event_frame_end is not None
        and event_frame_start - 1 <= sample_frame <= event_frame_end + 1
    ):
        return True
    sample_time = _number_or_none(
        _first_nonempty(
            sample.get("source_time_s"),
            sample.get("video_time_s"),
        )
    )
    event_time_start = _number_or_none(event.get("source_time_start_s"))
    event_time_end = _number_or_none(event.get("source_time_end_s"))
    return bool(
        sample_time is not None
        and event_time_start is not None
        and event_time_end is not None
        and event_time_start - 0.25
        <= sample_time
        <= event_time_end + 0.25
    )


def _record_check(
    checks: dict[str, dict[str, Any]],
    blockers: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    message: str,
    **details: Any,
) -> None:
    check = {"passed": bool(passed), "message": message, **details}
    checks[name] = check
    if not passed:
        blockers.append({"code": name, "message": message, **details})


def _required_status_fields(
    status: Mapping[str, Any],
    candidates: Mapping[str, Sequence[Sequence[str]]],
) -> dict[str, Any]:
    present: dict[str, str] = {}
    missing: list[str] = []
    for logical_name, paths in candidates.items():
        hit: tuple[str, ...] | None = None
        for path in paths:
            exists, _value = _path_exists(status, path)
            if exists:
                hit = tuple(path)
                break
        if hit is None:
            missing.append(logical_name)
        else:
            present[logical_name] = ".".join(hit)
    return {"present": present, "missing": missing}


def _path_exists(
    mapping: Mapping[str, Any],
    path: Sequence[str],
) -> tuple[bool, Any]:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _nested_value(node: Any, *path: str) -> Any:
    if not isinstance(node, Mapping):
        return None
    exists, value = _path_exists(node, path)
    return value if exists else None


def _node_value(node: Any, *keys: str) -> Any:
    if not isinstance(node, Mapping):
        return None
    return _first_nonempty(*(node.get(key) for key in keys))


def _flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], Any]]:
    rows: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        path = (*prefix, str(key))
        rows.append((path, value))
        if isinstance(value, Mapping):
            rows.extend(_flatten_mapping(value, path))
    return rows


def _status_group_snapshot(status: Mapping[str, Any], token: str) -> dict[str, Any]:
    direct = status.get(token)
    if isinstance(direct, Mapping):
        return dict(direct)
    return {
        ".".join(path): value
        for path, value in _flatten_mapping(status)
        if token.lower() in ".".join(path).lower()
    }


def _first_source_time(samples: Sequence[Mapping[str, Any]]) -> float | None:
    for sample in samples:
        value = _first_nonempty(
            sample.get("source_time_s"),
            sample.get("video_time_s"),
            sample.get("timestamp"),
        )
        converted = _number_or_none(value)
        if converted is not None:
            return converted
    return None


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if text else None


def _is_tensorrt(value: Any) -> bool:
    token = str(value or "").strip().lower().replace("_", "")
    return token in {"tensorrt", "trt", "tensorrtfp16"} or "tensorrt" in token


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_or_none(value: Any) -> int | None:
    if _is_int(value):
        return int(value)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mapping_int_total(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    total = 0
    for item in value.values():
        try:
            total += int(item)
        except (TypeError, ValueError):
            continue
    return total


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _contract_error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    **details: Any,
) -> None:
    errors.append({"code": code, "message": message, **details})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "JsonClient",
    "REPORT_TYPE",
    "WEB_ACCEPTANCE_SCHEMA_VERSION",
    "WebAcceptanceContractError",
    "aggregate_web_acceptance_report",
    "assert_web_acceptance_report",
    "build_web_acceptance_report",
    "run_authoritative_web_acceptance",
    "run_web_preflight",
    "validate_web_acceptance_report",
]

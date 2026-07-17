from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from defense.diagnostics.authoritative_manifest import (
    AuthoritativeAsset,
    AuthoritativeManifest,
)
from defense.diagnostics.web_acceptance_report import (
    _summarize_observations,
    aggregate_web_acceptance_report,
    build_web_acceptance_report,
    run_authoritative_web_acceptance,
    run_web_preflight,
    validate_web_acceptance_report,
)


@dataclass
class FakeResponse:
    status_code: int
    payload: Any

    def json(self) -> Any:
        return self.payload


class StaticFakeClient:
    def __init__(self, *, status: dict[str, Any]) -> None:
        self.status = status

    def get(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path.startswith("/api/status"):
            return FakeResponse(200, {"ok": True, "status": self.status})
        if path.startswith("/api/overlay"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "overlay": {"records": [], "latest_seq": 0, "run_id": 0},
                },
            )
        if path.startswith("/api/evidence/events"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "evidence": {
                        "count": 0,
                        "events": [],
                        "database": "fake.sqlite3",
                    },
                },
            )
        raise AssertionError(path)

    def post(self, path: str, **_kwargs: Any) -> FakeResponse:
        raise AssertionError(path)


class NeverEndingFakeClient(StaticFakeClient):
    def __init__(self, *, status: dict[str, Any]) -> None:
        super().__init__(status=status)
        self.run_id = 12

    def get(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path.startswith(f"/api/runs/{self.run_id}/overlay"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "overlay": {
                        "run_id": self.run_id,
                        "records": [],
                        "latest_seq": 0,
                    },
                },
            )
        if path.startswith("/api/status"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": True,
                        "source_ended": False,
                    },
                },
            )
        return super().get(path, **_kwargs)

    def post(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path == "/api/runs/start":
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "run_id": self.run_id,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": True,
                        "source_ended": False,
                    },
                },
            )
        if path == f"/api/runs/{self.run_id}/stop":
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": False,
                        "source_ended": False,
                    },
                },
            )
        raise AssertionError(path)


class CompletingFakeClient(StaticFakeClient):
    def __init__(self, *, status: dict[str, Any]) -> None:
        super().__init__(status=status)
        self.run_id = 21
        self.started = False

    def get(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path.startswith(f"/api/runs/{self.run_id}/overlay"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "overlay": {
                        "run_id": self.run_id,
                        "records": [
                            {
                                "seq": 1,
                                "run_id": self.run_id,
                                "source_epoch": 1,
                                "frame_idx": 30,
                                "source_time_s": 1.0,
                                "alert_confirmed": False,
                                "physical_alert_confirmed": False,
                                "module_a_alert_confirmed": True,
                                "attack_detected": False,
                                "attack_state_active": False,
                                "a3b_triggered": True,
                                "a3b_state": "confirmed",
                            }
                        ],
                        "latest_seq": 1,
                    },
                },
            )
        if path.startswith("/api/status") and self.started:
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": False,
                        "source_ended": True,
                        "source_time_s": 1.0,
                        "alert_confirmed": False,
                        "physical_alert_confirmed": False,
                        "module_a_alert_confirmed": True,
                        "attack_detected": False,
                        "attack_state_active": False,
                        "a3b_triggered": True,
                        "a3b_state": "confirmed",
                    },
                },
            )
        if path.startswith("/api/evidence/events"):
            events = (
                [
                    {
                        "channel": "a3b",
                        "evidence_event_key": "a3b-event-1",
                        "run_id": self.run_id,
                        "source_epoch": 1,
                        "source_type": "file",
                        "source": self.status["source"],
                        "source_frame_start": 29,
                        "source_frame_end": 31,
                        "source_time_start_s": 0.95,
                        "source_time_end_s": 1.05,
                    },
                    {
                        "channel": "ppe",
                        "evidence_event_key": "ppe-event-ignored",
                    },
                ]
                if self.started
                else []
            )
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "evidence": {
                        "count": len(events),
                        "events": events,
                        "database": "fake.sqlite3",
                    },
                },
            )
        return super().get(path, **_kwargs)

    def post(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path == "/api/runs/start":
            self.started = True
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "run_id": self.run_id,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": True,
                        "source_ended": False,
                    },
                },
            )
        if path == f"/api/runs/{self.run_id}/stop":
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": False,
                        "source_ended": True,
                    },
                },
            )
        raise AssertionError(path)


class WrongEvidenceLineageClient(CompletingFakeClient):
    def get(self, path: str, **kwargs: Any) -> FakeResponse:
        response = super().get(path, **kwargs)
        if path.startswith("/api/evidence/events") and self.started:
            events = response.payload["evidence"]["events"]
            for event in events:
                if event.get("channel") in {"module_a", "a3b"}:
                    event["run_id"] = self.run_id + 100
        return response


class ScenarioCompletingFakeClient(StaticFakeClient):
    def __init__(
        self,
        *,
        status: dict[str, Any],
        top_alert: bool,
        physical_confirmed: bool,
        a3b_triggered: bool,
        a3b_state: str,
        evidence_channels: tuple[str, ...],
    ) -> None:
        super().__init__(status=status)
        self.run_id = 31
        self.started = False
        self.evidence_channels = evidence_channels
        self.final_status = {
            **status,
            "run_id": self.run_id,
            "running": False,
            "source_ended": True,
            "source_time_s": 1.0,
            "frame_idx": 30,
            "alert_confirmed": physical_confirmed,
            "physical_alert_confirmed": physical_confirmed,
            "module_a_alert_confirmed": top_alert,
            "module_a_fresh_confirmed": physical_confirmed,
            "attack_detected": physical_confirmed,
            "attack_state_active": physical_confirmed,
            "module_a_primary_channel": (
                "adv" if physical_confirmed else "none"
            ),
            "a3b_triggered": a3b_triggered,
            "a3b_state": a3b_state,
            "a3b_triggered_source": (
                "rebuilt_media_confirmed"
                if a3b_state == "confirmed"
                else "observed_window"
                if a3b_triggered
                else "none"
            ),
            "a3b_observed_score": 0.78 if a3b_triggered else 0.0,
            "a3b_confirmed_score": (
                0.78 if a3b_state == "confirmed" else 0.0
            ),
            "a3b_confidence": (
                0.78 if a3b_state == "confirmed" else 0.0
            ),
            "a3b_display_score": 0.78 if a3b_triggered else 0.0,
        }

    def get(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path.startswith(f"/api/runs/{self.run_id}/overlay"):
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "overlay": {
                        "run_id": self.run_id,
                        "records": [
                            {
                                **self.final_status,
                                "seq": 1,
                            }
                        ],
                        "latest_seq": 1,
                    },
                },
            )
        if path.startswith("/api/status") and self.started:
            return FakeResponse(
                200,
                {"ok": True, "status": dict(self.final_status)},
            )
        if path.startswith("/api/evidence/events"):
            events = (
                [
                    {
                        "channel": channel,
                        "evidence_event_key": f"{channel}-event-{index}",
                        "run_id": self.run_id,
                        "source_epoch": 1,
                        "source_type": "file",
                        "source": self.status["source"],
                        "source_frame_start": 29,
                        "source_frame_end": 31,
                        "source_time_start_s": 0.95,
                        "source_time_end_s": 1.05,
                    }
                    for index, channel in enumerate(
                        self.evidence_channels,
                        start=1,
                    )
                ]
                if self.started
                else []
            )
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "evidence": {
                        "count": len(events),
                        "events": events,
                        "database": "fake.sqlite3",
                    },
                },
            )
        return super().get(path, **_kwargs)

    def post(self, path: str, **_kwargs: Any) -> FakeResponse:
        if path == "/api/runs/start":
            self.started = True
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "run_id": self.run_id,
                    "status": {
                        **self.status,
                        "run_id": self.run_id,
                        "running": True,
                        "source_ended": False,
                    },
                },
            )
        if path == f"/api/runs/{self.run_id}/stop":
            return FakeResponse(
                200,
                {"ok": True, "status": dict(self.final_status)},
            )
        raise AssertionError(path)


def _asset(
    *,
    asset_id: str,
    path: Path,
    category: str,
    order: int | None,
    attack_type: str | None = None,
) -> AuthoritativeAsset:
    content = path.read_bytes()
    expected_alert = category in {"a3b", "physical"}
    expected_a3b = category == "a3b"
    return AuthoritativeAsset(
        asset_id=asset_id,
        relative_path=path.name,
        canonical_path=str(path.resolve()),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        role=(
            "unique_model"
            if category == "model"
            else "a3b_target"
            if category == "a3b"
            else "physical_attack"
            if category == "physical"
            else "normal_video"
        ),
        label=f"label:{asset_id}",
        purpose="contract test",
        category=category,
        attack_type=attack_type,
        expected_module_a_alert=(
            None if category == "model" else expected_alert
        ),
        expected_a3b_trigger=None if category == "model" else expected_a3b,
        expected_module_a_evidence_events=(
            None
            if category == "model"
            else ">=1"
            if category in {"a3b", "physical"}
            else 0
        ),
        acceptance_order=order,
    )


def _manifest(tmp_path: Path) -> AuthoritativeManifest:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model = _asset(
        asset_id="model",
        path=model_path,
        category="model",
        order=None,
    )
    videos: list[AuthoritativeAsset] = []
    a3b_path = tmp_path / "a3b.mp4"
    a3b_path.write_bytes(b"a3b")
    videos.append(
        _asset(
            asset_id="a3b",
            path=a3b_path,
            category="a3b",
            order=1,
            attack_type="a3b_static_media",
        )
    )
    physical_types = (
        "adv_patch",
        "glare",
        "motion_blur",
        "occlusion",
        "visibility_degradation",
    )
    for index, attack_type in enumerate(physical_types, start=2):
        path = tmp_path / f"{attack_type}.mp4"
        path.write_bytes(attack_type.encode())
        videos.append(
            _asset(
                asset_id=attack_type,
                path=path,
                category="physical",
                order=index,
                attack_type=attack_type,
            )
        )
    for normal_index in range(30):
        path = tmp_path / f"normal-{normal_index}.mp4"
        path.write_bytes(f"normal-{normal_index}".encode())
        videos.append(
            _asset(
                asset_id=f"normal-{normal_index}",
                path=path,
                category="normal",
                order=7 + normal_index,
            )
        )
    return AuthoritativeManifest(
        schema_version=1,
        snapshot_date="2026-07-15",
        material_root=str(tmp_path.resolve()),
        unique_model=model,
        videos=tuple(videos),
        manifest_path=str((tmp_path / "manifest.json").resolve()),
    )


def _passing_status(
    tmp_path: Path,
    manifest: AuthoritativeManifest,
) -> dict[str, Any]:
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    source = manifest.asset_by_id("a3b").canonical_path
    return {
        "run_id": 0,
        "source_type": "file",
        "source": source,
        "source_epoch": 1,
        "backend": "tensorrt",
        "detector_queue_policy": "latest_only",
        "authoritative_model": {
            "locked": True,
            "metadata_valid": True,
            "backend": "tensorrt",
            "source": {
                "path": manifest.unique_model.canonical_path,
                "sha256": manifest.unique_model.sha256,
            },
            "engine": {
                "path": str(engine_path.resolve()),
                "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                "source_sha256": manifest.unique_model.sha256,
            },
        },
        "decoder": {
            "backend": "nvdec",
            "effective_backend": "nvdec",
            "codec": "h264",
            "gpu_device": "cuda:0",
            "decode_ms": {"p50": 1.0, "p95": 2.0},
            "gpu_to_cpu_copy_ms": {"p50": 0.1, "p95": 0.2},
            "fallback_count": 0,
            "fallback_reason": "",
        },
        "native": {
            "available": True,
            "version": "1.0.0",
            "binary_sha256": "1" * 64,
            "enabled_stages": ["a1", "a2"],
            "hit_counts": {"a1": 1},
            "fallback_counts": {"a1": 0},
            "fallback_reason": "",
        },
        "fps": 25.0,
        "processed_detection_frames": 90,
        "capture_frames_published": 100,
        "detection_source_coverage_ratio": 0.90,
        "source_frames_skipped_for_realtime": 0,
        "running": False,
        "source_ended": False,
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
    }


def _asset_report(asset: AuthoritativeAsset) -> dict[str, Any]:
    alert = bool(asset.expected_module_a_alert)
    a3b = bool(asset.expected_a3b_trigger)
    physical = asset.category == "physical"
    evidence = 1 if asset.expected_module_a_evidence_events == ">=1" else 0
    physical_evidence = evidence if physical else 0
    a3b_evidence = evidence if a3b else 0
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
            "source_ended": True,
            "timed_out": False,
            "completion_reason": "source_ended",
        },
        "source_identity": {
            "expected": {
                "canonical_path": asset.canonical_path,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
            },
            "before": {"matches_manifest": True},
            "after": {"matches_manifest": True},
        },
        "lineage": {
            "passed": True,
            "run_id": 1,
            "observed_source_epochs": [1],
            "errors": [],
        },
        "observations": {
            "alert_confirmed_observed": alert,
            "physical_alert_confirmed_observed": physical,
            "a3b_triggered_observed": a3b,
            "first_a3b_source_time_s": 1.0 if a3b else None,
            "a3b_internal_false_count": 0,
            "alert_evidence_correlated": evidence > 0,
            "module_a_evidence_event_count": evidence,
            "physical_evidence_event_count": physical_evidence,
            "a3b_evidence_event_count": a3b_evidence,
            "performance": {
                "detector_fps": 25.0,
                "detection_source_coverage_ratio": 0.90,
                "source_frames_skipped_for_realtime": 0,
                "decoder": {
                    "backend": "nvdec",
                    "effective_backend": "nvdec",
                    "fallback_count": 0,
                    "fallback_reason": "",
                },
                "native": {
                    "available": True,
                    "binary_sha256": "1" * 64,
                    "enabled_stages": ["a1"],
                    "hit_counts": {"a1": 1},
                    "fallback_counts": {"a1": 0},
                    "fallback_reason": "",
                },
            },
        },
        "runtime_contract": {
            "passed": True,
            "checks": {},
            "blockers": [],
        },
        "gates": {"all_contract_gates": {"passed": True}},
        "passed": True,
        "blockers": [],
    }


def _run_scenario_asset(
    tmp_path: Path,
    *,
    asset_id: str,
    top_alert: bool,
    physical_confirmed: bool,
    a3b_triggered: bool,
    a3b_state: str,
    evidence_channels: tuple[str, ...],
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(tmp_path)
    asset = manifest.asset_by_id(asset_id)
    status = _passing_status(tmp_path, manifest)
    status["source"] = asset.canonical_path
    report = run_authoritative_web_acceptance(
        ScenarioCompletingFakeClient(
            status=status,
            top_alert=top_alert,
            physical_confirmed=physical_confirmed,
            a3b_triggered=a3b_triggered,
            a3b_state=a3b_state,
            evidence_channels=evidence_channels,
        ),
        manifest,
        selected_asset_ids=[asset_id],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )
    assert len(report["assets"]) == 1
    return report["assets"][0]


def test_preflight_reports_production_contract_blockers(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    status["backend"] = "pytorch"
    status["detector_queue_policy"] = "fifo"
    status["authoritative_model"]["engine"]["source_sha256"] = "f" * 64
    status.pop("decoder")
    status.pop("native")

    result = run_web_preflight(
        StaticFakeClient(status=status),
        manifest,
        verify_status_artifact_files=False,
    )

    assert result["passed"] is False
    blocker_codes = {blocker["code"] for blocker in result["blockers"]}
    assert {
        "production_tensorrt",
        "latest_only",
        "authoritative_engine_binding",
        "decoder_status_fields",
        "native_status_fields",
        "fallback_visibility",
    }.issubset(blocker_codes)


def test_preflight_accepts_complete_status_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)

    result = run_web_preflight(
        StaticFakeClient(status=status),
        manifest,
        verify_status_artifact_files=True,
    )

    assert result["passed"] is True
    assert result["blockers"] == []
    assert all(check["passed"] is True for check in result["checks"].values())


def test_complete_report_aggregation_and_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    reports = [_asset_report(asset) for asset in manifest.ordered_videos]
    preflight = {"passed": True, "blockers": []}
    selected_ids = [asset.asset_id for asset in manifest.ordered_videos]

    summary = aggregate_web_acceptance_report(
        manifest=manifest,
        preflight=preflight,
        asset_reports=reports,
        selected_asset_ids=selected_ids,
    )
    report = build_web_acceptance_report(
        manifest=manifest,
        preflight=preflight,
        asset_reports=reports,
        selected_asset_ids=selected_ids,
    )

    assert summary["passed"] is True
    assert summary["categories"] == {
        "a3b": {"reported": 1, "passed": 1},
        "physical": {"reported": 5, "passed": 5},
        "normal": {"reported": 30, "passed": 30},
    }
    assert summary["normal_false_positive_videos"] == 0
    assert validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=True,
    ) == []


def test_report_identity_must_match_selected_manifest_assets(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    reports = [_asset_report(asset) for asset in manifest.ordered_videos]
    selected_ids = [asset.asset_id for asset in manifest.ordered_videos]
    reports[0]["asset_id"] = "forged-a3b"

    report = build_web_acceptance_report(
        manifest=manifest,
        preflight={"passed": True, "blockers": []},
        asset_reports=reports,
        selected_asset_ids=selected_ids,
    )

    assert report["summary"]["passed"] is False
    blocker_codes = {
        blocker["code"] for blocker in report["summary"]["blockers"]
    }
    assert "asset_report_identity_mismatch" in blocker_codes
    errors = validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=True,
    )
    error_codes = {error["code"] for error in errors}
    assert "selected_report_identity_mismatch" in error_codes
    assert "unknown_manifest_asset" in error_codes


def test_timeout_cannot_be_aggregated_or_validated_as_success(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    reports = [_asset_report(asset) for asset in manifest.ordered_videos]
    reports[0]["execution"]["source_ended"] = False
    reports[0]["execution"]["timed_out"] = True
    reports[0]["execution"]["completion_reason"] = "timeout"
    reports[0]["passed"] = False
    reports[0]["blockers"] = [{"code": "no_timeout"}]
    selected_ids = [asset.asset_id for asset in manifest.ordered_videos]
    preflight = {"passed": True, "blockers": []}

    report = build_web_acceptance_report(
        manifest=manifest,
        preflight=preflight,
        asset_reports=reports,
        selected_asset_ids=selected_ids,
    )

    assert report["summary"]["passed"] is False
    assert report["summary"]["timeout_count"] == 1
    assert report["summary"]["source_ended_count"] == 35
    blocker_codes = {
        blocker["code"] for blocker in report["summary"]["blockers"]
    }
    assert {"source_not_ended", "asset_timeouts"}.issubset(blocker_codes)
    assert validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=True,
    ) == []

    report["assets"][0]["passed"] = True
    errors = validate_web_acceptance_report(
        report,
        manifest=manifest,
        require_complete=True,
    )
    error_codes = {error["code"] for error in errors}
    assert "passed_without_source_end" in error_codes
    assert "timeout_marked_success" in error_codes


def test_full_runner_marks_never_ending_source_as_timeout(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    client = NeverEndingFakeClient(status=status)

    report = run_authoritative_web_acceptance(
        client,
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=0.01,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )

    assert len(report["assets"]) == 1
    asset_report = report["assets"][0]
    assert asset_report["execution"]["timed_out"] is True
    assert asset_report["execution"]["source_ended"] is False
    assert asset_report["passed"] is False
    assert report["summary"]["passed"] is False


def test_full_runner_records_live_source_end_and_module_a_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    client = CompletingFakeClient(status=status)

    report = run_authoritative_web_acceptance(
        client,
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )

    asset_report = report["assets"][0]
    assert asset_report["execution"]["source_ended"] is True
    assert asset_report["execution"]["timed_out"] is False
    assert asset_report["runtime_contract"]["passed"] is True
    assert asset_report["observations"]["alert_confirmed_observed"] is True
    assert (
        asset_report["observations"][
            "physical_alert_confirmed_observed"
        ]
        is False
    )
    assert asset_report["observations"]["a3b_triggered_observed"] is True
    assert (
        asset_report["observations"]["module_a_evidence_event_count"] == 1
    )
    assert asset_report["observations"]["physical_evidence_event_count"] == 0
    assert asset_report["observations"]["a3b_evidence_event_count"] == 1
    assert asset_report["passed"] is True
    assert report["summary"]["passed"] is False
    assert report["summary"]["complete_manifest_selection"] is False


def test_a3b_continuity_counts_false_between_confirmations_without_break() -> None:
    observations = _summarize_observations(
        status_samples=[],
        overlay_records=[
            {
                "seq": 1,
                "source_time_s": 1.0,
                "a3b_triggered": True,
                "a3b_confirmed_alert": True,
                "a3b_state": "confirmed",
            },
            {
                "seq": 2,
                "source_time_s": 1.2,
                "a3b_triggered": False,
                "a3b_confirmed_alert": False,
                "a3b_state": "normal",
            },
            {
                "seq": 3,
                "source_time_s": 1.4,
                "a3b_triggered": True,
                "a3b_confirmed_alert": True,
                "a3b_state": "confirmed",
            },
        ],
        bound_evidence_events=[],
        unbound_evidence_events=[],
    )

    assert observations["a3b_internal_false_count"] == 1
    assert observations["a3b_continuity_break_count"] == 0
    assert observations["a3b_continuity_segment_count"] == 1


@pytest.mark.parametrize(
    "failed_gate",
    [
        "border_suppressed",
        "camera_motion_suppressed",
        "physical_motion_suppressed",
        "rebuilt_result_stale",
        "rebuilt_candidate_disallowed",
        "rebuilt_policy_suppressed",
    ],
)
def test_a3b_continuity_does_not_count_false_across_hard_suppression_break(
    failed_gate: str,
) -> None:
    observations = _summarize_observations(
        status_samples=[],
        overlay_records=[
            {
                "seq": 1,
                "source_time_s": 1.0,
                "a3b_triggered": True,
                "a3b_confirmed_alert": True,
                "a3b_state": "confirmed",
            },
            {
                "seq": 2,
                "source_time_s": 1.2,
                "a3b_triggered": False,
                "a3b_confirmed_alert": False,
                "a3b_state": "suppressed",
                "a3b_debug": {"failed_gates": [failed_gate]},
            },
            {
                "seq": 3,
                "source_time_s": 1.4,
                "a3b_triggered": False,
                "a3b_confirmed_alert": False,
                "a3b_state": "normal",
            },
            {
                "seq": 4,
                "source_time_s": 1.6,
                "a3b_triggered": True,
                "a3b_confirmed_alert": True,
                "a3b_state": "confirmed",
            },
        ],
        bound_evidence_events=[],
        unbound_evidence_events=[],
    )

    assert observations["a3b_internal_false_count"] == 0
    assert observations["a3b_continuity_break_count"] == 1
    assert observations["a3b_continuity_segment_count"] == 2


def test_physical_asset_passes_with_physical_confirmation_and_evidence(
    tmp_path: Path,
) -> None:
    asset_report = _run_scenario_asset(
        tmp_path,
        asset_id="adv_patch",
        top_alert=True,
        physical_confirmed=True,
        a3b_triggered=False,
        a3b_state="normal",
        evidence_channels=("module_a",),
    )

    observations = asset_report["observations"]
    assert observations["alert_confirmed_observed"] is True
    assert observations["physical_alert_confirmed_observed"] is True
    assert observations["a3b_triggered_observed"] is False
    assert observations["module_a_evidence_event_count"] == 1
    assert observations["physical_evidence_event_count"] == 1
    assert observations["a3b_evidence_event_count"] == 0
    assert asset_report["gates"]["module_a_alert"]["passed"] is True
    assert (
        asset_report["gates"]["module_a_evidence_events"]["passed"]
        is True
    )
    assert asset_report["passed"] is True


def test_physical_asset_cannot_pass_from_a3b_alert_and_evidence(
    tmp_path: Path,
) -> None:
    asset_report = _run_scenario_asset(
        tmp_path,
        asset_id="adv_patch",
        top_alert=True,
        physical_confirmed=False,
        a3b_triggered=True,
        a3b_state="confirmed",
        evidence_channels=("a3b",),
    )

    observations = asset_report["observations"]
    assert observations["alert_confirmed_observed"] is True
    assert observations["physical_alert_confirmed_observed"] is False
    assert observations["a3b_triggered_observed"] is True
    assert observations["module_a_evidence_event_count"] == 1
    assert observations["physical_evidence_event_count"] == 0
    assert observations["a3b_evidence_event_count"] == 1
    assert asset_report["gates"]["module_a_alert"]["passed"] is False
    assert (
        asset_report["gates"]["module_a_evidence_events"]["passed"]
        is False
    )
    assert asset_report["passed"] is False


def test_a3b_asset_accepts_a3b_channel_evidence_when_confirmed(
    tmp_path: Path,
) -> None:
    confirmed = _run_scenario_asset(
        tmp_path,
        asset_id="a3b",
        top_alert=True,
        physical_confirmed=False,
        a3b_triggered=True,
        a3b_state="confirmed",
        evidence_channels=("a3b",),
    )

    confirmed_observations = confirmed["observations"]
    assert confirmed_observations["alert_confirmed_observed"] is True
    assert (
        confirmed_observations["physical_alert_confirmed_observed"]
        is False
    )
    assert confirmed_observations["a3b_triggered_observed"] is True
    assert confirmed_observations["module_a_evidence_event_count"] == 1
    assert confirmed_observations["physical_evidence_event_count"] == 0
    assert confirmed_observations["a3b_evidence_event_count"] == 1
    assert confirmed["passed"] is True


def test_a3b_asset_rejects_suspect_only_even_with_a3b_evidence(
    tmp_path: Path,
) -> None:
    suspect = _run_scenario_asset(
        tmp_path,
        asset_id="a3b",
        top_alert=False,
        physical_confirmed=False,
        a3b_triggered=True,
        a3b_state="suspect",
        evidence_channels=("a3b",),
    )

    suspect_observations = suspect["observations"]
    assert suspect_observations["alert_confirmed_observed"] is False
    assert (
        suspect_observations["physical_alert_confirmed_observed"]
        is False
    )
    # Raw suspect activity remains observable for diagnostics, while the
    # acceptance gate and top-level alert use confirmed-only state.
    assert suspect_observations["a3b_triggered_observed"] is True
    assert suspect_observations["a3b_confirmed_observed"] is False
    assert suspect_observations["module_a_evidence_event_count"] == 1
    assert suspect_observations["a3b_evidence_event_count"] == 1
    assert suspect["gates"]["module_a_alert"]["passed"] is False
    assert suspect["gates"]["a3b_trigger"]["passed"] is False
    assert suspect["passed"] is False


@pytest.mark.parametrize("evidence_channel", ["module_a", "a3b"])
def test_normal_asset_rejects_any_module_a_evidence_channel(
    tmp_path: Path,
    evidence_channel: str,
) -> None:
    asset_report = _run_scenario_asset(
        tmp_path,
        asset_id="normal-0",
        top_alert=False,
        physical_confirmed=False,
        a3b_triggered=False,
        a3b_state="normal",
        evidence_channels=(evidence_channel,),
    )

    observations = asset_report["observations"]
    assert observations["alert_confirmed_observed"] is False
    assert observations["physical_alert_confirmed_observed"] is False
    assert observations["a3b_triggered_observed"] is False
    assert observations["module_a_evidence_event_count"] == 1
    assert (
        observations["physical_evidence_event_count"]
        == (1 if evidence_channel == "module_a" else 0)
    )
    assert (
        observations["a3b_evidence_event_count"]
        == (1 if evidence_channel == "a3b" else 0)
    )
    assert (
        asset_report["gates"]["module_a_evidence_events"]["passed"]
        is False
    )
    assert asset_report["passed"] is False


def test_full_runner_blocks_low_performance_decoder_fallback_and_capture_skip(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    status["fps"] = 19.0
    status["detection_source_coverage_ratio"] = 0.67
    status["source_frames_skipped_for_realtime"] = 79
    status["decoder"].update(
        {
            "backend": "opencv",
            "effective_backend": "opencv",
            "fallback_count": 1,
            "fallback_reason": "nvdec_init_failed:synthetic",
        }
    )
    status["native"].update(
        {
            "hit_counts": {"a1": 0},
            "fallback_counts": {"a1": 1},
            "fallback_reason": "synthetic_fallback",
        }
    )

    report = run_authoritative_web_acceptance(
        CompletingFakeClient(status=status),
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )

    asset_report = report["assets"][0]
    assert asset_report["passed"] is False
    failed_gates = {
        name
        for name, gate in asset_report["gates"].items()
        if gate["passed"] is not True
    }
    assert {
        "detector_fps",
        "detection_source_coverage",
        "nvdec_effective_backend",
        "decoder_no_fallback",
        "capture_no_source_skip",
        "native_runtime_hit",
    }.issubset(failed_gates)


def test_full_runner_uses_compute_throughput_for_low_fps_file_source(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    status["fps"] = 24.0
    status["detector_compute_fps"] = 40.0

    report = run_authoritative_web_acceptance(
        CompletingFakeClient(status=status),
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )

    asset_report = report["assets"][0]
    performance = asset_report["observations"]["performance"]
    detector_gate = asset_report["gates"]["detector_fps"]

    assert performance["detector_fps"] == 24.0
    assert performance["detector_compute_fps"] == 40.0
    assert detector_gate["passed"] is True
    assert detector_gate["actual"] == 40.0


def test_full_runner_requires_source_bound_verified_derived_video(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)
    asset = manifest.asset_by_id("a3b")
    status["decoder"].update(
        {
            "derived_cache_used": True,
            "derived_cache_validation": "verified",
            "source_sha256": asset.sha256,
            "decode_source_sha256": "2" * 64,
            "derived_metadata_path": str(
                (tmp_path / "derived-metadata.json").resolve()
            ),
            "derived_metadata_sha256": "4" * 64,
            "source_asset_id": asset.asset_id,
            "source_role": asset.role,
            "source_label": asset.label,
            "source_attack_type": asset.attack_type,
            "derived_profile_id": "h264_nvenc_lossless_yuv420p_v1",
            "derived_profile_sha256": "5" * 64,
            "transcode_decode_backend": "ffmpeg_software_hevc",
            "transcode_encode_backend": "h264_nvenc",
            "derived_frame_parity": True,
            "derived_frame_count_match": True,
            "derived_fps_match": True,
            "derived_expected_frame_count": 100,
            "frames_decoded": 100,
            "eof": True,
        }
    )

    passing = run_authoritative_web_acceptance(
        CompletingFakeClient(status=status),
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )["assets"][0]

    assert passing["gates"]["derived_video_provenance"]["passed"] is True

    status["decoder"]["source_sha256"] = "3" * 64
    failing = run_authoritative_web_acceptance(
        CompletingFakeClient(status=status),
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )["assets"][0]

    assert failing["gates"]["derived_video_provenance"]["passed"] is False
    assert failing["passed"] is False


def test_full_runner_rejects_unbound_evidence_from_another_run(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    status = _passing_status(tmp_path, manifest)

    report = run_authoritative_web_acceptance(
        WrongEvidenceLineageClient(status=status),
        manifest,
        selected_asset_ids=["a3b"],
        asset_timeout_s=1.0,
        poll_interval_s=0.001,
        verify_status_artifact_files=True,
    )

    asset_report = report["assets"][0]
    assert asset_report["execution"]["source_ended"] is True
    assert asset_report["lineage"]["passed"] is False
    assert asset_report["evidence"]["bound_event_count"] == 0
    assert asset_report["evidence"]["unbound_event_count"] == 1
    assert asset_report["gates"]["runtime_lineage"]["passed"] is False
    assert asset_report["passed"] is False

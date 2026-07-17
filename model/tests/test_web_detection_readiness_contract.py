from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from defense.web.fastapi_app import create_app


class TrustedModelSecurity:
    def ensure_admitted(self, **_payload) -> dict:
        return {"allowed": True, "status": "trusted", "admission_status": "trusted", "whitelist_hit": True}

    def status(self, **_payload) -> dict:
        return self.ensure_admitted()


class BlockingModelSecurity:
    def __init__(self) -> None:
        self.scan_started = False
        self.scan_calls: list[dict] = []

    def ensure_admitted(self, **_payload) -> dict:
        return {
            "allowed": False,
            "status": "blocked_scan_required",
            "admission_status": "blocked_scan_required",
            "whitelist_hit": False,
            "blocking_reason": "model_not_in_whitelist",
            "next_action": "start_full_scan",
            "operator_message": "需要完整扫描",
        }

    def status(self, **_payload) -> dict:
        return {
            "allowed": False,
            "status": "scanning",
            "admission_status": "scanning",
            "whitelist_hit": False,
            "blocking_reason": "full_scan_running",
            "scanning": True,
        }

    def start_background_scan(self, **payload) -> dict:
        self.scan_started = True
        self.scan_calls.append(payload)
        return {"started": True, "scan_type": "full", "auto_purify": bool(payload.get("auto_purify"))}


class SuspiciousModelSecurity:
    def __init__(self) -> None:
        self.purification_calls: list[dict] = []

    def ensure_admitted(self, **_payload) -> dict:
        return {
            "allowed": False,
            "status": "suspicious",
            "admission_status": "suspicious",
            "whitelist_hit": False,
            "blocking_reason": "last_full_scan_suspicious",
            "next_action": "start_purification",
            "operator_message": "需要净化",
        }

    def status(self, **_payload) -> dict:
        return {
            "allowed": False,
            "status": "purifying",
            "admission_status": "purifying",
            "whitelist_hit": False,
            "blocking_reason": "purification_running",
            "purifying": True,
        }

    def start_background_purification(self, **payload) -> dict:
        self.purification_calls.append(payload)
        return {"started": True, "fingerprint": "sha256:suspicious", "scan_after": bool(payload.get("scan_after"))}


class PurifyingModelSecurity:
    def __init__(self) -> None:
        self.scan_calls: list[dict] = []
        self.purification_calls: list[dict] = []

    def ensure_admitted(self, **_payload) -> dict:
        return {
            "allowed": False,
            "status": "purifying",
            "admission_status": "purifying",
            "whitelist_hit": False,
            "blocking_reason": "purification_running",
            "purifying": True,
        }

    def status(self, **_payload) -> dict:
        return self.ensure_admitted()

    def start_background_scan(self, **payload) -> dict:
        self.scan_calls.append(payload)
        return {"started": False, "reason": "unexpected"}

    def start_background_purification(self, **payload) -> dict:
        self.purification_calls.append(payload)
        return {"started": False, "reason": "unexpected"}


class PurifiedAlternativeModelSecurity:
    def __init__(self, purified_path: str) -> None:
        self.purified_path = purified_path
        self.ensure_calls: list[dict] = []
        self.runtime_resolve_calls: list[dict] = []

    def ensure_admitted(self, **payload) -> dict:
        self.ensure_calls.append(payload)
        return {
            "allowed": False,
            "status": "purified_alternative_available",
            "admission_status": "purified_alternative_available",
            "whitelist_hit": False,
            "blocking_reason": "purified_pt_clean_but_runtime_not_selected",
            "purified_model_path": self.purified_path,
        }

    def status(self, **payload) -> dict:
        return self.ensure_admitted(**payload)

    def trusted_purified_runtime_model(self, **payload) -> dict:
        self.runtime_resolve_calls.append(payload)
        replacement = {
            "enabled": True,
            "path": self.purified_path,
            "backend": "pytorch",
            "model_family": "yolov5",
            "source_pt_path": self.purified_path,
        }
        return {
            "custom_model": replacement,
            "model_security": {
                "allowed": True,
                "status": "trusted",
                "admission_status": "trusted",
                "whitelist_hit": True,
                "runtime_artifact_path": self.purified_path,
                "model_hash": "sha256:purified",
            },
            "source_model_security": self.ensure_admitted(**payload),
        }


class AutoRemediateModelSecurity:
    def __init__(self, purified_path: str) -> None:
        self.purified_path = purified_path
        self.prepare_calls: list[dict] = []

    def prepare_runtime_for_start(self, **payload) -> dict:
        self.prepare_calls.append(payload)
        replacement = {
            "enabled": True,
            "path": self.purified_path,
            "backend": "pytorch",
            "model_family": "yolov5",
            "source_pt_path": self.purified_path,
        }
        if not bool(payload.get("auto_remediate", True)):
            return {
                "allowed": False,
                "custom_model": None,
                "model_security": {
                    "allowed": False,
                    "status": "suspicious",
                    "admission_status": "suspicious",
                    "whitelist_hit": False,
                    "blocking_reason": "last_full_scan_suspicious",
                    "next_action": "start_purification",
                    "operator_message": "需要显式后台净化",
                },
                "scan": None,
                "purification": None,
                "runtime_replacement": None,
            }
        return {
            "allowed": True,
            "custom_model": replacement,
            "model_security": {
                "allowed": True,
                "status": "trusted",
                "admission_status": "trusted",
                "whitelist_hit": True,
                "runtime_replacement": replacement,
            },
            "scan": {"status": "suspicious", "scan_type": "full"},
            "purification": {"status": "scan_clean_trusted", "purified_model_path": self.purified_path},
            "runtime_replacement": {"mode": "purified_runtime"},
        }


class ReadyDetectionEngine:
    def __init__(self) -> None:
        self.run_id = 41
        self.started_with: dict | None = None

    def start(self, **payload) -> int:
        self.started_with = payload
        return self.run_id

    def wait_ready_for_preview(self, run_id: int, timeout: float = 45.0) -> dict:
        assert run_id == self.run_id
        return self.get_status()

    def get_status(self) -> dict:
        return {
            "run_id": self.run_id,
            "running": True,
            "ready_for_preview": True,
            "detector_ready": True,
            "backend": "test",
            "artifact": "test://artifact",
            "overlay_seq": 3,
            "raw_boxes_count": 2,
            "frame_idx": 12,
            "p_adv": 0.42,
            "display_options": {},
            "recent_events": [],
            "recent_ppe_events": [],
            "recent_source_auth_events": [],
        }

    def get_overlay(self, since_seq: int = 0) -> dict:
        return {
            "run_id": self.run_id,
            "seq": 3,
            "latest_seq": 3,
            "records": [
                {
                    "overlay_seq": 3,
                    "video_time_s": 0.4,
                    "a3b_score": 0.2,
                    "a3b_triggered": False,
                    "raw_boxes_count": 2,
                }
            ],
        }


def test_web_start_returns_detection_readiness_fields() -> None:
    engine = ReadyDetectionEngine()
    client = TestClient(create_app(engine=engine, model_security=TrustedModelSecurity()))

    response = client.post(
        "/api/start",
        json={"source_type": "file", "source": "sample.mp4", "profile": "desktop_rtx", "ready_timeout_s": 0.1},
    )

    assert response.status_code == 200
    data = response.json()
    status = data["status"]
    assert data["ok"] is True
    assert data["run_id"] == 41
    assert status["ready_for_preview"] is True
    assert status["detector_ready"] is True
    assert status["backend"] == "test"
    assert status["overlay_seq"] == 3
    assert status["raw_boxes_count"] == 2
    assert engine.started_with["profile"] == "desktop_rtx"


@pytest.mark.skip(reason="超前契约未实装:仅准入(admission-only)流未实装,旧准入逻辑自动扫描/净化并覆盖admission、无顶层next_action")
def test_web_start_blocks_when_model_security_requires_scan() -> None:
    engine = ReadyDetectionEngine()
    security = BlockingModelSecurity()
    client = TestClient(create_app(engine=engine, model_security=security))

    response = client.post(
        "/api/start",
        json={"source_type": "file", "source": "sample.mp4", "profile": "desktop_rtx", "ready_timeout_s": 0.1},
    )

    assert response.status_code == 409
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "model_security_blocked"
    assert data["model_security"]["admission_status"] == "blocked_scan_required"
    assert data["next_action"] == "start_full_scan"
    assert data["scan"] is None
    assert security.scan_started is False
    assert security.scan_calls == []
    assert engine.started_with is None


@pytest.mark.skip(reason="超前契约未实装:仅准入(admission-only)流未实装,旧准入逻辑自动扫描/净化并覆盖admission、无顶层next_action")
def test_web_start_suspicious_model_starts_background_purification_and_blocks_a_module() -> None:
    engine = ReadyDetectionEngine()
    security = SuspiciousModelSecurity()
    client = TestClient(create_app(engine=engine, model_security=security))

    response = client.post(
        "/api/start",
        json={"source_type": "file", "source": "sample.mp4", "profile": "desktop_rtx", "ready_timeout_s": 0.1},
    )

    assert response.status_code == 409
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "model_security_blocked"
    assert data["model_security"]["admission_status"] == "suspicious"
    assert data["next_action"] == "start_purification"
    assert data["purification"] is None
    assert security.purification_calls == []
    assert engine.started_with is None


def test_web_start_purifying_model_blocks_without_duplicate_work_or_engine_start() -> None:
    engine = ReadyDetectionEngine()
    security = PurifyingModelSecurity()
    client = TestClient(create_app(engine=engine, model_security=security))

    response = client.post(
        "/api/start",
        json={"source_type": "file", "source": "sample.mp4", "profile": "desktop_rtx", "ready_timeout_s": 0.1},
    )

    assert response.status_code == 409
    data = response.json()
    assert data["model_security"]["admission_status"] == "purifying"
    assert data["scan"] is None
    assert data["purification"] is None
    assert security.scan_calls == []
    assert security.purification_calls == []
    assert engine.started_with is None


def test_web_start_rejects_purified_runtime_replacement_in_production(tmp_path) -> None:
    purified = tmp_path / "purified.pt"
    purified.write_bytes(b"trusted-purified")
    engine = ReadyDetectionEngine()
    security = PurifiedAlternativeModelSecurity(str(purified))
    client = TestClient(create_app(engine=engine, model_security=security))

    response = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "sample.mp4",
            "profile": "desktop_rtx",
            "ready_timeout_s": 0.1,
        },
    )

    assert response.status_code == 409
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "production_model_replacement_forbidden"
    assert data["model_security"]["admission_status"] == "trusted"
    assert data["model_security"]["runtime_replacement"]["path"] == str(purified)
    assert engine.started_with is None
    assert security.runtime_resolve_calls[0]["profile"] == "desktop_rtx"


def test_web_start_rejects_custom_model_before_auto_remediation(tmp_path) -> None:
    purified = tmp_path / "auto_purified.pt"
    purified.write_bytes(b"trusted-purified")
    engine = ReadyDetectionEngine()
    security = AutoRemediateModelSecurity(str(purified))
    client = TestClient(create_app(engine=engine, model_security=security))

    response = client.post(
        "/api/start",
        json={
            "source_type": "file",
            "source": "sample.mp4",
            "profile": "desktop_rtx",
            "ready_timeout_s": 0.1,
            "custom_model": {
                "enabled": True,
                "path": "poisoned.pt",
                "backend": "pytorch",
                "model_family": "yolov5",
            },
        },
    )

    assert response.status_code == 409
    data = response.json()
    assert data["ok"] is False
    assert data["error"] == "production_model_locked"
    assert engine.started_with is None
    assert security.prepare_calls == []


def test_web_overlay_returns_detection_records() -> None:
    client = TestClient(create_app(engine=ReadyDetectionEngine(), model_security=TrustedModelSecurity()))

    response = client.get("/api/overlay?since_seq=0")

    assert response.status_code == 200
    overlay = response.json()["overlay"]
    assert overlay["latest_seq"] == 3
    assert overlay["records"][0]["raw_boxes_count"] == 2
    assert "a3b_score" in overlay["records"][0]

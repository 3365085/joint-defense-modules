from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from defense.web.fastapi_app import create_app


class DummyEngine:
    def __init__(self) -> None:
        self.run_id = 7
        self.calls: list[tuple[int, str, dict]] = []

    def get_status(self) -> dict:
        return {"run_id": self.run_id, "running": True, "display_options": {}}

    def control_run(self, run_id: int, action: str, **payload) -> dict:
        self.calls.append((run_id, action, payload))
        return {"run_id": run_id, "running": True, "playback_paused": action == "pause", "display_options": {}}


def test_control_route_does_not_pass_duplicate_action() -> None:
    engine = DummyEngine()
    app = create_app(engine=engine)
    client = TestClient(app)

    response = client.post("/api/runs/7/control", json={"action": "pause", "source_time_s": 4.0})

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert engine.calls == [(7, "pause", {"source_time_s": 4.0})]


class RejectingEngine(DummyEngine):
    def control_run(self, run_id: int, action: str, **payload) -> dict:
        raise RuntimeError("run is not active")


def test_control_route_surfaces_inactive_run_conflict() -> None:
    app = create_app(engine=RejectingEngine())
    client = TestClient(app)

    response = client.post("/api/runs/7/control", json={"action": "play"})

    assert response.status_code == 409
    assert response.json()["detail"] == "run is not active"


class FakeModelSecurity:
    def ensure_admitted(self, *, profile: str, custom_model: dict) -> dict:
        return {
            "enabled": True,
            "allowed": True,
            "status": "trusted",
            "admission_status": "trusted",
            "whitelist_hit": True,
        }

    def status(self, *, profile: str, custom_model: dict) -> dict:
        return self.ensure_admitted(profile=profile, custom_model=custom_model)

    def _log_event(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None


class StartEngine(DummyEngine):
    def __init__(self, ready_status: dict) -> None:
        super().__init__()
        self.run_id = 12
        self.ready_status = ready_status
        self.start_calls: list[dict] = []

    def start(self, **kwargs) -> int:  # type: ignore[no-untyped-def]
        self.start_calls.append(kwargs)
        return self.run_id

    def wait_ready_for_preview(self, run_id: int, timeout: float) -> dict:
        assert run_id == self.run_id
        return {"run_id": run_id, "display_options": {}, **self.ready_status}

    def get_status(self) -> dict:
        return {"run_id": self.run_id, "running": False, "display_options": {}}


def test_start_route_surfaces_async_runtime_error_with_injected_engine() -> None:
    engine = StartEngine(
        {
            "running": False,
            "ready_for_preview": False,
            "error": "capture open failed",
        }
    )
    app = create_app(engine=engine, model_security=FakeModelSecurity(), bind_host="127.0.0.1")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/api/start",
        json={
            "source_type": "camera",
            "source": "camera:99",
        },
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "runtime_start_failed"
    assert payload["message"] == "capture open failed"
    assert engine.start_calls[0]["custom_model"] == {}


def test_start_route_surfaces_preview_ready_timeout_with_injected_engine() -> None:
    engine = StartEngine(
        {
            "running": True,
            "initializing": True,
            "prewarming": False,
            "ready_for_preview": False,
            "error": "",
        }
    )
    app = create_app(engine=engine, model_security=FakeModelSecurity(), bind_host="127.0.0.1")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/api/start",
        json={
            "source_type": "camera",
            "source": "camera:0",
            "ready_timeout_s": 0.01,
        },
    )

    assert response.status_code == 504
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "runtime_start_timeout"
    assert payload["message"] == "preview did not become ready"
    assert engine.start_calls[0]["custom_model"] == {}


@pytest.mark.parametrize(
    "bypass_field",
    [
        "test_bypass_model_security",
        "bypass_model_security_for_test",
        "bypass_model_security",
    ],
)
def test_production_start_rejects_model_security_test_bypass(bypass_field: str) -> None:
    engine = StartEngine({"ready_for_preview": True})
    app = create_app(engine=engine, model_security=FakeModelSecurity(), bind_host="127.0.0.1")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/api/start",
        json={
            "source_type": "camera",
            "source": "camera:0",
            bypass_field: True,
        },
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"] == "production_security_bypass_forbidden"
    assert engine.start_calls == []


from __future__ import annotations

from fastapi.testclient import TestClient

from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config
from defense.web.fastapi_app import create_app


class _Engine:
    def __init__(self) -> None:
        self.started_with = None

    def start(self, **kwargs):
        self.started_with = kwargs
        return 1

    def stop(self, **_kwargs):
        return None

    def get_status(self):
        return {"running": False}

    def wait_ready_for_preview(self, run_id: int, *, timeout: float):
        return {
            "run_id": run_id,
            "running": True,
            "ready_for_preview": True,
            "timeout": timeout,
        }


class _Security:
    def __init__(self) -> None:
        self.prepare_calls: list[dict] = []

    def prepare_runtime_for_start(self, *, profile, custom_model, auto_remediate):
        self.prepare_calls.append(
            {
                "profile": profile,
                "custom_model": custom_model,
                "auto_remediate": auto_remediate,
            }
        )
        return {
            "allowed": True,
            "custom_model": custom_model,
            "model_security": {"allowed": True, "status": "trusted"},
            "scan": None,
            "purification": None,
            "runtime_replacement": None,
        }

    def status(self, **_kwargs):
        return {"allowed": True, "status": "trusted"}


def test_production_web_allows_custom_model_after_b_admission() -> None:
    engine = _Engine()
    security = _Security()
    app = create_app(
        config_path=DEFAULT_CONFIG_PATH,
        engine=engine,
        model_security=security,
    )
    custom_model = {
        "enabled": True,
        "path": "other.engine",
        "backend": "tensorrt",
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "source_type": "file",
                "source": "unused.mp4",
                "profile": "desktop_rtx",
                "custom_model": custom_model,
            },
        )

    assert response.status_code == 200
    assert response.json()["model_security"]["status"] == "trusted"
    assert engine.started_with["custom_model"] == custom_model
    assert security.prepare_calls == [
        {
            "profile": "desktop_rtx",
            "custom_model": custom_model,
            "auto_remediate": False,
        }
    ]


def test_production_web_without_custom_selection_uses_default_model() -> None:
    engine = _Engine()
    app = create_app(
        config_path=DEFAULT_CONFIG_PATH,
        engine=engine,
        model_security=_Security(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "source_type": "file",
                "source": "unused.mp4",
                "profile": "desktop_rtx",
            },
        )

    assert response.status_code == 200
    assert engine.started_with is not None
    assert engine.started_with["custom_model"] == {}
    assert "allow_test_custom_model" not in engine.started_with


def test_production_web_rejects_model_security_bypass() -> None:
    engine = _Engine()
    app = create_app(
        config_path=DEFAULT_CONFIG_PATH,
        engine=engine,
        model_security=_Security(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "source_type": "file",
                "source": "unused.mp4",
                "profile": "desktop_rtx",
                "test_bypass_model_security": True,
            },
        )

    assert response.status_code == 403
    assert response.json()["error"] == "test_security_bypass_endpoint_required"
    assert engine.started_with is None


def test_localhost_test_entry_allows_explicit_custom_model_bypass() -> None:
    engine = _Engine()
    app = create_app(
        config_path=DEFAULT_CONFIG_PATH,
        engine=engine,
        model_security=_Security(),
        bind_host="127.0.0.1",
    )
    custom_model = {
        "enabled": True,
        "path": "D:/tmp/test-only.pt",
        "backend": "pytorch",
        "model_family": "ultralytics",
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/test/start",
            json={
                "source_type": "file",
                "source": "unused.mp4",
                "profile": "desktop_rtx",
                "custom_model": custom_model,
                "test_bypass_model_security": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["model_security"]["admission_status"] == "bypassed_for_test"
    assert engine.started_with["custom_model"] == custom_model
    assert engine.started_with["allow_test_custom_model"] is True


def test_ordinary_custom_model_uses_production_start_after_security_admission() -> None:
    engine = _Engine()
    app = create_app(
        config_path=DEFAULT_CONFIG_PATH,
        engine=engine,
        model_security=_Security(),
        bind_host="127.0.0.1",
    )
    custom_model = {
        "enabled": True,
        "path": "D:/tmp/test-only.pt",
        "backend": "pytorch",
        "model_family": "ultralytics",
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/start",
            json={
                "source_type": "file",
                "source": "unused.mp4",
                "profile": "desktop_rtx",
                "custom_model": custom_model,
            },
        )

    assert response.status_code == 200
    assert response.json()["model_security"]["status"] == "trusted"
    assert engine.started_with["custom_model"] == custom_model


def test_test_custom_model_override_only_changes_effective_config(tmp_path) -> None:
    model_path = tmp_path / "test-only.pt"
    model_path.write_bytes(b"test-only")

    config = load_runtime_config(
        config_path=DEFAULT_CONFIG_PATH,
        profile="desktop_rtx",
        custom_model={
            "enabled": True,
            "path": str(model_path),
            "backend": "pytorch",
            "model_family": "ultralytics",
        },
        allow_test_custom_model=True,
    )

    assert config["runtime"]["production_unique_model"] is False
    assert config["runtime"]["test_custom_model_bypass"] is True
    assert config["runtime"]["custom_model"]["enabled"] is True
    assert config["inference"]["artifacts"]["pytorch"] == [str(model_path)]

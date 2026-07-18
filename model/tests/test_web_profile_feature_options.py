from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from defense.web.fastapi_app import create_app


INDEX = Path(__file__).resolve().parents[1] / "src/defense/web/static/index.html"


class FakeModelSecurity:
    def prepare_runtime_for_start(
        self,
        *,
        profile: str,
        custom_model: dict,
        auto_remediate: bool,
    ) -> dict:
        assert auto_remediate is False
        return {
            "allowed": True,
            "custom_model": custom_model,
            "model_security": {
                "allowed": True,
                "admission_status": "trusted",
                "profile": profile,
            },
        }


class RecordingEngine:
    def __init__(self) -> None:
        self.run_id = 0
        self.start_calls: list[dict] = []
        self.stop_calls = 0

    def start(self, **kwargs) -> int:  # type: ignore[no-untyped-def]
        self.run_id += 1
        self.start_calls.append(kwargs)
        return self.run_id

    def wait_ready_for_preview(self, run_id: int, timeout: float) -> dict:
        return {
            "run_id": run_id,
            "running": True,
            "ready_for_preview": True,
            "display_options": {},
            "feature_options": self.start_calls[-1]["feature_options"] or {},
        }

    def get_status(self) -> dict:
        return {
            "run_id": self.run_id,
            "running": False,
            "display_options": {},
        }

    def stop(self) -> None:
        self.stop_calls += 1


@pytest.mark.parametrize("feature_options", [None, {}])
def test_start_passes_none_without_explicit_feature_overrides(
    feature_options: dict | None,
) -> None:
    engine = RecordingEngine()
    client = TestClient(
        create_app(engine=engine, model_security=FakeModelSecurity())
    )
    request_body: dict[str, object] = {
        "source_type": "camera",
        "source": "0",
        "profile": "desktop_rtx",
    }
    if feature_options is not None:
        request_body["feature_options"] = feature_options

    response = client.post(
        "/api/start",
        json=request_body,
    )

    assert response.status_code == 200
    assert engine.start_calls[-1]["feature_options"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"static_image_enabled": False},
        {"a3b_sensitivity": "sensitive"},
        {
            "static_image_enabled": False,
            "a3b_sensitivity": "sensitive",
        },
    ],
)
def test_start_preserves_explicit_feature_option_overrides(
    overrides: dict,
) -> None:
    engine = RecordingEngine()
    client = TestClient(
        create_app(engine=engine, model_security=FakeModelSecurity())
    )

    response = client.post(
        "/api/start",
        json={
            "source_type": "camera",
            "source": "0",
            "profile": "desktop_rtx",
            "feature_options": overrides,
        },
    )

    assert response.status_code == 200
    assert engine.start_calls[-1]["feature_options"] == overrides


def test_lifespan_stops_engine_from_finally_on_error() -> None:
    engine = RecordingEngine()
    app = create_app(engine=engine, model_security=FakeModelSecurity())

    async def exercise_lifespan() -> None:
        with pytest.raises(RuntimeError, match="lifespan failure"):
            async with app.router.lifespan_context(app):
                raise RuntimeError("lifespan failure")

    asyncio.run(exercise_lifespan())

    assert engine.stop_calls == 1


def test_frontend_only_sends_feature_options_after_explicit_user_input() -> None:
    source = INDEX.read_text(encoding="utf-8")
    feature_start = source.index("function featureOptions")
    feature_end = source.index("const a3bSensitivityLevels", feature_start)
    feature_source = source[feature_start:feature_end]
    start_start = source.index('    $("startBtn").onclick')
    start_end = source.index('    $("stopBtn").onclick', start_start)
    start_source = source[start_start:start_end]

    assert "if (featureOptionOverrides.static_image_enabled)" in feature_source
    assert "if (featureOptionOverrides.a3b_sensitivity)" in feature_source
    assert "feature_options: featureOptions()" not in start_source
    assert "if (Object.keys(featureOverrides).length)" in start_source
    assert "body.feature_options = featureOverrides;" in start_source


def test_frontend_marks_only_checkbox_and_slider_input_as_overrides() -> None:
    source = INDEX.read_text(encoding="utf-8")
    handlers_start = source.index('    $("enableStaticImage").onchange')
    handlers_end = source.index('    $("startBtn").onclick', handlers_start)
    handlers_source = source[handlers_start:handlers_end]

    assert "featureOptionOverrides.static_image_enabled = true;" in handlers_source
    assert "featureOptionOverrides.a3b_sensitivity = true;" in handlers_source


def test_frontend_prefers_effective_config_without_overwriting_user_input() -> None:
    source = INDEX.read_text(encoding="utf-8")
    apply_start = source.index("function applyEffectiveFeatureOptions")
    apply_end = source.index("function updateStartButtonState", apply_start)
    apply_source = source[apply_start:apply_end]

    assert "status?.module_a_effective_config || {}" in apply_source
    assert "if (!featureOptionOverrides.static_image_enabled)" in apply_source
    assert "if (!featureOptionOverrides.a3b_sensitivity)" in apply_source
    assert "effectiveConfig.a3b_sensitivity || featureStatus.a3b_sensitivity" in apply_source
    assert "hasEffectiveSensitivity && effectiveConfig.detector_impl" in apply_source
    assert "setA3bSensitivityCustom(effectiveConfig)" in apply_source


def test_frontend_routes_custom_model_by_persisted_bypass_switch() -> None:
    source = INDEX.read_text(encoding="utf-8")

    assert 'id="runtimeModelSummary"' in source
    assert 'id="runtimeModelIdentity"' in source
    assert 'id="resetCustomModelBtn"' not in source
    assert "function runtimeModelSelectionSummary" in source
    assert "默认使用 mask_bd_v4_clean_baseline.pt 及其绑定的 TensorRT FP16" in source
    assert "在安全中心开启后可选择其他模型，并由B模块执行安全准入" in source
    assert 'localStorage.getItem("moduleA.lastCustomModelEnabled")' in source
    assert "自定义运行模型" in source
    assert "启动前由B模块执行安全准入" in source
    assert 'const startEndpoint = bypassModelSecurity ? "/api/test/start" : "/api/start";' in source
    assert "body.test_bypass_model_security = true;" in source

from __future__ import annotations

from fastapi.testclient import TestClient

from defense.web.fastapi_app import _is_benign_windows_disconnect, create_app


class DummyEngine:
    def get_status(self) -> dict:
        return {"run_id": 0, "running": False, "display_options": {}}


def test_index_disables_browser_cache() -> None:
    client = TestClient(create_app(engine=DummyEngine()))

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["Pragma"] == "no-cache"


def test_static_assets_disable_browser_cache() -> None:
    client = TestClient(create_app(engine=DummyEngine()))

    response = client.get("/static/overlay_timeline.js")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"


def test_windows_proactor_client_reset_is_treated_as_benign() -> None:
    class WinReset(ConnectionResetError):
        @property
        def winerror(self) -> int:
            return 10054

    context = {
        "exception": WinReset("client disconnected"),
        "handle": "<Handle _ProactorBasePipeTransport._call_connection_lost(None)>",
        "message": "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
    }

    assert _is_benign_windows_disconnect(context) is True
    assert _is_benign_windows_disconnect({"exception": RuntimeError("real bug")}) is False

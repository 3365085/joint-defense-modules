from __future__ import annotations

from typing import Any


def require_keys(payload: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise AssertionError(f"{name} missing keys: {missing}")


def validate_status_response(payload: dict[str, Any]) -> None:
    require_keys(payload, ("ok", "status"), "status_response")
    if isinstance(payload.get("status"), dict):
        payload["status"].setdefault("running", False)


def validate_control_response(payload: dict[str, Any]) -> None:
    require_keys(payload, ("ok", "status"), "control_response")


def validate_overlay_response(payload: dict[str, Any]) -> None:
    require_keys(payload, ("ok", "overlay"), "overlay_response")
    overlay = payload.get("overlay") or {}
    if isinstance(overlay, dict):
        overlay.setdefault("records", [])
        overlay.setdefault("latest_seq", 0)


def websocket_status_payload(status: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return {"type": "status", "status": status, "overlay": overlay}


def websocket_completed_payload(status: dict[str, Any]) -> dict[str, Any]:
    return {"type": "completed", "status": status}

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from defense.runtime import MonitorEngine, PipelineCache, list_profiles, sample_sources, scan_camera_devices
from defense.runtime.config import DEFAULT_CONFIG_PATH, project_root
from defense.model_security import ModelSecurityService

from .contracts import websocket_completed_payload, websocket_status_payload
from .helpers import STATIC_DIR, enrich_status, json_default, normalize_profile, pick_file_dialog, test_source_connectivity
from .security import SecurityPolicy, require_http_access, require_ws_access, safe_current_media_path


def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=json.loads(json.dumps(data, ensure_ascii=False, default=json_default)),
        status_code=status_code,
    )


def _no_cache_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _engine(app: FastAPI) -> MonitorEngine:
    return app.state.engine


def _model_security(app: FastAPI) -> ModelSecurityService:
    return app.state.model_security


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def create_app(
    *,
    config_path: str | Path | None = None,
    engine: MonitorEngine | None = None,
    bind_host: str = "127.0.0.1",
) -> FastAPI:
    config = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    app = FastAPI(title="Module A Monitor", docs_url=None, redoc_url=None)
    app.state.config_path = config
    app.state.bind_host = bind_host
    app.state.security_policy = SecurityPolicy.from_env(bind_host)
    app.state.engine = engine or MonitorEngine(PipelineCache(config_path=config, root=project_root()))
    app.state.model_security = ModelSecurityService(config_path=config, root=project_root())

    @app.get("/")
    @app.get("/index.html")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8", headers=_no_cache_headers())

    @app.get("/static/{path:path}")
    async def static_file(path: str) -> FileResponse:
        target = (STATIC_DIR / path).resolve()
        root = STATIC_DIR.resolve()
        if root not in target.parents and target != root:
            raise HTTPException(status_code=403, detail="forbidden")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="static_not_found")
        media_type = "application/javascript" if target.suffix == ".js" else None
        if target.suffix == ".css":
            media_type = "text/css"
        return FileResponse(target, media_type=media_type, headers=_no_cache_headers())

    @app.get("/api/status")
    @app.get("/api/runs/current")
    async def status(request: Request) -> JSONResponse:
        status_payload = enrich_status(_engine(request.app).get_status())
        try:
            status_payload["model_security"] = _model_security(request.app).status(profile=str(status_payload.get("profile") or "default"), custom_model=status_payload.get("custom_model") or {})
        except Exception as exc:
            status_payload["model_security"] = {"status": "error", "error": str(exc)}
        return _json({"ok": True, "status": status_payload})

    @app.post("/api/runs/start")
    @app.post("/api/start")
    async def start(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        # Module B startup path: hash/fingerprint check only, then optional
        # background scan for unknown models. This never enters the per-frame
        # preview/detection hot path.
        try:
            ms = _model_security(request.app)
            ms_status = ms.status(
                profile=normalize_profile(payload.get("profile", "default")),
                custom_model=payload.get("custom_model") or {},
            )
            if ms_status.get("status") == "unknown":
                ms.start_background_scan(
                    scan_type="quick",
                    profile=normalize_profile(payload.get("profile", "default")),
                    custom_model=payload.get("custom_model") or {},
                )
        except Exception:
            pass
        engine = _engine(request.app)
        run_id = engine.start(
            source_type=str(payload.get("source_type", "file")),
            source=str(payload.get("source", "")),
            profile=normalize_profile(payload.get("profile", "default")),
            realtime=bool(payload.get("realtime", True)),
            feature_options=payload.get("feature_options") or {},
            custom_model=payload.get("custom_model") or {},
        )
        timeout = float(payload.get("ready_timeout_s", 45.0) or 45.0)
        status_payload = engine.wait_ready_for_preview(run_id, timeout=timeout)
        return _json({"ok": True, "run_id": run_id, "status": enrich_status(status_payload)})

    @app.post("/api/runs/{run_id}/stop")
    @app.post("/api/stop")
    async def stop(request: Request, run_id: int | None = None) -> JSONResponse:
        require_http_access(request)
        engine = _engine(request.app)
        if run_id is not None and int(run_id) != int(engine.run_id):
            raise HTTPException(status_code=409, detail="run_id does not match current run")
        engine.stop()
        return _json({"ok": True, "status": enrich_status(engine.get_status())})

    @app.post("/api/runs/{run_id}/control")
    async def control(request: Request, run_id: int) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        action = str(payload.get("action") or payload.get("command") or "")
        payload.pop("action", None)
        payload.pop("command", None)
        status_payload = _engine(request.app).control_run(run_id, action, **payload)
        return _json({"ok": True, "status": enrich_status(status_payload)})

    @app.get("/api/runs/{run_id}/overlay")
    @app.get("/api/overlay")
    async def overlay(request: Request, run_id: int | None = None, since_seq: int = 0) -> JSONResponse:
        engine = _engine(request.app)
        if run_id is not None and int(run_id) != int(engine.run_id):
            return _json({"ok": True, "overlay": {"run_id": engine.run_id, "records": [], "latest_seq": 0}})
        return _json({"ok": True, "overlay": engine.get_overlay(since_seq)})

    def mjpeg_generator(engine: MonitorEngine, run_id: int):
        last_seq = 0
        while int(run_id) == int(engine.run_id):
            seq, jpeg, running = engine.wait_latest_jpeg(last_seq, timeout=0.8)
            if jpeg is None:
                if not running:
                    break
                continue
            last_seq = seq
            yield b"--frame\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
            yield jpeg
            yield b"\r\n"

    @app.get("/api/runs/{run_id}/preview.mjpg")
    async def run_preview(request: Request, run_id: int) -> StreamingResponse:
        require_http_access(request)
        engine = _engine(request.app)
        if int(run_id) != int(engine.run_id):
            raise HTTPException(status_code=404, detail="run not found")
        return StreamingResponse(
            mjpeg_generator(engine, run_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, private", "Pragma": "no-cache"},
        )

    @app.get("/stream.mjpg")
    async def legacy_stream(request: Request) -> StreamingResponse:
        require_http_access(request)
        engine = _engine(request.app)
        return StreamingResponse(
            mjpeg_generator(engine, int(engine.run_id)),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, private", "Pragma": "no-cache"},
        )

    @app.get("/api/profiles")
    async def profiles(request: Request) -> JSONResponse:
        return _json({"ok": True, "profiles": list_profiles(request.app.state.config_path)})

    @app.get("/api/samples")
    async def samples() -> JSONResponse:
        return _json({"ok": True, "samples": sample_sources()})

    @app.get("/api/cameras")
    async def cameras() -> JSONResponse:
        devices = scan_camera_devices(8)
        for device in devices:
            device.setdefault("name", device.get("label", f"Camera {device.get('index', '')}"))
        return _json({"ok": True, "devices": devices})

    @app.post("/api/display-options")
    async def display_options(request: Request) -> JSONResponse:
        payload = await _body(request)
        options = payload.get("display_options") if isinstance(payload.get("display_options"), dict) else payload
        display = _engine(request.app).update_display_options(options or {})
        return _json({"ok": True, "display_options": display})

    @app.post("/api/pick-file")
    async def pick_file(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        return _json(pick_file_dialog(str(payload.get("mode", "video")), str(payload.get("current_path", ""))))

    @app.post("/api/probe")
    @app.post("/api/test-source")
    async def test_source(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        return _json(test_source_connectivity(str(payload.get("source_type", "file")), str(payload.get("source", ""))))

    @app.get("/media/current")
    async def current_media(request: Request) -> Any:
        require_http_access(request)
        status_payload = _engine(request.app).get_status()
        path = safe_current_media_path(status_payload)
        if path is None:
            return PlainTextResponse("current media not available", status_code=404)
        return FileResponse(path)


    @app.get("/api/model-security/status")
    async def model_security_status(request: Request, profile: str = "default") -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "model_security": _model_security(request.app).status(profile=normalize_profile(profile))})

    @app.post("/api/model-security/scan")
    async def model_security_scan(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        scan_type = str(payload.get("scan_type", "quick")).lower()
        if scan_type not in {"quick", "full"}:
            scan_type = "quick"
        background = bool(payload.get("background", True))
        profile = normalize_profile(payload.get("profile", "default"))
        custom_model = payload.get("custom_model") or {}
        if background:
            result = _model_security(request.app).start_background_scan(scan_type=scan_type, profile=profile, custom_model=custom_model)
            return _json({"ok": True, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model), "scan": result})
        report = _model_security(request.app).scan(scan_type=scan_type, profile=profile, custom_model=custom_model, trust_if_low_risk=bool(payload.get("trust_if_low_risk", False)))
        return _json({"ok": True, "report": report, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model)})

    @app.post("/api/model-security/scan/stop")
    async def model_security_scan_stop(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "scan": _model_security(request.app).stop_scan()})

    @app.get("/api/model-security/report")
    async def model_security_report(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "report": _model_security(request.app).latest_report()})

    @app.post("/api/model-security/trust")
    async def model_security_trust(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        rec = _model_security(request.app).trust_current(
            profile=normalize_profile(payload.get("profile", "default")),
            custom_model=payload.get("custom_model") or {},
            notes=str(payload.get("notes", "manual approval")),
        )
        return _json({"ok": True, "trust_record": rec, "model_security": _model_security(request.app).status()})

    @app.websocket("/ws/runs/{run_id}")
    async def run_socket(websocket: WebSocket, run_id: int) -> None:
        if not await require_ws_access(websocket):
            return
        await websocket.accept()
        engine = _engine(websocket.app)
        since_seq = 0
        try:
            while int(run_id) == int(engine.run_id):
                status_payload = enrich_status(engine.get_status())
                overlay_payload = engine.get_overlay(since_seq)
                since_seq = int(overlay_payload.get("latest_seq") or since_seq)
                await websocket.send_json(websocket_status_payload(status_payload, overlay_payload))
                if not status_payload.get("running"):
                    await websocket.send_json(websocket_completed_payload(status_payload))
                    break
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return

    return app

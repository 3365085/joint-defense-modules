from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from defense.runtime import MonitorEngine, PipelineCache, list_profiles, sample_sources, scan_camera_devices
from defense.runtime.config import DEFAULT_CONFIG_PATH, project_root
from defense.model_security import ModelSecurityService

from .contracts import websocket_completed_payload, websocket_status_payload
from .helpers import STATIC_DIR, enrich_status, jsonable, normalize_profile, pick_file_dialog, test_source_connectivity
from .security import SecurityPolicy, require_http_access, require_ws_access, safe_current_media_path


def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=jsonable(data),
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


def _is_benign_windows_disconnect(context: dict[str, Any]) -> bool:
    exc = context.get("exception")
    if not isinstance(exc, ConnectionResetError):
        return False
    if getattr(exc, "winerror", None) != 10054:
        return False
    handle_text = str(context.get("handle", ""))
    message = str(context.get("message", ""))
    return (
        "_ProactorBasePipeTransport._call_connection_lost" in handle_text
        or "_call_connection_lost" in handle_text
        or "connection_lost" in message
    )


def _install_asyncio_exception_filter() -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if _is_benign_windows_disconnect(context):
            return
        if previous_handler is not None:
            previous_handler(loop, context)
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    del app
    _install_asyncio_exception_filter()
    yield


async def _body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _start_blocked_payload(
    admission: dict[str, Any],
    *,
    scan_result: dict[str, Any] | None = None,
    purification_result: dict[str, Any] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "model_security_blocked",
        "message": message or "B模块模型安全准入未通过，已阻止A模块启动。",
        "model_security": admission,
        "scan": scan_result,
        "purification": purification_result,
    }


def _purified_runtime_from_service(
    service: ModelSecurityService,
    *,
    profile: str,
    custom_model: dict[str, Any],
) -> dict[str, Any] | None:
    resolver = getattr(service, "trusted_purified_runtime_model", None)
    if not callable(resolver):
        return None
    try:
        return resolver(profile=profile, custom_model=custom_model)
    except Exception:
        return None


def _resolve_model_security_start(
    service: ModelSecurityService,
    *,
    profile: str,
    custom_model: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    preparer = getattr(service, "prepare_runtime_for_start", None)
    if callable(preparer):
        prepared = preparer(profile=profile, custom_model=custom_model, auto_remediate=True)
        if bool(prepared.get("allowed", False)) and prepared.get("custom_model") is not None:
            return prepared["custom_model"], prepared.get("model_security", {}), {
                "scan": prepared.get("scan"),
                "purification": prepared.get("purification"),
                "runtime_replacement": prepared.get("runtime_replacement"),
            }
        return None, prepared.get("model_security", {}), {
            "scan": prepared.get("scan"),
            "purification": prepared.get("purification"),
            "runtime_replacement": prepared.get("runtime_replacement"),
        }

    admission = service.ensure_admitted(profile=profile, custom_model=custom_model)
    if bool(admission.get("allowed", False)):
        return custom_model, admission, None

    status = str(admission.get("admission_status") or admission.get("status") or "")
    if status == "purified_alternative_available":
        replacement = _purified_runtime_from_service(service, profile=profile, custom_model=custom_model)
        if replacement and replacement.get("custom_model") and replacement.get("model_security", {}).get("allowed"):
            model_security = dict(replacement["model_security"])
            model_security["runtime_replacement"] = {
                "enabled": True,
                "path": replacement["custom_model"].get("path"),
                "backend": replacement["custom_model"].get("backend"),
                "model_family": replacement["custom_model"].get("model_family"),
                "source_pt_path": replacement["custom_model"].get("source_pt_path"),
            }
            return replacement["custom_model"], model_security, {
                "scan": None,
                "purification": None,
                "runtime_replacement": {
                    "mode": "purified_runtime",
                    "source_model_security": replacement.get("source_model_security") or admission,
                },
            }

    return None, admission, None


def create_app(
    *,
    config_path: str | Path | None = None,
    engine: MonitorEngine | None = None,
    model_security: ModelSecurityService | None = None,
    bind_host: str = "127.0.0.1",
) -> FastAPI:
    config = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    app = FastAPI(title="Module A Monitor", docs_url=None, redoc_url=None, lifespan=_app_lifespan)
    app.state.config_path = config
    app.state.bind_host = bind_host
    app.state.security_policy = SecurityPolicy.from_env(bind_host)
    app.state.engine = engine or MonitorEngine(PipelineCache(config_path=config, root=project_root()))
    app.state.model_security = model_security or ModelSecurityService(config_path=config, root=project_root())

    @app.get("/")
    @app.get("/index.html")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8", headers=_no_cache_headers())

    @app.get("/model-security/logs")
    async def model_security_logs_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "model_security_logs.html", media_type="text/html; charset=utf-8", headers=_no_cache_headers())

    @app.get("/model-security")
    async def model_security_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "model_security.html", media_type="text/html; charset=utf-8", headers=_no_cache_headers())

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
        # Keep the high-frequency runtime status endpoint on the preview hot
        # path lightweight. Model-security fingerprinting can touch large model
        # artifacts, so the UI fetches it through /api/model-security/status.
        return _json({"ok": True, "status": enrich_status(_engine(request.app).get_status())})

    @app.post("/api/runs/start")
    @app.post("/api/start")
    async def start(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        profile = normalize_profile(payload.get("profile", "default"))
        custom_model = payload.get("custom_model") or {}
        model_security_service = _model_security(request.app)
        runtime_custom_model, admission, runtime_replacement = _resolve_model_security_start(
            model_security_service,
            profile=profile,
            custom_model=custom_model,
        )
        scan_result = runtime_replacement.get("scan") if isinstance(runtime_replacement, dict) else None
        purification_result = runtime_replacement.get("purification") if isinstance(runtime_replacement, dict) else None
        start_runtime_replacement = runtime_replacement.get("runtime_replacement") if isinstance(runtime_replacement, dict) else runtime_replacement
        if runtime_custom_model is None:
            admission_status = admission.get("admission_status")
            if admission_status == "blocked_scan_required" and scan_result is None:
                scan_result = model_security_service.start_background_scan(
                    scan_type="full",
                    profile=profile,
                    custom_model=custom_model,
                    auto_purify=True,
                )
                admission = model_security_service.status(profile=profile, custom_model=custom_model)
            elif admission_status == "suspicious" and purification_result is None:
                purification_result = model_security_service.start_background_purification(
                    profile=profile,
                    custom_model=custom_model,
                    scan_after=True,
                )
                admission = model_security_service.status(profile=profile, custom_model=custom_model)
            return _json(_start_blocked_payload(admission, scan_result=scan_result, purification_result=purification_result), status_code=409)
        engine = _engine(request.app)
        run_id = engine.start(
            source_type=str(payload.get("source_type", "file")),
            source=str(payload.get("source", "")),
            profile=profile,
            realtime=bool(payload.get("realtime", True)),
            feature_options=payload.get("feature_options") or {},
            custom_model=runtime_custom_model,
        )
        timeout = float(payload.get("ready_timeout_s", 45.0) or 45.0)
        status_payload = engine.wait_ready_for_preview(run_id, timeout=timeout)
        return _json(
            {
                "ok": True,
                "run_id": run_id,
                "status": enrich_status(status_payload),
                "model_security": admission,
                "model_security_runtime_replacement": start_runtime_replacement,
            }
        )

    @app.post("/api/runs/{run_id}/stop")
    @app.post("/api/stop")
    async def stop(request: Request, run_id: int | None = None) -> JSONResponse:
        require_http_access(request)
        engine = _engine(request.app)
        if run_id is not None and int(run_id) != int(engine.run_id):
            raise HTTPException(status_code=409, detail="run_id does not match current run")
        engine.stop(release_pipeline_cache=False)
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

    @app.post("/api/model-security/status")
    async def model_security_status_post(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        profile = normalize_profile(payload.get("profile", "default"))
        custom_model = payload.get("custom_model") or {}
        return _json(
            {
                "ok": True,
                "model_security": _model_security(request.app).status(
                    profile=profile,
                    custom_model=custom_model,
                ),
            }
        )

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
        auto_purify = bool(payload.get("auto_purify", scan_type == "full"))
        if background:
            result = _model_security(request.app).start_background_scan(
                scan_type=scan_type,
                profile=profile,
                custom_model=custom_model,
                auto_purify=auto_purify,
            )
            return _json({"ok": True, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model), "scan": result})
        report = _model_security(request.app).scan(scan_type=scan_type, profile=profile, custom_model=custom_model, trust_if_low_risk=bool(payload.get("trust_if_low_risk", False)))
        return _json({"ok": True, "report": report, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model)})

    @app.post("/api/model-security/scan/stop")
    async def model_security_scan_stop(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "scan": _model_security(request.app).stop_scan()})

    @app.post("/api/model-security/purify")
    async def model_security_purify(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        background = bool(payload.get("background", True))
        scan_after = bool(payload.get("scan_after", True))
        profile = normalize_profile(payload.get("profile", "default"))
        custom_model = payload.get("custom_model") or {}
        current_security = _model_security(request.app).status(profile=profile, custom_model=custom_model)
        if current_security.get("admission_status") != "suspicious":
            return _json(
                {
                    "ok": False,
                    "error": "purification_requires_suspicious_full_scan",
                    "model_security": current_security,
                },
                status_code=409,
            )
        if background:
            result = _model_security(request.app).start_background_purification(
                profile=profile,
                custom_model=custom_model,
                scan_after=scan_after,
            )
            return _json({"ok": True, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model), "purification": result})
        try:
            report = _model_security(request.app).purify(profile=profile, custom_model=custom_model, scan_after=scan_after)
        except ValueError as exc:
            return _json(
                {
                    "ok": False,
                    "error": str(exc),
                    "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model),
                },
                status_code=409,
            )
        return _json({"ok": True, "report": report, "model_security": _model_security(request.app).status(profile=profile, custom_model=custom_model)})

    @app.get("/api/model-security/report")
    async def model_security_report(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "report": _model_security(request.app).latest_report()})

    @app.get("/api/model-security/purification-report")
    async def model_security_purification_report(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "report": _model_security(request.app).latest_purification_report()})

    @app.get("/api/model-security/logs")
    async def model_security_logs(request: Request, limit: int = 80) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "logs": _model_security(request.app).recent_logs(limit=limit)})

    @app.post("/api/model-security/trust")
    async def model_security_trust(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json(
            {
                "ok": False,
                "error": "manual_trust_disabled",
                "message": "白名单只允许由原版 PT 完整扫描通过或净化模型复扫通过后自动写入；用户只能删除白名单记录，不能手工新增。",
            },
            status_code=403,
        )

    @app.get("/api/model-security/trust")
    async def model_security_trust_list(request: Request) -> JSONResponse:
        require_http_access(request)
        return _json({"ok": True, "trust": _model_security(request.app).trust_records()})

    @app.post("/api/model-security/trust/delete")
    async def model_security_trust_delete(request: Request) -> JSONResponse:
        require_http_access(request)
        payload = await _body(request)
        try:
            result = _model_security(request.app).delete_trust(str(payload.get("fingerprint", "")))
        except ValueError as exc:
            code = 409 if str(exc).startswith("trust_store_compromised") else 400
            return _json({"ok": False, "error": str(exc)}, status_code=code)
        return _json({"ok": True, "trust": result, "model_security": _model_security(request.app).status()})

    @app.post("/api/model-security/trust/clear")
    async def model_security_trust_clear(request: Request) -> JSONResponse:
        require_http_access(request)
        result = _model_security(request.app).clear_trust()
        return _json({"ok": True, "trust": result, "model_security": _model_security(request.app).status()})

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

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import cv2

from defense.runtime import open_capture, resolve_source_path
from defense.runtime.config import project_root

STATIC_DIR = Path(__file__).resolve().parent / "static"

PROFILE_ALIASES = {
    "full_gpu": "desktop_rtx",
    "balanced_gpu": "desktop_debug_onnx",
    "edge_onnx": "edge_fast",
}


def normalize_profile(value: str | None) -> str:
    value = str(value or "default").strip() or "default"
    return PROFILE_ALIASES.get(value, value)


def json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _sanitize_json_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _sanitize_json_value(value.item())
        if isinstance(value, np.ndarray):
            return [_sanitize_json_value(item) for item in value.tolist()]
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def jsonable(data: Any) -> Any:
    sanitized = _sanitize_json_value(data)
    return json.loads(json.dumps(sanitized, ensure_ascii=False, default=json_default, allow_nan=False))


def _file_response_payload(path: Path, *, source_type: str = "file") -> dict[str, Any]:
    return {"source_type": source_type, "source": str(path), "path": str(path)}


def pick_file_dialog(mode: str, current_path: str = "") -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return {"ok": True, "path": current_path or "", "message": "File picker unavailable; enter a path manually."}

    result: dict[str, Any] = {"ok": True, "path": current_path or ""}
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        initialdir = str(resolve_source_path(current_path).parent) if current_path else str(project_root())
        if mode == "model":
            filetypes = [("Model files", "*.engine *.onnx *.pt *.pth"), ("All files", "*.*")]
        else:
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        chosen = filedialog.askopenfilename(initialdir=initialdir, filetypes=filetypes)
        if chosen:
            result.update(_file_response_payload(Path(chosen), source_type="file"))
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return result


def test_source_connectivity(source_type: str, source: str) -> dict[str, Any]:
    cap = None
    try:
        cap = open_capture(source_type, source, timeout_ms=3000)
        ok = False
        frame = None
        attempts = 15 if str(source_type or "").lower() in {"camera", "rtsp"} else 1
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            threading.Event().wait(0.08)
        if not ok or frame is None:
            return {"ok": True, "reachable": False, "message": "Source opened but no valid frame was read."}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return {
            "ok": True,
            "reachable": True,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "fps": fps,
            "message": f"Source is reachable: {frame.shape[1]}x{frame.shape[0]}, FPS={fps:.1f}",
        }
    except Exception as exc:
        return {"ok": True, "reachable": False, "error": str(exc), "message": str(exc)}
    finally:
        if cap is not None:
            cap.release()


def enrich_status(status: dict[str, Any]) -> dict[str, Any]:
    out = dict(status)
    recent_events = list(out.get("recent_events") or [])
    recent_ppe_events = list(out.get("recent_ppe_events") or [])
    recent_source_events = list(out.get("recent_source_auth_events") or [])
    out["recent_events"] = recent_events
    out["recent_ppe_events"] = recent_ppe_events
    out["recent_source_auth_events"] = recent_source_events
    out["alert_event_count"] = int(out.get("alert_event_count") or len(recent_events))
    out["ppe_event_count"] = int(out.get("ppe_event_count") or len(recent_ppe_events))
    active_a3b = 1 if out.get("a3b_triggered") else 0
    out["source_event_count"] = int(out.get("source_event_count") or (len(recent_source_events) + active_a3b))
    out.setdefault("realtime", True)
    out.setdefault("video_time_s", 0.0)
    out.setdefault("overlay_seq", out.get("frame_idx", 0))
    return out

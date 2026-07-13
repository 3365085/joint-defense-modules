"""Module A Tuning Tool — offline per-frame signal visualizer with tunable thresholds.

Run from repo root:
    pixi run python model/tools/module_a_tuning_app.py

Open http://127.0.0.1:8765 in your browser.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# ── bootstrap ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from defense.module_a import ModuleADetector  # noqa: E402
from defense.module_a.types import ModuleAInput  # noqa: E402
from defense.runtime.config import load_runtime_config  # noqa: E402

# ── overlay font (Chinese-capable) ────────────────────────────────────────
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)
_FONT_CACHE: dict[int, Any] = {}


def _get_font(size: int):
    if size not in _FONT_CACHE:
        font = None
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def _draw_overlay(frame: np.ndarray, fd: dict[str, Any]) -> np.ndarray:
    """Draw alert/signal HUD onto a BGR frame, return annotated BGR frame."""
    alert = bool(fd.get("alert_confirmed"))
    susp = bool(fd.get("suspicious"))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Status banner
    if alert:
        banner_color = (220, 40, 60, 220)
        status_text = "⚠ 告警 ALERT"
    elif susp:
        banner_color = (230, 170, 40, 200)
        status_text = "可疑 SUSPICIOUS"
    else:
        banner_color = (40, 160, 90, 160)
        status_text = "正常 NORMAL"

    font_big = _get_font(max(20, h // 22))
    font_small = _get_font(max(14, h // 36))
    bh = int(h * 0.085)
    draw.rectangle([0, 0, w, bh], fill=banner_color)
    draw.text((12, bh // 2 - font_big.size // 2), f"帧 {fd.get('frame_idx', 0)}  {status_text}",
              font=font_big, fill=(255, 255, 255, 255))

    # Reason codes (right side of banner)
    reasons = fd.get("reason_codes", []) or []
    if reasons:
        rtext = " | ".join(reasons[:3])
        draw.text((12, bh + 6), rtext, font=font_small, fill=(255, 230, 120, 255))

    # Signal readout (bottom-left)
    sigs = [
        ("motion", fd.get("motion_score", 0)),
        ("flow_loc", fd.get("flow_local_ratio", 0)),
        ("temporal", fd.get("temporal_local_max", 0)),
        ("blur", fd.get("blur_score", 0)),
        ("overexp", fd.get("overexposure_ratio", 0)),
        ("flash", fd.get("temporal_flash_ratio", 0)),
    ]
    line = "  ".join(f"{k}={float(v):.2f}" for k, v in sigs)
    ty = h - font_small.size - 10
    draw.rectangle([0, ty - 6, w, h], fill=(0, 0, 0, 140))
    draw.text((10, ty), line, font=font_small, fill=(180, 230, 255, 255))

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return out

# ── attack video registry ────────────────────────────────────────────────
ATTACK_VIDEOS = {
    "glare": "D:/联合防御模块/素材/物理扰动攻击视频/glare/raw_glare_attacked.mp4",
    "motion_blur": "D:/联合防御模块/素材/物理扰动攻击视频/motion_blur/raw_motion_blur_attacked.mp4",
    "occlusion": "D:/联合防御模块/素材/物理扰动攻击视频/occlusion/raw_occlusion_attacked.mp4",
    "visibility": "D:/联合防御模块/素材/物理扰动攻击视频/visibility_degradation/raw_visibility_degradation_attacked.mp4",
    "adv_patch": "D:/联合防御模块/素材/物理扰动攻击视频/adv_patch/raw_adv_patch_attacked.mp4",
    "normal_outdoor": "D:/联合防御模块/素材/手机随意录制的视频/固定镜头室外视频.mp4",
}

# Default tuned thresholds
DEFAULT_TUNING = {
    # A1 overexposure
    "glare_ratio_threshold": 0.06,
    "glare_flash_diff_threshold": 30.0,
    "glare_flash_ratio_threshold": 0.08,
    # A3b
    "a3b_high_score_bypass_threshold": 0.70,
    # A4 target_anchored
    "flow_local_anomaly_threshold": 0.68,
    "roi_blur_threshold": 0.60,
    "track_drop_threshold": 0.40,
    "paired_temporal_motion_threshold": 0.18,
    "motion_score_threshold": 0.35,
    # Alert state
    "alert_window": 7,
    "alert_trigger_count": 3,
    "attack_state_hold_frames": 4,
}

# Tunable config keys → human-readable names and ranges
TUNING_PARAMS = {
    # A1
    "glare_ratio_threshold": {"name": "A1 静态过曝阈值", "min": 0.01, "max": 0.30, "step": 0.01, "default": 0.06},
    "glare_flash_diff_threshold": {"name": "A1 闪烁灰度差", "min": 5.0, "max": 80.0, "step": 1.0, "default": 30.0},
    "glare_flash_ratio_threshold": {"name": "A1 闪烁比例阈值", "min": 0.01, "max": 0.30, "step": 0.01, "default": 0.08},
    # A3b
    "a3b_high_score_bypass_threshold": {"name": "A3b 高分旁路阈值", "min": 0.40, "max": 0.90, "step": 0.01, "default": 0.70},
    # A4
    "flow_local_anomaly_threshold": {"name": "A4 flow_local 阈值", "min": 0.40, "max": 0.90, "step": 0.01, "default": 0.68},
    "roi_blur_threshold": {"name": "A4 blur 阈值", "min": 0.10, "max": 1.00, "step": 0.01, "default": 0.60},
    "track_drop_threshold": {"name": "A4 track_drop 阈值", "min": 0.10, "max": 1.00, "step": 0.01, "default": 0.40},
    "paired_temporal_motion_threshold": {"name": "A4 temporal 阈值", "min": 0.05, "max": 0.60, "step": 0.01, "default": 0.18},
    "motion_score_threshold": {"name": "A4 motion_score 阈值", "min": 0.10, "max": 1.00, "step": 0.01, "default": 0.35},
    # Alert
    "alert_window": {"name": "告警窗口大小", "min": 3, "max": 15, "step": 1, "default": 7},
    "alert_trigger_count": {"name": "告警触发帧数", "min": 1, "max": 10, "step": 1, "default": 3},
    "attack_state_hold_frames": {"name": "告警保持帧数", "min": 0, "max": 20, "step": 1, "default": 4},
}


def _build_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Build a module_a config patch from threshold overrides.

    All thresholds live under the 'module_a' key as detector_setup.py reads
    from config.get('module_a', config).
    """
    module_cfg = {}
    for key, value in overrides.items():
        if key in ("alert_window", "alert_trigger_count", "attack_state_hold_frames"):
            module_cfg[key] = value
        elif key.startswith("glare_"):
            module_cfg[key] = value
        elif key.startswith("a3b_"):
            module_cfg[key] = value
        elif key.startswith("flow_local_"):
            module_cfg[key] = value
        elif key.startswith("roi_") or key.startswith("track_") or key.startswith("paired_") or key.startswith("motion_"):
            module_cfg[key] = value
    return {"module_a": module_cfg}


def run_video(
    video_path: str,
    tuning: dict[str, Any],
    max_frames: int = 0,
) -> dict[str, Any]:
    """Run Module A on a video with threshold overrides, return per-frame signals."""
    config_patch = _build_config(tuning)
    cfg = load_runtime_config(profile="desktop_rtx", feature_options={})
    # Apply tuning overrides
    for key, value in config_patch.items():
        cfg[key] = value

    detector = ModuleADetector(config=cfg)
    detector.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": f"Cannot open: {video_path}"}

    frames: list[dict[str, Any]] = []
    alert_count = 0
    suspicious_count = 0
    frame_idx = 0
    started = time.perf_counter()

    # Cache overlay frames globally for the HTTP server
    global _cached_frames
    _cached_frames.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        item = ModuleAInput(frame=frame, frame_idx=frame_idx)
        result = detector.process(item)

        frame_data = {
            "frame_idx": frame_idx,
            "alert_confirmed": bool(result.alert_confirmed),
            "attack_state_active": bool(result.attack_state_active),
            "p_adv": float(result.p_adv),
            "p_media": float(result.features.get("static_media", {}).get("p_media", 0)),
            # A1
            "overexposure_ratio": float(result.features.get("overexposure", {}).get("ratio", 0)),
            "is_glare": bool(result.features.get("overexposure", {}).get("is_glare", False)),
            "static_glare": bool(result.features.get("overexposure", {}).get("static_glare", False)),
            "temporal_flash": bool(result.features.get("overexposure", {}).get("temporal_flash", False)),
            "temporal_flash_ratio": float(result.features.get("overexposure", {}).get("temporal_flash_ratio", 0)),
            # A2
            "temporal_local_max": float(result.features.get("temporal", {}).get("local_max", 0)),
            "temporal_change": float(result.features.get("temporal", {}).get("change_rate", 0)),
            # A3
            "motion_score": float(result.features.get("flow", {}).get("motion_score", 0)),
            "flow_local_ratio": float(result.features.get("flow", {}).get("local_max_ratio", 0)),
            "blur_score": float(result.features.get("blur", {}).get("score", 0)),
            "track_score": float(result.features.get("track", {}).get("score", 0)),
            "confidence_drop_score": float(result.features.get("track", {}).get("confidence_drop_score", 0)),
            "light_flow_score": float(result.features.get("flow", {}).get("light_flow_score", 0)),
            # A3b
            "a3b_triggered": bool(result.features.get("static_media", {}).get("triggered", False)),
            "a3b_score": float(result.features.get("static_media", {}).get("score", 0)),
            # A4
            "suspicious": bool(result.single_frame_suspicious),
            "reason_codes": list(result.reason_codes),
        }
        frames.append(frame_data)

        # Cache overlay frame as JPEG
        overlaid = _draw_overlay(frame, frame_data)
        _, buf = cv2.imencode(".jpg", overlaid, [cv2.IMWRITE_JPEG_QUALITY, 78])
        _cached_frames[frame_idx] = buf.tobytes()

        if frame_data["alert_confirmed"]:
            alert_count += 1
        if frame_data["suspicious"]:
            suspicious_count += 1
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - started

    return {
        "ok": True,
        "video": video_path,
        "total_frames": frame_idx,
        "wall_seconds": round(elapsed, 3),
        "fps_effective": round(frame_idx / elapsed, 1) if elapsed > 0 else 0,
        "alert_frames": alert_count,
        "suspicious_frames": suspicious_count,
        "frames": frames,
    }


# ── HTTP server ─────────────────────────────────────────────────────────
current_result: dict[str, Any] = {}
result_lock = threading.Lock()
current_tuning = dict(DEFAULT_TUNING)

# Cached overlay frames (frame_idx -> jpeg bytes)
_cached_frames: dict[int, bytes] = {}


class TuningHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/params":
            self._send_json(json.dumps({"ok": True, "params": TUNING_PARAMS}).encode())
        elif parsed.path == "/api/result":
            with result_lock:
                data = json.dumps({"ok": True, "result": current_result}).encode()
            self._send_json(data)
        elif parsed.path == "/api/videos":
            self._send_json(json.dumps({"ok": True, "videos": ATTACK_VIDEOS}).encode())
        elif parsed.path.startswith("/api/frame/"):
            # Serve a cached overlay frame
            try:
                idx = int(parsed.path.rsplit("/", 1)[1])
            except (ValueError, IndexError):
                self._send_json(b'{"ok": false, "error": "bad index"}', 400)
                return
            frame_bytes = _cached_frames.get(idx)
            if frame_bytes is None:
                self._send_json(b'{"ok": false, "error": "frame not cached"}', 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(frame_bytes)
        elif parsed.path == "/":
            self._serve_html()
        else:
            self._send_json(b'{"ok": false, "error": "not found"}', 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            body = self._read_body()
            video = str(body.get("video", "")).strip()
            max_frames = int(body.get("max_frames", 0))
            tuning = dict(body.get("tuning", DEFAULT_TUNING))
            if not video or video not in ATTACK_VIDEOS:
                self._send_json(json.dumps({"ok": False, "error": "Invalid video"}).encode(), 400)
                return
            path = ATTACK_VIDEOS[video]
            if not Path(path).exists():
                self._send_json(json.dumps({"ok": False, "error": f"File not found: {path}"}).encode(), 400)
                return
            with result_lock:
                current_tuning = dict(tuning)
            # Run synchronously (CUDA context is per-thread on Windows)
            result = run_video(path, tuning, max_frames)
            with result_lock:
                global current_result
                current_result = result
            self._send_json(json.dumps({"ok": True, "result": result}).encode())
        else:
            self._send_json(b'{"ok": false, "error": "not found"}', 404)

    def _serve_html(self) -> None:
        html = HTML_CONTENT.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, fmt: str, *args) -> None:
        pass  # silence logs


HTML_CONTENT = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Module A 检测调参工具</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0c0e11;--surface:#151920;--surface2:#1c222a;--border:#2a3240;
  --text:#e8ecf0;--text2:#8d99a8;--accent:#47cf8e;--danger:#f05565;
  --warn:#f0b840;--blue:#5ba8f5;--radius:6px;
}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex}

/* ── 侧边栏 ── */
.sidebar{width:280px;min-width:280px;height:100vh;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sidebar-header{padding:14px 16px 10px;border-bottom:1px solid var(--border);flex-shrink:0}
.sidebar-header h1{font-size:15px;font-weight:700;letter-spacing:.3px}
.sidebar-header p{font-size:11px;color:var(--text2);margin-top:2px}
.sidebar-scroll{flex:1;overflow-y:auto;padding:10px 14px 20px}
.group{margin-bottom:14px}
.group-title{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--border)}
.param-row{margin-bottom:10px}
.param-label{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.param-label span:first-child{font-size:11px;color:var(--text2)}
.param-label .param-val{font-size:12px;font-weight:600;color:var(--accent);min-width:40px;text-align:right}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;border-radius:2px;background:var(--border);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--accent);border:2px solid var(--surface);cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.4)}
select{width:100%;height:32px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface2);color:var(--text);font-size:12px;padding:0 8px;outline:none;cursor:pointer}
select:focus{border-color:var(--accent)}
.num-input{width:100%;height:30px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface2);color:var(--text);font-size:12px;padding:0 8px;outline:none}
.num-input:focus{border-color:var(--accent)}
.run-btn{width:100%;height:36px;border:none;border-radius:var(--radius);background:var(--accent);color:#111;font-size:13px;font-weight:700;cursor:pointer;margin-top:6px;transition:opacity .15s}
.run-btn:hover{opacity:.88}
.run-btn:disabled{opacity:.4;cursor:wait}
.run-status{font-size:11px;color:var(--text2);margin-top:6px;min-height:16px}
.run-status.running{color:var(--warn)}
.run-status.done{color:var(--accent)}
.run-status.error{color:var(--danger)}

/* ── 主区域 ── */
.main{flex:1;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* 顶部统计 */
.stats-bar{display:flex;gap:1px;background:var(--border);flex-shrink:0}
.stat{flex:1;background:var(--surface);padding:8px 14px;text-align:center}
.stat b{font-size:20px;display:block;font-weight:700}
.stat span{font-size:10px;color:var(--text2)}
.stat.alert b{color:var(--danger)}
.stat.susp b{color:var(--warn)}

/* 上下分割 */
.content{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* 视频 + 信号 并排 */
.top-row{display:flex;flex:1;min-height:0;border-bottom:1px solid var(--border)}
.player-pane{flex:3;display:flex;flex-direction:column;background:var(--bg);border-right:1px solid var(--border);min-width:0}
.player-pane .pane-header{padding:6px 12px;font-size:11px;font-weight:600;color:var(--text2);background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0;display:flex;justify-content:space-between;align-items:center}
.player-img-wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#000;min-height:0}
.player-img-wrap img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.player-controls{display:flex;align-items:center;gap:6px;padding:6px 10px;background:var(--surface);border-top:1px solid var(--border);flex-shrink:0;flex-wrap:wrap}
.ctrl-btn{height:28px;padding:0 12px;border:1px solid var(--border);border-radius:4px;background:var(--surface2);color:var(--text);font-size:11px;cursor:pointer;white-space:nowrap;flex-shrink:0}
.ctrl-btn:hover{background:var(--border)}
.ctrl-btn.playing{background:var(--warn);color:#111;border-color:var(--warn)}
.ctrl-btn.jump{background:var(--danger);color:#fff;border-color:var(--danger);opacity:.85}
.ctrl-btn.jump:hover{opacity:1}
#frameSlider{flex:1;min-width:80px;margin:0}
.frame-info{font-size:11px;color:var(--text2);min-width:90px;text-align:right;flex-shrink:0}
.speed-row{display:flex;align-items:center;gap:4px;font-size:10px;color:var(--text2)}
.spd-btn{height:22px;padding:0 6px;border:1px solid var(--border);border-radius:3px;background:transparent;color:var(--text2);font-size:10px;cursor:pointer}
.spd-btn.active{background:var(--accent);color:#111;border-color:var(--accent);font-weight:600}

/* 信号图 */
.chart-pane{flex:2;display:flex;flex-direction:column;background:var(--surface);min-width:0}
.chart-pane .pane-header{padding:6px 12px;font-size:11px;font-weight:600;color:var(--text2);border-bottom:1px solid var(--border);flex-shrink:0}
.chart-body{flex:1;padding:6px;cursor:crosshair;min-height:0;position:relative}
.chart-body canvas{width:100%;height:100%;display:block}
.legend{display:flex;gap:10px;padding:4px 12px;flex-shrink:0;flex-wrap:wrap}
.legend i{width:10px;height:3px;border-radius:1px;display:inline-block;vertical-align:middle;margin-right:3px}
.legend span{font-size:10px;color:var(--text2)}

/* 数据表 */
.table-pane{height:200px;min-height:120px;overflow:auto;background:var(--surface);flex-shrink:0;border-top:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:11px;table-layout:auto}
th{position:sticky;top:0;z-index:1;background:var(--surface2);padding:4px 8px;text-align:left;border-bottom:1px solid var(--border);color:var(--text2);font-weight:600;white-space:nowrap}
td{padding:3px 8px;border-bottom:1px solid #161b22;white-space:nowrap;color:var(--text2)}
tr:hover td{background:rgba(255,255,255,.03)}
tr.hl td{background:rgba(71,207,142,.08)}
td.c-alert{color:var(--danger);font-weight:700}
td.c-susp{color:var(--warn);font-weight:600}
td.c-a3b{color:var(--blue)}

/* 空状态 */
.empty-state{flex:1;display:flex;align-items:center;justify-content:center;color:var(--text2);font-size:14px;text-align:center;padding:40px}
</style>
</head>
<body>

<!-- 侧边栏 -->
<div class="sidebar">
  <div class="sidebar-header">
    <h1>Module A 调参工具</h1>
    <p>离线视频检测 + 告警叠加回放</p>
  </div>
  <div class="sidebar-scroll">
    <div class="group">
      <div class="group-title">视频源</div>
      <select id="videoSelect"><option value="">选择视频...</option></select>
      <div class="param-row" style="margin-top:8px">
        <div class="param-label"><span>最大帧数 (0=全部)</span></div>
        <input class="num-input" type="number" id="maxFrames" value="0" min="0">
      </div>
      <button class="run-btn" id="runBtn" onclick="runVideo()">运行检测</button>
      <div class="run-status" id="status"></div>
    </div>
    <div class="group"><div class="group-title">A1 过曝检测</div><div id="a1Params"></div></div>
    <div class="group"><div class="group-title">A3b 翻拍检测</div><div id="a3bParams"></div></div>
    <div class="group"><div class="group-title">A4 目标锚定融合</div><div id="a4Params"></div></div>
    <div class="group"><div class="group-title">告警状态机</div><div id="alertParams"></div></div>
  </div>
</div>

<!-- 主区域 -->
<div class="main">
  <!-- 统计栏 -->
  <div class="stats-bar" id="statsBar" style="display:none">
    <div class="stat alert"><b id="sAlert">0</b><span>告警帧</span></div>
    <div class="stat susp"><b id="sSusp">0</b><span>可疑帧</span></div>
    <div class="stat"><b id="sTotal">0</b><span>总帧</span></div>
    <div class="stat"><b id="sFps">0</b><span>处理fps</span></div>
  </div>

  <div class="content" id="contentArea">
    <!-- 空状态 -->
    <div class="empty-state" id="emptyState">
      选择左侧视频 → 调整阈值 → 点击「运行检测」<br>
      <span style="font-size:11px;margin-top:8px;display:block;opacity:.6">检测完成后可逐帧回放带告警叠加的画面</span>
    </div>

    <!-- 上部：视频+信号 -->
    <div class="top-row" id="topRow" style="display:none">
      <div class="player-pane">
        <div class="pane-header">
          <span>视频回放</span>
          <span id="frameLabel" style="color:var(--text)">帧 0</span>
        </div>
        <div class="player-img-wrap">
          <img id="playerFrame" alt="frame" src="">
        </div>
        <div class="player-controls">
          <button class="ctrl-btn" id="playBtn" onclick="togglePlay()">播放</button>
          <button class="ctrl-btn" onclick="stepFrame(-1)" title="上一帧 (←)">◀</button>
          <button class="ctrl-btn" onclick="stepFrame(1)" title="下一帧 (→)">▶</button>
          <input type="range" id="frameSlider" min="0" max="0" value="0" oninput="seekFrame(this.value)">
          <span class="frame-info" id="frameInfo">0 / 0</span>
        </div>
        <div class="player-controls" style="border-top:none;padding-top:0">
          <div class="speed-row">
            <span>速度</span>
            <button class="spd-btn" onclick="setSpeed(5)" id="spd5">5</button>
            <button class="spd-btn active" onclick="setSpeed(15)" id="spd15">15</button>
            <button class="spd-btn" onclick="setSpeed(30)" id="spd30">30</button>
            <button class="spd-btn" onclick="setSpeed(60)" id="spd60">60</button>
          </div>
          <div style="flex:1"></div>
          <button class="ctrl-btn jump" onclick="jumpAlert(-1)" title="上一告警 (p)">◀ 上一告警</button>
          <button class="ctrl-btn jump" onclick="jumpAlert(1)" title="下一告警 (n)">下一告警 ▶</button>
        </div>
      </div>

      <div class="chart-pane">
        <div class="pane-header">信号时序图 <span style="font-weight:400;opacity:.6">(点击跳转)</span></div>
        <div class="chart-body" id="chartBody" onclick="chartClick(event)">
          <canvas id="signalChart"></canvas>
        </div>
        <div class="legend" id="legendBar"></div>
      </div>
    </div>

    <!-- 下部：数据表 -->
    <div class="table-pane" id="tablePane" style="display:none">
      <table><thead><tr id="tableHeader"></tr></thead><tbody id="tableBody"></tbody></table>
    </div>
  </div>
</div>

<script>
const PG={
  a1:["glare_ratio_threshold","glare_flash_diff_threshold","glare_flash_ratio_threshold"],
  a3b:["a3b_high_score_bypass_threshold"],
  a4:["flow_local_anomaly_threshold","roi_blur_threshold","track_drop_threshold","paired_temporal_motion_threshold","motion_score_threshold"],
  alert:["alert_window","alert_trigger_count","attack_state_hold_frames"],
};
const P={};
let frames=[], curIdx=0, playing=false, playTimer=null, playFps=15;
const SIG=[
  {k:"motion_score",c:"#47cf8e",n:"motion"},
  {k:"flow_local_ratio",c:"#5ba8f5",n:"flow_local"},
  {k:"temporal_local_max",c:"#f0b840",n:"temporal"},
  {k:"blur_score",c:"#e87840",n:"blur"},
  {k:"overexposure_ratio",c:"#d4c540",n:"overexp",s:5},
  {k:"temporal_flash_ratio",c:"#f05565",n:"flash",s:5},
];

// Init
fetch("/api/params").then(r=>r.json()).then(d=>{Object.assign(P,d.params);renderControls();});
fetch("/api/videos").then(r=>r.json()).then(d=>{
  const s=document.getElementById("videoSelect");
  for(const[k]of Object.entries(d.videos)){const o=document.createElement("option");o.value=k;o.textContent=k;s.appendChild(o);}
});
// Legend
(function(){
  const lb=document.getElementById("legendBar");
  for(const s of SIG){
    const sp=document.createElement("span");
    const i=document.createElement("i");i.style.background=s.c;
    sp.appendChild(i);sp.appendChild(document.createTextNode(s.n));
    lb.appendChild(sp);
  }
})();

function renderControls(){
  for(const[grp,keys]of Object.entries(PG)){
    const div=document.getElementById(grp+"Params");if(!div)continue;div.innerHTML="";
    for(const k of keys){
      const p=P[k];if(!p)continue;
      const isInt=Number.isInteger(p.step)&&Number.isInteger(p.default);
      const row=document.createElement("div");row.className="param-row";
      const lbl=document.createElement("div");lbl.className="param-label";
      const name=document.createElement("span");name.textContent=p.name;
      const val=document.createElement("span");val.className="param-val";val.id="v_"+k;
      val.textContent=isInt?p.default:p.default.toFixed(2);
      lbl.appendChild(name);lbl.appendChild(val);
      const inp=document.createElement("input");inp.type="range";inp.id="r_"+k;
      inp.min=p.min;inp.max=p.max;inp.step=p.step;inp.value=p.default;
      inp.oninput=()=>{const v=parseFloat(inp.value);val.textContent=isInt?Math.round(v):v.toFixed(2);P[k].default=v;};
      row.appendChild(lbl);row.appendChild(inp);div.appendChild(row);
    }
  }
}
function getTuning(){const t={};for(const k of Object.keys(P))t[k]=P[k].default;return t;}

// ── Run ──
function runVideo(){
  const v=document.getElementById("videoSelect").value;
  if(!v){alert("请先选择视频");return;}
  const mf=parseInt(document.getElementById("maxFrames").value)||0;
  const st=document.getElementById("status"),btn=document.getElementById("runBtn");
  st.textContent="处理中...";st.className="run-status running";btn.disabled=true;
  stopPlay();
  document.getElementById("statsBar").style.display="none";
  document.getElementById("topRow").style.display="none";
  document.getElementById("tablePane").style.display="none";
  document.getElementById("emptyState").style.display="flex";
  fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({video:v,max_frames:mf,tuning:getTuning()})
  }).then(r=>r.json()).then(d=>{
    btn.disabled=false;
    if(d.ok&&d.result&&d.result.ok){st.textContent=d.result.total_frames+"帧 / "+d.result.wall_seconds+"s";st.className="run-status done";showResult(d.result);}
    else{st.textContent="错误: "+(d.error||d.result?.error||"未知");st.className="run-status error";}
  }).catch(e=>{btn.disabled=false;st.textContent="失败: "+e;st.className="run-status error";});
}

function showResult(r){
  document.getElementById("emptyState").style.display="none";
  document.getElementById("sAlert").textContent=r.alert_frames;
  document.getElementById("sSusp").textContent=r.suspicious_frames;
  document.getElementById("sTotal").textContent=r.total_frames;
  document.getElementById("sFps").textContent=r.fps_effective;
  document.getElementById("statsBar").style.display="flex";
  frames=r.frames;curIdx=0;
  document.getElementById("topRow").style.display="flex";
  document.getElementById("frameSlider").max=frames.length-1;
  showFrame(0);
  drawChart();
  renderTable();
  document.getElementById("tablePane").style.display="block";
}

// ── Player ──
function showFrame(i){
  if(i<0||i>=frames.length)return;
  curIdx=i;
  document.getElementById("playerFrame").src="/api/frame/"+i;
  document.getElementById("frameSlider").value=i;
  const f=frames[i];
  let tag="正常";
  if(f.alert_confirmed)tag="告警";else if(f.suspicious)tag="可疑";
  document.getElementById("frameLabel").textContent="帧 "+i+" — "+tag;
  document.getElementById("frameLabel").style.color=f.alert_confirmed?"var(--danger)":f.suspicious?"var(--warn)":"var(--text)";
  document.getElementById("frameInfo").textContent=i+" / "+(frames.length-1);
  // table highlight
  document.querySelectorAll("#tableBody tr.hl").forEach(r=>r.classList.remove("hl"));
  const rows=document.querySelectorAll("#tableBody tr");
  if(rows[i]){rows[i].classList.add("hl");rows[i].scrollIntoView({block:"nearest"});}
  drawChart();
}
function seekFrame(v){showFrame(parseInt(v));}
function stepFrame(d){showFrame(curIdx+d);}
function togglePlay(){playing?stopPlay():startPlay();}
function startPlay(){playing=true;document.getElementById("playBtn").textContent="暂停";document.getElementById("playBtn").classList.add("playing");tick();}
function stopPlay(){playing=false;if(playTimer){clearTimeout(playTimer);playTimer=null;}document.getElementById("playBtn").textContent="播放";document.getElementById("playBtn").classList.remove("playing");}
function tick(){if(!playing)return;playTimer=setTimeout(()=>{if(curIdx<frames.length-1){showFrame(curIdx+1);tick();}else stopPlay();},1000/playFps);}
function setSpeed(fps){playFps=fps;document.querySelectorAll(".spd-btn").forEach(b=>b.classList.remove("active"));const e=document.getElementById("spd"+fps);if(e)e.classList.add("active");if(playing){stopPlay();startPlay();}}
function jumpAlert(dir){
  if(!frames.length)return;
  if(dir>0){for(let i=curIdx+1;i<frames.length;i++){if(frames[i].alert_confirmed){showFrame(i);return;}}}
  else{for(let i=curIdx-1;i>=0;i--){if(frames[i].alert_confirmed){showFrame(i);return;}}}
}
document.addEventListener("keydown",e=>{
  if(!frames.length||e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  if(e.key==="ArrowRight")stepFrame(1);
  else if(e.key==="ArrowLeft")stepFrame(-1);
  else if(e.key===" "){e.preventDefault();togglePlay();}
  else if(e.key==="n")jumpAlert(1);
  else if(e.key==="p")jumpAlert(-1);
});

// ── Chart ──
function drawChart(){
  const wrap=document.getElementById("chartBody");
  const c=document.getElementById("signalChart");
  const W=wrap.clientWidth-12,H=wrap.clientHeight-12;
  if(W<10||H<10)return;
  c.width=W;c.height=H;
  const ctx=c.getContext("2d");
  const n=frames.length;if(!n)return;
  ctx.clearRect(0,0,W,H);

  // background bands
  for(const f of frames){
    const x=(f.frame_idx/n)*W;const bw=Math.max(1.5,W/n);
    if(f.alert_confirmed){ctx.fillStyle="rgba(240,85,101,.15)";ctx.fillRect(x,0,bw,H);}
    else if(f.suspicious){ctx.fillStyle="rgba(240,184,64,.07)";ctx.fillRect(x,0,bw,H);}
  }
  // grid
  ctx.strokeStyle="#1e2530";ctx.lineWidth=1;
  for(let i=1;i<4;i++){const y=(i/4)*H;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  // signals
  for(const sig of SIG){
    ctx.beginPath();ctx.strokeStyle=sig.c;ctx.lineWidth=1.5;ctx.globalAlpha=.85;
    let first=true;
    for(const f of frames){
      const x=(f.frame_idx/n)*W;
      let v=f[sig.k]||0;if(sig.s)v=Math.min(1,v*sig.s);
      const y=H-v*H;
      if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
    }
    ctx.stroke();ctx.globalAlpha=1;
  }
  // playhead
  if(curIdx>=0&&curIdx<n){
    const px=(curIdx/n)*W;
    ctx.strokeStyle="rgba(255,255,255,.65)";ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();ctx.setLineDash([]);
  }
}
function chartClick(e){
  if(!frames.length)return;
  const c=document.getElementById("signalChart"),rect=c.getBoundingClientRect();
  const x=e.clientX-rect.left,ratio=x/c.width;
  showFrame(Math.max(0,Math.min(frames.length-1,Math.round(ratio*frames.length))));
}
window.addEventListener("resize",()=>{if(frames.length)drawChart();});

// ── Table ──
function renderTable(){
  const TH=document.getElementById("tableHeader"),TB=document.getElementById("tableBody");
  const cols=[
    {k:"frame_idx",n:"帧"},{k:"alert_confirmed",n:"告警",cls:"c-alert"},
    {k:"suspicious",n:"可疑",cls:"c-susp"},
    {k:"motion_score",n:"motion",f:v=>v.toFixed(3)},
    {k:"flow_local_ratio",n:"flow_loc",f:v=>v.toFixed(3)},
    {k:"temporal_local_max",n:"temporal",f:v=>v.toFixed(3)},
    {k:"blur_score",n:"blur",f:v=>v.toFixed(3)},
    {k:"track_score",n:"track",f:v=>v.toFixed(3)},
    {k:"overexposure_ratio",n:"overexp",f:v=>v.toFixed(4)},
    {k:"temporal_flash_ratio",n:"flash",f:v=>v.toFixed(4)},
    {k:"a3b_triggered",n:"A3b",cls:"c-a3b"},
    {k:"reason_codes",n:"原因"},
  ];
  TH.innerHTML="";TB.innerHTML="";
  for(const c of cols){const th=document.createElement("th");th.textContent=c.n;TH.appendChild(th);}
  for(const fr of frames){
    const tr=document.createElement("tr");tr.style.cursor="pointer";
    tr.onclick=()=>showFrame(fr.frame_idx);
    for(const c of cols){
      const td=document.createElement("td");const v=fr[c.k];
      if(c.k==="reason_codes")td.textContent=(v||[]).join(", ");
      else if(c.k==="alert_confirmed"||c.k==="suspicious"||c.k==="a3b_triggered"){td.textContent=v?"Y":"";if(v&&c.cls)td.className=c.cls;}
      else if(c.f)td.textContent=c.f(v);
      else td.textContent=v??"";
      tr.appendChild(td);
    }
    TB.appendChild(tr);
  }
}
</script>
</body>
</html>"""


def main() -> None:
    port = 8766
    for _ in range(10):
        try:
            server = HTTPServer(("127.0.0.1", port), TuningHandler)
            break
        except PermissionError:
            port += 1
    print(f"Module A Tuning Tool → http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()

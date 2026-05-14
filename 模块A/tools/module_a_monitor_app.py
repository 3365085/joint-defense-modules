"""Local web monitor for Module A live/video-stream inference."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("PyYAML is required in the pixi environment.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFENSE_DIR = PROJECT_ROOT / "defense"
sys.path.insert(0, str(PROJECT_ROOT))
EVIDENCE_ROOT = PROJECT_ROOT / "异常记录"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from defense.module_a.backends import create_detector_backend  # noqa: E402
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def draw_hud(frame: Any, info: dict[str, Any], frame_idx: int, effective: bool) -> Any:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (10, 10, 15), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    layer = info.get("layer_triggered", "NONE")
    timing = float(info.get("timing_ms", 0.0))
    attack = bool(info.get("attack_detected", info.get("is_attack", False)))
    alert = bool(info.get("alert_confirmed", False))

    cv2.putText(
        frame,
        f"FRAME {frame_idx:05d} | {layer} | {timing:.1f}ms",
        (10, 23),
        font,
        0.55,
        (255, 210, 120),
        1,
    )

    state = "ALERT CONFIRMED" if alert else "MONITORING"
    color = (0, 0, 255) if alert else (0, 220, 80)
    cv2.putText(frame, state, (10, 50), font, 0.6, color, 2)

    tags: list[str] = []
    if effective:
        tags.append("EFFECTIVE")
    if attack:
        tags.append("ATTACK")
    if info.get("attack_state_active") and not attack:
        tags.append("STATE_ACTIVE")
    if tags:
        cv2.putText(frame, " | ".join(tags), (10, 73), font, 0.45, (255, 255, 255), 1)

    if alert:
        thick = 6 if frame_idx % 2 == 0 else 3
        cv2.rectangle(frame, (thick, thick), (w - thick, h - thick), (0, 0, 255), thick)
    return frame


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Module A 监控台</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --line: #353d45;
      --text: #f4f7fa;
      --muted: #a9b4bf;
      --accent: #46d39a;
      --danger: #ff4d5e;
      --warn: #ffc857;
      --blue: #6bb8ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
    }
    .videoShell, .side {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
    }
    .videoShell { min-width: 0; }
    .videoHeader, .panelHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 52px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .title {
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .pill {
      min-width: 74px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .pill.live { color: var(--accent); border-color: rgba(70, 211, 154, .45); }
    .pill.alert { color: var(--danger); border-color: rgba(255, 77, 94, .6); }
    .stage {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 360px;
      aspect-ratio: 16 / 9;
      background: #080a0c;
    }
    .stage img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }
    .stage img[src] { display: block; }
    .empty {
      display: grid;
      gap: 8px;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      padding: 20px;
      text-align: center;
    }
    .empty strong {
      color: var(--text);
      font-size: 18px;
    }
    .empty .hint {
      max-width: 360px;
      line-height: 1.7;
    }
    .side {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .panel {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    label {
      display: block;
      margin: 10px 0 6px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select, button {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #11161b;
      color: var(--text);
      font-size: 13px;
      padding: 0 10px;
      min-width: 0;
    }
    input { overflow: hidden; text-overflow: ellipsis; }
    button {
      cursor: pointer;
      font-weight: 650;
      background: #242c34;
    }
    button.primary { background: #1b5f46; border-color: #2a8a68; }
    button.danger { background: #5e1d29; border-color: #9d3545; }
    button.secondary { min-width: 86px; background: #27313a; border-color: #46535f; }
    button:disabled { cursor: wait; opacity: .68; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .fieldRow {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
      align-items: center;
    }
    .fieldRow.single { grid-template-columns: 1fr; }
    .sourceHint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .connectStatus {
      min-height: 20px;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .connectStatus.ok { color: var(--accent); }
    .connectStatus.bad { color: var(--danger); }
    .connectStatus.testing { color: var(--warn); }
    .toggleGrid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      margin-top: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #11161b;
    }
    .toggleLine {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 26px;
      margin: 0;
      color: var(--text);
      font-size: 13px;
    }
    .toggleLine input {
      width: 16px;
      height: 16px;
      flex: 0 0 auto;
    }
    .verdict {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
      background: #11161b;
    }
    .verdict b {
      display: block;
      font-size: 18px;
      margin-bottom: 4px;
    }
    .verdict span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .verdict.alert {
      border-color: rgba(255, 77, 94, .65);
      background: rgba(255, 77, 94, .08);
    }
    .verdict.live {
      border-color: rgba(70, 211, 154, .5);
      background: rgba(70, 211, 154, .07);
    }
    .verdict.ppe-warn {
      border-color: rgba(255, 200, 87, .72);
      background: rgba(255, 200, 87, .09);
    }
    .verdict.ppe-ok {
      border-color: rgba(107, 184, 255, .45);
      background: rgba(107, 184, 255, .07);
    }
    /* ---------------------------------------------------------------
       Three-card branch panel (Task 8.1 / Requirements 10.1, 10.2).

       Each branch (p_adv / p_safety / p_synth) gets an independent card
       with: title, score display, score bar, state label, reason text,
       and a highlight border controlled by the card-<severity> class.
    --------------------------------------------------------------- */
    .branchCards {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }
    .branchCard {
      border: 2px solid var(--line);
      border-radius: 10px;
      padding: 12px 12px 10px;
      background: #11161b;
      transition: border-color .15s ease-in-out, background-color .15s ease-in-out;
    }
    .branchCard.card-idle {
      border-color: var(--line);
    }
    .branchCard.card-warning {
      border-color: rgba(255, 200, 87, .72);
      background: rgba(255, 200, 87, .07);
    }
    .branchCard.card-confirmed {
      border-color: rgba(255, 77, 94, .72);
      background: rgba(255, 77, 94, .10);
    }
    .branchCard.card-suppressed {
      border-color: rgba(107, 184, 255, .55);
      background: rgba(107, 184, 255, .06);
    }
    .branchCard.card-missing {
      border-color: rgba(169, 180, 191, .30);
      background: #10141a;
      opacity: .85;
    }
    .branchCard .cardHeader {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 6px;
    }
    .branchCard .cardTitle {
      font-size: 14px;
      font-weight: 650;
      color: var(--text);
    }
    .branchCard .cardState {
      font-size: 12px;
      color: var(--muted);
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #181d22;
    }
    .branchCard.card-warning .cardState {
      color: #ffd48a;
      border-color: rgba(255, 200, 87, .55);
    }
    .branchCard.card-confirmed .cardState {
      color: #ffd0d5;
      border-color: rgba(255, 77, 94, .65);
    }
    .branchCard.card-suppressed .cardState {
      color: #cfe7ff;
      border-color: rgba(107, 184, 255, .55);
    }
    .branchCard.card-missing .cardState {
      color: var(--muted);
    }
    .branchCard .cardScoreRow {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin: 4px 0;
    }
    .branchCard .cardScore {
      font-size: 22px;
      line-height: 1.1;
      font-weight: 650;
      font-variant-numeric: tabular-nums;
      color: var(--text);
    }
    .branchCard .cardBadges {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .branchCard .cardBadge {
      font-size: 11px;
      color: #cfe7ff;
      border: 1px solid rgba(107, 184, 255, .45);
      background: rgba(107, 184, 255, .08);
      padding: 2px 7px;
      border-radius: 999px;
    }
    .branchCard .cardBar {
      position: relative;
      width: 100%;
      height: 6px;
      border-radius: 4px;
      background: #1d242b;
      overflow: hidden;
      margin-top: 2px;
    }
    .branchCard .cardBarFill {
      position: absolute;
      top: 0; left: 0; bottom: 0;
      width: 0%;
      background: var(--muted);
      transition: width .15s ease-out, background-color .15s ease-out;
    }
    .branchCard.card-warning .cardBarFill { background: var(--warn); }
    .branchCard.card-confirmed .cardBarFill { background: var(--danger); }
    .branchCard.card-suppressed .cardBarFill { background: var(--blue); }
    .branchCard.card-idle .cardBarFill { background: var(--accent); }
    .branchCard.card-missing .cardBarFill { background: transparent; }
    .branchCard .cardDetail {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.55;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }
    .branchCard .cardReason {
      font-size: 12px;
      color: var(--text);
      line-height: 1.55;
      margin-top: 4px;
      overflow-wrap: anywhere;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      min-height: 62px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #11161b;
    }
    .metric b {
      display: block;
      font-size: 20px;
      line-height: 24px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .events {
      overflow: auto;
      padding: 10px 14px 14px;
    }
    .event {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
      background: #11161b;
      font-size: 12px;
      color: var(--muted);
    }
    .event.alert { border-color: rgba(255, 77, 94, .48); }
    .event.ppe { border-color: rgba(255, 200, 87, .58); }
    .event strong {
      display: block;
      color: var(--text);
      margin-bottom: 3px;
      font-size: 14px;
    }
    .event .main {
      color: var(--text);
      font-size: 13px;
      line-height: 1.5;
      margin: 6px 0 8px;
    }
    .event .sub {
      color: var(--muted);
      line-height: 1.5;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      color: #dce7f2;
      background: #20262b;
      font-size: 11px;
    }
    .chip.danger { color: #ffd9dd; border-color: rgba(255,77,94,.5); }
    .chip.warn { color: #fff0c4; border-color: rgba(255,200,87,.58); }
    .evidenceLinks {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .evidenceLinks a, .evidenceLinks span {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      color: #cfe5ff;
      text-decoration: none;
      background: #17212a;
      font-size: 12px;
    }
    .evidenceLinks span { color: var(--muted); }
    details {
      margin-top: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    summary { cursor: pointer; }
    code {
      display: block;
      margin-top: 6px;
      color: #b8c7d6;
      line-height: 1.45;
    }
    .err {
      margin-top: 10px;
      min-height: 18px;
      color: var(--danger);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
        gap: 10px;
        min-height: auto;
        padding: 8px;
      }
      .videoHeader, .panelHeader {
        min-height: 46px;
        padding: 8px 10px;
      }
      .title { font-size: 15px; }
      .pill {
        min-width: auto;
        padding: 4px 8px;
      }
      .stage {
        min-height: 210px;
        aspect-ratio: 4 / 3;
      }
      .panel { padding: 10px; }
      .events { padding: 8px 10px 12px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    /* Collapsible section styles */
    .collapsible-header {
      cursor: pointer;
      user-select: none;
    }
    .collapsible-header .collapse-icon {
      display: inline-block;
      transition: transform .15s ease;
      font-size: 11px;
    }
    .collapsible-header.collapsed .collapse-icon {
      transform: rotate(-90deg);
    }
    .collapsible-content {
      display: block;
    }
    .collapsible-content.collapsed {
      display: none;
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="videoShell">
      <div class="videoHeader">
        <div class="title">模块A监控台</div>
        <div id="statePill" class="pill">未运行</div>
      </div>
      <div class="stage">
        <img id="stream" alt="实时检测画面" />
        <div id="empty" class="empty">
          <strong>等待输入源</strong>
          <div class="hint">选择 MP4、RTSP 或摄像头后开始检测。告警会在画面上用红框标出。</div>
        </div>
      </div>
    </section>
    <aside class="side">
      <div class="panelHeader">
        <div class="title">输入源</div>
        <div id="profilePill" class="pill">全速GPU</div>
      </div>
      <section class="panel">
        <label>输入类型</label>
        <select id="sourceType">
          <option value="file">MP4 文件路径</option>
          <option value="rtsp">RTSP / HTTP 视频流</option>
          <option value="camera">本机摄像头编号</option>
        </select>
        <label id="sourceValueLabel">MP4 文件</label>
        <div id="pathRow" class="fieldRow">
          <input id="sourceValue" value="materials/训练素材/分类/12_监控视角_仓库巡检/015_clean_baseline_single_worker_normal_6f9897da7479.mp4" />
          <button class="secondary" id="browseBtn" type="button">浏览</button>
          <button class="secondary" id="testSourceBtn" type="button" style="display:none;">连通检测</button>
        </div>
        <div id="cameraRow" class="fieldRow" style="display:none;">
          <select id="cameraSelect"></select>
          <button class="secondary" id="scanBtn" type="button">扫描</button>
        </div>
        <div id="sourceHint" class="sourceHint">点击“浏览”可打开资源管理器选择本机 MP4。</div>
        <div id="connectionStatus" class="connectStatus"></div>
        <label>算力档位</label>
        <select id="profile">
          <option value="full_gpu">全速GPU（TensorRT）</option>
          <option value="balanced_gpu">均衡GPU（省算力）</option>
          <option value="edge_onnx">边缘ONNX（兼容）</option>
        </select>
        <label>检测模型</label>
        <div class="toggleGrid">
          <label class="toggleLine">
            <input id="enableCustomModel" type="checkbox" />
            使用自定义模型
          </label>
        </div>
        <div id="customModelPanel" style="display:none;">
          <label>模型文件</label>
          <div class="fieldRow">
            <input id="customModelPath" value="" placeholder="选择 .engine / .onnx / .pt" />
            <button class="secondary" id="browseModelBtn" type="button">浏览</button>
          </div>
          <label>模型后端</label>
          <select id="customModelBackend">
            <option value="auto">自动识别后缀</option>
            <option value="tensorrt">TensorRT Engine</option>
            <option value="onnx">ONNX Runtime</option>
            <option value="pytorch">PyTorch</option>
          </select>
          <label>模型类型</label>
          <select id="customModelFamily">
            <option value="ultralytics">Ultralytics YOLOv8/YOLO11</option>
            <option value="yolov5">原版 YOLOv5</option>
          </select>
          <div class="sourceHint">自定义模型只替换目标检测器，A1-A4、静态媒介检测和伪造流提示仍使用当前模块参数。</div>
        </div>
        <label style="display:flex;align-items:center;gap:8px;margin-top:12px;">
          <input id="realtime" type="checkbox" checked style="width:16px;height:16px;" />
          <span id="realtimeText">MP4按视频FPS播放</span>
        </label>
        <label>画面叠加</label>
        <div class="toggleGrid">
          <label class="toggleLine">
            <input id="showBoxes" type="checkbox" checked />
            显示目标框和置信度
          </label>
          <label class="toggleLine">
            <input id="showModuleHud" type="checkbox" checked />
            显示模块A状态文字
          </label>
          <label class="toggleLine">
            <input id="showPpeHud" type="checkbox" checked />
            显示安全帽画面提示
          </label>
        </div>
        <label>检测功能</label>
        <div class="toggleGrid">
          <label class="toggleLine">
            <input id="enableStaticImage" type="checkbox" checked />
            静态媒介欺骗检测
          </label>
          <label class="toggleLine" title="伪造视频流检测仍在开发中，暂不建议开启">
            <input id="enableSynthWarning" type="checkbox" disabled />
            伪造视频流告警 <span style="color:#888;font-size:12px;">（开发中，暂停用）</span>
          </label>
        </div>
        <div class="row">
          <button class="primary" id="startBtn">开始检测</button>
          <button class="danger" id="stopBtn">停止</button>
        </div>
        <div class="err" id="error"></div>
      </section>
      <section class="panel">
        <!-- Task 8.1 / Requirements 10.1, 10.2: right-panel three-card
             layout. Cards are rendered from status.branch_cards (built by
             build_branch_cards in Python) so the UI and Python contract
             cannot drift apart. -->
        <div id="branchCards" class="branchCards" data-testid="branch-cards">
          <div class="branchCard card-idle" data-branch="p_adv" data-testid="branch-card-p_adv">
            <div class="cardHeader">
              <div class="cardTitle">物理对抗扰动（p_adv）</div>
              <div class="cardState">系统待机</div>
            </div>
            <div class="cardScoreRow">
              <div class="cardScore">--</div>
              <div class="cardBadges"></div>
            </div>
            <div class="cardBar"><div class="cardBarFill" style="width:0%"></div></div>
            <div class="cardDetail">尚未开始检测。</div>
            <div class="cardReason"></div>
          </div>
          <div class="branchCard card-idle" data-branch="p_safety" data-testid="branch-card-p_safety">
            <div class="cardHeader">
              <div class="cardTitle">翻拍/假目标检测（A3b）</div>
              <div class="cardState">系统待机</div>
            </div>
            <div class="cardScoreRow">
              <div class="cardScore">--</div>
              <div class="cardBadges"></div>
            </div>
            <div class="cardBar"><div class="cardBarFill" style="width:0%"></div></div>
            <div class="cardDetail">检测画面中是否出现屏幕翻拍、照片等非真人目标。</div>
            <div class="cardReason"></div>
          </div>
          <div class="branchCard card-idle" data-branch="p_synth" data-testid="branch-card-p_synth">
            <div class="cardHeader">
              <div class="cardTitle">视频源真实性（p_synth）</div>
              <div class="cardState">系统待机</div>
            </div>
            <div class="cardScoreRow">
              <div class="cardScore">--</div>
              <div class="cardBadges"></div>
            </div>
            <div class="cardBar"><div class="cardBarFill" style="width:0%"></div></div>
            <div class="cardDetail">视频源真实性检测通道。</div>
            <div class="cardReason"></div>
          </div>
        </div>
        <div class="metrics">
          <div class="metric"><b id="frame">0</b><span>当前帧</span></div>
          <div class="metric"><b id="pAdv">0.000</b><span>攻击概率</span></div>
          <div class="metric"><b id="pSynth">0.000</b><span>伪造流概率</span></div>
          <div class="metric"><b id="timing">0.0</b><span>处理耗时 ms</span></div>
          <div class="metric"><b id="alerts">0</b><span>模块A告警</span></div>
          <div class="metric"><b id="ppeAlerts">0</b><span>安全帽告警</span></div>
          <div class="metric"><b id="sourceAlerts">0</b><span>伪造流告警</span></div>
          <div class="metric"><b id="ppeCounts">0/0</b><span>人员/安全帽</span></div>
        </div>
      </section>
      <div class="panelHeader collapsible-header" onclick="toggleSection(this)">
        <div class="title"><span class="collapse-icon">▼</span> 模块A扰动告警</div>
        <div id="backendPill" class="pill">-</div>
      </div>
      <section class="events collapsible-content" id="events"></section>
      <div class="panelHeader collapsible-header" onclick="toggleSection(this)">
        <div class="title"><span class="collapse-icon">▼</span> 安全帽业务告警</div>
        <div id="ppePill" class="pill">独立通道</div>
      </div>
      <section class="events collapsible-content" id="ppeEvents"></section>
      <div class="panelHeader collapsible-header" onclick="toggleSection(this)">
        <div class="title"><span class="collapse-icon">▼</span> 生成/伪造视频流提示</div>
        <div id="sourcePill" class="pill">独立通道</div>
      </div>
      <section class="events collapsible-content" id="sourceEvents"></section>
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let streamNonce = 0;
    let cameraDevicesLoaded = false;

    function toggleSection(headerEl) {
      headerEl.classList.toggle("collapsed");
      const content = headerEl.nextElementSibling;
      if (content) content.classList.toggle("collapsed");
    }

    const profileLabels = {
      full_gpu: "全速GPU",
      balanced_gpu: "均衡GPU",
      edge_onnx: "边缘ONNX",
    };

    function profileLabel(value) {
      return profileLabels[value] || value || "-";
    }

    function setError(text) {
      $("error").textContent = text || "";
    }

    function setConnectionStatus(text, state) {
      const node = $("connectionStatus");
      node.textContent = text || "";
      node.className = "connectStatus " + (state || "");
    }

    function reasonTags(reason) {
      const text = reason || "";
      const tags = [];
      const add = (key, label) => { if (text.includes(key) && !tags.includes(label)) tags.push(label); };
      add("overexposure", "强光/过曝");
      add("temporal_texture_change", "时序纹理突变");
      add("local_temporal_texture_change", "局部纹理突变");
      add("motion_artifact", "运动伪影");
      add("paired_temporal_flow_change", "光流不一致");
      add("light_flow", "轻光流异常");
      add("blur", "清晰度退化");
      add("track_consistency_drop", "目标轨迹异常");
      add("classifier_adv", "分类器确认");
      return tags.length ? tags : ["模块A确认"];
    }

    function reasonSummary(reason) {
      const tags = reasonTags(reason);
      const core = tags.filter((tag) => tag !== "分类器确认");
      if (!reason) return "模块A已连续触发，建议复核该时间段。";
      return "检测到" + core.slice(0, 3).join("、") + "，已形成连续帧确认告警。";
    }

    function reasonText(reason) {
      if (!reason) return "";
      return reasonTags(reason).join("\u3001");
    }

    function stateText(status) {
      if (status.alert_confirmed) return ["确认告警", "检测到疑似物理扰动，建议复核画面与事件列表。", "alert"];
      if (status.running) return ["检测中", "视频流正在分析，暂无确认告警。", "live"];
      if ((status.frame_idx || 0) > 0) return ["已停止", "上一次检测已结束，可切换输入源重新开始。", ""];
      return ["系统待机", "尚未开始检测。", ""];
    }

    // Task 8.1 / Requirements 10.1, 10.2 — render the three branch cards
    // (p_adv / p_safety / p_synth) from the status.branch_cards payload
    // emitted by build_branch_cards() in Python. Each card is selected by
    // its data-branch attribute so the Python contract and the HTML skeleton
    // stay in lock-step without ad-hoc DOM lookup per branch.
    function renderBranchCards(cards) {
      if (!Array.isArray(cards)) return;
      const container = $("branchCards");
      if (!container) return;
      cards.forEach((card) => {
        const node = container.querySelector(
          `.branchCard[data-branch="${card.branch}"]`
        );
        if (!node) return;
        const borderClass = card.border_class || "card-idle";
        node.className = "branchCard " + borderClass;
        const titleEl = node.querySelector(".cardTitle");
        if (titleEl) titleEl.textContent = card.title || "";
        const stateEl = node.querySelector(".cardState");
        if (stateEl) stateEl.textContent = card.state || "";
        const scoreEl = node.querySelector(".cardScore");
        if (scoreEl) scoreEl.textContent = card.score_display || "--";
        const fillEl = node.querySelector(".cardBarFill");
        if (fillEl) {
          const ratio = Math.max(0, Math.min(1, Number(card.score_bar_ratio) || 0));
          fillEl.style.width = (ratio * 100).toFixed(1) + "%";
        }
        const detailEl = node.querySelector(".cardDetail");
        if (detailEl) detailEl.textContent = card.state_detail || "";
        const reasonEl = node.querySelector(".cardReason");
        if (reasonEl) {
          reasonEl.textContent = card.reason_text
            ? "最近原因：" + card.reason_text
            : "";
        }
        const badgesEl = node.querySelector(".cardBadges");
        if (badgesEl) {
          const badges = Array.isArray(card.badges) ? card.badges : [];
          badgesEl.innerHTML = badges
            .map((b) => `<span class="cardBadge">${b}</span>`)
            .join("");
        }
      });
    }

    function evidenceLinks(evt) {
      if (!evt.evidence_saved) {
        return '<div class="evidenceLinks"><span>证据保存中，事件结束后生成视频和帧</span></div>';
      }
      const links = [];
      if (evt.evidence_clip_url) {
        links.push(`<a href="${evt.evidence_clip_url}" target="_blank">查看证据视频</a>`);
      }
      if (evt.evidence_representative_url) {
        links.push(`<a href="${evt.evidence_representative_url}" target="_blank">查看代表帧</a>`);
      }
      if (evt.evidence_frames_dir) {
        links.push(`<span>逐帧已保存 ${evt.evidence_saved_frame_count || 0} 张</span>`);
      }
      return `<div class="evidenceLinks">${links.join("") || "<span>证据已保存</span>"}</div>`;
    }

    function a3bSourceText(source) {
      const key = String(source || "none");
      const labels = {
        fast: "快速确认",
        replay: "持续确认",
        static_media_fast: "快速确认",
        static_media_replay: "持续确认",
        static_image: "静态图像",
        none: "未触发",
      };
      return labels[key] || key;
    }

    function a3bDetails(evt) {
      const detail = evt.a3b_detail || {};
      if (!detail || Object.keys(detail).length === 0) return "";
      const replay = detail.replay || {};
      const fast = detail.fast || {};
      const occlusion = detail.occlusion || {};
      const bits = [];
      if (detail.media_type) bits.push(`类型 ${detail.media_type}`);
      if (Number.isFinite(Number(detail.live_score))) bits.push(`观察值=${Number(detail.live_score).toFixed(3)}`);
      if (Number.isFinite(Number(detail.score))) bits.push(`确认值=${Number(detail.score).toFixed(3)}`);
      if (detail.triggered_source) bits.push(`来源 ${a3bSourceText(detail.triggered_source)}`);
      if (fast.window) bits.push(`快速 ${fast.votes || 0}/${fast.trigger_count || fast.window}`);
      if (replay.window) bits.push(`持续 ${replay.votes || 0}/${replay.trigger_count || replay.window}`);
      if (Number.isFinite(Number(replay.bbox_area))) bits.push(`候选面积 ${Number(replay.bbox_area).toFixed(0)}`);
      if (Number.isFinite(Number(replay.target_iou))) bits.push(`目标IoU ${Number(replay.target_iou).toFixed(2)}`);
      return bits.length ? `<div class="sub">A3b细节：${bits.join(" · ")}</div>` : "";
    }

    async function api(path, body) {
      const res = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: body ? {"Content-Type": "application/json"} : {},
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await res.json();
      if (!res.ok || data.ok === false) {
        throw new Error(data.error || res.statusText);
      }
      return data;
    }

    function currentSourceValue() {
      if ($("sourceType").value === "camera") {
        return $("cameraSelect").value || $("sourceValue").value || "0";
      }
      return $("sourceValue").value;
    }

    function drawingOptions() {
      return {
        show_boxes: $("showBoxes").checked,
        show_module_hud: $("showModuleHud").checked,
        show_ppe_hud: $("showPpeHud").checked,
      };
    }

    function featureOptions() {
      // 伪造视频流检测仍在开发中，强制关闭（Web 端代码层面硬约束）
      const synthEnabled = false;
      return {
        static_image_enabled: $("enableStaticImage").checked,
        source_authenticity_enabled: synthEnabled,
        synth_strong_warning_enabled: synthEnabled,
      };
    }

    function customModelOptions() {
      return {
        enabled: $("enableCustomModel").checked,
        path: $("customModelPath").value,
        backend: $("customModelBackend").value,
        model_family: $("customModelFamily").value,
      };
    }

    function loadLastPaths() {
      try {
        const source = localStorage.getItem("moduleA.lastSourcePath");
        const model = localStorage.getItem("moduleA.lastModelPath");
        const modelEnabled = localStorage.getItem("moduleA.lastCustomModelEnabled");
        if (source) $("sourceValue").value = source;
        if (model) $("customModelPath").value = model;
        if (modelEnabled === "true") {
          $("enableCustomModel").checked = true;
          $("customModelPanel").style.display = "block";
        }
      } catch (_) {}
    }

    function saveLastPaths() {
      try {
        if ($("sourceValue").value) localStorage.setItem("moduleA.lastSourcePath", $("sourceValue").value);
        if ($("customModelPath").value) localStorage.setItem("moduleA.lastModelPath", $("customModelPath").value);
        localStorage.setItem("moduleA.lastCustomModelEnabled", $("enableCustomModel").checked ? "true" : "false");
      } catch (_) {}
    }

    async function syncDrawingOptions() {
      await api("/api/display-options", { display_options: drawingOptions() });
    }

    function updateSourceControls() {
      const type = $("sourceType").value;
      $("pathRow").style.display = type === "camera" ? "none" : "grid";
      $("cameraRow").style.display = type === "camera" ? "grid" : "none";
      $("browseBtn").style.display = type === "file" ? "block" : "none";
      $("testSourceBtn").style.display = type === "rtsp" ? "block" : "none";
      $("pathRow").className = type === "file" || type === "rtsp" ? "fieldRow" : "fieldRow single";
      $("realtime").disabled = type !== "file";
      $("realtimeText").textContent = type === "file" ? "MP4按视频FPS播放" : "实时流模式";
      setConnectionStatus("", "");
      if (type === "file") {
        $("sourceValueLabel").textContent = "MP4 文件";
        $("sourceHint").textContent = "点击“浏览”可打开资源管理器选择本机 MP4，也可以手动粘贴路径。";
      } else if (type === "rtsp") {
        $("sourceValueLabel").textContent = "RTSP / HTTP 地址";
        $("sourceHint").textContent = "输入摄像头或视频服务地址，先点“连通检测”，通过后再开始检测。";
      } else {
        $("sourceValueLabel").textContent = "本机摄像头";
        $("sourceHint").textContent = "点击“扫描”识别本机可打开的摄像头，然后从列表选择。";
        if (!cameraDevicesLoaded) scanCameras();
      }
    }

    async function pickFile() {
      setError("正在打开资源管理器...");
      $("browseBtn").disabled = true;
      try {
        const data = await api("/api/pick-file", {mode: "video", current_path: $("sourceValue").value || ""});
        if (data.path) {
          $("sourceValue").value = data.path;
          saveLastPaths();
          setError("");
        }
      } catch (err) {
        setError(err.message);
      } finally {
        $("browseBtn").disabled = false;
      }
    }

    async function pickModelFile() {
      setError("正在打开模型文件选择器...");
      $("browseModelBtn").disabled = true;
      try {
        const data = await api("/api/pick-file", {mode: "model", current_path: $("customModelPath").value || ""});
        if (data.path) {
          $("customModelPath").value = data.path;
          saveLastPaths();
          setError("");
        }
      } catch (err) {
        setError(err.message);
      } finally {
        $("browseModelBtn").disabled = false;
      }
    }

    async function scanCameras() {
      setError("正在扫描本机摄像头...");
      setConnectionStatus("", "");
      $("scanBtn").disabled = true;
      $("cameraSelect").innerHTML = '<option value="">扫描中...</option>';
      try {
        const data = await api("/api/cameras");
        const devices = data.devices || [];
        cameraDevicesLoaded = true;
        if (!devices.length) {
          $("cameraSelect").innerHTML = '<option value="">未发现可打开摄像头</option>';
          setError("未发现可打开摄像头，请确认摄像头未被其他软件占用。");
          return;
        }
        $("cameraSelect").innerHTML = devices.map((device) => {
          const detail = device.width && device.height ? ` · ${device.width}x${device.height}` : "";
          return `<option value="${device.index}">${device.name}${detail}</option>`;
        }).join("");
        setError("");
      } catch (err) {
        setError(err.message);
      } finally {
        $("scanBtn").disabled = false;
      }
    }

    async function testSource() {
      setError("");
      setConnectionStatus("正在检测连通性...", "testing");
      $("testSourceBtn").disabled = true;
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        const res = await fetch("/api/test-source", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            source_type: $("sourceType").value,
            source: currentSourceValue(),
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          throw new Error(data.error || res.statusText);
        }
        if (data.reachable) {
          setConnectionStatus(data.message || "连接可用。", "ok");
        } else {
          setConnectionStatus(data.message || "连接不可用。", "bad");
        }
      } catch (err) {
        if (err.name === "AbortError") {
          setConnectionStatus("连通检测超时（15秒），请检查地址是否正确。", "bad");
        } else {
          setConnectionStatus(err.message || "Failed to fetch", "bad");
        }
      } finally {
        $("testSourceBtn").disabled = false;
      }
    }

    $("sourceType").onchange = updateSourceControls;
    $("browseBtn").onclick = pickFile;
    $("browseModelBtn").onclick = pickModelFile;
    $("scanBtn").onclick = scanCameras;
    $("testSourceBtn").onclick = testSource;
    $("enableCustomModel").onchange = () => {
      $("customModelPanel").style.display = $("enableCustomModel").checked ? "block" : "none";
    };
    $("cameraSelect").onchange = () => {};
    ["showBoxes", "showModuleHud", "showPpeHud"].forEach((id) => {
      $(id).onchange = async () => {
        try {
          await syncDrawingOptions();
        } catch (err) {
          setError(err.message);
        }
      };
    });

    $("startBtn").onclick = async () => {
      setError("");
      try {
        await syncDrawingOptions();
        const body = {
          source_type: $("sourceType").value,
          source: currentSourceValue(),
          profile: $("profile").value,
          realtime: $("realtime").checked,
          feature_options: featureOptions(),
          custom_model: customModelOptions(),
        };
        saveLastPaths();
        await api("/api/start", body);
        streamNonce += 1;
        $("stream").src = "/stream.mjpg?nonce=" + streamNonce;
        $("empty").style.display = "none";
        $("stream").style.display = "block";
      } catch (err) {
        setError(err.message);
      }
    };

    $("stopBtn").onclick = async () => {
      setError("");
      try {
        await api("/api/stop", {});
        $("stream").removeAttribute("src");
        $("stream").style.display = "none";
        $("empty").style.display = "block";
      } catch (err) {
        setError(err.message);
      }
    };

    async function refresh() {
      try {
        const data = await api("/api/status");
        const status = data.status;
        const pill = $("statePill");
        const [title, text, stateClass] = stateText(status);
        pill.textContent = title;
        pill.className = "pill " + stateClass;
        renderBranchCards(status.branch_cards || []);
        $("profilePill").textContent = profileLabel(status.profile || $("profile").value);
        $("backendPill").textContent = status.backend || "-";
        $("frame").textContent = status.frame_idx ?? 0;
        $("pAdv").textContent = Number(status.p_adv || 0).toFixed(3);
        $("pSynth").textContent = Number(status.p_synth || 0).toFixed(3);
        $("timing").textContent = Number(status.timing_ms || 0).toFixed(1);
        $("alerts").textContent = status.alert_event_count || 0;
        $("ppeAlerts").textContent = status.ppe_event_count || 0;
        $("sourceAlerts").textContent = status.source_authenticity_event_count || 0;
        $("ppeCounts").textContent = `${status.ppe_person_count || 0}/${status.ppe_helmet_count || 0}`;
        $("ppePill").textContent = status.ppe_warning ? "未戴帽" : "业务通道";
        $("ppePill").className = "pill " + (status.ppe_warning ? "alert" : "");
        $("sourcePill").textContent = status.source_authenticity_warning ? "疑似伪造" : "独立通道";
        $("sourcePill").className = "pill " + (status.source_authenticity_warning ? "alert" : "");
        const featureStatus = status.feature_options || {};
        if (status.running) {
          $("enableStaticImage").checked = featureStatus.static_image_enabled !== false;
          // 伪造视频流告警强制保持关闭，即使后端历史状态为 true 也不回写
          $("enableSynthWarning").checked = false;
          $("enableSynthWarning").disabled = true;
          const customStatus = status.custom_model || {};
          $("enableCustomModel").checked = customStatus.enabled === true;
          $("customModelPath").value = customStatus.path || $("customModelPath").value;
          $("customModelBackend").value = customStatus.backend || "auto";
          $("customModelFamily").value = customStatus.model_family || "ultralytics";
          $("customModelPanel").style.display = $("enableCustomModel").checked ? "block" : "none";
        }
        const displayOptions = status.display_options || {};
        $("showBoxes").checked = displayOptions.show_boxes !== false;
        $("showModuleHud").checked = displayOptions.show_module_hud !== false;
        $("showPpeHud").checked = displayOptions.show_ppe_hud !== false;
        if ((status.running || (status.frame_idx || 0) > 0) && !$("stream").getAttribute("src")) {
          streamNonce += 1;
          $("stream").src = "/stream.mjpg?nonce=" + streamNonce;
          $("stream").style.display = "block";
          $("empty").style.display = "none";
        }
        if (status.error) setError(status.error);
        const events = status.recent_events || [];
        $("events").innerHTML = events.map((evt) => {
          const tags = reasonTags(evt.reason);
          return `
          <div class="event alert">
            <strong>告警 #${evt.event_id}</strong>
            <div class="main">${reasonSummary(evt.reason)}</div>
            <div class="sub">帧 ${evt.trigger_frame} - ${evt.last_alert_frame} · 峰值概率 ${Number(evt.peak_p_adv || 0).toFixed(3)}</div>
            <div class="chips">
              <span class="chip danger">高风险</span>
              ${tags.map((tag) => `<span class="chip">${tag}</span>`).join("")}
            </div>
            ${a3bDetails(evt)}
            ${evidenceLinks(evt)}
            <details>
              <summary>技术细节</summary>
              <code>${evt.reason || "module_a"}</code>
            </details>
          </div>
        `;
        }).join("") || '<div class="empty"><strong>暂无确认告警</strong><div class="hint">一旦连续帧触发，告警会出现在这里。</div></div>';
        const ppeEvents = status.recent_ppe_events || [];
        $("ppeEvents").innerHTML = ppeEvents.map((evt) => `
          <div class="event ppe">
            <strong>安全帽告警 #${evt.event_id}</strong>
            <div class="main">检测到人员未佩戴安全帽，已按连续帧规则确认。</div>
            <div class="sub">帧 ${evt.trigger_frame} - ${evt.last_warning_frame} · 人员 ${evt.person_count} · 裸头 ${evt.head_count || 0} · 安全帽 ${evt.helmet_count}</div>
            <div class="chips">
              <span class="chip warn">业务安全</span>
              <span class="chip">未戴安全帽</span>
              <span class="chip">独立于模块A</span>
            </div>
            ${evidenceLinks(evt)}
          </div>
        `).join("") || '<div class="empty"><strong>暂无安全帽告警</strong><div class="hint">该通道只判断人员是否佩戴安全帽，不代表物理扰动。</div></div>';
        const sourceEvents = status.recent_source_auth_events || [];
        $("sourceEvents").innerHTML = sourceEvents.map((evt) => `
          <div class="event alert">
            <strong>伪造流提示 #${evt.event_id}</strong>
            <div class="main">输入视频流疑似存在生成、伪造或重放特征。</div>
            <div class="sub">帧 ${evt.trigger_frame} - ${evt.last_warning_frame} · 峰值概率 ${Number(evt.peak_p_synth || 0).toFixed(3)}</div>
            <div class="chips">
              <span class="chip danger">源真实性</span>
              <span class="chip">${evt.reason || "clip级统计异常"}</span>
              <span class="chip">独立于模块A</span>
            </div>
            ${evidenceLinks(evt)}
          </div>
        `).join("") || '<div class="empty"><strong>暂无伪造流提示</strong><div class="hint">该通道只提示输入源真实性风险，不代表物理扰动。</div></div>';
      } catch (err) {
        setError(err.message);
      }
    }

    setInterval(refresh, 750);
    loadLastPaths();
    updateSourceControls();
    refresh();
  </script>
</body>
</html>
"""


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text.strip().strip('"'))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def merge_profile(
    module_config: dict[str, Any], profile_config_path: Path, profile_name: str
) -> dict[str, Any]:
    profile_config = load_yaml(profile_config_path)
    profiles = profile_config.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown runtime profile: {profile_name}")
    profile = profiles[profile_name] or {}
    detector_backend = profile.get("detector_backend")
    if detector_backend:
        module_config.setdefault("inference", {})["backend"] = detector_backend
    for key, value in (profile.get("module_overrides") or {}).items():
        module_config.setdefault("module_a", {})[key] = parse_scalar(str(value))
    return profile


def apply_feature_options(
    module_config: dict[str, Any], feature_options: dict[str, Any] | None
) -> None:
    options = feature_options or {}
    module_a = module_config.setdefault("module_a", {})
    if "static_image_enabled" in options:
        module_a["static_image_enabled"] = bool(options["static_image_enabled"])
    if "source_authenticity_enabled" in options:
        # Merged checkbox: enabling source_authenticity also enables the
        # synth classifier gate so both features activate together.
        enabled = bool(options["source_authenticity_enabled"])
        module_a["source_authenticity_enabled"] = enabled
        module_a["synth_classifier_enabled"] = enabled
    # synth_strong_warning_enabled is a Monitor_App UI-level filter only.
    # It does not flip pipeline-side gates.


def infer_backend_from_model_path(path: Path, fallback: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".engine":
        return "tensorrt"
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    return fallback


def normalize_custom_model_options(custom_model: dict[str, Any] | None) -> dict[str, Any]:
    data = custom_model or {}
    enabled = bool(data.get("enabled", False))
    path_text = str(data.get("path", "")).strip()
    backend = str(data.get("backend", "auto")).strip().lower()
    family = str(data.get("model_family", "ultralytics")).strip().lower()
    if family not in {"ultralytics", "yolov5"}:
        family = "ultralytics"
    return {
        "enabled": enabled,
        "path": path_text,
        "backend": backend,
        "model_family": family,
    }


def apply_custom_model(
    module_config: dict[str, Any], custom_model: dict[str, Any] | None
) -> dict[str, Any]:
    options = normalize_custom_model_options(custom_model)
    if not options["enabled"]:
        return options

    path = resolve_path(options["path"])
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"自定义模型文件不存在: {path}")

    inference = module_config.setdefault("inference", {})
    current_backend = str(inference.get("backend", "tensorrt")).lower()
    backend = options["backend"]
    if backend == "auto":
        backend = infer_backend_from_model_path(path, current_backend)
    if backend not in {"tensorrt", "onnx", "pytorch"}:
        raise ValueError("自定义模型后端必须是 auto、tensorrt、onnx 或 pytorch")

    artifact_key = "engine" if backend == "tensorrt" else backend
    inference["backend"] = backend
    inference["model_family"] = options["model_family"]
    inference["artifacts"] = {artifact_key: [str(path)]}
    options.update({"path": str(path), "backend": backend})
    return options


def info_reason(info: dict[str, Any]) -> str:
    feature_details = info.get("details", {}).get("module_a_features", {})
    fusion = feature_details.get("fusion", {})
    if fusion.get("reason"):
        return str(fusion["reason"])
    reason_codes = info.get("reason_codes", [])
    return ",".join(str(item) for item in reason_codes)


# --------------------------------------------------------------------------
# Branch cards (Task 8.1 / Requirements 10.1, 10.2)
# --------------------------------------------------------------------------
#
# ``build_branch_cards`` is a pure function that converts the monitor status
# dict (as returned by ``MonitorState.get_status``) into a three-card payload
# consumed by the Monitor_App right panel. Keeping the mapping in Python (as
# opposed to only in JS) makes it testable without starting the web server,
# and lets the three cards share a single authoritative contract with the
# triple-channel ``info`` fields:
#
# * ``p_adv`` card   ← ``info["p_adv"]`` + ``info["reason_codes"]`` + the
#   existing 3/5 ``alert_confirmed`` state machine.
# * ``p_safety`` card ← currently falls back to the business-side PPE signals
#   because the pipeline stamps ``info["p_safety"] = None`` as a structural
#   placeholder (see ``VideoDefensePipeline._run_detection``). When/if a
#   numeric ``p_safety`` score lands in the pipeline contract, the card will
#   automatically start showing it without further UI changes.
# * ``p_synth`` card ← ``info["p_synth"]`` + the Source_Authenticity suppress
#   flag from ``info["details"]["module_a_features"]["source_authenticity"]``.
#
# Missing channels are rendered as "未启用/数据缺失: <reason>" using the
# ``<branch>_missing_reason`` fields; no silent fallback to zero.
_BRANCH_ID_P_ADV = "p_adv"
_BRANCH_ID_P_SAFETY = "p_safety"
_BRANCH_ID_P_SYNTH = "p_synth"


def _format_score(value: Any) -> str:
    """Render a branch score for the HUD card.

    ``None`` maps to ``"--"`` so missing channels are visually distinct from
    a real ``0.000`` reading. Non-numeric inputs fall back to ``"--"`` rather
    than raising, to keep the status endpoint crash-free.
    """
    if value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _score_bar_ratio(value: Any) -> float:
    """Clamp a branch score to ``[0.0, 1.0]`` for the score bar width."""
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _card_border(severity: str) -> str:
    """Pick a border class name matching the design.md colour contract.

    design.md §7: "确认告警时对应卡片边框变红" — confirmed 红 / warning 橙 /
    其它灰。``missing`` cards land on ``card-missing`` (muted) so operators
    can tell disabled branches apart from merely-idle ones.
    """
    if severity == "confirmed":
        return "card-confirmed"
    if severity == "warning":
        return "card-warning"
    if severity == "missing":
        return "card-missing"
    if severity == "suppressed":
        return "card-suppressed"
    return "card-idle"


def _build_p_adv_card(status: dict[str, Any]) -> dict[str, Any]:
    missing_reason = status.get("p_adv_missing_reason") or ""
    score = status.get("p_adv_display", status.get("p_adv"))
    score_value = _score_bar_ratio(score)
    alert_confirmed = bool(status.get("alert_confirmed", False))
    attack_state_active = bool(status.get("attack_state_active", False))
    reason_text = str(status.get("reason", ""))

    if missing_reason:
        return {
            "branch": _BRANCH_ID_P_ADV,
            "title": "物理对抗扰动（p_adv）",
            "score": None,
            "score_display": "--",
            "score_bar_ratio": 0.0,
            "state": "数据缺失",
            "state_detail": f"未启用/数据缺失: {missing_reason}",
            "severity": "missing",
            "border_class": _card_border("missing"),
            "reason_text": "",
            "enabled": False,
            "missing_reason": missing_reason,
            "suppressed": False,
            "badges": [],
        }

    if alert_confirmed:
        state = "Confirmed_Alert"
        state_detail = "连续帧确认告警，建议立即复核"
        severity = "confirmed"
    elif attack_state_active:
        state = "Warning"
        state_detail = "单帧触发命中，正在等待连续帧确认"
        severity = "warning"
    elif score_value >= 0.55:
        state = "候选观察"
        state_detail = "物理扰动概率升高，正在等待连续帧确认"
        severity = "warning"
    else:
        state = "OK"
        state_detail = "未触发物理扰动检测"
        severity = "idle"

    return {
        "branch": _BRANCH_ID_P_ADV,
        "title": "物理对抗扰动（p_adv）",
        "score": float(score) if score is not None else None,
        "score_display": _format_score(score),
        "score_bar_ratio": _score_bar_ratio(score),
        "state": state,
        "state_detail": state_detail,
        "severity": severity,
        "border_class": _card_border(severity),
        "reason_text": reason_text,
        "enabled": True,
        "missing_reason": "",
        "suppressed": False,
        "badges": [],
    }


def _build_p_safety_card(status: dict[str, Any]) -> dict[str, Any]:
    # Replaced: the p_safety card now shows A3b static media detection status
    # instead of the old PPE business-side placeholder. The card reads from
    # the ``a3b_*`` fields populated in ``_run_capture`` from
    # ``info["details"]["module_a_features"]["static_media"]``.
    missing_reason = status.get("p_safety_missing_reason") or ""
    confirmed_score = status.get("a3b_score")
    raw_live_score = status.get("a3b_live_score_display", status.get("a3b_live_score", confirmed_score))
    triggered = bool(status.get("a3b_triggered", False))
    media_type = str(status.get("a3b_media_type", "normal"))
    trigger_count = int(status.get("a3b_trigger_count", 0))
    p_media = float(status.get("a3b_p_media", 0.0) or 0.0)
    replay_state = status.get("a3b_replay_state") or {}
    fast_state = status.get("a3b_fast_state") or {}
    occlusion_state = status.get("a3b_occlusion_state") or {}
    candidate = bool(replay_state.get("candidate", False) or fast_state.get("candidate", False))
    score = confirmed_score if triggered else (raw_live_score if candidate else 0.0)

    if missing_reason or score is None:
        # Branch is disabled or no data yet (empty status before first frame).
        effective_reason = missing_reason or "a3b_no_data"
        return {
            "branch": _BRANCH_ID_P_SAFETY,
            "title": "翻拍/假目标检测（A3b）",
            "score": None,
            "score_display": "--",
            "score_bar_ratio": 0.0,
            "state": "未启用" if missing_reason else "数据缺失",
            "state_detail": f"未启用: {missing_reason}" if missing_reason else "等待首帧数据",
            "severity": "missing",
            "border_class": _card_border("missing"),
            "reason_text": "",
            "enabled": False,
            "missing_reason": effective_reason,
            "suppressed": False,
            "badges": [],
        }

    fast_votes = int(fast_state.get("votes", 0) or 0)
    fast_need = int(fast_state.get("trigger_count", 0) or 0)
    replay_votes = int(replay_state.get("votes", 0) or 0)
    replay_need = int(replay_state.get("trigger_count", 0) or 0)
    source_text = {
        "fast": "快速确认",
        "replay": "持续确认",
        "static_media_fast": "快速确认",
        "static_media_replay": "持续确认",
        "static_image": "静态图像",
        "none": "未触发",
    }.get(str(status.get("a3b_triggered_source", "none")), str(status.get("a3b_triggered_source", "none")))
    observer_text = (
        f"观察值 p_media={p_media:.3f}，快速 {fast_votes}/{fast_need}，持续 {replay_votes}/{replay_need}"
    )

    if triggered:
        severity = "confirmed"
        state = "确认告警"
        state_detail = (
            f"主分=确认报警概率 {_format_score(score)} · {source_text} · "
            f"{observer_text} · 类型 {media_type} · 累计 {trigger_count}"
        )
    elif candidate:
        severity = "warning"
        state = "候选观察"
        state_detail = (
            f"主分=候选报警概率 {_format_score(score)} · {observer_text} · 正在等待连续确认"
        )
    else:
        severity = "idle"
        state = "正常"
        state_detail = f"主分=确认报警概率 0.000 · {observer_text} · 未进入报警候选"

    return {
        "branch": _BRANCH_ID_P_SAFETY,
        "title": "翻拍/假目标检测（A3b）",
        "score": float(score),
        "score_display": _format_score(score),
        "score_bar_ratio": _score_bar_ratio(score),
        "state": state,
        "state_detail": state_detail,
        "severity": severity,
        "border_class": _card_border(severity),
        "reason_text": (
            f"{source_text}；观察值 p_media={p_media:.3f}；确认分={float(confirmed_score or 0.0):.3f}"
            if triggered
            else (
                f"候选观察；观察值 p_media={p_media:.3f}；候选分={float(raw_live_score or 0.0):.3f}"
                if candidate
                else ""
            )
        ),
        "enabled": True,
        "missing_reason": "",
        "suppressed": False,
        "badges": ["Confirmed"] if triggered else (["Candidate"] if candidate else []),
    }


def _build_p_synth_card(status: dict[str, Any]) -> dict[str, Any]:
    missing_reason = status.get("p_synth_missing_reason") or ""
    score = status.get("p_synth")
    enabled = bool(status.get("source_authenticity_enabled", False))
    suppressed = bool(status.get("source_authenticity_suppressed_by_p_adv", False))
    warning = bool(status.get("source_authenticity_warning", False))
    confirmed = bool(status.get("source_authenticity_confirmed", False))
    reason_text = str(status.get("source_authenticity_reason", ""))

    if not enabled or (score is None and missing_reason):
        effective_reason = missing_reason or (
            "source_authenticity_disabled" if not enabled else "p_synth_unavailable"
        )
        return {
            "branch": _BRANCH_ID_P_SYNTH,
            "title": "视频源真实性（p_synth）",
            "score": None,
            "score_display": "--",
            "score_bar_ratio": 0.0,
            "state": "数据缺失" if enabled else "未启用",
            "state_detail": f"未启用/数据缺失: {effective_reason}",
            "severity": "missing",
            "border_class": _card_border("missing"),
            "reason_text": "",
            "enabled": enabled,
            "missing_reason": effective_reason,
            "suppressed": False,
            "badges": [],
        }

    badges: list[str] = []
    if suppressed:
        # design.md §7: 物理扰动活跃时显示"已被物理扰动抑制"，但保留数值展示
        badges.append("抑制中")
        severity = "suppressed"
        state = "抑制中"
        state_detail = "物理扰动活跃，已抑制 p_synth 强告警（数值保留用于记录）"
    elif confirmed:
        severity = "confirmed"
        state = "Confirmed_Alert"
        state_detail = "clip 级连续帧确认，疑似伪造视频源"
    elif warning:
        severity = "warning"
        state = "Warning"
        state_detail = "clip 级单次命中，等待连续帧确认"
    else:
        severity = "idle"
        state = "OK"
        state_detail = "clip 级统计未触发"

    return {
        "branch": _BRANCH_ID_P_SYNTH,
        "title": "视频源真实性（p_synth）",
        "score": float(score) if score is not None else None,
        "score_display": _format_score(score),
        "score_bar_ratio": _score_bar_ratio(score),
        "state": state,
        "state_detail": state_detail,
        "severity": severity,
        "border_class": _card_border(severity),
        "reason_text": reason_text,
        "enabled": enabled,
        "missing_reason": "",
        "suppressed": suppressed,
        "badges": badges,
    }


def build_branch_cards(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the right-panel three-card payload from a monitor status dict.

    Cards are returned in the fixed order ``[p_adv, p_safety, p_synth]`` to
    match the design.md §7 specification. Each card carries:

    * ``branch`` — stable branch id (``p_adv`` / ``p_safety`` / ``p_synth``).
    * ``title`` — human-readable Chinese title.
    * ``score`` / ``score_display`` / ``score_bar_ratio`` — numeric score,
      formatted string, and ``[0, 1]`` ratio for the score bar width.
    * ``state`` / ``state_detail`` — alarm state label and longer detail.
    * ``severity`` / ``border_class`` — severity tag and CSS class driving
      the highlight border (red / orange / grey / muted).
    * ``reason_text`` — most recent reason string, if any.
    * ``enabled`` / ``missing_reason`` — whether the branch is active and
      why it is missing when not.
    * ``suppressed`` / ``badges`` — ``p_synth`` carries the ``抑制中`` badge
      when Source_Authenticity has been suppressed by a physical-perturbation
      alert (design.md §7, Requirement 1.4).

    The function is intentionally pure so the Monitor_App UI test can assert
    the card contract without spinning up the HTTP server.
    """
    status = status or {}
    return [
        _build_p_adv_card(status),
        _build_p_safety_card(status),
        _build_p_synth_card(status),
    ]


def normalize_label(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def label_matches(label: str, hints: tuple[str, ...]) -> bool:
    normalized = normalize_label(label)
    return any(hint in normalized for hint in hints)


def summarize_ppe_from_detections(detections: Any) -> dict[str, Any]:
    person_hints = ("person", "worker", "pedestrian")
    helmet_hints = ("helmet", "hardhat", "hard_hat", "safety_helmet")
    bare_head_hints = ("head", "no_helmet", "without_helmet", "bare_head")

    person_count = 0
    helmet_count = 0
    head_count = 0
    other_counts: dict[str, int] = {}
    for cls_id, confidence in zip(detections.classes, detections.confidences):
        label = str(detections.names.get(int(cls_id), f"class_{int(cls_id)}"))
        other_counts[label] = other_counts.get(label, 0) + 1
        if float(confidence) < 0.25:
            continue
        if label_matches(label, bare_head_hints):
            head_count += 1
        elif label_matches(label, helmet_hints):
            helmet_count += 1
        elif label_matches(label, person_hints):
            person_count += 1

    missing_helmet_count = 0
    if person_count > 0:
        missing_helmet_count = max(person_count - helmet_count, 0)
    if head_count > 0:
        missing_helmet_count = max(missing_helmet_count, head_count)

    candidate = (person_count > 0 or head_count > 0) and missing_helmet_count > 0
    reason = ""
    if candidate:
        if head_count > 0:
            reason = "检测到裸头/头部目标，且安全帽证据不足"
        elif helmet_count == 0:
            reason = "检测到人员，但未检测到安全帽"
        else:
            reason = "人员数量多于安全帽数量"
    elif person_count > 0:
        reason = "检测到人员，安全帽数量满足当前检测结果"
    else:
        reason = "未检测到人员"

    return {
        "person_count": person_count,
        "helmet_count": helmet_count,
        "head_count": head_count,
        "missing_helmet_count": missing_helmet_count,
        "candidate": candidate,
        "reason": reason,
        "class_counts": other_counts,
    }


class SafetyHelmetState:
    """Business-rule PPE warning state, separate from Module A perturbation alerts."""

    def __init__(self, window: int = 12, trigger_count: int = 8, hold_frames: int = 12):
        self.window = max(1, int(window))
        self.trigger_count = max(1, min(int(trigger_count), self.window))
        self.hold_frames = max(0, int(hold_frames))
        self.queue: deque[int] = deque(maxlen=self.window)
        self.hold_remaining = 0

    def reset(self) -> None:
        self.queue.clear()
        self.hold_remaining = 0

    def update(self, ppe: dict[str, Any]) -> dict[str, Any]:
        candidate = bool(ppe.get("candidate", False))
        self.queue.append(1 if candidate else 0)
        confirmed = sum(self.queue) >= self.trigger_count
        if confirmed:
            self.hold_remaining = self.hold_frames
        elif self.hold_remaining > 0:
            self.hold_remaining -= 1
        warning = bool(confirmed or self.hold_remaining > 0)
        return {
            **ppe,
            "warning": warning,
            "confirmed": bool(confirmed),
            "state_active": warning,
            "window_positive": int(sum(self.queue)),
            "window": self.window,
            "trigger_count": self.trigger_count,
            "hold_remaining": int(self.hold_remaining),
        }


def draw_ppe_hud(frame, ppe: dict[str, Any]):
    if not ppe.get("warning"):
        return frame
    h, w = frame.shape[:2]
    color = (0, 190, 255)
    cv2.rectangle(frame, (6, h - 62), (w - 6, h - 8), color, 3)
    cv2.rectangle(frame, (8, h - 60), (w - 8, h - 32), (20, 35, 45), -1)
    cv2.putText(
        frame,
        "PPE WARNING: NO HELMET",
        (18, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"person={ppe.get('person_count', 0)} helmet={ppe.get('helmet_count', 0)}",
        (18, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def safe_path_part(text: str, fallback: str = "source", max_len: int = 80) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", text.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:max_len].strip("._-") or fallback


def source_slug(source_type: str, source: str) -> str:
    if source_type == "file":
        path = Path(source.strip().strip('"'))
        name = path.stem or path.name
        return safe_path_part(name, "mp4")
    if source_type == "camera":
        return safe_path_part(f"camera_{source}", "camera")
    if source_type == "rtsp":
        parsed = urlparse(source)
        label = parsed.hostname or source[:40]
        return safe_path_part(f"stream_{label}", "stream")
    return safe_path_part(source_type, "source")


def evidence_file_url(path: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(EVIDENCE_ROOT.resolve())
    except ValueError:
        return ""
    return "/evidence/" + quote(rel_path.as_posix())


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, default=json_default)


def write_image_file(path: Path, image: Any, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    path.write_bytes(encoded.tobytes())


def build_a3b_detail(status: dict[str, Any]) -> dict[str, Any]:
    reason = str(status.get("reason", ""))
    replay = status.get("a3b_replay_state")
    should_attach = bool(status.get("a3b_triggered", False)) or any(
        token in reason
        for token in ("static_media", "static_image", "A3b", "翻拍", "假目标")
    )
    if not should_attach:
        return {}
    detail: dict[str, Any] = {
        "score": status.get("a3b_score"),
        "live_score": status.get("a3b_live_score"),
        "p_media": status.get("a3b_p_media"),
        "triggered": bool(status.get("a3b_triggered", False)),
        "media_type": str(status.get("a3b_media_type", "normal")),
        "trigger_count": int(status.get("a3b_trigger_count", 0)),
        "triggered_source": str(status.get("a3b_triggered_source", "none")),
        "bbox": status.get("a3b_bbox"),
    }
    fast = status.get("a3b_fast_state")
    occlusion = status.get("a3b_occlusion_state")
    if isinstance(replay, dict):
        detail["replay"] = replay
    if isinstance(fast, dict):
        detail["fast"] = fast
    if isinstance(occlusion, dict):
        detail["occlusion"] = occlusion
    return detail


def monitor_frame_record(
    channel: str, frame_idx: int, active: bool, status: dict[str, Any]
) -> dict[str, Any]:
    return {
        "frame": int(frame_idx),
        "channel": channel,
        "active": bool(active),
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "p_adv": float(status.get("p_adv", 0.0) or 0.0),
        "alert_confirmed": bool(status.get("alert_confirmed", False)),
        "attack_detected": bool(status.get("attack_detected", False)),
        "attack_state_active": bool(status.get("attack_state_active", False)),
        "module_a_reason": str(status.get("reason", "")),
        "a3b_detail": build_a3b_detail(status),
        "ppe_warning": bool(status.get("ppe_warning", False)),
        "ppe_candidate": bool(status.get("ppe_candidate", False)),
        "ppe_person_count": int(status.get("ppe_person_count", 0)),
        "ppe_helmet_count": int(status.get("ppe_helmet_count", 0)),
        "ppe_head_count": int(status.get("ppe_head_count", 0)),
        "ppe_missing_helmet_count": int(status.get("ppe_missing_helmet_count", 0)),
        "ppe_reason": str(status.get("ppe_reason", "")),
        "p_synth": float(status.get("p_synth", 0.0)),
        "source_authenticity_enabled": bool(status.get("source_authenticity_enabled", False)),
        "source_authenticity_warning": bool(status.get("source_authenticity_warning", False)),
        "source_authenticity_confirmed": bool(status.get("source_authenticity_confirmed", False)),
        "source_authenticity_available": bool(status.get("source_authenticity_available", False)),
        "source_authenticity_reason": str(status.get("source_authenticity_reason", "")),
        "timing_ms": float(status.get("timing_ms", 0.0)),
        "detector_inference_ms": float(status.get("detector_inference_ms", 0.0)),
        "module_a_timing_ms": float(status.get("module_a_timing_ms", 0.0)),
        "backend": status.get("backend"),
        "profile": status.get("profile"),
    }


class ChannelEvidenceRecorder:
    def __init__(
        self,
        session_dir: Path,
        channel: str,
        channel_label: str,
        fps: float = 25.0,
        pre_frames: int = 15,
        post_frames: int = 30,
        max_events: int = 20,
        max_clip_frames: int = 360,
        pre_frame_stride: int = 3,
    ):
        self.session_dir = session_dir
        self.channel = channel
        self.channel_label = channel_label
        self.fps = self._valid_fps(fps)
        self.pre_buffer: deque[dict[str, Any]] = deque(maxlen=max(0, int(pre_frames)))
        self.post_frames = max(0, int(post_frames))
        self.max_events = max(0, int(max_events))
        self.max_clip_frames = max(1, int(max_clip_frames))
        self.pre_frame_stride = max(1, int(pre_frame_stride))
        self.current_event: dict[str, Any] | None = None
        self.saved_events: list[dict[str, Any]] = []
        self.saved_count = 0

    @staticmethod
    def _valid_fps(fps: float) -> float:
        value = float(fps or 0.0)
        if value < 1.0 or value > 120.0:
            return 25.0
        return value

    def set_fps(self, fps: float) -> None:
        self.fps = self._valid_fps(fps)

    def wants_frame(self, frame_idx: int, active: bool) -> bool:
        return bool(
            active or self.current_event is not None or frame_idx % self.pre_frame_stride == 0
        )

    def record_frame(
        self,
        frame_idx: int,
        image: Any,
        active: bool,
        event_id: int | None,
        score: float,
        reason: str,
        record: dict[str, Any],
        started_at: str | None = None,
    ) -> dict[str, Any] | None:
        entry = {
            "frame": int(frame_idx),
            "image": image.copy(),
            "record": dict(record),
            "active": bool(active),
            "score": float(score),
            "reason": str(reason),
        }

        completed: dict[str, Any] | None = None
        if active:
            if self.current_event is not None and self.current_event.get("in_post", False):
                completed = self.finalize_current()
            if self.current_event is None:
                if self.saved_count >= self.max_events:
                    self.pre_buffer.append(entry)
                    return None
                self.current_event = self._create_event(
                    event_id or (self.saved_count + 1),
                    frame_idx,
                    started_at=started_at,
                )
                for buffered in self.pre_buffer:
                    self._append_entry(self.current_event, buffered)
            self.current_event["post_remaining"] = self.post_frames
            self.current_event["in_post"] = False
            self._append_entry(self.current_event, entry)
        elif self.current_event is not None:
            if int(self.current_event.get("post_remaining", 0)) > 0:
                self.current_event["in_post"] = True
                self._append_entry(self.current_event, entry)
                self.current_event["post_remaining"] = (
                    int(self.current_event.get("post_remaining", 0)) - 1
                )
            else:
                completed = self.finalize_current()

        self.pre_buffer.append(entry)
        return completed

    def finalize_current(self) -> dict[str, Any] | None:
        if self.current_event is None:
            return None
        summary = self._finalize_event(self.current_event)
        self.current_event = None
        self.saved_count += 1
        self.saved_events.append(summary)
        return summary

    def close(self) -> dict[str, Any] | None:
        return self.finalize_current()

    def _create_event(
        self, event_id: int, trigger_frame: int, started_at: str | None = None
    ) -> dict[str, Any]:
        event_name = f"event_{int(event_id):03d}"
        event_dir = self.session_dir / self.channel / event_name
        return {
            "event_id": int(event_id),
            "event_name": event_name,
            "event_dir": event_dir,
            "channel": self.channel,
            "channel_label": self.channel_label,
            "trigger_frame": int(trigger_frame),
            "start_frame": int(trigger_frame),
            "end_frame": int(trigger_frame),
            "last_active_frame": int(trigger_frame),
            "peak_frame": int(trigger_frame),
            "peak_score": 0.0,
            "reason": "",
            "frames": [],
            "records": [],
            "active_frames": [],
            "post_remaining": self.post_frames,
            "clip_truncated": False,
            "last_appended_frame": None,
            # Task 8.3 / Requirement 10.5: capture the wall-clock timestamp
            # at event start so the branch summary writer can share the
            # same "timestamp" across all branches that triggered on the
            # same frame.
            "started_at": str(started_at)
            if started_at
            else datetime.now().isoformat(timespec="milliseconds"),
        }

    def _append_entry(self, event: dict[str, Any], entry: dict[str, Any]) -> None:
        frame_idx = int(entry["frame"])
        if event.get("last_appended_frame") == frame_idx:
            return
        event["last_appended_frame"] = frame_idx
        event["start_frame"] = min(int(event["start_frame"]), frame_idx)
        event["end_frame"] = max(int(event["end_frame"]), frame_idx)
        event["records"].append(entry["record"])
        if len(event["frames"]) < self.max_clip_frames:
            event["frames"].append((frame_idx, entry["image"]))
        else:
            event["clip_truncated"] = True
        if entry.get("active"):
            event["last_active_frame"] = frame_idx
            event["active_frames"].append(frame_idx)
        score = float(entry.get("score", 0.0))
        if score >= float(event.get("peak_score", 0.0)):
            event["peak_score"] = score
            event["peak_frame"] = frame_idx
            event["reason"] = str(entry.get("reason", ""))

    def _finalize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_dir: Path = event["event_dir"]
        frames_dir = event_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames: list[tuple[int, Any]] = event.get("frames", [])

        clip_path = event_dir / "clip.mp4"
        if frames:
            height, width = frames[0][1].shape[:2]
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"无法创建异常证据视频: {clip_path}")
            try:
                for _, image in frames:
                    writer.write(image)
            finally:
                writer.release()

        frame_paths: list[str] = []
        for frame_idx, image in frames:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            write_image_file(frame_path, image, quality=88)
            frame_paths.append(str(frame_path))

        representative_path = event_dir / "representative.jpg"
        representative_image = None
        peak_frame = int(event.get("peak_frame", event.get("trigger_frame", 0)))
        for frame_idx, image in frames:
            if frame_idx == peak_frame:
                representative_image = image
                break
        if representative_image is None and frames:
            representative_image = frames[min(len(frames) - 1, len(frames) // 2)][1]
        if representative_image is not None:
            write_image_file(representative_path, representative_image, quality=92)

        frame_records_path = event_dir / "frame_records.jsonl"
        with frame_records_path.open("w", encoding="utf-8") as fp:
            for record in event.get("records", []):
                json.dump(record, fp, ensure_ascii=False, default=json_default)
                fp.write("\n")

        summary = {
            "event_id": int(event["event_id"]),
            "event_name": str(event["event_name"]),
            "channel": self.channel,
            "channel_label": self.channel_label,
            "event_dir": str(event_dir),
            "start_frame": int(event["start_frame"]),
            "trigger_frame": int(event["trigger_frame"]),
            "last_active_frame": int(event["last_active_frame"]),
            "end_frame": int(event["end_frame"]),
            "duration_frames": int(event["end_frame"]) - int(event["start_frame"]) + 1,
            "clip_frame_count": len(frames),
            "clip_truncated": bool(event.get("clip_truncated", False)),
            "active_frame_count": len(event.get("active_frames", [])),
            "active_frames": event.get("active_frames", [])[:80],
            "peak_score": float(event.get("peak_score", 0.0)),
            "peak_frame": peak_frame,
            "reason": str(event.get("reason", "")),
            "clip_path": str(clip_path) if frames else None,
            "clip_url": evidence_file_url(clip_path) if frames else "",
            "representative_frame_path": str(representative_path)
            if representative_image is not None
            else None,
            "representative_frame_url": evidence_file_url(representative_path)
            if representative_image is not None
            else "",
            "frames_dir": str(frames_dir),
            "saved_frame_count": len(frame_paths),
            "frame_paths": frame_paths[:120],
            "frame_records_path": str(frame_records_path),
            # Task 8.3 / Requirement 10.5: carry the event-start timestamp
            # through the summary so MonitorEvidenceSession can use it as
            # the shared "timestamp" field in branch summary files when
            # multiple channels trigger on the same frame.
            "started_at": str(event.get("started_at", "")),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json_file(event_dir / "summary.json", summary)
        return summary


class MonitorEvidenceSession:
    # Task 8.3 / Requirement 10.3: stable channel → branch-summary folder
    # mapping. The detailed session evidence still lives under
    # 异常记录/监控台/<session>/<channel>/event_XXX/, but for scripts that
    # want to scan "one JSON per confirmed event" across all sessions, the
    # branch summary JSONs are laid out flat under
    # 异常记录/<branch>/<source_label>/<event>.json.
    CHANNEL_TO_BRANCH: dict[str, str] = {
        "module_a": "p_adv",
        "safety_helmet": "p_safety",
        "source_authenticity": "p_synth",
    }

    def __init__(
        self,
        source_type: str,
        source: str,
        profile: str,
        custom_model: dict[str, Any] | None = None,
        evidence_root: Path | None = None,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evidence_root = Path(evidence_root) if evidence_root is not None else EVIDENCE_ROOT
        self.source_label = source_slug(source_type, source)
        self.session_dir = self.evidence_root / "监控台" / f"{stamp}_{self.source_label}"
        self.manifest_path = self.session_dir / "events.json"
        self.source_type = source_type
        self.source = source
        self.profile = profile
        self.custom_model = normalize_custom_model_options(custom_model)
        self.module_a = ChannelEvidenceRecorder(self.session_dir, "module_a", "模块A扰动告警")
        self.ppe = ChannelEvidenceRecorder(self.session_dir, "safety_helmet", "安全帽业务告警")
        self.source_auth = ChannelEvidenceRecorder(
            self.session_dir, "source_authenticity", "生成/伪造视频疑似告警"
        )
        # Task 8.3 / Requirement 10.5: map trigger_frame → shared
        # timestamp. When multiple channels trigger on the same frame,
        # their branch summary JSONs share exactly the same "timestamp"
        # field (and naturally the same "frame_id" via trigger_frame).
        self._shared_event_ts: dict[int, str] = {}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    @property
    def saved_event_count(self) -> int:
        return (
            len(self.module_a.saved_events)
            + len(self.ppe.saved_events)
            + len(self.source_auth.saved_events)
        )

    def set_fps(self, fps: float) -> None:
        self.module_a.set_fps(fps)
        self.ppe.set_fps(fps)
        self.source_auth.set_fps(fps)

    def wants_frame(self, frame_idx: int, status: dict[str, Any]) -> bool:
        return (
            self.module_a.wants_frame(frame_idx, bool(status.get("alert_confirmed", False)))
            or self.ppe.wants_frame(frame_idx, bool(status.get("ppe_warning", False)))
            or self.source_auth.wants_frame(
                frame_idx, bool(status.get("source_authenticity_warning", False))
            )
        )

    def record_frame(
        self, frame_idx: int, image: Any, status: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        completed: list[tuple[str, dict[str, Any]]] = []

        module_active = bool(status.get("alert_confirmed", False))
        ppe_active = bool(status.get("ppe_warning", False))
        source_active = bool(status.get("source_authenticity_warning", False))

        # Task 8.3 / Requirement 10.5: compute the shared wall-clock
        # timestamp once per frame so that any branch whose event *starts*
        # on this frame will record the same "timestamp" in its branch
        # summary JSON.
        shared_ts = self._shared_event_ts.get(int(frame_idx))
        if shared_ts is None and (module_active or ppe_active or source_active):
            shared_ts = datetime.now().isoformat(timespec="milliseconds")
            self._shared_event_ts[int(frame_idx)] = shared_ts

        if self.module_a.wants_frame(frame_idx, module_active):
            module_summary = self.module_a.record_frame(
                frame_idx=frame_idx,
                image=image,
                active=module_active,
                event_id=status.get("current_alert_event_id"),
                score=float(status.get("p_adv", 0.0)),
                reason=str(status.get("reason", "")),
                record=monitor_frame_record("module_a", frame_idx, module_active, status),
                started_at=shared_ts if module_active else None,
            )
            if module_summary:
                completed.append(("module_a", module_summary))

        if self.ppe.wants_frame(frame_idx, ppe_active):
            ppe_summary = self.ppe.record_frame(
                frame_idx=frame_idx,
                image=image,
                active=ppe_active,
                event_id=status.get("current_ppe_event_id"),
                score=float(status.get("ppe_missing_helmet_count", 0)),
                reason=str(status.get("ppe_reason", "")),
                record=monitor_frame_record("safety_helmet", frame_idx, ppe_active, status),
                started_at=shared_ts if ppe_active else None,
            )
            if ppe_summary:
                completed.append(("safety_helmet", ppe_summary))

        if self.source_auth.wants_frame(frame_idx, source_active):
            source_summary = self.source_auth.record_frame(
                frame_idx=frame_idx,
                image=image,
                active=source_active,
                event_id=status.get("current_source_auth_event_id"),
                score=float(status.get("p_synth", 0.0)),
                reason=str(status.get("source_authenticity_reason", "")),
                record=monitor_frame_record(
                    "source_authenticity", frame_idx, source_active, status
                ),
                started_at=shared_ts if source_active else None,
            )
            if source_summary:
                completed.append(("source_authenticity", source_summary))

        # Task 8.3 / Requirements 10.3, 10.5: after each finalize, flush a
        # branch summary JSON to 异常记录/<branch>/<source_label>/<event>.json.
        for channel, summary in completed:
            self.write_branch_event_summary(channel, summary)

        if completed:
            self._write_manifest()
        return completed

    def close(self) -> list[tuple[str, dict[str, Any]]]:
        completed: list[tuple[str, dict[str, Any]]] = []
        for channel, recorder in (
            ("module_a", self.module_a),
            ("safety_helmet", self.ppe),
            ("source_authenticity", self.source_auth),
        ):
            summary = recorder.close()
            if summary:
                completed.append((channel, summary))
                self.write_branch_event_summary(channel, summary)
        self._write_manifest()
        return completed

    # ------------------------------------------------------------------
    # Task 8.3 / Requirements 10.3, 10.5
    # ------------------------------------------------------------------
    def branch_summary_dir(self, channel: str) -> Path:
        """Return the directory that holds branch summary JSONs for one
        channel, i.e. 异常记录/<branch>/<source_label>/."""
        branch = self.CHANNEL_TO_BRANCH.get(channel, channel)
        return self.evidence_root / branch / self.source_label

    def branch_summary_path(self, channel: str, event_name: str, event_id: int) -> Path:
        """Full path of the branch summary JSON for a given event.

        The file name follows ``<event_name>_<event_id>.json`` so that
        three simultaneous triggers (p_adv / p_safety / p_synth) produce
        three sibling files under the three branch folders and the
        filename collisions across branches are naturally avoided (the
        branch folder already disambiguates them).
        """
        safe_name = safe_path_part(str(event_name), fallback="event")
        return self.branch_summary_dir(channel) / f"{safe_name}_{int(event_id):03d}.json"

    def write_branch_event_summary(self, channel: str, summary: dict[str, Any]) -> Path:
        """Write a thin "one JSON per confirmed event" file for the given
        branch.

        See design.md §7 and Requirements 10.3 / 10.5. The detailed
        session evidence (clip.mp4 / frames/ / summary.json) stays under
        the existing 异常记录/监控台/<session>/<channel>/event_XXX/
        folder. This function adds an **extra** flat per-branch layout
        under 异常记录/<branch>/<source_label>/<event>.json for downstream
        scripts that want to scan by branch rather than by session.
        """
        branch = self.CHANNEL_TO_BRANCH.get(channel, channel)
        event_id = int(summary.get("event_id", 0))
        event_name = str(summary.get("event_name", f"event_{event_id:03d}"))
        trigger_frame = int(summary.get("trigger_frame", summary.get("start_frame", 0)))

        # The shared timestamp is the one stamped at event start, which
        # matches across branches that triggered on the same frame
        # (Requirement 10.5). Fall back to the summary's own started_at
        # and finally to saved_at so older events still get a value.
        timestamp = str(
            self._shared_event_ts.get(trigger_frame)
            or summary.get("started_at")
            or summary.get("saved_at")
            or datetime.now().isoformat(timespec="milliseconds")
        )

        payload = {
            "branch": branch,
            "channel": channel,
            "source_label": self.source_label,
            "source_type": self.source_type,
            "source": self.source,
            "profile": self.profile,
            "event_id": event_id,
            "event_name": event_name,
            # Task 8.3 hard field: "frame_id" is the event trigger frame
            # so that three branches triggered on the same frame share
            # the same frame_id (Requirement 10.5).
            "frame_id": trigger_frame,
            "timestamp": timestamp,
            "trigger_frame": trigger_frame,
            "start_frame": int(summary.get("start_frame", trigger_frame)),
            "end_frame": int(summary.get("end_frame", trigger_frame)),
            "peak_frame": int(summary.get("peak_frame", trigger_frame)),
            "peak_score": float(summary.get("peak_score", 0.0)),
            "reason": str(summary.get("reason", "")),
            "clip_path": summary.get("clip_path"),
            "representative_frame_path": summary.get("representative_frame_path"),
            "session_event_dir": summary.get("event_dir"),
            "session_dir": str(self.session_dir),
        }
        path = self.branch_summary_path(channel, event_name, event_id)
        write_json_file(path, payload)
        return path

    def _write_manifest(self) -> None:
        write_json_file(
            self.manifest_path,
            {
                "session_dir": str(self.session_dir),
                "source_type": self.source_type,
                "source": self.source,
                "profile": self.profile,
                "custom_model": self.custom_model,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "module_a_event_count": len(self.module_a.saved_events),
                "safety_helmet_event_count": len(self.ppe.saved_events),
                "source_authenticity_event_count": len(self.source_auth.saved_events),
                "events": {
                    "module_a": self.module_a.saved_events,
                    "safety_helmet": self.ppe.saved_events,
                    "source_authenticity": self.source_auth.saved_events,
                },
            },
        )


class PipelineCache:
    def __init__(self, module_config_path: Path, profile_config_path: Path):
        self.module_config_path = module_config_path
        self.profile_config_path = profile_config_path
        self._lock = threading.Lock()
        self._profile_name: str | None = None
        self._feature_key: tuple[bool, bool] | None = None
        self._custom_model_key: tuple[bool, str, str, str] | None = None
        self._pipeline: VideoDefensePipeline | None = None
        self._backend_name: str | None = None
        self._artifact_path: str | None = None

    def get(
        self,
        profile_name: str,
        feature_options: dict[str, Any] | None = None,
        custom_model: dict[str, Any] | None = None,
    ) -> tuple[VideoDefensePipeline, str, str]:
        options = feature_options or {}
        feature_key = (
            bool(options.get("static_image_enabled", True)),
            bool(options.get("source_authenticity_enabled", False)),
        )
        custom_options = normalize_custom_model_options(custom_model)
        custom_model_key = (
            bool(custom_options["enabled"]),
            str(custom_options["path"]),
            str(custom_options["backend"]),
            str(custom_options["model_family"]),
        )
        with self._lock:
            if (
                self._pipeline is not None
                and self._profile_name == profile_name
                and self._feature_key == feature_key
                and self._custom_model_key == custom_model_key
            ):
                self._pipeline.reset()
                return self._pipeline, self._backend_name or "", self._artifact_path or ""

            module_config = load_yaml(self.module_config_path)
            merge_profile(module_config, self.profile_config_path, profile_name)
            apply_feature_options(module_config, feature_options)
            resolved_custom_options = apply_custom_model(module_config, custom_model)
            detector_backend = create_detector_backend(module_config, PROJECT_ROOT)
            pipeline = VideoDefensePipeline(detector_backend, config=module_config)
            pipeline.warmup(getattr(pipeline, "warmup_frames", 0))
            pipeline.reset()

            self._profile_name = profile_name
            self._feature_key = feature_key
            self._custom_model_key = (
                bool(resolved_custom_options["enabled"]),
                str(resolved_custom_options["path"]),
                str(resolved_custom_options["backend"]),
                str(resolved_custom_options["model_family"]),
            )
            self._pipeline = pipeline
            self._backend_name = detector_backend.backend
            self._artifact_path = str(detector_backend.artifact_path)
            return pipeline, self._backend_name, self._artifact_path


class MonitorState:
    def __init__(self, cache: PipelineCache):
        self.cache = cache
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.status: dict[str, Any] = self._empty_status()
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=20)
        self.recent_ppe_events: deque[dict[str, Any]] = deque(maxlen=20)
        self.recent_source_auth_events: deque[dict[str, Any]] = deque(maxlen=20)
        self.current_event: dict[str, Any] | None = None
        self.current_ppe_event: dict[str, Any] | None = None
        self.current_source_auth_event: dict[str, Any] | None = None
        self.event_seq = 0
        self.ppe_event_seq = 0
        self.source_auth_event_seq = 0
        self.ppe_state = SafetyHelmetState()
        self.evidence_session: MonitorEvidenceSession | None = None
        self.display_options: dict[str, bool] = {
            "show_boxes": True,
            "show_module_hud": True,
            "show_ppe_hud": True,
        }

    @staticmethod
    def _empty_status() -> dict[str, Any]:
        return {
            "running": False,
            "source_type": None,
            "source": None,
            "profile": None,
            "backend": None,
            "artifact": None,
            "evidence_root": str(EVIDENCE_ROOT),
            "evidence_session_dir": None,
            "evidence_manifest_path": None,
            "evidence_saved_event_count": 0,
            "frame_idx": 0,
            "p_adv": 0.0,
            "p_adv_missing_reason": "",
            "p_safety": None,
            "p_safety_missing_reason": "静态媒介检测未启用",
            "a3b_score": 0.0,
            "a3b_triggered": False,
            "a3b_media_type": "normal",
            "a3b_trigger_count": 0,
            "a3b_live_score": 0.0,
            "a3b_p_media": 0.0,
            "a3b_bbox": None,
            "a3b_replay_state": {},
            "a3b_fast_state": {},
            "a3b_occlusion_state": {},
            "a3b_triggered_source": "none",
            "timing_ms": 0.0,
            "detector_inference_ms": 0.0,
            "module_a_timing_ms": 0.0,
            "alert_confirmed": False,
            "attack_detected": False,
            "attack_state_active": False,
            "reason": "",
            "alert_event_count": 0,
            "current_alert_event_id": None,
            "ppe_warning": False,
            "ppe_candidate": False,
            "ppe_person_count": 0,
            "ppe_helmet_count": 0,
            "ppe_head_count": 0,
            "ppe_missing_helmet_count": 0,
            "ppe_reason": "",
            "ppe_event_count": 0,
            "current_ppe_event_id": None,
            "p_synth": 0.0,
            "p_synth_missing_reason": "source_authenticity_disabled",
            "source_authenticity_enabled": False,
            "source_authenticity_warning": False,
            "source_authenticity_confirmed": False,
            "source_authenticity_available": False,
            "source_authenticity_suppressed_by_p_adv": False,
            "source_authenticity_reason": "",
            "source_authenticity_event_count": 0,
            "current_source_auth_event_id": None,
            "feature_options": {
                "static_image_enabled": True,
                "source_authenticity_enabled": False,
                # Task 8.2 / Requirements 10.2: Monitor_App UI-level strong
                # warning toggle. When False (default) the monitor suppresses
                # source_authenticity_warning promotion in the branch card and
                # the Monitor_App event list. The run-level frame_events.jsonl
                # and the pipeline contract are not affected.
                "synth_strong_warning_enabled": False,
            },
            "custom_model": normalize_custom_model_options(None),
            "fps": 0.0,
            "error": "",
            "started_at": None,
            "stopped_at": None,
            "display_options": {
                "show_boxes": True,
                "show_module_hud": True,
                "show_ppe_hud": True,
            },
            "recent_events": [],
            "recent_ppe_events": [],
            "recent_source_auth_events": [],
        }

    def start(
        self,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool = True,
        feature_options: dict[str, Any] | None = None,
        custom_model: dict[str, Any] | None = None,
    ) -> None:
        self.stop()
        feature_options = {
            "static_image_enabled": bool((feature_options or {}).get("static_image_enabled", True)),
            "source_authenticity_enabled": bool(
                (feature_options or {}).get("source_authenticity_enabled", False)
            ),
            # Task 8.2 / Requirements 10.2: UI-level toggle, default False.
            # Kept on MonitorState (not the pipeline) so the raw info
            # contract stays untouched even when this is flipped.
            "synth_strong_warning_enabled": bool(
                (feature_options or {}).get("synth_strong_warning_enabled", False)
            ),
        }
        custom_model_options = normalize_custom_model_options(custom_model)
        self.stop_event.clear()
        with self.lock:
            self.latest_jpeg = None
            self.recent_events.clear()
            self.recent_ppe_events.clear()
            self.recent_source_auth_events.clear()
            self.current_event = None
            self.current_ppe_event = None
            self.current_source_auth_event = None
            self.event_seq = 0
            self.ppe_event_seq = 0
            self.source_auth_event_seq = 0
            self.ppe_state.reset()
            self.evidence_session = MonitorEvidenceSession(
                source_type,
                source,
                profile,
                custom_model=custom_model_options,
            )
            self.status = self._empty_status()
            self.status.update(
                {
                    "running": True,
                    "source_type": source_type,
                    "source": source,
                    "profile": profile,
                    "realtime": bool(realtime),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "evidence_session_dir": str(self.evidence_session.session_dir),
                    "evidence_manifest_path": str(self.evidence_session.manifest_path),
                    "feature_options": dict(feature_options),
                    "custom_model": dict(custom_model_options),
                    "source_authenticity_enabled": bool(
                        feature_options["source_authenticity_enabled"]
                    ),
                    "display_options": dict(self.display_options),
                }
            )
        self.worker = threading.Thread(
            target=self._run_capture,
            args=(
                source_type,
                source,
                profile,
                bool(realtime),
                feature_options,
                custom_model_options,
            ),
            name="module-a-monitor-capture",
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        worker = self.worker
        if worker and worker.is_alive():
            worker.join(timeout=3.0)
        self.worker = None
        with self.lock:
            self.status["running"] = False
            self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.condition.notify_all()

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.status)
            payload["display_options"] = dict(self.display_options)
            payload["recent_events"] = list(self.recent_events)
            payload["recent_ppe_events"] = list(self.recent_ppe_events)
            payload["recent_source_auth_events"] = list(self.recent_source_auth_events)
            # Task 8.1 / Requirements 10.1, 10.2 — derive the three-card
            # payload from the already-locked snapshot so the JS side can
            # render it directly without duplicating the mapping.
            payload["branch_cards"] = build_branch_cards(payload)
            return payload

    def get_display_options(self) -> dict[str, bool]:
        with self.lock:
            return dict(self.display_options)

    def update_display_options(self, options: dict[str, Any]) -> dict[str, bool]:
        allowed = {
            "show_boxes": True,
            "show_module_hud": True,
            "show_ppe_hud": True,
        }
        with self.lock:
            for key, default in allowed.items():
                if key in options:
                    self.display_options[key] = bool(options.get(key))
                elif key not in self.display_options:
                    self.display_options[key] = default
            self.status["display_options"] = dict(self.display_options)
            self.condition.notify_all()
            return dict(self.display_options)

    def get_latest_jpeg(self) -> bytes | None:
        with self.condition:
            if self.latest_jpeg is None and self.status.get("running"):
                self.condition.wait(timeout=2.0)
            return self.latest_jpeg

    def _open_capture(self, source_type: str, source: str) -> cv2.VideoCapture:
        if source_type == "camera":
            try:
                capture_source: int | str = int(str(source).strip())
            except ValueError as exc:
                raise ValueError("摄像头输入值必须是编号，例如 0 或 1") from exc
        elif source_type == "file":
            path = resolve_path(source)
            if not path.exists():
                raise FileNotFoundError(f"MP4文件不存在: {path}")
            capture_source = str(path)
        elif source_type == "rtsp":
            capture_source = source.strip()
            if not capture_source:
                raise ValueError("RTSP/HTTP视频流地址不能为空")
        else:
            raise ValueError(f"不支持的输入类型: {source_type}")

        cap = open_probe_capture(source_type, str(capture_source), timeout_ms=5000)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开输入源: {source}")
        configure_capture_runtime(cap, source_type)
        return cap

    def _run_capture(
        self,
        source_type: str,
        source: str,
        profile: str,
        realtime: bool,
        feature_options: dict[str, Any],
        custom_model: dict[str, Any],
    ) -> None:
        cap: cv2.VideoCapture | None = None
        evidence_session = self.evidence_session
        try:
            pipeline, backend, artifact = self.cache.get(
                profile,
                feature_options=feature_options,
                custom_model=custom_model,
            )
            cap = self._open_capture(source_type, source)
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_fps < 1 or source_fps > 120:
                source_fps = 25.0
            if evidence_session is not None:
                evidence_session.set_fps(source_fps)
            # Adaptive frame skip for high-FPS sources: if the source is
            # faster than the pipeline can process, skip frames to maintain
            # real-time playback speed. Target: process at most 30 fps worth
            # of frames regardless of source fps.
            max_process_fps = 30.0
            frame_skip = max(1, int(source_fps / max_process_fps))
            frame_period = 1.0 / (source_fps / frame_skip)
            fps_estimator = deque(maxlen=30)
            frame_idx = 0
            skip_counter = 0
            with torch.no_grad():
                while not self.stop_event.is_set():
                    grabbed_at = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        if source_type == "file":
                            break
                        time.sleep(0.05)
                        continue
                    # Skip frames to keep up with real-time on high-fps sources.
                    skip_counter += 1
                    if skip_counter < frame_skip:
                        continue
                    skip_counter = 0
                    # Early downscale: if the source is larger than 1280 on any
                    # axis, resize to 1280 max BEFORE passing to the pipeline.
                    # This prevents 4K frames from burning CPU on the 640 resize
                    # inside the pipeline (cv2.resize 4K→640 is ~5ms; 1280→640
                    # is ~0.5ms). The detection quality at 640×640 is identical.
                    h_src, w_src = frame.shape[:2]
                    max_input = 1280
                    if h_src > max_input or w_src > max_input:
                        scale = max_input / max(h_src, w_src)
                        frame = cv2.resize(
                            frame,
                            (int(w_src * scale), int(h_src * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    frame_640 = cv2.resize(frame, (640, 640))
                    _, detections, info = pipeline.process_frame(frame_640)
                    ppe = self.ppe_state.update(summarize_ppe_from_detections(detections))
                    display_options = self.get_display_options()
                    boxed_frame = detections.plot()
                    rendered = (
                        boxed_frame.copy()
                        if display_options.get("show_boxes", True)
                        else frame_640.copy()
                    )
                    if display_options.get("show_module_hud", True):
                        rendered = draw_hud(rendered, info, frame_idx, effective=False)
                    if display_options.get("show_ppe_hud", True):
                        rendered = draw_ppe_hud(rendered, ppe)
                    hud = rendered
                    ok, encoded = cv2.imencode(".jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if not ok:
                        raise RuntimeError("JPEG编码失败")
                    fps_estimator.append(time.perf_counter() - grabbed_at)
                    fps = 1.0 / (sum(fps_estimator) / len(fps_estimator)) if fps_estimator else 0.0
                    # Triple-channel contract (Requirements 1.1 / 1.6): each
                    # branch carries its own numeric score plus a missing
                    # reason so the Monitor_App cards can render "未启用/数据
                    # 缺失" instead of silently showing 0.
                    p_adv_raw = info.get("p_adv")
                    static_media_details = (
                        info.get("details", {})
                        .get("module_a_features", {})
                        .get("static_media", {})
                    )
                    status_update = {
                        "running": True,
                        "source_type": source_type,
                        "source": source,
                        "profile": profile,
                        "realtime": bool(realtime),
                        "feature_options": dict(feature_options),
                        "custom_model": dict(custom_model),
                        "backend": backend,
                        "artifact": artifact,
                        "evidence_root": str(EVIDENCE_ROOT),
                        "evidence_session_dir": str(evidence_session.session_dir)
                        if evidence_session
                        else None,
                        "evidence_manifest_path": str(evidence_session.manifest_path)
                        if evidence_session
                        else None,
                        "frame_idx": frame_idx,
                        "p_adv": float(p_adv_raw) if p_adv_raw is not None else None,
                        "p_adv_display": float(info.get("p_adv_display", p_adv_raw or 0.0))
                        if p_adv_raw is not None
                        else None,
                        "p_adv_missing_reason": str(info.get("p_adv_missing_reason", "")),
                        "p_safety": info.get("p_safety"),
                        "p_safety_missing_reason": "静态媒介检测未启用"
                        if not feature_options.get("static_image_enabled", True)
                        else "",
                        "a3b_score": float(
                            static_media_details.get("score", 0.0)
                        ),
                        "a3b_triggered": bool(
                            static_media_details.get("triggered", False)
                        ),
                        "a3b_media_type": str(
                            static_media_details.get("media_type", "normal")
                        ),
                        "a3b_trigger_count": int(
                            static_media_details.get("trigger_count", 0)
                        ),
                        "a3b_live_score": float(
                            static_media_details.get(
                                "live_score",
                                max(
                                    float(static_media_details.get("score", 0.0)),
                                    float(static_media_details.get("p_media", 0.0)),
                                    float(static_media_details.get("classifier_score", 0.0)),
                                ),
                            )
                        ),
                        "a3b_live_score_display": float(
                            static_media_details.get(
                                "live_score_display",
                                static_media_details.get(
                                    "live_score",
                                    max(
                                        float(static_media_details.get("score", 0.0)),
                                        float(static_media_details.get("p_media", 0.0)),
                                        float(static_media_details.get("classifier_score", 0.0)),
                                    ),
                                ),
                            )
                        ),
                        "a3b_p_media": float(static_media_details.get("p_media", 0.0)),
                        "a3b_bbox": static_media_details.get("p_media_bbox"),
                        "a3b_replay_state": dict(
                            static_media_details.get("p_media_replay_state", {})
                        ),
                        "a3b_fast_state": dict(
                            static_media_details.get("p_media_fast_state", {})
                        ),
                        "a3b_occlusion_state": dict(
                            static_media_details.get("p_media_occlusion_state", {})
                        ),
                        "a3b_triggered_source": str(
                            static_media_details.get("triggered_source", "none")
                        ),
                        "timing_ms": float(info.get("timing_ms", 0.0)),
                        "detector_inference_ms": float(info.get("detector_inference_ms", 0.0)),
                        "module_a_timing_ms": float(info.get("module_a_timing_ms", 0.0)),
                        "alert_confirmed": bool(info.get("alert_confirmed", False)),
                        "attack_detected": bool(info.get("attack_detected", False)),
                        "attack_state_active": bool(info.get("attack_state_active", False)),
                        "reason": info_reason(info),
                        "alert_event_count": self.event_seq,
                        "ppe_warning": bool(ppe.get("warning", False)),
                        "ppe_candidate": bool(ppe.get("candidate", False)),
                        "ppe_confirmed": bool(ppe.get("confirmed", False)),
                        "ppe_person_count": int(ppe.get("person_count", 0)),
                        "ppe_helmet_count": int(ppe.get("helmet_count", 0)),
                        "ppe_head_count": int(ppe.get("head_count", 0)),
                        "ppe_missing_helmet_count": int(ppe.get("missing_helmet_count", 0)),
                        "ppe_reason": str(ppe.get("reason", "")),
                        "ppe_event_count": self.ppe_event_seq,
                        "ppe_window_positive": int(ppe.get("window_positive", 0)),
                        "ppe_window": int(ppe.get("window", 0)),
                        "ppe_class_counts": ppe.get("class_counts", {}),
                        # p_synth 遵循三路分路契约（见 requirements 1.1/1.6）：分路缺失
                        # 时会被 pipeline 写成 None + missing_reason，此处回落到 0.0 仅
                        # 用于前端展示，不改变 info 里落盘的 None 语义。
                        "p_synth": float(info.get("p_synth") or 0.0),
                        "p_synth_missing_reason": str(info.get("p_synth_missing_reason", "")),
                        "source_authenticity_enabled": bool(
                            info.get("source_authenticity_enabled", False)
                        ),
                        "source_authenticity_warning": bool(
                            info.get("source_authenticity_warning", False)
                        ),
                        "source_authenticity_confirmed": bool(
                            info.get("source_authenticity_confirmed", False)
                        ),
                        "source_authenticity_available": bool(
                            info.get("source_authenticity_available", False)
                        ),
                        # Task 1.3 / Requirement 1.4: Source_Authenticity 抑制
                        # 字段从 details.module_a_features.source_authenticity
                        # 取出，供 Monitor_App 在 p_synth 卡片显示“抑制中”。
                        "source_authenticity_suppressed_by_p_adv": bool(
                            info.get("details", {})
                            .get("module_a_features", {})
                            .get("source_authenticity", {})
                            .get("suppressed_by_p_adv", False)
                        ),
                        "source_authenticity_reason": str(
                            info.get("source_authenticity_reason", "")
                        ),
                        "source_authenticity_event_count": self.source_auth_event_seq,
                        "fps": fps,
                        "source_fps": source_fps,
                        "display_options": display_options,
                        "error": "",
                    }
                    # Task 8.2 / Requirements 10.2: Monitor_App UI-level
                    # strong-warning filter. When 伪造视频流强告警模式 is off
                    # (default), we demote source_authenticity_warning /
                    # source_authenticity_confirmed to False so the p_synth
                    # branch card stays idle and the event list does not
                    # accrue new Source_Authenticity entries. The numeric
                    # p_synth, enabled flag, suppression flag, and reason
                    # text are preserved so operators still see the score
                    # and so the run-level frame_events.jsonl (written by
                    # tools/run_experiment.py) is unaffected.
                    if not feature_options.get("synth_strong_warning_enabled", False):
                        status_update["source_authenticity_warning"] = False
                        status_update["source_authenticity_confirmed"] = False
                    self._update_event_state(status_update)
                    self._update_ppe_event_state(status_update)
                    self._update_source_auth_event_state(status_update)
                    if evidence_session is not None and evidence_session.wants_frame(
                        frame_idx, status_update
                    ):
                        evidence_frame = draw_hud(
                            boxed_frame.copy(), info, frame_idx, effective=False
                        )
                        evidence_frame = draw_ppe_hud(evidence_frame, ppe)
                        completed_evidence = evidence_session.record_frame(
                            frame_idx, evidence_frame, status_update
                        )
                        for channel, summary in completed_evidence:
                            self._attach_evidence_summary(channel, summary)
                        status_update["evidence_saved_event_count"] = (
                            evidence_session.saved_event_count
                        )
                    with self.condition:
                        self.latest_jpeg = encoded.tobytes()
                        self.status.update(status_update)
                        self.status["recent_events"] = list(self.recent_events)
                        self.status["recent_ppe_events"] = list(self.recent_ppe_events)
                        self.status["recent_source_auth_events"] = list(
                            self.recent_source_auth_events
                        )
                        self.condition.notify_all()
                    frame_idx += 1
                    if realtime:
                        elapsed = time.perf_counter() - grabbed_at
                        if elapsed > frame_period * 1.25:
                            # Latest-frame priority: when processing falls behind
                            # the source cadence, discard buffered frames instead
                            # of replaying them slowly. This preserves real-time
                            # wall-clock behavior for MP4/RTSP/camera inputs.
                            extra_drop = min(12, max(0, int(elapsed / max(frame_period, 1e-3)) - 1))
                            for _ in range(extra_drop):
                                try:
                                    if not cap.grab():
                                        break
                                except Exception:
                                    break
                        elif elapsed < frame_period:
                            time.sleep(frame_period - elapsed)
        except Exception as exc:
            with self.condition:
                self.status["running"] = False
                self.status["error"] = str(exc)
                self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
                self.condition.notify_all()
        finally:
            if evidence_session is not None:
                try:
                    completed_evidence = evidence_session.close()
                    for channel, summary in completed_evidence:
                        self._attach_evidence_summary(channel, summary)
                except Exception as exc:
                    with self.condition:
                        self.status["error"] = f"异常证据保存失败: {exc}"
            if cap is not None:
                cap.release()
            with self.condition:
                self.status["running"] = False
                if evidence_session is not None:
                    self.status["evidence_session_dir"] = str(evidence_session.session_dir)
                    self.status["evidence_manifest_path"] = str(evidence_session.manifest_path)
                    self.status["evidence_saved_event_count"] = evidence_session.saved_event_count
                    self.status["recent_events"] = list(self.recent_events)
                    self.status["recent_ppe_events"] = list(self.recent_ppe_events)
                    self.status["recent_source_auth_events"] = list(self.recent_source_auth_events)
                self.status["stopped_at"] = datetime.now().isoformat(timespec="seconds")
                self.condition.notify_all()

    def _update_event_state(self, status: dict[str, Any]) -> None:
        alert = bool(status.get("alert_confirmed", False))
        frame_idx = int(status.get("frame_idx", 0))
        p_adv = float(status.get("p_adv", 0.0))
        reason = str(status.get("reason", ""))
        if alert and self.current_event is None:
            self.event_seq += 1
            self.current_event = {
                "event_id": self.event_seq,
                "trigger_frame": frame_idx,
                "last_alert_frame": frame_idx,
                "peak_frame": frame_idx,
                "peak_p_adv": p_adv,
                "reason": reason,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            a3b_detail = build_a3b_detail(status)
            if a3b_detail:
                self.current_event["a3b_detail"] = a3b_detail
            self.recent_events.appendleft(dict(self.current_event))
        elif alert and self.current_event is not None:
            self.current_event["last_alert_frame"] = frame_idx
            if p_adv >= float(self.current_event.get("peak_p_adv", 0.0)):
                self.current_event["peak_p_adv"] = p_adv
                self.current_event["peak_frame"] = frame_idx
                self.current_event["reason"] = reason
            a3b_detail = build_a3b_detail(status)
            if a3b_detail:
                self.current_event["a3b_detail"] = a3b_detail
            if self.recent_events:
                self.recent_events[0] = dict(self.current_event)
        elif not alert:
            self.current_event = None
        status["alert_event_count"] = self.event_seq
        status["current_alert_event_id"] = (
            int(self.current_event["event_id"])
            if alert and self.current_event is not None
            else None
        )

    def _update_ppe_event_state(self, status: dict[str, Any]) -> None:
        warning = bool(status.get("ppe_warning", False))
        frame_idx = int(status.get("frame_idx", 0))
        if warning and self.current_ppe_event is None:
            self.ppe_event_seq += 1
            self.current_ppe_event = {
                "event_id": self.ppe_event_seq,
                "trigger_frame": frame_idx,
                "last_warning_frame": frame_idx,
                "person_count": int(status.get("ppe_person_count", 0)),
                "helmet_count": int(status.get("ppe_helmet_count", 0)),
                "head_count": int(status.get("ppe_head_count", 0)),
                "missing_helmet_count": int(status.get("ppe_missing_helmet_count", 0)),
                "reason": str(status.get("ppe_reason", "")),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.recent_ppe_events.appendleft(dict(self.current_ppe_event))
        elif warning and self.current_ppe_event is not None:
            self.current_ppe_event["last_warning_frame"] = frame_idx
            self.current_ppe_event["person_count"] = int(status.get("ppe_person_count", 0))
            self.current_ppe_event["helmet_count"] = int(status.get("ppe_helmet_count", 0))
            self.current_ppe_event["head_count"] = int(status.get("ppe_head_count", 0))
            self.current_ppe_event["missing_helmet_count"] = int(
                status.get("ppe_missing_helmet_count", 0)
            )
            self.current_ppe_event["reason"] = str(status.get("ppe_reason", ""))
            if self.recent_ppe_events:
                self.recent_ppe_events[0] = dict(self.current_ppe_event)
        elif not warning:
            self.current_ppe_event = None
        status["ppe_event_count"] = self.ppe_event_seq
        status["current_ppe_event_id"] = (
            int(self.current_ppe_event["event_id"])
            if warning and self.current_ppe_event is not None
            else None
        )

    def _update_source_auth_event_state(self, status: dict[str, Any]) -> None:
        warning = bool(status.get("source_authenticity_warning", False))
        frame_idx = int(status.get("frame_idx", 0))
        p_synth = float(status.get("p_synth", 0.0))
        reason = str(status.get("source_authenticity_reason", ""))
        if warning and self.current_source_auth_event is None:
            self.source_auth_event_seq += 1
            self.current_source_auth_event = {
                "event_id": self.source_auth_event_seq,
                "trigger_frame": frame_idx,
                "last_warning_frame": frame_idx,
                "peak_frame": frame_idx,
                "peak_p_synth": p_synth,
                "reason": reason,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.recent_source_auth_events.appendleft(dict(self.current_source_auth_event))
        elif warning and self.current_source_auth_event is not None:
            self.current_source_auth_event["last_warning_frame"] = frame_idx
            if p_synth >= float(self.current_source_auth_event.get("peak_p_synth", 0.0)):
                self.current_source_auth_event["peak_p_synth"] = p_synth
                self.current_source_auth_event["peak_frame"] = frame_idx
                self.current_source_auth_event["reason"] = reason
            if self.recent_source_auth_events:
                self.recent_source_auth_events[0] = dict(self.current_source_auth_event)
        elif not warning:
            self.current_source_auth_event = None
        status["source_authenticity_event_count"] = self.source_auth_event_seq
        status["current_source_auth_event_id"] = (
            int(self.current_source_auth_event["event_id"])
            if warning and self.current_source_auth_event is not None
            else None
        )

    def _attach_evidence_summary(self, channel: str, summary: dict[str, Any]) -> None:
        if channel == "module_a":
            target_events = self.recent_events
        elif channel == "safety_helmet":
            target_events = self.recent_ppe_events
        else:
            target_events = self.recent_source_auth_events
        event_id = int(summary.get("event_id", 0))
        for index, event in enumerate(target_events):
            if int(event.get("event_id", -1)) == event_id:
                updated = dict(event)
                updated.update(
                    {
                        "evidence_saved": True,
                        "evidence_event_dir": summary.get("event_dir"),
                        "evidence_clip_path": summary.get("clip_path"),
                        "evidence_clip_url": summary.get("clip_url"),
                        "evidence_representative_path": summary.get("representative_frame_path"),
                        "evidence_representative_url": summary.get("representative_frame_url"),
                        "evidence_frames_dir": summary.get("frames_dir"),
                        "evidence_saved_frame_count": summary.get("saved_frame_count", 0),
                    }
                )
                target_events[index] = updated
                return


class MonitorRequestHandler(BaseHTTPRequestHandler):
    server_version = "ModuleAMonitor/1.0"

    @property
    def monitor(self) -> MonitorState:
        return self.server.monitor_state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/status":
            write_json_response(
                self, HTTPStatus.OK, {"ok": True, "status": self.monitor.get_status()}
            )
            return
        if parsed.path == "/stream.mjpg":
            self._send_mjpeg()
            return
        if parsed.path.startswith("/evidence/"):
            self._send_evidence_file(parsed.path)
            return
        if parsed.path == "/api/sample-sources":
            write_json_response(self, HTTPStatus.OK, {"ok": True, "sources": sample_sources()})
            return
        if parsed.path == "/api/cameras":
            write_json_response(self, HTTPStatus.OK, {"ok": True, "devices": scan_camera_devices()})
            return
        write_json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                source_type = str(payload.get("source_type", "file"))
                source = str(payload.get("source", ""))
                profile = str(payload.get("profile", "full_gpu"))
                realtime = bool(payload.get("realtime", True))
                feature_options = payload.get("feature_options", {})
                if not isinstance(feature_options, dict):
                    feature_options = {}
                custom_model = payload.get("custom_model", {})
                if not isinstance(custom_model, dict):
                    custom_model = {}
                self.monitor.start(
                    source_type,
                    source,
                    profile,
                    realtime,
                    feature_options=feature_options,
                    custom_model=custom_model,
                )
                write_json_response(
                    self, HTTPStatus.OK, {"ok": True, "status": self.monitor.get_status()}
                )
                return
            if parsed.path == "/api/stop":
                self.monitor.stop()
                write_json_response(
                    self, HTTPStatus.OK, {"ok": True, "status": self.monitor.get_status()}
                )
                return
            if parsed.path == "/api/pick-file":
                mode = str(payload.get("mode", "video"))
                current_path = str(payload.get("current_path", ""))
                write_json_response(
                    self, HTTPStatus.OK, {"ok": True, "path": pick_local_file(mode, current_path)}
                )
                return
            if parsed.path == "/api/test-source":
                source_type = str(payload.get("source_type", "file"))
                source = str(payload.get("source", ""))
                result = test_source_connectivity(source_type, source)
                write_json_response(self, HTTPStatus.OK, {"ok": True, **result})
                return
            if parsed.path == "/api/display-options":
                options = payload.get("display_options", payload)
                if not isinstance(options, dict):
                    raise ValueError("display_options 必须是对象")
                display_options = self.monitor.update_display_options(options)
                write_json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "display_options": display_options,
                        "status": self.monitor.get_status(),
                    },
                )
                return
            write_json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
        except Exception as exc:
            write_json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        if "/api/status" in message or "/stream.mjpg" in message:
            return
        print(f"[monitor] {self.address_string()} - {message}")

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_mjpeg(self) -> None:
        parsed = urlparse(self.path)
        _ = parse_qs(parsed.query)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sent_frame = -1
        min_period = 1.0 / 15.0
        last_sent_at = 0.0
        while True:
            status = self.monitor.get_status()
            current_frame = int(status.get("frame_idx", -1) or -1)
            if current_frame == last_sent_frame:
                if not status.get("running"):
                    break
                time.sleep(0.03)
                continue
            elapsed = time.perf_counter() - last_sent_at
            if elapsed < min_period:
                time.sleep(min_period - elapsed)
            jpeg = self.monitor.get_latest_jpeg()
            if jpeg is None:
                if not status.get("running"):
                    break
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_sent_frame = current_frame
                last_sent_at = time.perf_counter()
            except (BrokenPipeError, ConnectionResetError):
                break

    def _send_evidence_file(self, request_path: str) -> None:
        rel_text = unquote(request_path[len("/evidence/") :]).replace("/", os.sep)
        root = EVIDENCE_ROOT.resolve()
        path = (root / rel_text).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            write_json_response(self, HTTPStatus.FORBIDDEN, {"ok": False, "error": "Forbidden"})
            return
        if not path.exists() or not path.is_file():
            write_json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))


def pick_local_file(mode: str = "video", current_path: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - platform guard
        raise RuntimeError("当前环境无法打开本机文件选择窗口，请手动粘贴 MP4 路径。") from exc

    current = Path(str(current_path).strip().strip("\"")).expanduser() if current_path else None
    initial_dir: Path | None = None
    if current is not None:
        if current.is_file():
            initial_dir = current.parent
        elif current.is_dir():
            initial_dir = current
        elif str(current):
            parent = current.parent
            if parent.exists():
                initial_dir = parent
    if initial_dir is None:
        if mode == "model":
            initial_dir = PROJECT_ROOT / "baseline_training" / "runs"
        else:
            initial_dir = PROJECT_ROOT / "experiments"
    if not initial_dir.exists():
        initial_dir = PROJECT_ROOT

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    root.update()
    try:
        if mode == "model":
            title = "选择自定义检测模型"
            filetypes = [
                ("模型文件", "*.engine *.onnx *.pt *.pth"),
                ("TensorRT Engine", "*.engine"),
                ("ONNX 模型", "*.onnx"),
                ("PyTorch 权重", "*.pt *.pth"),
                ("所有文件", "*.*"),
            ]
        else:
            title = "选择要检测的 MP4 / 视频文件"
            filetypes = [
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("MP4 文件", "*.mp4"),
                ("所有文件", "*.*"),
            ]
        path = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=str(initial_dir),
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    if not path:
        raise RuntimeError("未选择文件。")
    return str(Path(path).resolve())


def open_probe_capture(source_type: str, source: str, timeout_ms: int = 8000) -> cv2.VideoCapture:
    if source_type == "camera":
        try:
            index = int(str(source).strip())
        except ValueError as exc:
            raise ValueError("摄像头输入值必须是编号，例如 0 或 1。") from exc
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(index)
        return cap

    if source_type == "file":
        return cv2.VideoCapture(str(resolve_path(source)))

    if source_type != "rtsp":
        raise ValueError(f"不支持的输入类型: {source_type}")

    url = source.strip()
    if not url:
        raise ValueError("RTSP/HTTP 视频流地址不能为空。")
    if not url.lower().startswith(("rtsp://", "http://", "https://")):
        raise ValueError("请输入 rtsp://、http:// 或 https:// 开头的视频流地址。")

    # 强制使用 TCP 传输（避免 UDP 丢包导致卡顿）+ 低延迟选项
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    )

    params: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
    if params:
        try:
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except Exception:
            pass
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def configure_capture_runtime(cap: cv2.VideoCapture, source_type: str) -> None:
    # 最小化内部缓冲，拿到最新帧而非堆积的旧帧
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if source_type == "rtsp":
        # RTSP 流额外优化：减少 FFmpeg 内部缓冲延迟
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        except Exception:
            pass
        return
    if source_type != "camera":
        return
    for prop, value in (
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (cv2.CAP_PROP_FPS, 15),
    ):
        try:
            cap.set(prop, value)
        except Exception:
            pass


def test_source_connectivity(source_type: str, source: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if source_type == "file":
            path = resolve_path(source)
            if not path.exists():
                return {"reachable": False, "message": f"文件不存在：{path}"}
            if not path.is_file():
                return {"reachable": False, "message": f"路径不是文件：{path}"}

        # Use a longer timeout for connectivity probes (10s) to accommodate
        # LAN cameras and HTTP MJPEG streams that may be slow to respond.
        probe_timeout = 10000 if source_type == "rtsp" else 8000
        cap = open_probe_capture(source_type, source, timeout_ms=probe_timeout)
        try:
            if not cap.isOpened():
                return {
                    "reachable": False,
                    "message": "无法打开输入源。请确认地址格式正确且设备可达。",
                }
            ok, frame = cap.read()
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if ok and frame is not None:
                height, width = frame.shape[:2]
            if not ok:
                return {
                    "reachable": False,
                    "message": f"已打开但无法读取视频帧，耗时 {elapsed_ms} ms。",
                    "elapsed_ms": elapsed_ms,
                }
            source_label = {"file": "文件", "camera": "摄像头", "rtsp": "视频流"}.get(
                source_type, "输入源"
            )
            detail = f"{width}x{height}" if width and height else "已读取首帧"
            return {
                "reachable": True,
                "message": f"{source_label}连通正常：{detail}，耗时 {elapsed_ms} ms。",
                "elapsed_ms": elapsed_ms,
                "width": width,
                "height": height,
            }
        finally:
            cap.release()
    except Exception as exc:
        return {"reachable": False, "message": str(exc)}


def windows_camera_names() -> list[str]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image') -and $_.Name } | "
        "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
    except Exception:
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        return [str(item) for item in data if item]
    return []


def scan_camera_devices(max_index: int = 8) -> list[dict[str, Any]]:
    names = windows_camera_names()
    devices: list[dict[str, Any]] = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for index in range(max(1, max_index)):
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if ok and frame is not None:
                height, width = frame.shape[:2]
            if not ok and width <= 0 and height <= 0:
                continue
            friendly = names[len(devices)] if len(devices) < len(names) else ""
            name = f"{friendly}（设备 {index}）" if friendly else f"摄像头设备 {index}"
            devices.append(
                {
                    "index": str(index),
                    "name": name,
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2) if fps > 0 else 0,
                }
            )
        finally:
            cap.release()
    return devices


def sample_sources() -> list[dict[str, str]]:
    candidates = [
        ("干净基线示例", "samples/clean_baseline.mp4"),
        ("对抗补丁示例", "samples/adv_patch_attacked.mp4"),
        ("强光眩光示例", "samples/glare_attacked.mp4"),
        ("可见性退化示例", "samples/visibility_degradation_attacked.mp4"),
        ("运动模糊示例", "samples/motion_blur_attacked.mp4"),
        ("局部遮挡示例", "samples/occlusion_attacked.mp4"),
        ("official_glare", "experiments/defense_results/official_v1/glare/raw_glare_attacked.mp4"),
        (
            "official_yolov5_glare",
            "experiments/official/module_a_yolov5_v1_20260505/attack_assets/physical/glare/raw_glare_attacked.mp4",
        ),
        (
            "official_visibility",
            "experiments/defense_results/official_v1/visibility_degradation/raw_visibility_degradation_attacked.mp4",
        ),
        (
            "official_yolov5_visibility",
            "experiments/official/module_a_yolov5_v1_20260505/attack_assets/physical/visibility_degradation/raw_visibility_degradation_attacked.mp4",
        ),
        (
            "official_motion",
            "experiments/defense_results/official_v1/motion_blur/raw_motion_blur_attacked.mp4",
        ),
        (
            "official_yolov5_motion",
            "experiments/official/module_a_yolov5_v1_20260505/attack_assets/physical/motion_blur/raw_motion_blur_attacked.mp4",
        ),
        (
            "official_occlusion",
            "experiments/defense_results/official_v1/occlusion/raw_occlusion_attacked.mp4",
        ),
        (
            "official_yolov5_occlusion",
            "experiments/official/module_a_yolov5_v1_20260505/attack_assets/physical/occlusion/raw_occlusion_attacked.mp4",
        ),
        (
            "clean_baseline",
            "materials/训练素材/分类/12_监控视角_仓库巡检/015_clean_baseline_single_worker_normal_6f9897da7479.mp4",
        ),
    ]
    return [
        {"name": name, "path": path} for name, path in candidates if (PROJECT_ROOT / path).exists()
    ]


def parse_args(
    argv: list[str] | None = None,
    *,
    default_open_browser: bool = False,
    default_auto_port: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Module A web monitor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--module-config", default="experiments/configs/module_a_baseline.yaml")
    parser.add_argument(
        "--profile-config", default="experiments/configs/module_a_runtime_profiles.yaml"
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=default_open_browser,
        help="Open the local web console automatically after the server starts.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_false",
        dest="open_browser",
        help="Do not open the local web console automatically.",
    )
    parser.add_argument(
        "--auto-port",
        action="store_true",
        default=default_auto_port,
        help="If the requested port is busy, try the next local ports automatically.",
    )
    return parser.parse_args(argv)


def create_server(host: str, port: int, *, auto_port: bool) -> ThreadingHTTPServer:
    last_error: OSError | None = None
    ports = range(port, port + 20) if auto_port else (port,)
    for candidate in ports:
        try:
            return ThreadingHTTPServer((host, candidate), MonitorRequestHandler)
        except OSError as exc:
            last_error = exc
            if not auto_port:
                break
    assert last_error is not None
    raise last_error


def open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(0.6)
        webbrowser.open(url)

    threading.Thread(target=_open, name="module-a-open-browser", daemon=True).start()


def main(
    argv: list[str] | None = None,
    *,
    default_open_browser: bool = False,
    default_auto_port: bool = False,
) -> None:
    args = parse_args(
        argv,
        default_open_browser=default_open_browser,
        default_auto_port=default_auto_port,
    )
    module_config_path = resolve_path(args.module_config)
    profile_config_path = resolve_path(args.profile_config)
    cache = PipelineCache(module_config_path, profile_config_path)
    server = create_server(args.host, args.port, auto_port=bool(args.auto_port))
    server.monitor_state = MonitorState(cache)  # type: ignore[attr-defined]
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    if args.open_browser:
        open_browser_later(url)
    print("=" * 72, flush=True)
    print("Module A integrated web console", flush=True)
    print(f"URL: {url}", flush=True)
    print("输入支持：MP4文件路径、RTSP/HTTP视频流、本机摄像头编号", flush=True)
    print("按 Ctrl+C 停止", flush=True)
    print("=" * 72, flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.monitor_state.stop()  # type: ignore[attr-defined]
        server.server_close()


if __name__ == "__main__":
    main()


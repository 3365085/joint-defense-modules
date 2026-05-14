from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from ab_runtime_policy import (  # noqa: E402
    DEFAULT_A_STATUS_URL,
    DEFAULT_A_STREAM_URL,
    CombinedSnapshot,
    RuntimeTrigger,
    build_combined_snapshot,
    describe_runtime_policy,
)

DEFAULT_PORT = 7870

HTML_PAGE = r"""<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>模块AB联合监控台</title>
  <style>
    body { margin: 0; font-family: Microsoft YaHei, sans-serif; background: #0b1020; color: #e8eefc; }
    header, main, footer { padding: 16px 20px; }
    header { position: sticky; top: 0; background: rgba(8,13,28,.9); border-bottom: 1px solid rgba(140,164,224,.18); backdrop-filter: blur(10px); }
    h1, h2 { margin: 0 0 10px; }
    .sub { color: #9cabd4; font-size: 13px; line-height: 1.6; }
    .row { display: grid; gap: 14px; grid-template-columns: 1.4fr 1fr; margin-top: 14px; }
    .card { border: 1px solid rgba(140,164,224,.18); border-radius: 14px; background: rgba(18,26,51,.95); padding: 14px; }
    .grid3 { display: grid; gap: 12px; grid-template-columns: repeat(3, 1fr); }
    .grid2 { display: grid; gap: 12px; grid-template-columns: repeat(2, 1fr); }
    .k { color: #9cabd4; font-size: 12px; margin-bottom: 6px; }
    .v { font-size: 20px; font-weight: 700; }
    .small { color: #9cabd4; font-size: 12px; line-height: 1.55; margin-top: 6px; white-space: pre-wrap; }
    .pill { display: inline-block; padding: 5px 10px; border-radius: 999px; border: 1px solid rgba(140,164,224,.18); background: rgba(255,255,255,.05); }
    .green { color: #33d17a; } .yellow { color: #f6c445; } .orange { color: #ff9f43; } .red { color: #ff5c7a; } .blue { color: #64a8ff; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; align-items: center; }
    input { min-width: 320px; padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(140,164,224,.18); background: #08102a; color: #e8eefc; }
    button, a.btn { padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(140,164,224,.18); background: linear-gradient(180deg, #21355f 0, #18294a 100%); color: #e8eefc; text-decoration: none; cursor: pointer; }
    .stream { width: 100%; min-height: 320px; border-radius: 14px; border: 1px solid rgba(140,164,224,.18); background: #050911; object-fit: contain; }
    .mono { font-family: Consolas, Courier New, monospace; font-size: 12px; line-height: 1.55; white-space: pre-wrap; color: #d8e5ff; }
    .list { display: grid; gap: 8px; }
    .item, .trigger { border: 1px solid rgba(140,164,224,.18); border-radius: 12px; background: rgba(8,13,28,.55); padding: 10px 12px; }
    .item-title, .trigger-title { font-weight: 700; margin-bottom: 4px; }
    .item-text, .trigger-text { color: #9cabd4; font-size: 12px; line-height: 1.55; }
    .trigger { border-left-width: 4px; border-left-style: solid; }
    .trigger.warn { border-left-color: #f6c445; } .trigger.high { border-left-color: #ff9f43; } .trigger.critical { border-left-color: #ff5c7a; }
    footer { color: #9cabd4; font-size: 12px; }
    @media (max-width: 1200px) { .row, .grid3, .grid2 { grid-template-columns: 1fr; } input { min-width: 220px; flex: 1; } }
  </style>
</head>
<body>
<header>
  <h1>模块AB联合监控台</h1>
  <div class='sub'>A 负责视频流实时检测；B 负责模型安全门、自检与运行期复核。这里是统一总览，不替代 A 的原始检测页。</div>
  <div class='toolbar'>
    <input id='aUrl' type='text' />
    <button onclick='saveConfig()'>保存连接</button>
    <button onclick='refreshNow()'>立即刷新</button>
    <a class='btn' href='http://127.0.0.1:7860/' target='_blank' rel='noreferrer'>打开模块A</a>
  </div>
</header>
<main>
  <section class='card'>
    <h2>联合总览</h2>
    <div class='grid3'>
      <div class='card'><div class='k'>联合级别</div><div id='overallLevel' class='v'>-</div><div id='overallReason' class='small'></div></div>
      <div class='card'><div class='k'>推荐动作</div><div id='overallAction' class='v' style='font-size:16px'>-</div><div id='overallActionDetail' class='small'></div></div>
      <div class='card'><div class='k'>A / B 状态</div><div id='statusPill' class='pill'>-</div><div id='statusSmall' class='small'></div></div>
    </div>
  </section>
  <section class='row'>
    <div class='card'>
      <h2>模块A实时画面</h2>
      <div class='row' style='grid-template-columns: 1.2fr .8fr;'>
        <img id='stream' class='stream' alt='模块A直播预览' />
        <div>
          <div class='card'><div class='k'>A核心指标</div><div class='mono' id='aMetrics'>-</div></div>
          <div class='card'><div class='k'>B运行期触发策略</div><div class='list' id='policyList'></div></div>
        </div>
      </div>
    </div>
    <div class='card'>
      <h2>模块B安全门摘要</h2>
      <div class='grid2'>
        <div class='card'><div class='k'>绿色模型</div><div class='mono' id='bModel'>-</div></div>
        <div class='card'><div class='k'>安全门结果</div><div class='mono' id='bGate'>-</div></div>
      </div>
      <div class='card'><div class='k'>摘要片段</div><div class='mono' id='bSummary'>-</div></div>
    </div>
  </section>
  <section class='card'>
    <h2>启动自检与运行期触发</h2>
    <div class='grid2'>
      <div class='card'><div class='k'>启动自检</div><div class='list' id='startupTriggers'></div></div>
      <div class='card'><div class='k'>运行期触发</div><div class='list' id='runtimeTriggers'></div></div>
    </div>
  </section>
</main>
<footer>如果 A 使用的不是 B 绿色模型，联合面板会标记模型侧风险；如果 A 出现确认告警、A3b、源真实性或实时性异常，B 会进入复核建议链路。</footer>
<script>
const DEFAULT_A_URL = %DEFAULT_A_STATUS_URL%;
const DEFAULT_STREAM = %DEFAULT_A_STREAM_URL%;
const LS_KEY = 'module_ab_a_url';
function $(id){ return document.getElementById(id); }
function levelClass(level){ const v = String(level || '').toLowerCase(); if(v.includes('red')) return 'red'; if(v.includes('orange')) return 'orange'; if(v.includes('yellow')) return 'yellow'; if(v.includes('green')) return 'green'; if(v.includes('blue')) return 'blue'; return ''; }
function esc(v){ return String(v ?? '').replace(/[&<>\"']/g, function(s){ if(s === '&') return '&amp;'; if(s === '<') return '&lt;'; if(s === '>') return '&gt;'; if(s === '\"') return '&quot;'; return '&#39;'; }); }
async function api(path, options){ const res = await fetch(path, Object.assign({ headers: {'Content-Type':'application/json'} }, options || {})); const data = await res.json(); if(!res.ok || data.ok === false) throw new Error(data.error || res.statusText); return data; }
function renderTriggers(id, items){ const node = $(id); if(!items || !items.length){ node.innerHTML = '<div class="small">暂无触发项。</div>'; return; } node.innerHTML = items.map(function(item){ return '<div class="trigger ' + esc(item.severity || 'warn') + '"><div class="trigger-title">' + esc(item.title || item.code) + '</div><div class="trigger-text">' + esc(item.reason || '') + '</div></div>'; }).join(''); }
function renderPolicies(items){ $('policyList').innerHTML = (items || []).map(function(item){ return '<div class="item"><div class="item-title">' + esc(item.case || '') + '</div><div class="item-text">' + esc(item.why || '') + '</div></div>'; }).join(''); }
function renderMetrics(status){ var lines = []; lines.push('帧号: ' + (status.frame_idx ?? 0)); lines.push('A3b: ' + Number(status.a3b_live_score || status.a3b_score || 0).toFixed(3) + ' / 触发=' + (status.a3b_triggered ? '是' : '否')); lines.push('p_adv: ' + Number(status.p_adv || 0).toFixed(3) + ' | p_synth: ' + Number(status.p_synth || 0).toFixed(3)); lines.push('时延: ' + Number(status.timing_ms || 0).toFixed(1) + ' ms | FPS: ' + Number(status.fps || 0).toFixed(1)); lines.push('告警数: ' + (status.alert_event_count || 0) + ' | 源真实性: ' + (status.source_authenticity_warning ? '异常' : '正常')); lines.push('原因: ' + (status.reason || status.p_adv_missing_reason || '')); lines.push('模型: ' + (status.artifact || (status.custom_model && status.custom_model.path) || '-')); $('aMetrics').textContent = lines.join('\n'); }
function renderState(state){ $('aUrl').value = state.a_status_url || DEFAULT_A_URL; $('stream').src = state.stream_url || DEFAULT_STREAM; $('overallLevel').textContent = state.combined_level || '-'; $('overallLevel').className = 'v ' + levelClass(state.combined_level); $('overallReason').textContent = state.combined_reason || ''; $('overallAction').textContent = state.recommended_action || '-'; $('overallActionDetail').textContent = state.timestamp_text || ''; $('statusPill').className = 'pill ' + levelClass(state.combined_level); $('statusPill').textContent = state.a_available ? 'A在线 / B已加载' : 'A离线 / B已加载'; $('statusSmall').textContent = 'A状态: ' + (state.a_status_url || '-') + ' | B安全门: ' + ((state.b_summary && state.b_summary.decision_level) || '-'); $('bModel').textContent = (state.b_summary && state.b_summary.green_model ? state.b_summary.green_model : '-') + '\n存在: ' + ((state.b_summary && state.b_summary.green_model_exists) ? '是' : '否'); $('bGate').textContent = '级别: ' + ((state.b_summary && state.b_summary.decision_level) || '-') + '\n分数: ' + ((state.b_summary && state.b_summary.decision_score) ?? '-'); $('bSummary').textContent = (state.b_summary && state.b_summary.summary_excerpt) || '-'; renderTriggers('startupTriggers', state.startup_triggers || []); renderTriggers('runtimeTriggers', state.runtime_triggers || []); renderPolicies(state.runtime_policy || []); renderMetrics(state.a_status || {}); }
async function refreshNow(){ try { const data = await api('/api/state'); renderState(data.state); } catch(err) { $('overallLevel').textContent = '离线'; $('overallReason').textContent = err.message || '刷新失败'; $('statusPill').className = 'pill red'; $('statusPill').textContent = '不可用'; } }
async function saveConfig(){ const aUrl = $('aUrl').value.trim() || DEFAULT_A_URL; localStorage.setItem(LS_KEY, aUrl); await api('/api/config', { method: 'POST', body: JSON.stringify({ a_status_url: aUrl }) }); await refreshNow(); }
async function init(){ const saved = localStorage.getItem(LS_KEY) || DEFAULT_A_URL; $('aUrl').value = saved; await api('/api/config', { method: 'POST', body: JSON.stringify({ a_status_url: saved }) }); await refreshNow(); setInterval(refreshNow, 1500); }
init().catch(function(err){ $('overallLevel').textContent = '启动失败'; $('overallReason').textContent = err.message || '初始化失败'; });
</script>
</body>
</html>""".replace("%DEFAULT_A_STATUS_URL%", json.dumps(DEFAULT_A_STATUS_URL)).replace("%DEFAULT_A_STREAM_URL%", json.dumps(DEFAULT_A_STREAM_URL));

class CombinedMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poll_interval = 1.5
        self._a_status_url = DEFAULT_A_STATUS_URL
        self._previous_status: dict[str, Any] | None = None
        self._state: dict[str, Any] = self._build_state_unlocked()
        self._worker: threading.Thread | None = None

    def _build_state_unlocked(self) -> dict[str, Any]:
        snapshot = build_combined_snapshot(self._a_status_url, self._previous_status, self._poll_interval)
        self._previous_status = snapshot.a_status.copy()
        return self._snapshot_to_state(snapshot)

    @staticmethod
    def _trigger_to_dict(trigger: RuntimeTrigger) -> dict[str, Any]:
        return {
            'code': trigger.code,
            'title': trigger.title,
            'reason': trigger.reason,
            'severity': trigger.severity,
            'evidence': trigger.evidence,
        }

    def _snapshot_to_state(self, snapshot: CombinedSnapshot) -> dict[str, Any]:
        host = urlparse(self._a_status_url).netloc or '127.0.0.1:7860'
        return {
            'a_status_url': self._a_status_url,
            'stream_url': DEFAULT_A_STREAM_URL.replace('127.0.0.1:7860', host),
            'timestamp': snapshot.timestamp,
            'timestamp_text': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snapshot.timestamp)),
            'a_available': snapshot.a_available,
            'a_status': snapshot.a_status,
            'b_summary': snapshot.b_summary,
            'startup_triggers': [self._trigger_to_dict(item) for item in snapshot.startup_triggers],
            'runtime_triggers': [self._trigger_to_dict(item) for item in snapshot.runtime_triggers],
            'runtime_policy': describe_runtime_policy(),
            'combined_level': snapshot.combined_level,
            'combined_reason': snapshot.combined_reason,
            'recommended_action': snapshot.recommended_action,
        }

    def refresh(self) -> dict[str, Any]:
        with self._lock:
            self._state = self._build_state_unlocked()
            return self._state.copy()

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.copy()

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            url = str(payload.get('a_status_url') or self._a_status_url).strip()
            if url:
                self._a_status_url = url
            interval = payload.get('poll_interval_sec')
            if interval is not None:
                try:
                    self._poll_interval = max(0.5, float(interval))
                except (TypeError, ValueError):
                    pass
            self._state = self._build_state_unlocked()
            return self._state.copy()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:
                pass
            self._stop.wait(self._poll_interval)


class Handler(BaseHTTPRequestHandler):
    server_version = 'ModuleABMonitor/1.0'

    @property
    def monitor(self) -> CombinedMonitor:
        return self.server.monitor_state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send_bytes(HTML_PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if parsed.path == '/api/state':
            self._send_json({'ok': True, 'state': self.monitor.refresh()})
            return
        if parsed.path == '/api/config':
            self._send_json({'ok': True, 'state': self.monitor.get_state()})
            return
        self._send_json({'ok': False, 'error': 'Not found'}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == '/api/config':
                self._send_json({'ok': True, 'state': self.monitor.update_config(payload)})
                return
            if parsed.path == '/api/refresh':
                self._send_json({'ok': True, 'state': self.monitor.refresh()})
                return
            self._send_json({'ok': False, 'error': 'Not found'}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({'ok': False, 'error': str(exc)}, HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get('Content-Length', '0') or '0')
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'), 'application/json; charset=utf-8', status)

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        if '/api/state' in message or '/api/config' in message:
            return
        print(f'[ab-monitor] {self.address_string()} - {message}')


def run(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    monitor = CombinedMonitor()
    monitor.start()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.monitor_state = monitor  # type: ignore[attr-defined]
    url = f'http://127.0.0.1:{port}/'
    print(f'[ab-monitor] listening on {url}')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='模块AB联合监控台')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--no-open', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(port=args.port, open_browser=not args.no_open)







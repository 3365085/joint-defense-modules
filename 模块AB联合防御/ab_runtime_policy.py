from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODULE_B_ROOT = ROOT / "模块B"
B_HANDOFF_ROOT = MODULE_B_ROOT / "clean_green_engineering_handoff_2026-05-09"
B_SECURITY_REPORT = B_HANDOFF_ROOT / "artifacts" / "current_best" / "security_report.json"
B_FULL_FLOW_SUMMARY = B_HANDOFF_ROOT / "artifacts" / "current_best" / "FULL_FLOW_GREEN_SUMMARY.md"
B_GREEN_MODEL = B_HANDOFF_ROOT / "artifacts" / "current_best" / "best2_purified_semantic_fixed_2026-05-09.pt"
DEFAULT_A_STATUS_URL = "http://127.0.0.1:7860/api/status"
DEFAULT_A_STREAM_URL = "http://127.0.0.1:7860/stream.mjpg"
PATH_SUFFIX_RE = re.compile(r"best2_purified_semantic_fixed_2026-05-09\.pt$", re.IGNORECASE)


@dataclass
class RuntimeTrigger:
    code: str
    title: str
    reason: str
    severity: str = "warn"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedSnapshot:
    timestamp: float
    a_available: bool
    a_status: dict[str, Any]
    b_summary: dict[str, Any]
    startup_triggers: list[RuntimeTrigger]
    runtime_triggers: list[RuntimeTrigger]
    combined_level: str
    combined_reason: str
    recommended_action: str


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def read_text_excerpt(path: Path, limit: int = 280) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeDecodeError):
        return ""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def fetch_a_status(a_status_url: str = DEFAULT_A_STATUS_URL, timeout: float = 2.5) -> tuple[bool, dict[str, Any]]:
    request = Request(a_status_url, headers={"User-Agent": "ModuleABMonitor/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("ok") and isinstance(payload.get("status"), dict):
            return True, payload["status"]
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False, {}
    return False, {}


def _normalize_text(value: str) -> str:
    return str(value or "").replace("/", "\\").strip().casefold()


def _path_contains_green_model(path_text: str) -> bool:
    if not path_text:
        return False
    normalized = _normalize_text(path_text)
    return bool(PATH_SUFFIX_RE.search(normalized)) or "clean_green_engineering_handoff_2026-05-09" in normalized


def load_module_b_summary() -> dict[str, Any]:
    report = read_json(B_SECURITY_REPORT)
    decision = report.get("decision", {}) if isinstance(report, dict) else {}
    provenance = report.get("summaries", {}).get("provenance", {}) if isinstance(report, dict) else {}
    model_path = str(provenance.get("model_path", B_GREEN_MODEL))
    return {
        "root": str(B_HANDOFF_ROOT),
        "green_model": str(B_GREEN_MODEL),
        "green_model_exists": B_GREEN_MODEL.exists(),
        "security_report": str(B_SECURITY_REPORT),
        "full_flow_summary": str(B_FULL_FLOW_SUMMARY),
        "decision_level": str(decision.get("level", "Unknown")),
        "decision_score": decision.get("score"),
        "report_exists": B_SECURITY_REPORT.exists(),
        "summary_excerpt": read_text_excerpt(B_FULL_FLOW_SUMMARY),
        "report": report,
        "provenance_model_path": model_path,
        "provenance_model_exists": Path(model_path).exists() if model_path else False,
    }


def describe_runtime_policy() -> list[dict[str, str]]:
    return [
        {"case": "开机自检", "why": "确认 A 服务、B 安全门报告、绿色模型文件是否都可用。"},
        {"case": "模型切换", "why": "A 的自定义模型路径变化时，B 立即复核是否换成了未验证模型。"},
        {"case": "告警尖峰", "why": "A3b、p_adv、源真实性或确认告警突然升高时，B 用安全门档案做复核。"},
        {"case": "实时性异常", "why": "FPS 降低、单帧耗时升高或帧推进停滞时，B 用来判断是模型侧还是输入侧拖慢。"},
        {"case": "稳态巡检", "why": "长时间运行时定期做轻量复核，防止配置漂移或模型被替换。"},
    ]


def build_startup_triggers(a_available: bool, a_status: dict[str, Any], b_summary: dict[str, Any]) -> list[RuntimeTrigger]:
    triggers: list[RuntimeTrigger] = []
    if not a_available:
        triggers.append(RuntimeTrigger("a_unreachable", "A 服务不可达", "模块A 的状态接口当前无法访问，联合面板只能展示离线状态。", "critical"))
    if not b_summary.get("report_exists"):
        triggers.append(RuntimeTrigger("b_report_missing", "B 安全门报告缺失", "未找到 security_report.json，无法确认绿色门状态。", "critical"))
    if not b_summary.get("green_model_exists"):
        triggers.append(RuntimeTrigger("b_model_missing", "B 绿色模型缺失", "绿色模型文件不存在，联合模块无法验证模型侧安全基线。", "critical"))
    if str(b_summary.get("decision_level", "")).lower() != "green":
        triggers.append(RuntimeTrigger("b_gate_not_green", "B 安全门非 Green", f"当前安全门级别为 {b_summary.get('decision_level', 'Unknown')}。", "warn", {"score": b_summary.get("decision_score")}))
    current_model = str(a_status.get("artifact") or a_status.get("custom_model", {}).get("path") or "")
    if current_model and not _path_contains_green_model(current_model):
        triggers.append(RuntimeTrigger("a_model_not_green", "A 当前模型非绿色模型", "模块A 当前使用的模型路径看起来不是 B 的绿色交付模型。", "warn", {"model": current_model}))
    return triggers


def build_runtime_triggers(a_status: dict[str, Any], previous_status: dict[str, Any] | None, b_summary: dict[str, Any], poll_interval_sec: float) -> list[RuntimeTrigger]:
    triggers: list[RuntimeTrigger] = []
    if not a_status:
        return triggers
    current_model = str(a_status.get("artifact") or a_status.get("custom_model", {}).get("path") or "")
    previous_model = str(previous_status.get("artifact") or previous_status.get("custom_model", {}).get("path") or "") if previous_status else ""
    if current_model and previous_model and _normalize_text(current_model) != _normalize_text(previous_model):
        triggers.append(RuntimeTrigger("model_swapped", "模型路径变更", "检测到 A 的模型路径变化，需要让 B 复核当前模型是否仍是已验证版本。", "high", {"current": current_model, "previous": previous_model}))
    fps = float(a_status.get("fps") or 0.0)
    timing_ms = float(a_status.get("timing_ms") or 0.0)
    if a_status.get("running") and ((fps > 0 and fps < 8.0) or timing_ms >= 160.0):
        triggers.append(RuntimeTrigger("runtime_slowdown", "实时性异常", "当前帧率偏低或单帧时延过高，建议由 B 复核模型链路和当前配置。", "warn", {"fps": fps, "timing_ms": timing_ms}))
    frame_idx = int(a_status.get("frame_idx") or 0)
    previous_frame_idx = int(previous_status.get("frame_idx") or 0) if previous_status else -1
    if a_status.get("running") and previous_status and frame_idx == previous_frame_idx and frame_idx > 0 and poll_interval_sec >= 1.0:
        triggers.append(RuntimeTrigger("frame_stall", "帧推进停滞", "帧号长时间未推进，B 可用于区分输入流卡顿和模型侧阻塞。", "warn", {"frame_idx": frame_idx}))
    if a_status.get("alert_confirmed") or a_status.get("a3b_triggered"):
        triggers.append(RuntimeTrigger("security_scan", "运行期安全复核", "A 已出现确认告警或 A3b 触发，需要把 B 的绿色门档案带入复核。", "high", {"alert_confirmed": bool(a_status.get("alert_confirmed")), "a3b_triggered": bool(a_status.get("a3b_triggered")), "reason": a_status.get("reason", "")}))
    if float(a_status.get("p_adv") or 0.0) >= 0.55 and not a_status.get("alert_confirmed"):
        triggers.append(RuntimeTrigger("candidate_review", "高风险候选", "A 的物理扰动候选分数较高，但尚未完全确认，适合由 B 做一次轻量复核。", "warn", {"p_adv": float(a_status.get("p_adv") or 0.0)}))
    if a_status.get("source_authenticity_warning"):
        triggers.append(RuntimeTrigger("source_auth_review", "源真实性异常", "运行中出现视频源真实性风险，建议调用 B 的模型安全门档案做交叉检查。", "warn"))
    custom_model = a_status.get("custom_model")
    if isinstance(custom_model, dict):
        custom_path = str(custom_model.get("path") or "")
        if custom_path and not _path_contains_green_model(custom_path):
            triggers.append(RuntimeTrigger("custom_model_risk", "自定义模型风险", "当前自定义模型不是绿色交付模型，建议运行 B 复核。", "warn", {"path": custom_path}))
    if str(b_summary.get("decision_level", "")).lower() != "green":
        triggers.append(RuntimeTrigger("b_gate_attention", "B 安全门关注", "B 安全门不是 Green，运行期不宜把当前模型当成完全可信基线。", "warn", {"level": b_summary.get("decision_level"), "score": b_summary.get("decision_score")}))
    return triggers


def decide_combined_level(startup_triggers: list[RuntimeTrigger], runtime_triggers: list[RuntimeTrigger], a_status: dict[str, Any], b_summary: dict[str, Any]) -> tuple[str, str, str]:
    if any(trigger.severity == "critical" for trigger in startup_triggers):
        return "Red", "启动自检未通过，需要先处理模型/服务基线问题。", "先修复启动层问题，再恢复实时检测。"
    if a_status.get("alert_confirmed"):
        return "Red", "模块A 已确认攻击或告警。", "立即复核视频源、模型路径和证据。"
    if any(trigger.severity == "high" for trigger in runtime_triggers):
        return "Orange", "运行期出现高风险事件，建议 B 介入复核。", "查看 B 触发原因并重新确认模型基线。"
    if runtime_triggers:
        return "Yellow", "当前存在候选异常，建议保持观察。", "继续监控 A3b、p_adv、帧率与模型路径。"
    if str(b_summary.get("decision_level", "")).lower() == "green" and a_status.get("running"):
        return "Green", "A 正常运行，B 安全门也保持绿色。", "维持当前配置即可。"
    if a_status.get("running"):
        return "Blue", "A 正在运行，但尚未形成明显异常。", "继续观察实时状态。"
    return "Idle", "当前没有运行中的视频任务。", "可启动模块A或切换模型后再观察。"


def build_combined_snapshot(a_status_url: str = DEFAULT_A_STATUS_URL, previous_status: dict[str, Any] | None = None, poll_interval_sec: float = 1.5) -> CombinedSnapshot:
    a_available, a_status = fetch_a_status(a_status_url)
    b_summary = load_module_b_summary()
    startup_triggers = build_startup_triggers(a_available, a_status, b_summary)
    runtime_triggers = build_runtime_triggers(a_status, previous_status, b_summary, poll_interval_sec)
    level, reason, action = decide_combined_level(startup_triggers, runtime_triggers, a_status, b_summary)
    return CombinedSnapshot(time.time(), a_available, a_status, b_summary, startup_triggers, runtime_triggers, level, reason, action)

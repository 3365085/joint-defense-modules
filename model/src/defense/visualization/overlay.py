from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PPE_LABEL_TEXT = {
    "helmet": "安全帽",
    "head": "裸头",
    "person": "人员",
}

PPE_LABEL_COLORS = {
    "helmet": (0, 220, 80),
    "head": (0, 150, 255),
    "person": (255, 210, 80),
}

_PERSON_LABELS = {"person", "worker", "human", "pedestrian"}


_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(2, int(size))
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(path, size=size)
            return _FONT_CACHE[size]
        except Exception:
            continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _draw_text_cn(frame: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int], *, size: int = 18) -> None:
    _draw_texts_cn(frame, [(text, pos, color, size)])


def _draw_texts_cn(
    frame: np.ndarray,
    items: list[tuple[str, tuple[int, int], tuple[int, int, int], int]],
) -> None:
    if not items:
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for text, pos, color, size in items:
        b, g, r = color
        draw.text(pos, str(text), font=_font(size), fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _text_width(text: str, size: int) -> int:
    try:
        return int(round(float(_font(size).getlength(str(text)))))
    except Exception:
        return len(str(text)) * max(1, int(size) // 2)


def _fit_text_for_width(text: str, max_width: int, size: int) -> str:
    value = str(text)
    if max_width <= 0 or _text_width(value, size) <= max_width:
        return value
    suffix = "..."
    budget = max(0, int(max_width) - _text_width(suffix, size))
    if budget <= 0:
        return suffix
    low = 0
    high = len(value)
    while low < high:
        mid = (low + high + 1) // 2
        if _text_width(value[:mid], size) <= budget:
            low = mid
        else:
            high = mid - 1
    return f"{value[:low].rstrip()}{suffix}"


def _reason_text(values: Any) -> str:
    labels = {
        "glare": "强光干扰",
        "occlusion": "遮挡干扰",
        "motion_blur": "运动模糊",
        "local_flow": "局部运动异常",
        "confidence_drop": "置信度下降",
        "observed_window": "窗口观察触发",
        "single_strong": "单帧强触发",
        "bare_head_without_matched_helmet": "裸头未匹配安全帽",
        "B_BLIND_GLARE_BLIND": "强光/眩光遮蔽",
    }
    if not values:
        return ""
    if isinstance(values, str):
        items = re.split(r"[,;，；\s]+", values)
    else:
        items = values if isinstance(values, (list, tuple, set)) else [values]
    return "、".join(labels.get(str(item), str(item)) for item in list(items)[:4])


def _ui_scale(h: int, w: int) -> float:
    """按短边相对 720p 归一化的 UI 缩放, UHD 下自动放大字体/卡片, 保持视觉比例一致。"""
    return max(0.6, min(3.0, min(h, w) / 720.0))


def _rounded_panel(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *, radius: int = 14, color: tuple[int, int, int] = (24, 22, 20), alpha: float = 0.62,
) -> None:
    """深色半透明矩形面板(卡片底)。就地混合到 frame。radius 参数保留但不再画圆角。"""
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, frame[y1:y2, x1:x2], 1 - alpha, 0, frame[y1:y2, x1:x2])


# 状态配色(BGR): 克制的绿/琥珀/红, 文字统一近白, 不用刺眼纯红大字。
_STATE_STYLE = {
    "clear":   {"accent": (120, 200, 120), "label": "监控中"},
    "suspect": {"accent": (60, 190, 245),  "label": "疑似异常"},
    "held":    {"accent": (70, 170, 255),  "label": "告警保持"},
    "alert":   {"accent": (70, 90, 245),   "label": "告警确认"},
}


def draw_hud(frame: np.ndarray, info: dict[str, Any], frame_idx: int, *, effective: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    s = _ui_scale(h, w)
    p_adv = info.get("p_adv")
    p_adv_text = "--" if p_adv is None else f"{float(p_adv):.2f}"
    detect_fps = info.get("detect_fps")
    alert = bool(info.get("alert_confirmed", False))
    attack = bool(info.get("attack_detected", info.get("is_attack", False)))
    held = bool(info.get("alert_display_held", False))

    key = "held" if (held and not alert) else ("alert" if alert else ("suspect" if attack else "clear"))
    style = _STATE_STYLE[key]
    accent = style["accent"]

    pad = int(16 * s)
    title_sz = int(30 * s)
    body_sz = int(19 * s)
    small_sz = int(16 * s)
    line_gap = int(10 * s)

    # 面板尺寸: 标题行 + 指标行 (+ 可选原因行)
    reason_codes = info.get("reason_codes") or info.get("details", {}).get("reason_codes") or []
    last_reason_codes = info.get("alert_last_reason_codes") or []
    reason_line = ""
    if held and last_reason_codes:
        reason_line = f"原因 · {_reason_text(last_reason_codes)}"
    elif reason_codes and (alert or attack or held):
        reason_line = f"原因 · {_reason_text(reason_codes)}"

    metrics = f"p_adv {p_adv_text}"
    if detect_fps is not None:
        metrics += f"    检测 {float(detect_fps):.0f} fps"
    metrics += f"    帧 {frame_idx:05d}"

    panel_x = pad
    panel_y = pad
    panel_w = max(
        _text_width(style["label"], title_sz) + int(64 * s),
        _text_width(metrics, body_sz) + pad * 2,
        _text_width(reason_line, small_sz) + pad * 2 if reason_line else 0,
    )
    panel_w = min(panel_w, w - pad * 2)
    rows_h = title_sz + line_gap + body_sz + (line_gap + small_sz if reason_line else 0)
    panel_h = rows_h + pad * 2
    _rounded_panel(frame, panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
                   color=(28, 24, 22), alpha=0.60)

    ty = panel_y + pad
    items = [
        (style["label"], (panel_x + pad, ty), accent, title_sz),
        (metrics, (panel_x + pad, ty + title_sz + line_gap), (235, 235, 235), body_sz),
    ]
    if reason_line:
        items.append((_fit_text_for_width(reason_line, panel_w - pad * 2, small_sz),
                      (panel_x + pad, ty + title_sz + line_gap + body_sz + line_gap), (200, 200, 200), small_sz))
    _draw_texts_cn(frame, items)

    # 告警: 稳定描边(不闪烁), 颜色随状态
    if alert or held:
        t = max(2, int(3 * s))
        cv2.rectangle(frame, (t, t), (w - t, h - t), accent, t)
    return frame


def _draw_label(
    frame: np.ndarray,
    box: list[int],
    text: str,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    s = _ui_scale(h, w)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font_size = max(12, int(round(15 * s)))
    pad_x = max(4, int(font_size * 0.4))
    pad_y = max(2, int(font_size * 0.22))
    try:
        bbox = _font(font_size).getbbox(text)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
    except Exception:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_h = max(font_size + pad_y * 2, th + pad_y * 2)
    label_w = tw + pad_x * 2
    # 标签优先放框上方, 顶部空间不足则放框内顶部
    label_y1 = y1 - label_h if y1 - label_h >= 0 else y1
    label_y2 = label_y1 + label_h
    lx2 = min(w - 1, x1 + label_w)
    _rounded_panel(frame, x1, label_y1, lx2, label_y2, radius=int(4 * s), color=color, alpha=0.92)
    _draw_text_cn(frame, text, (x1 + pad_x, label_y1 + pad_y), (15, 15, 15), size=font_size)


def _is_person_track(track: dict[str, Any]) -> bool:
    labels = (
        track.get("label"),
        track.get("stable_label"),
        track.get("evidence_label"),
    )
    return any(str(label or "").lower() in _PERSON_LABELS for label in labels)


def draw_ppe_boxes(
    frame: np.ndarray,
    tracks: list[dict[str, Any]] | None,
    *,
    show_person_boxes: bool = True,
) -> np.ndarray:
    if not tracks:
        return frame
    rendered = frame
    for track in tracks:
        if not show_person_boxes and _is_person_track(track):
            continue
        box = track.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        label = str(track.get("label", "object"))
        color = PPE_LABEL_COLORS.get(label, (200, 200, 200))
        confidence = float(track.get("confidence", 0.0) or 0.0)
        suffix = " 保持" if int(track.get("misses", 0) or 0) > 0 else ""
        small = " 远距" if track.get("is_small") else ""
        evidence_label = str(track.get("evidence_label") or "")
        promoted_label = str(track.get("promoted_label") or "")
        if promoted_label == "head":
            display_label = "裸头增强"
        elif promoted_label == "helmet":
            display_label = "安全帽增强"
        elif label == "head" and (evidence_label == "head" or not bool(track.get("hold_eligible", True))):
            display_label = "弱裸头"
        elif label == "helmet" and evidence_label == "helmet":
            display_label = "弱安全帽"
        else:
            display_label = PPE_LABEL_TEXT.get(label, label)
        text = f"{display_label}{confidence:.2f}{small}{suffix}"
        _draw_label(rendered, [int(v) for v in box], text, color, 1)
    return rendered


def draw_media_box(frame: np.ndarray, info: dict[str, Any] | None) -> np.ndarray:
    """框出 A3b 检出的翻拍/静态媒体可疑区域(demo 行为)。bbox 为 640 空间, 按当前帧尺寸缩放。
    仅在 media_confirmed 时画, 避免干扰。纯显示, 不改检测。"""
    if not info:
        return frame
    details = info.get("details", {})
    a3b = details.get("a3b", {}) if isinstance(details, dict) else {}
    if not isinstance(a3b, dict) or not a3b.get("media_confirmed"):
        return frame
    bbox = a3b.get("p_media_bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return frame
    h, w = frame.shape[:2]
    sx, sy = w / 640.0, h / 640.0
    x1, y1, x2, y2 = [int(round(v * s)) for v, s in zip(bbox, (sx, sy, sx, sy))]
    color = (0, 140, 255)  # 橙红, 区别于 PPE 框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    score = float(a3b.get("p_media_confirmed_score", a3b.get("p_media", 0.0)) or 0.0)
    _draw_label(frame, [x1, y1, x2, y2], f"疑似翻拍/静态媒体 {score:.2f}", color, 2)
    return frame


def draw_ppe_hud(frame: np.ndarray, ppe: dict[str, Any] | None) -> np.ndarray:
    if not ppe:
        return frame
    h, w = frame.shape[:2]
    s = _ui_scale(h, w)
    person_count = int(ppe.get("effective_person_count", ppe.get("person_count", 0)) or 0)
    helmet_count = int(ppe.get("helmet_count", 0) or 0)
    head_count = int(ppe.get("head_count", 0) or 0)
    missing_helmet = int(ppe.get("missing_helmet_count", 0) or 0)

    pad = int(16 * s)
    sz = int(20 * s)
    gap = int(28 * s)
    warn = missing_helmet > 0

    # 底部右对齐的精炼计数条: 人员 / 安全帽 / 裸头 (+ 未戴帽告警)
    chips = [
        ("人员", person_count, (235, 210, 120)),
        ("安全帽", helmet_count, (120, 210, 130)),
        ("裸头", head_count, (90, 175, 245)),
    ]
    if warn:
        chips.append(("未戴帽", missing_helmet, (70, 90, 245)))

    parts = [f"{name} {val}" for name, val, _ in chips]
    line = "    ".join(parts)
    line_w = _text_width(line, sz)
    panel_w = min(line_w + pad * 2, w - pad * 2)
    panel_h = sz + pad * 2
    panel_x2 = w - pad
    panel_x1 = panel_x2 - panel_w
    panel_y2 = h - pad
    panel_y1 = panel_y2 - panel_h
    _rounded_panel(frame, panel_x1, panel_y1, panel_x2, panel_y2,
                   radius=int(12 * s), color=(28, 24, 22), alpha=0.58)

    # 逐段上色渲染(数字用对应类别色, 未戴帽标红)
    items = []
    cx = panel_x1 + pad
    ty = panel_y1 + pad
    for name, val, col in chips:
        seg = f"{name} {val}"
        items.append((seg, (cx, ty), col, sz))
        cx += _text_width(seg, sz) + gap
    _draw_texts_cn(frame, items)
    return frame


def render_preview(
    frame: np.ndarray,
    *,
    info: dict[str, Any] | None = None,
    ppe: dict[str, Any] | None = None,
    ppe_tracks: list[dict[str, Any]] | None = None,
    display_options: dict[str, Any] | None = None,
    frame_idx: int = 0,
) -> np.ndarray:
    options = display_options or {}
    rendered = frame.copy()
    if options.get("show_boxes", True):
        rendered = draw_ppe_boxes(
            rendered,
            ppe_tracks,
            show_person_boxes=options.get("show_person_boxes", True),
        )
    if options.get("show_boxes", True) and info is not None:
        rendered = draw_media_box(rendered, info)
    if options.get("show_module_hud", True) and info is not None:
        rendered = draw_hud(rendered, info, frame_idx)
    if options.get("show_ppe_hud", True) and ppe is not None:
        rendered = draw_ppe_hud(rendered, ppe)
    return rendered


def encode_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return encoded.tobytes()

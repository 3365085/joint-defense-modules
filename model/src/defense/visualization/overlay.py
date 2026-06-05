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
    }
    if not values:
        return ""
    if isinstance(values, str):
        items = re.split(r"[,;，；\s]+", values)
    else:
        items = values if isinstance(values, (list, tuple, set)) else [values]
    return "、".join(labels.get(str(item), str(item)) for item in list(items)[:4])


def draw_hud(frame: np.ndarray, info: dict[str, Any], frame_idx: int, *, effective: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (10, 10, 15), -1)
    frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)

    layer = str(info.get("layer_triggered", "NORMAL"))
    timing = float(info.get("timing_ms", 0.0) or 0.0)
    p_adv = info.get("p_adv")
    p_adv_text = "N/A" if p_adv is None else f"{float(p_adv):.2f}"
    alert = bool(info.get("alert_confirmed", False))
    attack = bool(info.get("attack_detected", info.get("is_attack", False)))

    text_items = [
        (f"帧 {frame_idx:05d} | 层 {layer} | 物理扰动={p_adv_text} | {timing:.1f}毫秒", (10, 7), (0, 0, 255), 18)
    ]
    state = "告警确认" if alert else ("疑似异常" if attack else "监控中")
    text_items.append((state, (10, 35), (0, 0, 255), 18))

    reason_codes = info.get("reason_codes") or info.get("details", {}).get("reason_codes") or []
    if reason_codes:
        text = _reason_text(reason_codes)
        text_items.append((text[:80], (10, 62), (0, 0, 255), 15))
    elif effective:
        text_items.append(("证据帧", (10, 62), (0, 0, 255), 15))
    _draw_texts_cn(frame, text_items)

    if alert:
        thick = 2 if frame_idx % 2 == 0 else 1
        cv2.rectangle(frame, (thick, thick), (w - thick, h - thick), (0, 0, 255), thick)
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
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font_size = max(6, min(9, int(round(min(h, w) / 180))))
    pad_x = max(2, font_size // 3)
    pad_y = max(1, font_size // 5)
    try:
        bbox = _font(font_size).getbbox(text)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
    except Exception:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
    label_h = max(7, th + pad_y * 2)
    label_y1 = max(0, y1 - label_h)
    label_y2 = min(h - 1, label_y1 + label_h)
    cv2.rectangle(frame, (x1, label_y1), (min(w - 1, x1 + tw + pad_x * 2), label_y2), color, -1)
    _draw_text_cn(frame, text, (x1 + pad_x, label_y1 + pad_y), (0, 0, 0), size=font_size)


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


def draw_ppe_hud(frame: np.ndarray, ppe: dict[str, Any] | None) -> np.ndarray:
    if not ppe:
        return frame
    h, w = frame.shape[:2]
    y0 = h - 86
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (10, 10, 15), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    warning = bool(ppe.get("warning") or ppe.get("confirmed"))
    color = (0, 0, 255)
    suppression = ppe.get("helmet_fp_suppression", {}) if isinstance(ppe.get("helmet_fp_suppression"), dict) else {}
    weak_head_count = len(suppression.get("weak_head_indices", suppression.get("suppressed_head_indices", [])) or [])
    weak_helmet_count = len(suppression.get("weak_helmet_indices", []) or [])
    weak_person_count = len(suppression.get("weak_person_indices", []) or [])
    promoted_head_count = int(ppe.get("promoted_head_count", 0) or 0)
    promoted_helmet_count = int(ppe.get("promoted_helmet_count", 0) or 0)
    promoted_person_count = int(ppe.get("promoted_person_count", 0) or 0)
    raw_person_count = int(ppe.get("raw_person_count", ppe.get("person_count", 0)) or 0)
    inferred_person_count = int(ppe.get("inferred_person_count", raw_person_count) or 0)
    effective_person_count = int(ppe.get("effective_person_count", ppe.get("person_count", 0)) or 0)
    text = (
        f"安全帽 原始人={raw_person_count} 有效人={effective_person_count} 推断人={inferred_person_count} "
        f"安全帽={int(ppe.get('helmet_count', 0) or 0)} "
        f"裸头={int(ppe.get('head_count', 0) or 0)} "
        f"弱证据={weak_person_count}/{weak_head_count}/{weak_helmet_count} "
        f"时序增强={promoted_person_count}/{promoted_head_count}/{promoted_helmet_count} "
        f"未戴帽={int(ppe.get('missing_helmet_count', 0) or 0)}"
    )
    text_items = [(text, (10, h - 62), color, 17)]
    reason = str(ppe.get("reason", ""))[:90]
    if reason:
        text_items.append((_reason_text(reason), (10, h - 32), color, 15))
    _draw_texts_cn(frame, text_items)
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

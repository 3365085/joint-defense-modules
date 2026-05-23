from __future__ import annotations

from typing import Any

import cv2
import numpy as np

PPE_LABEL_TEXT = {
    "helmet": "helmet",
    "head": "head",
    "person": "person",
}

PPE_LABEL_COLORS = {
    "helmet": (0, 220, 80),
    "head": (0, 150, 255),
    "person": (255, 210, 80),
}


def draw_hud(frame: np.ndarray, info: dict[str, Any], frame_idx: int, *, effective: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (10, 10, 15), -1)
    frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    layer = str(info.get("layer_triggered", "NORMAL"))
    timing = float(info.get("timing_ms", 0.0) or 0.0)
    p_adv = info.get("p_adv")
    p_adv_text = "N/A" if p_adv is None else f"{float(p_adv):.2f}"
    alert = bool(info.get("alert_confirmed", False))
    attack = bool(info.get("attack_detected", info.get("is_attack", False)))

    cv2.putText(
        frame,
        f"FRAME {frame_idx:05d} | {layer} | p_adv={p_adv_text} | {timing:.1f}ms",
        (10, 24),
        font,
        0.52,
        (255, 230, 150),
        1,
        cv2.LINE_AA,
    )
    state = "ALERT" if alert else ("SUSPICIOUS" if attack else "MONITORING")
    color = (0, 0, 255) if alert else ((0, 180, 255) if attack else (0, 220, 100))
    cv2.putText(frame, state, (10, 52), font, 0.52, color, 1, cv2.LINE_AA)

    reason_codes = info.get("reason_codes") or info.get("details", {}).get("reason_codes") or []
    if reason_codes:
        text = ",".join(str(v) for v in reason_codes[:4])
        cv2.putText(frame, text[:80], (10, 78), font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    elif effective:
        cv2.putText(frame, "EVIDENCE", (10, 78), font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

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
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    label_y1 = max(0, y1 - th - 7)
    cv2.rectangle(frame, (x1, label_y1), (min(w - 1, x1 + tw + 6), y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 3, max(th + 1, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_ppe_boxes(frame: np.ndarray, tracks: list[dict[str, Any]] | None) -> np.ndarray:
    if not tracks:
        return frame
    rendered = frame
    for track in tracks:
        box = track.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        label = str(track.get("label", "object"))
        color = PPE_LABEL_COLORS.get(label, (200, 200, 200))
        confidence = float(track.get("confidence", 0.0) or 0.0)
        suffix = " held" if int(track.get("misses", 0) or 0) > 0 else ""
        small = " far" if track.get("is_small") else ""
        evidence_label = str(track.get("evidence_label") or "")
        promoted_label = str(track.get("promoted_label") or "")
        if promoted_label == "head":
            display_label = "head+"
        elif promoted_label == "helmet":
            display_label = "helmet+"
        elif label == "head" and (evidence_label == "head" or not bool(track.get("hold_eligible", True))):
            display_label = "weakHead"
        elif label == "helmet" and evidence_label == "helmet":
            display_label = "weakHelmet"
        else:
            display_label = PPE_LABEL_TEXT.get(label, label)
        text = f"{display_label}#{int(track.get('track_id', 0))} {confidence:.2f}{small}{suffix}"
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    warning = bool(ppe.get("warning") or ppe.get("confirmed"))
    color = (0, 0, 255) if warning else (0, 220, 100)
    suppression = ppe.get("helmet_fp_suppression", {}) if isinstance(ppe.get("helmet_fp_suppression"), dict) else {}
    weak_head_count = len(suppression.get("weak_head_indices", suppression.get("suppressed_head_indices", [])) or [])
    weak_helmet_count = len(suppression.get("weak_helmet_indices", []) or [])
    promoted_head_count = int(ppe.get("promoted_head_count", 0) or 0)
    promoted_helmet_count = int(ppe.get("promoted_helmet_count", 0) or 0)
    raw_person_count = int(ppe.get("raw_person_count", ppe.get("person_count", 0)) or 0)
    inferred_person_count = int(ppe.get("inferred_person_count", raw_person_count) or 0)
    text = (
        f"PPE rawP={raw_person_count} inferP={inferred_person_count} "
        f"helmet={int(ppe.get('helmet_count', 0) or 0)} "
        f"head={int(ppe.get('head_count', 0) or 0)} "
        f"weakH={weak_head_count}/{weak_helmet_count} "
        f"prom={promoted_head_count}/{promoted_helmet_count} "
        f"missing={int(ppe.get('missing_helmet_count', 0) or 0)}"
    )
    cv2.putText(frame, text, (10, h - 52), font, 0.45, color, 1, cv2.LINE_AA)
    reason = str(ppe.get("reason", ""))[:90]
    if reason:
        cv2.putText(frame, reason, (10, h - 24), font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
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
        rendered = draw_ppe_boxes(rendered, ppe_tracks)
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

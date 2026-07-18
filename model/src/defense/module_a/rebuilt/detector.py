from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any

import cv2
import numpy as np

from .. import native_bridge as _NATIVE_BRIDGE
from ..types import ROI, ModuleAInput, ModuleAResult
from .a4_artifact import (
    A4ArtifactValidationError,
    load_a4_artifact_metadata,
    validate_a4_artifact_metadata,
)
from .a4_patch_features import (
    A4_PATCH_FEATURE_NAMES,
    extract_a4_patch_features,
)
from .target_anchored import TargetAnchoredAnalyzer

A4_FEATURE_SCHEMA_VERSION = "rebuilt-a4-96-v4"
_A4_PHYSICAL_FEATURE_NAMES: tuple[str, ...] = (
    "a1.delta_h",
    "a1.delta_h_roi_max",
    "a1.delta_h_local_max",
    "a1.delta_h_target_contrast",
    "a1.a1_feature_score",
    "a2.change_t",
    "a2.change_t_roi_max",
    "a2.change_t_local_max",
    "a2.change_t_without_motion_target",
    "a2.a2_feature_score",
    "a3.f_flow",
    "a3.flow_local_anomaly_ratio",
    "a3.flow_residual",
    "a3.flow_shape_score",
    "a3.flow_target_relation",
    "a3.a3_feature_score",
)
A4_FEATURE_NAMES: tuple[str, ...] = (
    *_A4_PHYSICAL_FEATURE_NAMES,
    *A4_PATCH_FEATURE_NAMES,
    *tuple(
        name.replace("a4_patch.", "a4_patch_delta.", 1)
        for name in A4_PATCH_FEATURE_NAMES
    ),
)

_A3B_GLOBAL_WORKER_LOCK = threading.Lock()
_A3B_GLOBAL_WORKER_TOKENS: set[int] = set()
_A3B_GLOBAL_WORKER_TOKEN_SEQ = 0
_A3B_GLOBAL_WORKER_HARD_LIMIT = 2


def _try_acquire_a3b_global_worker_token(limit: int) -> int | None:
    global _A3B_GLOBAL_WORKER_TOKEN_SEQ
    with _A3B_GLOBAL_WORKER_LOCK:
        if len(_A3B_GLOBAL_WORKER_TOKENS) >= max(1, int(limit)):
            return None
        _A3B_GLOBAL_WORKER_TOKEN_SEQ += 1
        token = _A3B_GLOBAL_WORKER_TOKEN_SEQ
        _A3B_GLOBAL_WORKER_TOKENS.add(token)
        return token


def _release_a3b_global_worker_token(token: int) -> None:
    with _A3B_GLOBAL_WORKER_LOCK:
        _A3B_GLOBAL_WORKER_TOKENS.discard(int(token))


def _a3b_global_live_worker_count() -> int:
    with _A3B_GLOBAL_WORKER_LOCK:
        return len(_A3B_GLOBAL_WORKER_TOKENS)


try:
    _NATIVE = _NATIVE_BRIDGE.require_native()
except RuntimeError:
    _NATIVE = None


def _file_sha256(path: _Path | str | None) -> str | None:
    if not path:
        return None
    resolved = _Path(path)
    if not resolved.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest().upper()


def _resolve_rebuilt_data_dir() -> _Path:
    """Resolve main-project data for A4 and RAFT auxiliary artifacts.

    Candidate order (first existing wins):
      1. ``MODULE_A_ARTIFACT_DIR`` env var
      2. ``model/runtime/artifacts/module_a``
      3. ``defense/module_a/rebuilt/data`` (bundled next to this package)
      4. ``model/data`` (main project data dir)
    Returns candidate 2 as the conventional default when none exist. Runtime
    initialization treats these locations as read-only and never downloads or
    builds missing RAFT assets.
    """
    import os

    here = _Path(__file__).resolve()
    candidates: list[_Path] = []
    env_dir = os.environ.get("MODULE_A_ARTIFACT_DIR")
    if env_dir:
        candidates.append(_Path(env_dir))
    candidates.append(here.parents[4] / "runtime" / "artifacts" / "module_a")
    candidates.append(here.parent / "data")              # defense/module_a/rebuilt/data
    candidates.append(here.parents[4] / "data")          # model/data
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return here.parents[4] / "runtime" / "artifacts" / "module_a"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _projection_peak_lines(
    values: np.ndarray,
    limit: int,
    min_gap: int,
) -> list[tuple[int, float]]:
    """Return strong projection-line centers without scanning bins in Python."""

    if values.size == 0:
        return []
    vmax = float(np.max(values))
    if vmax <= 1e-6:
        return []
    norm = values.astype(np.float32) / vmax
    threshold = max(0.18, float(np.percentile(norm, 88)) * 0.82)
    above_threshold = norm >= threshold
    padded = np.concatenate(
        (
            np.asarray([False], dtype=np.bool_),
            above_threshold,
            np.asarray([False], dtype=np.bool_),
        )
    )
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    groups: list[tuple[int, float]] = []
    for start, end in zip(
        transitions[0::2],
        transitions[1::2],
        strict=True,
    ):
        segment = norm[start:end]
        weights = segment + 1e-4
        center = int(
            round(
                float(
                    np.average(
                        np.arange(start, end),
                        weights=weights,
                    )
                )
            )
        )
        groups.append((center, float(np.max(segment))))
    groups.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for center, strength in groups:
        if all(
            abs(center - old_center) >= min_gap
            for old_center, _ in selected
        ):
            selected.append((center, strength))
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: item[0])
    return selected


def _score(value: float, floor: float, ceil: float) -> float:
    if ceil <= floor:
        return 1.0 if value >= ceil else 0.0
    return _clamp((float(value) - floor) / (ceil - floor))


def _bbox_area(box: tuple[int, int, int, int] | list[float] | None) -> float:
    if not box or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: tuple[int, int, int, int] | list[float] | None,
              b: tuple[int, int, int, int] | list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _bbox_proximity(a: tuple[int, int, int, int] | list[float] | None,
                    b: tuple[int, int, int, int] | list[float] | None,
                    diag: float) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    dist = math.hypot(acx - bcx, acy - bcy)
    return _clamp(1.0 - dist / max(1.0, diag * 0.35))


def _expand_box(box: tuple[int, int, int, int], width: int, height: int,
                margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    dx, dy = int(bw * margin), int(bh * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    )


def _clip_box(box: tuple[int, int, int, int] | list[float],
              width: int,
              height: int,
              min_size: int = 8) -> tuple[int, int, int, int] | None:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return (x1, y1, x2, y2)


def _hist_lbp(lbp: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if box is not None:
        x1, y1, x2, y2 = box
        patch = lbp[y1:y2, x1:x2]
    else:
        patch = lbp
    if patch.size == 0:
        return np.zeros(32, dtype=np.float32)
    hist = cv2.calcHist([patch.astype(np.uint8)], [0], None, [32], [0, 256])
    hist = hist.reshape(-1).astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(32, dtype=np.float32)
    return hist / total


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(0.5 * np.abs(a - b).sum())


def _best_grid_value(
    value_map: np.ndarray,
    grid: int = 8,
) -> tuple[float, float, tuple[int, int, int, int]]:
    h, w = value_map.shape[:2]
    cell_w = max(8, w // max(1, grid))
    cell_h = max(8, h // max(1, grid))
    best = 0.0
    total = 0.0
    count = 0
    best_box = (0, 0, w, h)
    for y in range(0, h, cell_h):
        for x in range(0, w, cell_w):
            x2 = min(w, x + cell_w)
            y2 = min(h, y + cell_h)
            patch = value_map[y:y2, x:x2]
            if patch.size == 0:
                continue
            val = float(np.mean(patch))
            total += val
            count += 1
            if val > best:
                best = val
                best_box = (x, y, x2, y2)
    mean = total / max(1, count)
    return best, mean, best_box


def _roi_anchor_weight(roi: ROI, width: int, height: int, has_person_context: bool) -> float:
    box = _clip_box(roi.bbox, width, height, min_size=4)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    area_ratio = _bbox_area(box) / max(1.0, width * height)
    label = str(roi.label or "").lower()
    conf = float(roi.confidence) if roi.confidence is not None else 1.0
    touches_edge = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
    head_like = label in {"head", "helmet"}

    # Small head/helmet boxes on the image border are common shelf/window false
    # detections in hand-held normal videos. They should not anchor A1/A2/A3.
    if head_like and touches_edge and area_ratio < 0.018 and not has_person_context:
        return 0.0
    if touches_edge and area_ratio < 0.006 and conf < 0.80:
        return 0.0
    if area_ratio < 0.006 and conf < 0.45:
        return 0.25
    return 1.0


def _target_relation(
    candidate_box: tuple[int, int, int, int] | list[float] | None,
    rois: list[ROI],
    width: int,
    height: int,
) -> tuple[float, float, float, bool]:
    if not candidate_box or not rois:
        return 0.0, 0.0, 0.0, False
    diag = math.hypot(width, height)
    best_iou = 0.0
    best_prox = 0.0
    has_person_context = any(
        str(roi.label or "").lower() == "person"
        and _bbox_area(roi.bbox) / max(1.0, width * height) >= 0.025
        for roi in rois
    )
    for roi in rois:
        anchor_weight = _roi_anchor_weight(roi, width, height, has_person_context)
        if anchor_weight <= 0.0:
            continue
        iou = _bbox_iou(candidate_box, roi.bbox) * anchor_weight
        prox = _bbox_proximity(candidate_box, roi.bbox, diag) * anchor_weight
        best_iou = max(best_iou, iou)
        best_prox = max(best_prox, prox)
    relation = max(_score(best_iou, 0.02, 0.35), best_prox * 0.85)
    return relation, best_iou, best_prox, bool(best_iou >= 0.02 or best_prox >= 0.35)


@dataclass
class _MediaTrack:
    bbox: tuple[int, int, int, int]
    stable_count: int = 1
    lifetime: int = 1
    miss_count: int = 0
    last_area: float = 0.0


class ModuleADetector:
    """Production rebuilt Module A detector using the shared result contract."""

    _A3B_CLOSE_JOIN_BUDGET_SECONDS = 1.0

    def _refresh_native_status(self) -> None:
        status = _NATIVE_BRIDGE.status()
        status.update(
            {
                "hit_counts": dict(self.native_hit_counts),
                "fallback_counts": dict(self.native_fallback_counts),
                "enabled_stages": sorted(
                    stage
                    for stage, count in self.native_hit_counts.items()
                    if int(count) > 0
                ),
                "last_error": self.native_last_error,
            }
        )
        self.native_status = status

    def _native_call(
        self,
        stage: str,
        function_name: str,
        *args: Any,
    ) -> Any | None:
        if _NATIVE is None:
            self.native_fallback_counts[stage] += 1
            self._native_status_dirty = True
            return None
        try:
            function = getattr(_NATIVE, function_name)
            value = function(*args)
        except Exception as exc:
            self.native_fallback_counts[stage] += 1
            self.native_last_error = (
                f"{stage}:{function_name}:{type(exc).__name__}:{exc}"
            )
            self._native_status_dirty = True
            return None
        self.native_hit_counts[stage] += 1
        self._native_status_dirty = True
        return value

    def __init__(self, config: dict[str, Any] | None = None):
        module_config = (config or {}).get("module_a", config or {})
        self.native_hit_counts: dict[str, int] = {
            "a1": 0,
            "a2": 0,
            "a3": 0,
            "a3b": 0,
            "blind": 0,
        }
        self.native_fallback_counts: dict[str, int] = {
            "a1": 0,
            "a2": 0,
            "a3": 0,
            "a3b": 0,
            "blind": 0,
        }
        self.native_last_error: str | None = None
        self.native_status: dict[str, Any] = {}
        self._refresh_native_status()
        self.frame_size = int(module_config.get("frame_size", 640))
        self.theta_adv = float(module_config.get("rebuilt_theta_adv", 0.65))
        # Rule fallback keeps the configured threshold.  A bound production
        # classifier may carry a threshold selected by grouped CV/heldout
        # gates; keep it separate so classifier calibration does not silently
        # change fallback behaviour.
        self.a4_classifier_decision_threshold = float(self.theta_adv)
        self.theta_media = float(module_config.get("rebuilt_theta_media", 0.55))
        self.theta_media_raw = float(module_config.get("rebuilt_theta_media_raw", 0.50))
        # A3b 静态媒体/翻拍作为独立报警触发器: demo 内核移植时禁用(背景结构墙/门框会误报, 原计划改喂
        # XGBoost 但未完成)→ 主项目原有的静态图片/翻拍检测能力回退。config 控制是否恢复(默认关=保持
        # demo 现状零风险); 开启后 media 候选可经 N-of-M 独立确认报警。启用前须留出集验证误报不破 7.4%。
        # 默认开: 收紧门(edge/border_contrast)+持续段门(_media_run>=15)已使留出集48段验证
        # 误报0%/召回90.5%不变(glare4-4), 且找回手机/电脑屏翻拍检测。设False回退demo禁用现状。
        self._a3b_independent_trigger = bool(
            module_config.get("rebuilt_a3b_independent_trigger", True)
        )
        # A3b 独立触发的收紧候选门(2026-07-12 数据方案): 直接用 media_candidate_allowed 会让干净视频
        # 误报81.5%(墙/门框/施工纹理经 screen_like_evidence 后门穿过)。实测可分离维度是 edge(边缘密度,
        # 反向: 真媒体插入≈0.56 vs 施工繁忙纹理≥0.64)与 border_contrast(正向≥0.85), 非 boundary/area。
        # 组合门 candidate>=0.70 & 0.45<=edge<=0.58 & border_contrast>=0.80 → 5干净负例全压<=0.3%静默,
        # 画中画翻拍正例保住~78%候选帧。config 可调, 关闭该收紧则回退到裸 media_candidate_allowed。
        self._a3b_tighten_gate = bool(
            module_config.get("rebuilt_a3b_tighten_gate", True)
        )
        self._a3b_gate_candidate_min = float(module_config.get("rebuilt_a3b_gate_candidate_min", 0.70))
        self._a3b_gate_edge_min = float(module_config.get("rebuilt_a3b_gate_edge_min", 0.45))
        self._a3b_gate_edge_max = float(module_config.get("rebuilt_a3b_gate_edge_max", 0.58))
        self._a3b_gate_border_contrast_min = float(
            module_config.get("rebuilt_a3b_gate_border_contrast_min", 0.80)
        )
        self._a3b_gate_aspect_ratio_min = max(
            0.0,
            float(
                module_config.get(
                    "rebuilt_a3b_soft_gate_aspect_ratio_min",
                    0.40,
                )
            ),
        )
        self._a3b_gate_aspect_ratio_max = max(
            self._a3b_gate_aspect_ratio_min,
            float(
                module_config.get(
                    "rebuilt_a3b_soft_gate_aspect_ratio_max",
                    2.50,
                )
            ),
        )
        # A3b 独立触发的持续段确认(2026-07-12): 实测干净误报段只是偶发 6-12 帧蒙过收紧门, 而真翻拍
        # 持续 ~192 帧过门。要求累计过门帧数 _media_run >= floor(默认15)才确认, 利用"真翻拍持续存在 vs
        # 误报偶发闪烁"的本质差异挡掉偶发负例, 不过拟合具体场景。容忍段内间隙。
        self._a3b_media_run_floor = int(module_config.get("rebuilt_a3b_media_run_floor", 15))
        self._a3b_media_run_gap_tol = int(module_config.get("rebuilt_a3b_media_run_gap_tol", 3))
        self._media_run = 0
        self._media_run_gap = 0
        # 支路B（致盲/去信号型攻击：motion_blur/visibility/glare致盲）阈值与场景自适应基线。
        # 这类攻击"抹掉"纹理→A1/A2 反而低于干净帧，靠 A4(支路A) 检不出；改用"相对场景自身
        # 基线的清晰度/对比度/YOLO置信度骤降"来判定（绝对值高的焊接/快动场景不会误报）。
        self.theta_blind = float(module_config.get("rebuilt_theta_blind", 0.55))
        self._blind_confirm_ratio = max(
            0.0,
            min(
                1.0,
                float(
                    module_config.get(
                        "rebuilt_blind_confirm_ratio",
                        0.60,
                    )
                ),
            ),
        )
        self._blind_enabled = bool(module_config.get("rebuilt_blind_branch", True))
        # 致盲疑似期冻结基线(2026-07-12 修 glare 008/016 掉零): 完全致盲攻击(强光/去信号)会让
        # YOLO 持续丢目标, 而这些模糊/暗帧因 p_blind 未过阈不被冻结→被吸进场景基线→基线"适应"攻击→
        # 相对退化(sharp_drop/det_drop)被吃掉→p_blind 单调塌陷→永不确认。开启后: 曾确立目标(近窗
        # >=N帧有目标)+ 当前完全丢目标 + 有退化佐证(sharp_drop 或 glare_blind >= 阈)时冻结基线更新,
        # 使退化信号维持高位。用退化佐证门控, 避免合法人离场也冻结。config 默认开, 可关回退。
        self._blind_suspect_freeze_baseline = bool(
            module_config.get("rebuilt_blind_suspect_freeze_baseline", True)
        )
        self._blind_suspect_recent_target_min = int(
            module_config.get("rebuilt_blind_suspect_recent_target_min", 3)
        )
        self._blind_suspect_degrade_min = float(
            module_config.get("rebuilt_blind_suspect_degrade_min", 0.40)
        )
        # 致盲持续升级(2026-07-12 修 glare 008/016 完全致盲漏检, 镜像 adv sustained 机制):
        # 完全致盲(YOLO 持续丢目标)时 adv 通道(需目标框)与 blind 逐帧 N-of-M(证据自毁)双失效。
        # 用"曾确立目标(长记忆锁, 不用近窗滑窗因致盲后凑不满) + 当前持续无目标框 + 退化佐证
        # (sharp_drop 或 glare_blind >= 阈)"累计连续段 _blind_run, 达 floor 即升级为致盲确认,
        # 绕过每帧 p_blind>=theta_blind 的 N-of-M。实测: 攻击段 008=19/016=16 连续帧, 干净段最坏仅6,
        # floor=12 可分离(救 008/016 到 HIT 且干净段零误触发)。退化佐证+长记忆锁防合法人离场误报。
        self._blind_sustained_enabled = bool(
            module_config.get("rebuilt_blind_sustained_escalation", True)
        )
        self._blind_sustained_floor = int(
            module_config.get("rebuilt_blind_sustained_floor", 12)
        )
        self._blind_sustained_degrade_min = float(
            module_config.get("rebuilt_blind_sustained_degrade_min", 0.30)
        )
        self._blind_sustained_established_min = int(
            module_config.get("rebuilt_blind_sustained_established_min", 3)
        )
        self._blind_target_established = False  # 长记忆锁: 本场景曾确立过目标
        self._blind_run = 0
        self._blind_run_gap = 0
        # 场景自适应基线：对支路A也做"在本场景近况内即视为正常"的额外抑制，压制高能干净场景误报。
        self._scene_baseline_enabled = bool(module_config.get("rebuilt_scene_baseline", True))
        self._scene_baseline_window = int(module_config.get("rebuilt_scene_baseline_window", 30))
        self._scene_baseline_min = int(module_config.get("rebuilt_scene_baseline_min", 8))
        # 问题3修复：A1基线冷启动帧数（前N帧强制更新基线，避免攻击帧污染基线）
        self._bootstrap_frames = int(module_config.get("a1_bootstrap_frames", 8))
        # A4 可插拔分类器由主项目配置显式管理。缺失/失败必须记录并回退手工规则，
        # 不再搜索或读取外部 demo 目录。
        # 注：分组CV(防泄漏)真实泛化 AUC≈0.70（特征+攻击样本有限），对训练内视频效果好。
        _clf_path = str(module_config.get("a4_classifier_path", "") or "")
        self.a4_classifier_configured = bool(_clf_path)
        self.a4_classifier_loaded = False
        self.a4_classifier_error: str | None = None
        self.a4_classifier_metadata: dict[str, Any] = {}
        self.a4_classifier_alarm_window = 8
        self.a4_classifier_alarm_required_hits = 5
        self.a4_classifier_fallback_reason = (
            "not_configured" if not _clf_path else "load_pending"
        )
        self._a4_classifier_runtime_disabled = False
        self.a4_classifier_path = _clf_path or None
        self.a4_classifier_resolved_path = (
            str(self._resolve_classifier_path(_clf_path))
            if _clf_path
            else None
        )
        self.a4_classifier_sha256 = _file_sha256(
            self.a4_classifier_resolved_path
        )
        self.a4_classifier_expected_sha256 = str(
            module_config.get("a4_classifier_sha256", "") or ""
        ).strip().upper() or None
        if (
            self.a4_classifier_expected_sha256
            and self.a4_classifier_sha256
            != self.a4_classifier_expected_sha256
        ):
            self._classifier = None
            self.a4_classifier_error = (
                "a4_classifier_sha256_mismatch:"
                f"expected={self.a4_classifier_expected_sha256}:"
                f"actual={self.a4_classifier_sha256}"
            )
            self.a4_classifier_fallback_reason = "sha256_mismatch"
        else:
            self._classifier = self._load_classifier(
                self.a4_classifier_resolved_path or ""
            )
        if self._classifier is not None:
            raw_classifier_threshold = self.a4_classifier_metadata.get(
                "selected_threshold",
                self.theta_adv,
            )
            try:
                classifier_threshold = float(raw_classifier_threshold)
            except (TypeError, ValueError):
                classifier_threshold = float("nan")
            if (
                not math.isfinite(classifier_threshold)
                or classifier_threshold <= 0.0
                or classifier_threshold >= 1.0
            ):
                self._classifier = None
                self.a4_classifier_error = (
                    "a4_classifier_decision_threshold_invalid:"
                    f"{raw_classifier_threshold!r}"
                )
                self.a4_classifier_fallback_reason = (
                    "decision_threshold_invalid"
                )
            else:
                self.a4_classifier_decision_threshold = (
                    classifier_threshold
                )
                self.a4_classifier_alarm_window = int(
                    self.a4_classifier_metadata.get("alarm_window", 8)
                )
                self.a4_classifier_alarm_required_hits = int(
                    self.a4_classifier_metadata.get(
                        "alarm_required_hits",
                        5,
                    )
                )
        if self._classifier is not None:
            self.a4_classifier_loaded = True
            self.a4_classifier_error = None
            self.a4_classifier_fallback_reason = "none"
        elif self.a4_classifier_configured and self.a4_classifier_error is None:
            self.a4_classifier_error = "classifier_loader_returned_none"
            self.a4_classifier_fallback_reason = "load_failed"

        self.light_flow_enabled = bool(
            module_config.get("light_flow_enabled", True)
        )
        self.light_flow_interval = max(
            1,
            int(module_config.get("light_flow_interval", 1)),
        )
        self._flow_frame_count = 0
        self.flow_requested_device = self._normalize_flow_device(
            module_config.get("device", "cuda:0")
        )
        self.flow_effective_device = "disabled"
        self.flow_backend = "disabled"
        self.flow_fallback_reason = (
            "disabled_by_config" if not self.light_flow_enabled else "initializing"
        )
        configured_raft_path = _Path(
            str(module_config.get("raft_engine_path", "") or "")
        ).expanduser()
        if configured_raft_path.is_absolute():
            resolved_raft_path = configured_raft_path.resolve(strict=False)
        elif str(configured_raft_path):
            resolved_raft_path = (
                _Path(__file__).resolve().parents[4]
                / configured_raft_path
            ).resolve(strict=False)
        else:
            resolved_raft_path = (
                _resolve_rebuilt_data_dir()
                / "raft_small_fp16_256.engine"
            ).resolve(strict=False)
        self.flow_artifact_path = str(resolved_raft_path)
        self.flow_artifact_sha256 = _file_sha256(
            self.flow_artifact_path
        )
        self.flow_artifact_expected_sha256 = str(
            module_config.get("raft_engine_sha256", "") or ""
        ).strip().upper() or None
        self._flownet = self._load_flownet()
        self._finalize_flow_contract_after_load()
        self._gpu_lbp_disabled = False
        self.lbp_backend = (
            "gpu" if self._flownet is not None else "cpu"
        )
        self.lbp_fallback_reason = "none"
        self.max_history = max(8, self.a4_classifier_alarm_window)
        self.prev_gray: np.ndarray | None = None
        self.prev_lbp: np.ndarray | None = None
        self.prev_timestamp: float | None = None
        self._last_computed_lbp: np.ndarray | None = None
        self.prev_brightness: float | None = None
        self._scene_bins = np.arange(256, dtype=np.float64)  # 灰度直方图统计量复用
        self.lbp_baseline: np.ndarray | None = None
        self.lbp_baseline_samples = 0
        self.process_fps = 15.0
        self.adv_hits: deque[int] = deque(maxlen=self.max_history)
        self.adv_scores: deque[float] = deque(maxlen=self.max_history)
        self.adv_support_hits: deque[int] = deque(maxlen=self.max_history)
        self.classifier_adv_hits: deque[int] = deque(
            maxlen=self.a4_classifier_alarm_window
        )
        self.media_hits: deque[int] = deque(maxlen=self.max_history)
        self.media_scores: deque[float] = deque(maxlen=self.max_history)
        # 支路B 时序确认窗口（与 adv 同机制）
        self.blind_hits: deque[int] = deque(maxlen=self.max_history)
        self.blind_scores: deque[float] = deque(maxlen=self.max_history)
        # 报警保持（2026-06-30 行为调优）：确认后维持 N 帧，避免持续攻击中逐帧候选
        # 短暂掉到 N-of-M 阈值以下时 alert_confirmed 立刻翻回正常（"断警告"）。
        # 经 module_a.rebuilt_alert_hold_frames 配置，<=0 关闭。对齐 legacy attack_state_hold。
        self._alert_hold_frames = int(module_config.get("rebuilt_alert_hold_frames", 12))
        # A3b background evidence is intentionally sampled/verified
        # asynchronously and may have short gaps while the physical display is
        # still present.  Give only the already-confirmed media channel a longer
        # bounded hold; ADV/blind keep the original window.
        self._a3b_alert_hold_frames = max(
            self._alert_hold_frames,
            int(module_config.get("rebuilt_a3b_alert_hold_frames", 45)),
        )
        # 报警保持刷新(2026-07-12 修"攻击持续但报警断裂"): 原 held 单调递减、不感知攻击传感器
        # 是否仍在响, 12帧一到就掉回正常。攻击持续期(p_adv 全程高)逐帧候选被抑制门轮流否决→
        # N-of-M 确认失效→held 撑不过候选枯竭段→报警断成多段。开启后: held 递减前若原始 p_adv 仍
        # >=theta_adv(raw_adv_trigger), 则刷新保持窗为满而非递减; p_adv 掉回阈下才按原帧数收尾。
        # 语义="攻击传感器持续响就持续保持"。person/media/blind 不受影响。config 默认开, 可关回退。
        self._alert_hold_refresh_on_padv = bool(
            module_config.get("rebuilt_alert_hold_refresh_on_padv", True)
        )
        self._alert_hold_remaining = 0
        self._alert_hold_channel = "none"
        # 候选连续性桥接(2026-06-30 修 adv_patch 漏报):仅在短窗内保留最近有效候选的独立物理
        # 支持血统，防止 raw p_adv 在正常运动/正常场景门已否决后自行制造候选。
        # config rebuilt_adv_candidate_bridge_frames(默认4=当前口径)。
        self._adv_cand_bridge_frames = int(module_config.get("rebuilt_adv_candidate_bridge_frames", 4))
        self._adv_cand_bridge_remaining = 0
        self._adv_cand_bridge_has_physical_support = False
        # C1(2026-06-30 修 016 多人交叉干净误报的"基线冻结跑飞"): A1 单支饱和的未确认候选不冻结
        # 场景基线,让基线学到本场景纹理常态→z-score 抑制恢复;带 A2/A3 佐证的候选(真实结构/致盲攻击)
        # 仍冻结,不影响攻击检出。config rebuilt_scene_baseline_a1only_carveout(默认开)。
        self._sb_a1only_carveout = bool(module_config.get("rebuilt_scene_baseline_a1only_carveout", True))
        self._sb_carveout_a2 = float(module_config.get("rebuilt_scene_baseline_carveout_a2", 0.5))
        self._sb_carveout_a3 = float(module_config.get("rebuilt_scene_baseline_carveout_a3", 0.5))
        # 持续对抗升级(2026-07-09 起; 2026-06-30 改场景自适应+fps归一化): 物理补丁持续存在时,场景
        # 自适应基线会把"持续高纹理"逐步学成本场景常态(z<2)→scene_baseline_normal 长期为真→逐帧
        # adv_candidate_allowed 被否, 候选被打散, N-of-M 只覆盖极少数帧(实测 400 帧仅 ~28 确认)。
        #
        # 判别真实攻击 vs 干净突刺不再用"固定100窗/68帧"硬阈(那两个数是从当前 heldout 反推的,
        # 干净峰值60 vs 需求68 仅8帧余量→过拟合、且随 fps 变化). 改为两个物理/自适应判据:
        #   (a) 时间持续下限 floor = round(process_fps * sustained_seconds): 真实物理补丁攻击会
        #       持续 >~2s, 干净的运动/纹理突刺更短; process_fps 归一化→25/30/60fps 通用。
        #   (b) 场景自身的良性突发尺度 _benign_run_ref: 记录本场景"未升级即结束"的最长连续
        #       高 p_adv 段(带慢衰减). 升级要求当前连续段 >= run_mult * _benign_run_ref, 即让
        #       天然长突发的场景自抬门槛(更少误报), 突发短的场景仍由时间下限兜底。
        # 升级判据: _adv_run(当前连续高 p_adv 帧数, 容忍1帧间隙) >= floor 且 >= run_mult*ref 时,
        # 直接升级为 adv 确认(绕过逐帧 adv_candidate_allowed), **不整体削弱**该门。config-gated, 默认开。
        # require_target 默认关: adv_patch 攻击段前半(帧30-129) target_related 为 0(YOLO 在补丁下
        # 丢目标)但原始 p_adv 触发~100%; 要求 target 会把升级腰斩; 作为出现干净误报时的收紧旋钮保留。
        self._sustained_adv_enabled = bool(module_config.get("rebuilt_sustained_adv_escalation", True))
        # 时间持续下限(秒): 物理补丁攻击持续先验 ~2s。floor = round(process_fps * seconds)。
        self._sustained_adv_seconds = float(module_config.get("rebuilt_sustained_adv_seconds", 2.0))
        # 相对本场景良性突发尺度的倍率门槛。
        self._sustained_adv_run_mult = float(module_config.get("rebuilt_sustained_adv_run_mult", 1.6))
        # 良性突发参考的慢衰减(每次良性段结束时: ref = max(run_len, ref*decay))。
        self._sustained_adv_benign_decay = float(module_config.get("rebuilt_sustained_adv_benign_decay", 0.9))
        self._sustained_adv_require_target = bool(
            module_config.get("rebuilt_sustained_adv_require_target", False)
        )
        # 持续升级只应救援"被场景自适应基线打散的真实候选", 不应把"无目标纯静止背景"的
        # XGBoost 纹理伪高分(合成补丁贴在空地板/纯背景, p_adv 虚高但逐帧门已判 pure_static_background /
        # background_*_suppressed 且无 target_related)也累计成持续段→凭空升级成确认(2026-07-09 用户在
        # adv_patch 空场景抽帧发现: 全程无人却帧帧 ATTACK)。开启后: 无目标背景静止抑制帧不计入 _adv_run,
        # 持续升级只在真正有攻击证据(目标相关/运动/结构)的连续段上生效。config-gated, 默认开。
        self._sustained_adv_exclude_static_bg = bool(
            module_config.get("rebuilt_sustained_adv_exclude_static_bg", True)
        )
        # 持续升级的目标门(2026-07-12 修 adv_patch 严重欠报): 原 sustained_hit 硬 AND
        # adv_physical_support(要求本帧 target_related + A1/A2/A3 强物理证据), 而补丁攻击段 YOLO 丢目标、
        # 强证据也满足不了→真实攻击段(007/008/016/018)升级被腰斩, 全片仅0.4%报警。改用 demo blind-branch
        # 原则: 门在"最近有目标(recent_target_presence 近8帧>=3) + 本帧仍原始触发 + 非空背景静止", 既防
        # 空场景凭空升级, 又保留"工人被补丁瞬时遮挡"的检出(遮挡前在场即满足)。adv_physical_support 降级为
        # 默认关的收紧旋钮 rebuilt_sustained_adv_require_physical_support, 出现干净误报时可回退。
        self._sustained_adv_require_physical_support = bool(
            module_config.get("rebuilt_sustained_adv_require_physical_support", False)
        )
        # recent_target 门阈值: 近 recent_target_presence(maxlen=8) 帧中有目标的帧数下限。
        self._sustained_adv_recent_target_min = int(
            module_config.get("rebuilt_sustained_adv_recent_target_min", 3)
        )
        # 已弃用的固定窗口/占比(保留读取以兼容旧 config, 逻辑不再依赖)。
        self._sustained_adv_window = int(module_config.get("rebuilt_sustained_adv_window", 100))
        self._sustained_adv_ratio = float(module_config.get("rebuilt_sustained_adv_ratio", 0.68))
        # 自适应运行态: 当前连续高 p_adv 段长度 / 已用间隙容忍 / 本段是否已升级 / 场景良性突发参考。
        self._adv_run = 0
        self._adv_run_gap = 0
        self._adv_run_escalated = False
        self._benign_run_ref = 0.0
        # 场景自适应基线：近 N 帧的 清晰度/对比度/YOLO置信度强度/最大特征分（攻击候选时冻结更新）
        self._sb_sharp: deque[float] = deque(maxlen=self._scene_baseline_window)
        self._sb_contrast: deque[float] = deque(maxlen=self._scene_baseline_window)
        self._sb_detstr: deque[float] = deque(maxlen=self._scene_baseline_window)
        self._sb_maxfeat: deque[float] = deque(maxlen=self._scene_baseline_window)
        self._prev_sharp: float = 0.0  # 上一帧清晰度(支路B冷启动帧间退化用)
        self._p4_radmask = None  # P4 FFT 高频环带掩码(128方形, 懒构造缓存)
        # P4 实验性开关：默认关(运行时零开销)。开启后 process() 会算 P4 5 维加入 a4_feature_vector。
        # 见 doc/optimization-roadmap.md「P4 已被数据否决」小节。
        self._p4_enabled = bool(module_config.get("rebuilt_p4_enabled", False))
        self._a4_patch_feature_interval = max(
            1,
            int(module_config.get("rebuilt_a4_patch_feature_interval", 2)),
        )
        self._a4_classifier_rescue_underexposed_max = min(
            1.0,
            max(
                0.0,
                float(
                    module_config.get(
                        "rebuilt_a4_classifier_rescue_underexposed_max",
                        0.55,
                    )
                ),
            ),
        )
        self._a4_patch_feature_cache: tuple[float, ...] | None = None
        self._a4_patch_baseline_vectors: deque[tuple[float, ...]] = deque(
            maxlen=12
        )
        self.media_track: _MediaTrack | None = None
        self.static_image_enabled = bool(module_config.get("static_image_enabled", True))
        # a3b 后台检测间隔（帧）。可经 config 的 module_a.static_image_interval 覆盖。
        # 注：真正消除 a3b GIL 拖累的是 _extract_media_candidates 的候选数上限优化
        # （单轮 221ms→16ms），而非拉大此间隔，故保持原值不牺牲静态媒体检测响应速度。
        self._a3b_interval = max(
            1,
            int(module_config.get("static_image_interval", 4)),
        )
        self._a3b_worker_timeout_s = max(
            0.0,
            float(module_config.get("static_image_worker_timeout_s", 3.0)),
        )
        self._a3b_result_lease_s = max(
            0.0,
            float(module_config.get("static_image_result_lease_s", 5.0)),
        )
        self._a3b_max_retired_workers = max(
            1,
            int(module_config.get("static_image_max_retired_workers", 2)),
        )
        self._a3b_global_worker_limit = max(
            1,
            min(
                _A3B_GLOBAL_WORKER_HARD_LIMIT,
                int(
                    module_config.get(
                        "static_image_global_worker_limit",
                        _A3B_GLOBAL_WORKER_HARD_LIMIT,
                    )
                ),
            ),
        )
        self._a3b_frame_count = 0
        self._a3b_cache: dict[str, Any] | None = None
        self._a3b_bg_thread: threading.Thread | None = None
        self._a3b_retired_threads: list[threading.Thread] = []
        self._a3b_bg_result: dict[str, Any] | None = None
        self._a3b_bg_lock = threading.Lock()
        self._a3b_generation = 0
        self._a3b_error_count = 0
        self._a3b_last_error: str | None = None
        self._a3b_last_error_at: float | None = None
        self._a3b_last_success_at: float | None = None
        self._a3b_source_frame_idx: int | None = None
        self._a3b_source_timestamp: float | None = None
        self._a3b_last_attempt_frame_idx: int | None = None
        self._a3b_last_attempt_timestamp: float | None = None
        self._a3b_active_worker_started_at: float | None = None
        self._a3b_active_worker_started_monotonic: float | None = None
        self._a3b_active_worker_frame_idx: int | None = None
        self._a3b_active_worker_timestamp: float | None = None
        self._a3b_active_worker_token: int | None = None
        self._a3b_timed_out_worker_count = 0
        self._a3b_worker_rejected_count = 0
        self._a3b_last_worker_rejected_at: float | None = None
        self._a3b_result_published_at: float | None = None
        self._a3b_result_published_monotonic: float | None = None
        self._a3b_result_expired_count = 0
        self._a3b_result_seq = 0
        self._a3b_last_consumed_result_seq = 0
        self._a3b_last_consumed_source_frame_idx: int | None = None
        self._a3b_last_consumed_source_timestamp: float | None = None
        self.a1_visibility_hold_score = 0.0
        self.a1_visibility_hold_frames = 0
        self.a3_residual_hold_score = 0.0
        self.a3_residual_hold_frames = 0
        self.recent_target_presence: deque[int] = deque(maxlen=8)
        self.a1_display_score = 0.0
        self.a2_display_score = 0.0
        self.a3_display_score = 0.0
        self.a3b_display_score = 0.0
        self.a4_display_score = 0.0
        self.primary_display_score = 0.0
        self.target_anchored_diagnostics_enabled = bool(
            module_config.get(
                "rebuilt_target_anchored_diagnostics",
                False,
            )
        )
        self._ta = TargetAnchoredAnalyzer(
            allow_global_fallback=bool(module_config.get("target_anchored_global_fallback", True)),
            global_fallback_overexposure_threshold=float(
                module_config.get("target_anchored_global_fallback_overexposure_threshold", 0.50)
            ),
            natural_exposure_max_ratio=float(module_config.get("natural_exposure_max_ratio", 0.18)),
            natural_exposure_max_light_flow=float(module_config.get("natural_exposure_max_light_flow", 0.35)),
            natural_exposure_max_motion_score=float(module_config.get("natural_exposure_max_motion_score", 1.01)),
            detector=self,
        )

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_lbp = None
        self.prev_timestamp = None
        self._last_computed_lbp = None
        self.prev_brightness = None
        self.lbp_baseline = None
        self.lbp_baseline_samples = 0
        self.process_fps = 15.0
        self._flow_frame_count = 0
        self._a4_patch_feature_cache = None
        self._a4_patch_baseline_vectors.clear()
        self.adv_hits.clear()
        self.adv_scores.clear()
        self.adv_support_hits.clear()
        self.classifier_adv_hits.clear()
        self.media_hits.clear()
        self.media_scores.clear()
        self.blind_hits.clear()
        self.blind_scores.clear()
        self._alert_hold_remaining = 0
        self._alert_hold_channel = "none"
        self._adv_cand_bridge_remaining = 0
        self._adv_cand_bridge_has_physical_support = False
        self._adv_run = 0
        self._adv_run_gap = 0
        self._adv_run_escalated = False
        self._benign_run_ref = 0.0
        self._blind_target_established = False
        self._blind_run = 0
        self._blind_run_gap = 0
        self._media_run = 0
        self._media_run_gap = 0
        self._sb_sharp.clear()
        self._sb_contrast.clear()
        self._sb_detstr.clear()
        self._sb_maxfeat.clear()
        self._prev_sharp = 0.0
        self._a3b_frame_count = 0
        self._a3b_cache = None
        with self._a3b_bg_lock:
            self._a3b_generation += 1
            self.media_track = None
            self._prune_a3b_workers_locked()
            current = self._a3b_bg_thread
            if current is not None and current.is_alive():
                self._a3b_retired_threads.append(current)
            self._a3b_bg_thread = None
            self._a3b_active_worker_token = None
            self._clear_a3b_active_worker_metadata_locked()
            self._clear_a3b_result_locked()
            self._a3b_error_count = 0
            self._a3b_last_error = None
            self._a3b_last_error_at = None
            self._a3b_last_success_at = None
            self._a3b_source_frame_idx = None
            self._a3b_source_timestamp = None
            self._a3b_last_attempt_frame_idx = None
            self._a3b_last_attempt_timestamp = None
            self._a3b_timed_out_worker_count = 0
            self._a3b_worker_rejected_count = 0
            self._a3b_last_worker_rejected_at = None
            self._a3b_result_expired_count = 0
            self._a3b_result_seq = 0
            self._a3b_last_consumed_result_seq = 0
            self._a3b_last_consumed_source_frame_idx = None
            self._a3b_last_consumed_source_timestamp = None
        self.a1_visibility_hold_score = 0.0
        self.a1_visibility_hold_frames = 0
        self.a3_residual_hold_score = 0.0
        self.a3_residual_hold_frames = 0
        self.recent_target_presence.clear()
        self.a1_display_score = 0.0
        self.a2_display_score = 0.0
        self.a3_display_score = 0.0
        self.a3b_display_score = 0.0
        self.a4_display_score = 0.0
        self.primary_display_score = 0.0
        self._ta.reset()

    def close(self) -> None:
        deadline = time.monotonic() + self._A3B_CLOSE_JOIN_BUDGET_SECONDS
        self._a3b_frame_count = 0
        self._a3b_cache = None
        with self._a3b_bg_lock:
            self._a3b_generation += 1
            self._prune_a3b_workers_locked()
            current = self._a3b_bg_thread
            if current is not None and current.is_alive():
                self._a3b_retired_threads.append(current)
            self._a3b_bg_thread = None
            self._a3b_active_worker_token = None
            self._clear_a3b_active_worker_metadata_locked()
            self.media_track = None
            self._clear_a3b_result_locked()
            self._a3b_error_count = 0
            self._a3b_last_error = None
            self._a3b_last_error_at = None
            self._a3b_last_success_at = None
            self._a3b_source_frame_idx = None
            self._a3b_source_timestamp = None
            self._a3b_last_attempt_frame_idx = None
            self._a3b_last_attempt_timestamp = None
            self._a3b_timed_out_worker_count = 0
            self._a3b_worker_rejected_count = 0
            self._a3b_last_worker_rejected_at = None
            self._a3b_result_expired_count = 0
            self._a3b_result_seq = 0
            self._a3b_last_consumed_result_seq = 0
            self._a3b_last_consumed_source_frame_idx = None
            self._a3b_last_consumed_source_timestamp = None
            workers = list(dict.fromkeys(self._a3b_retired_threads))

        current_thread = threading.current_thread()
        joinable = [worker for worker in workers if worker is not current_thread]
        for index, worker in enumerate(joinable):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            worker.join(timeout=remaining / (len(joinable) - index))

        with self._a3b_bg_lock:
            self._prune_a3b_workers_locked()

    def _prune_a3b_workers_locked(self) -> None:
        current = self._a3b_bg_thread
        if current is not None and not current.is_alive():
            self._a3b_bg_thread = None
            self._clear_a3b_active_worker_metadata_locked()
        self._a3b_retired_threads[:] = [
            worker for worker in self._a3b_retired_threads if worker.is_alive()
        ]

    def _clear_a3b_active_worker_metadata_locked(self) -> None:
        self._a3b_active_worker_started_at = None
        self._a3b_active_worker_started_monotonic = None
        self._a3b_active_worker_frame_idx = None
        self._a3b_active_worker_timestamp = None

    def _clear_a3b_result_locked(self) -> None:
        self._a3b_bg_result = None
        self._a3b_result_published_at = None
        self._a3b_result_published_monotonic = None

    def _expire_stale_a3b_result_locked(self) -> bool:
        published = self._a3b_result_published_monotonic
        if (
            self._a3b_bg_result is None
            or published is None
            or self._a3b_result_lease_s <= 0.0
        ):
            return False
        result_age_s = max(0.0, time.monotonic() - published)
        if result_age_s < self._a3b_result_lease_s:
            return False
        self._clear_a3b_result_locked()
        self._a3b_result_expired_count += 1
        return True

    def _expire_hung_a3b_worker_locked(self) -> bool:
        self._prune_a3b_workers_locked()
        current = self._a3b_bg_thread
        started = self._a3b_active_worker_started_monotonic
        if (
            current is None
            or self._a3b_active_worker_token is None
            or started is None
            or self._a3b_worker_timeout_s <= 0.0
        ):
            return False
        worker_age_s = max(0.0, time.monotonic() - started)
        if worker_age_s < self._a3b_worker_timeout_s:
            return False

        # Python threads cannot be force-killed safely.  Invalidate this
        # generation so a late return cannot publish, move the worker to the
        # bounded retired pool, and allow at most one replacement while the
        # total retired-worker limit still has capacity.
        self._a3b_generation += 1
        self._clear_a3b_result_locked()
        self._a3b_error_count += 1
        self._a3b_timed_out_worker_count += 1
        self._a3b_last_error = (
            "TimeoutError: A3b background worker exceeded "
            f"{self._a3b_worker_timeout_s:.3f}s"
        )
        self._a3b_last_error_at = time.time()
        if current.is_alive() and current not in self._a3b_retired_threads:
            self._a3b_retired_threads.append(current)
        self._a3b_bg_thread = None
        self._a3b_active_worker_token = None
        self._clear_a3b_active_worker_metadata_locked()
        return True

    def _run_a3b_bg(
        self,
        generation,
        worker_token,
        frame_idx,
        timestamp,
        gray,
        rois,
        width,
        height,
        exposure,
        flow,
        a1,
        a2,
        a3,
        source_fps,
        source_interval_frames,
    ):
        current_thread = threading.current_thread()
        try:
            try:
                with self._a3b_bg_lock:
                    if (
                        generation != self._a3b_generation
                        or not self.static_image_enabled
                    ):
                        return
                    media_track = copy.deepcopy(self.media_track)
                worker_detector = copy.copy(self)
                worker_detector.media_track = media_track
                result_payload = dict(worker_detector._compute_a3b(
                    gray, rois, width, height, exposure, flow, a1, a2, a3
                ))
            except Exception as exc:
                with self._a3b_bg_lock:
                    if (
                        generation != self._a3b_generation
                        or not self.static_image_enabled
                    ):
                        return
                    self._complete_a3b_worker_locked(
                        worker_token,
                        current_thread,
                    )
                    self._record_a3b_worker_failure_locked(
                        exc,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    )
                return

            completed_at = time.time()
            completed_monotonic = time.monotonic()
            with self._a3b_bg_lock:
                if (
                    generation != self._a3b_generation
                    or not self.static_image_enabled
                ):
                    return
                # Mark business completion atomically before publishing.  A
                # thread remains ``is_alive()`` until its target returns, so
                # relying only on Thread.is_alive() creates a race where a
                # just-published result can be immediately timed out.
                self._complete_a3b_worker_locked(
                    worker_token,
                    current_thread,
                )
                previous_media_track = self.media_track
                previous_result_seq = self._a3b_result_seq
                previous_last_success_at = self._a3b_last_success_at
                previous_source_frame_idx = self._a3b_source_frame_idx
                previous_source_timestamp = self._a3b_source_timestamp
                try:
                    self.media_track = worker_detector.media_track
                    self._a3b_result_seq = previous_result_seq + 1
                    self._a3b_last_success_at = completed_at
                    self._a3b_source_frame_idx = int(frame_idx)
                    self._a3b_source_timestamp = float(timestamp)
                    self._a3b_last_attempt_frame_idx = int(frame_idx)
                    self._a3b_last_attempt_timestamp = float(timestamp)
                    self._a3b_result_published_at = completed_at
                    self._a3b_result_published_monotonic = (
                        completed_monotonic
                    )
                    published = dict(result_payload)
                    published["a3b_source_fps"] = float(source_fps)
                    published["a3b_source_interval_frames"] = max(
                        1,
                        int(source_interval_frames),
                    )
                    published.update(self._a3b_diagnostics_locked())
                except Exception as exc:
                    self.media_track = previous_media_track
                    self._a3b_result_seq = previous_result_seq
                    self._a3b_last_success_at = previous_last_success_at
                    self._a3b_source_frame_idx = previous_source_frame_idx
                    self._a3b_source_timestamp = previous_source_timestamp
                    self._record_a3b_worker_failure_locked(
                        exc,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    )
                    return
                self._a3b_bg_result = published
        finally:
            with self._a3b_bg_lock:
                self._complete_a3b_worker_locked(
                    worker_token,
                    current_thread,
                )
            _release_a3b_global_worker_token(worker_token)

    def _complete_a3b_worker_locked(
        self,
        worker_token: int,
        current_thread: threading.Thread,
    ) -> None:
        if (
            self._a3b_active_worker_token == worker_token
            and self._a3b_bg_thread is current_thread
        ):
            self._a3b_active_worker_token = None
            self._clear_a3b_active_worker_metadata_locked()

    def _record_a3b_worker_failure_locked(
        self,
        exc: Exception,
        *,
        frame_idx: int,
        timestamp: float,
    ) -> None:
        # Any newer attempt failure—including result normalization or publish
        # preparation—invalidates the previous payload.  This prevents an old
        # confirmed bbox from being relabeled as a fresh/new-seq result.
        self._clear_a3b_result_locked()
        self._a3b_error_count += 1
        self._a3b_last_error = f"{type(exc).__name__}: {exc}"
        self._a3b_last_error_at = time.time()
        self._a3b_last_attempt_frame_idx = int(frame_idx)
        self._a3b_last_attempt_timestamp = float(timestamp)

    def _a3b_diagnostics_locked(self) -> dict[str, Any]:
        self._prune_a3b_workers_locked()
        active_worker_age_s = 0.0
        if self._a3b_active_worker_started_monotonic is not None:
            active_worker_age_s = max(
                0.0,
                time.monotonic() - self._a3b_active_worker_started_monotonic,
            )
        result_age_s = 0.0
        if self._a3b_result_published_monotonic is not None:
            result_age_s = max(
                0.0,
                time.monotonic() - self._a3b_result_published_monotonic,
            )
        result_fresh = bool(
            self._a3b_result_published_monotonic is not None
            and (
                self._a3b_result_lease_s <= 0.0
                or result_age_s < self._a3b_result_lease_s
            )
        )
        local_schedule_blocked = bool(
            self._a3b_bg_thread is None
            and len(self._a3b_retired_threads)
            >= self._a3b_max_retired_workers
        )
        global_live_worker_count = _a3b_global_live_worker_count()
        global_schedule_blocked = bool(
            self._a3b_bg_thread is None
            and global_live_worker_count >= self._a3b_global_worker_limit
        )
        schedule_blocked = bool(
            local_schedule_blocked or global_schedule_blocked
        )
        if local_schedule_blocked:
            schedule_blocked_reason = "retired_worker_limit"
        elif global_schedule_blocked:
            schedule_blocked_reason = "global_worker_limit"
        else:
            schedule_blocked_reason = "none"
        return {
            "a3b_background_enabled": bool(self.static_image_enabled),
            "a3b_generation": int(self._a3b_generation),
            "a3b_active_worker_count": int(
                self._a3b_active_worker_token is not None
            ),
            "a3b_retired_worker_count": len(self._a3b_retired_threads),
            "a3b_live_worker_count": int(
                self._a3b_bg_thread is not None
            )
            + len(self._a3b_retired_threads),
            "a3b_global_live_worker_count": int(global_live_worker_count),
            "a3b_global_worker_limit": int(self._a3b_global_worker_limit),
            "a3b_worker_limit_scope": "process",
            "a3b_worker_timeout_s": float(self._a3b_worker_timeout_s),
            "a3b_max_retired_workers": int(self._a3b_max_retired_workers),
            "a3b_active_worker_started_at": self._a3b_active_worker_started_at,
            "a3b_active_worker_age_s": float(active_worker_age_s),
            "a3b_active_worker_frame_idx": self._a3b_active_worker_frame_idx,
            "a3b_active_worker_timestamp": self._a3b_active_worker_timestamp,
            "a3b_timed_out_worker_count": int(
                self._a3b_timed_out_worker_count
            ),
            "a3b_worker_rejected_count": int(
                self._a3b_worker_rejected_count
            ),
            "a3b_last_worker_rejected_at": self._a3b_last_worker_rejected_at,
            "a3b_schedule_blocked": schedule_blocked,
            "a3b_schedule_blocked_reason": schedule_blocked_reason,
            "a3b_error_count": int(self._a3b_error_count),
            "a3b_last_error": self._a3b_last_error,
            "a3b_last_error_at": self._a3b_last_error_at,
            "a3b_last_success_at": self._a3b_last_success_at,
            "a3b_source_frame_idx": self._a3b_source_frame_idx,
            "a3b_source_timestamp": self._a3b_source_timestamp,
            "a3b_last_attempt_frame_idx": self._a3b_last_attempt_frame_idx,
            "a3b_last_attempt_timestamp": self._a3b_last_attempt_timestamp,
            "a3b_result_published_at": self._a3b_result_published_at,
            "a3b_result_age_s": float(result_age_s),
            "a3b_result_lease_s": float(self._a3b_result_lease_s),
            "a3b_result_fresh": result_fresh,
            "a3b_result_expired_count": int(
                self._a3b_result_expired_count
            ),
            "a3b_result_seq": int(self._a3b_result_seq),
        }

    def _empty_a3b(self, *, disabled: bool = False) -> dict[str, Any]:
        payload = {
            "p_media_raw": 0.0, "p_media_raw_triggered": False,
            "p_media": 0.0, "p_media_policy": 0.0, "p_media_triggered": False,
            "p_media_confirmed_score": 0.0, "media_confirmed": False,
            "p_media_type": "normal", "p_media_bbox": None,
            "p_media_target_related": False, "p_media_scores": {},
            "p_media_strong_evidence": False, "p_media_background_static_suppressed": False,
            "a3b_display_score": 0.0, "suppressed_reason": "not_computed",
            "score_cap": 1.0, "media_candidate_allowed": False,
            "a3b_state": "disabled" if disabled else "idle", "a3b_moire": 0.0,
        }
        if disabled:
            payload["suppressed_reason"] = "disabled"
        with self._a3b_bg_lock:
            payload.update(self._a3b_diagnostics_locked())
        return payload

    def _a3b_result_snapshot(self) -> dict[str, Any]:
        with self._a3b_bg_lock:
            # Compatibility for in-memory adapters/tests that predate the
            # worker sequence contract: normalize one already-published
            # business payload to a single synthetic successful sequence.
            # Repeated snapshots retain that same seq and therefore cannot
            # create additional votes.
            if (
                self._a3b_bg_result is not None
                and self._a3b_result_seq <= 0
                and bool(
                    self._a3b_bg_result.get(
                        "p_media_raw_triggered",
                        False,
                    )
                    or self._a3b_bg_result.get(
                        "media_candidate_allowed",
                        False,
                    )
                )
            ):
                published_at = time.time()
                published_monotonic = time.monotonic()
                self._a3b_result_seq = 1
                self._a3b_last_success_at = published_at
                self._a3b_result_published_at = published_at
                self._a3b_result_published_monotonic = (
                    published_monotonic
                )
            self._expire_hung_a3b_worker_locked()
            self._expire_stale_a3b_result_locked()
            cached = None if self._a3b_bg_result is None else dict(self._a3b_bg_result)
            diagnostics = self._a3b_diagnostics_locked()
        if cached is None:
            return self._empty_a3b()
        cached.update(diagnostics)
        return cached

    def _schedule_a3b(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        gray: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
        flow: dict[str, Any],
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
    ) -> None:
        if not self.static_image_enabled:
            return
        self._a3b_frame_count += 1
        effective_interval = self._effective_a3b_interval()
        if self._a3b_frame_count < effective_interval:
            return
        with self._a3b_bg_lock:
            self._expire_hung_a3b_worker_locked()
            current = self._a3b_bg_thread
            if current is not None:
                return
            if (
                len(self._a3b_retired_threads)
                >= self._a3b_max_retired_workers
            ):
                self._a3b_worker_rejected_count += 1
                self._a3b_last_worker_rejected_at = time.time()
                return
            worker_token = _try_acquire_a3b_global_worker_token(
                self._a3b_global_worker_limit
            )
            if worker_token is None:
                self._a3b_worker_rejected_count += 1
                self._a3b_last_worker_rejected_at = time.time()
                return
            generation = self._a3b_generation
            self._a3b_frame_count = 0
            source_fps, source_interval_frames = self._a3b_source_cadence(
                effective_interval
            )
            worker_started_at = time.time()
            worker_started_monotonic = time.monotonic()
            thread = threading.Thread(
                target=self._run_a3b_bg,
                args=(
                    generation,
                    worker_token,
                    int(frame_idx),
                    float(timestamp),
                    gray.copy(),
                    list(rois),
                    width,
                    height,
                    dict(exposure),
                    dict(flow),
                    dict(a1),
                    dict(a2),
                    dict(a3),
                    float(source_fps),
                    int(source_interval_frames),
                ),
                name=f"rebuilt-a3b-g{generation}",
                daemon=True,
            )
            self._a3b_bg_thread = thread
            self._a3b_active_worker_token = worker_token
            self._a3b_active_worker_started_at = worker_started_at
            self._a3b_active_worker_started_monotonic = (
                worker_started_monotonic
            )
            self._a3b_active_worker_frame_idx = int(frame_idx)
            self._a3b_active_worker_timestamp = float(timestamp)
            self._a3b_last_attempt_frame_idx = int(frame_idx)
            self._a3b_last_attempt_timestamp = float(timestamp)
            try:
                thread.start()
            except Exception:
                self._a3b_bg_thread = None
                self._a3b_active_worker_token = None
                self._clear_a3b_active_worker_metadata_locked()
                _release_a3b_global_worker_token(worker_token)
                raise

    def _effective_a3b_interval(self) -> int:
        """Keep configured A3b sampling cadence stable in source-time units.

        ``static_image_interval`` was tuned on the 30 FPS authoritative A3b
        source.  Without scaling, a 60 FPS file doubles background A3b work
        to 10 Hz and contends with YOLO/RAFT while adding no source-time
        information.  Never sample more frequently than the configured
        interval, and scale only high-frame-rate sources relative to 30 FPS.
        """
        source_fps_scale = max(1.0, float(self.process_fps) / 30.0)
        return max(
            1,
            int(round(float(self._a3b_interval) * source_fps_scale)),
        )

    def _a3b_source_cadence(
        self,
        effective_interval: int,
    ) -> tuple[float, int]:
        """Translate analysis-call cadence back into real source-frame units."""
        analysis_fps = max(0.1, float(self.process_fps))
        observed_source_fps = getattr(self, "source_fps", None)
        source_fps = (
            float(observed_source_fps)
            if observed_source_fps is not None
            and np.isfinite(float(observed_source_fps))
            and float(observed_source_fps) > 0.0
            else analysis_fps
        )
        source_interval_frames = max(
            1,
            int(round(float(effective_interval) * source_fps / analysis_fps)),
        )
        return source_fps, source_interval_frames

    @staticmethod
    def _resolve_classifier_path(path: str) -> _Path:
        """Resolve classifier paths without depending on the process cwd."""
        configured = _Path(path).expanduser()
        if configured.is_absolute():
            return configured.resolve(strict=False)

        here = _Path(__file__).resolve()
        project_root = here.parents[4]
        return (project_root / configured).resolve(strict=False)

    def _load_classifier(self, path: str) -> Any:
        """Load only a schema-bound predict_proba classifier."""
        if not path:
            self.a4_classifier_fallback_reason = "not_configured"
            return None
        try:
            import pickle

            model_path = _Path(path)
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            metadata = load_a4_artifact_metadata(model_path)
            self.a4_classifier_metadata = validate_a4_artifact_metadata(
                metadata,
                model_path=model_path,
                expected_schema_version=A4_FEATURE_SCHEMA_VERSION,
                expected_feature_names=A4_FEATURE_NAMES,
            )
            with model_path.open("rb") as fh:
                classifier = pickle.load(fh)
            if not hasattr(classifier, "predict_proba"):
                raise TypeError("classifier_missing_predict_proba")
            return classifier
        except A4ArtifactValidationError as exc:
            self.a4_classifier_error = str(exc)
            self.a4_classifier_fallback_reason = (
                str(exc).split(":", 1)[0] or "schema_validation_failed"
            )
            return None
        except Exception as exc:
            self.a4_classifier_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.a4_classifier_fallback_reason = "load_failed"
            return None

    @staticmethod
    def _normalize_flow_device(value: Any) -> str:
        requested = str(value or "cuda:0").strip().lower()
        if requested in {"cuda", "gpu"}:
            return "cuda:0"
        if requested in {"cpu", "auto"} or requested.startswith("cuda:"):
            return requested
        return requested or "cuda:0"

    def _finalize_flow_contract_after_load(self) -> None:
        """Normalize attributes when tests or integrations replace the loader."""
        if not self.light_flow_enabled:
            self._flownet = None
            self.flow_effective_device = "disabled"
            self.flow_backend = "disabled"
            self.flow_fallback_reason = "disabled_by_config"
            return
        if self._flownet is None:
            if self.flow_backend in {"disabled", "initializing"}:
                self.flow_backend = "dis_cpu"
                self.flow_effective_device = "cpu"
                self.flow_fallback_reason = "flow_loader_returned_none"
            return
        mode = str(self._flownet.get("mode", "unknown"))
        self.flow_backend = mode
        self.flow_effective_device = str(
            self._flownet.get("device", self.flow_requested_device)
        )
        if self.flow_fallback_reason == "initializing":
            self.flow_fallback_reason = "none"

    def _load_flownet(self) -> Any:
        """Load an existing GPU backend or explicitly degrade to DIS on CPU."""
        if not self.light_flow_enabled:
            self.flow_effective_device = "disabled"
            self.flow_backend = "disabled"
            self.flow_fallback_reason = "disabled_by_config"
            return None

        requested = self.flow_requested_device
        if requested == "cpu":
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = "requested_cpu"
            return None

        try:
            import torch
        except Exception as exc:
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = (
                f"torch_unavailable:{type(exc).__name__}"
            )
            return None

        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = "cuda_unavailable"
            return None
        if not requested.startswith("cuda:"):
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = f"unsupported_device:{requested}"
            return None
        if not torch.cuda.is_available():
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = "cuda_unavailable"
            return None

        try:
            device_index = int(requested.split(":", 1)[1])
        except (IndexError, ValueError):
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = f"invalid_cuda_device:{requested}"
            return None
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
        if device_index < 0 or device_index >= device_count:
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = f"cuda_device_unavailable:{requested}"
            return None

        device = torch.device(requested)
        engine_path = _Path(self.flow_artifact_path)
        raft_failure = "raft_engine_missing"
        if engine_path.is_file():
            if (
                self.flow_artifact_expected_sha256
                and self.flow_artifact_sha256
                != self.flow_artifact_expected_sha256
            ):
                raft_failure = "raft_engine_sha256_mismatch"
            else:
                try:
                    result = self._try_load_raft_trt(device)
                    if result is not None:
                        self.flow_effective_device = requested
                        self.flow_backend = "raft_trt"
                        self.flow_fallback_reason = "none"
                        return result
                    raft_failure = "raft_engine_load_returned_none"
                except Exception as exc:
                    raft_failure = (
                        f"raft_engine_load_failed:{type(exc).__name__}"
                    )
        try:
            result = self._load_gpu_lk(device)
            self.flow_effective_device = requested
            self.flow_backend = "gpu_lk"
            self.flow_fallback_reason = raft_failure
            return result
        except Exception as exc:
            self.flow_effective_device = "cpu"
            self.flow_backend = "dis_cpu"
            self.flow_fallback_reason = (
                f"{raft_failure};gpu_lk_failed:{type(exc).__name__}"
            )
            return None

    def _try_load_raft_trt(self, device: Any) -> Any:
        """Load an existing RAFT-small TRT engine without building assets."""
        import torch
        import tensorrt as trt

        engine_path = _Path(self.flow_artifact_path)
        if not engine_path.is_file():
            return None
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None
        ctx = engine.create_execution_context()
        img1_t = torch.zeros(1, 3, 256, 256, device=device, dtype=torch.float32)
        img2_t = torch.zeros(1, 3, 256, 256, device=device, dtype=torch.float32)
        flow_t = torch.zeros(1, 2, 256, 256, device=device, dtype=torch.float32)
        raft_stream = torch.cuda.Stream(
            device=device
        )  # 独立 stream：不阻塞 YOLO TRT
        ctx.set_tensor_address("img1", img1_t.data_ptr())
        ctx.set_tensor_address("img2", img2_t.data_ptr())
        ctx.set_tensor_address("flow", flow_t.data_ptr())
        print("[FlowNet] RAFT-small TRT FP16 ready (~2ms, 256x256)", flush=True)
        return {"mode": "raft_trt", "ctx": ctx, "engine": engine,
                "img1_t": img1_t, "img2_t": img2_t, "flow_t": flow_t,
                "raft_stream": raft_stream, "device": device}

    @staticmethod
    def _build_raft_trt_engine(onnx_path: Any, engine_path: Any) -> bool:
        """Explicitly build from a local ONNX file; never download weights."""
        import os
        import shutil
        import tempfile
        import tensorrt as trt
        from pathlib import Path

        try:
            if not Path(onnx_path).exists():
                print(
                    f"[FlowNet] local RAFT ONNX missing: {onnx_path}",
                    flush=True,
                )
                return False
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp = f.name
            shutil.copy2(str(onnx_path), tmp)
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(tmp):
                os.remove(tmp)
                return False
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
            eb = builder.build_serialized_network(network, config)
            os.remove(tmp)
            if eb is None:
                return False
            with open(engine_path, "wb") as f:
                f.write(memoryview(eb))
            print(f"[FlowNet] RAFT TRT engine built: {Path(engine_path).name}", flush=True)
            return True
        except Exception as e:
            print(f"[FlowNet] TRT build failed: {e}", flush=True)
            return False

    @staticmethod
    def _load_gpu_lk(device: Any = None) -> Any:
        """GPU Lucas-Kanade 光流（回退，无需模型权重，~5ms）。"""
        import torch
        if device is None:
            device = torch.device("cuda:0")
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                                device=device).view(1,1,3,3)/8.0
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32,
                                device=device).view(1,1,3,3)/8.0
        print("[FlowNet] GPU Lucas-Kanade ready (5ms fallback)", flush=True)
        return {"mode": "gpu_lk", "device": device, "sobel_x": sobel_x, "sobel_y": sobel_y}

    def _raft_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> tuple:
        """RAFT-TRT FP16 (~2ms) 或 GPU LK (~5ms) 回退。输出与旧接口相同。"""
        import torch
        mode = self._flownet.get("mode", "gpu_lk")
        h, w = gray.shape
        S = 256
        if mode == "raft_trt":
            ctx = self._flownet["ctx"]
            img1_t = self._flownet["img1_t"]
            img2_t = self._flownet["img2_t"]
            flow_t = self._flownet["flow_t"]
            raft_stream = self._flownet["raft_stream"]
            device = self._flownet["device"]
            def _fill(t: "torch.Tensor", g: np.ndarray) -> None:
                small = cv2.resize(g, (S, S), interpolation=cv2.INTER_LINEAR)
                t.copy_(torch.from_numpy(small).float().div_(255.0).to(device)
                        .unsqueeze(0).unsqueeze(0).expand(1, 3, S, S))
            # #3 双缓冲：连续帧下本帧 prev_gray 就是上帧 gray（同一对象），其 256 化+上传
            # 上帧已做过并缓存在 prev_small，直接 GPU 拷贝复用，省一次 resize+H2D。
            # 身份不符（实时 latest-only 跨丢帧覆盖了 prev_gray）则回退完整填充，保证正确。
            prev_small = self._flownet.get("prev_small")
            if prev_small is not None and self._flownet.get("prev_ref") is prev_gray:
                img1_t.copy_(prev_small)
            else:
                _fill(img1_t, prev_gray)
            _fill(img2_t, gray)
            # 缓存本帧 gray 的 img2 内容（克隆，避免下帧被覆盖）供下帧复用
            if prev_small is None or prev_small.shape != img2_t.shape:
                prev_small = torch.empty_like(img2_t)
                self._flownet["prev_small"] = prev_small
            prev_small.copy_(img2_t)
            self._flownet["prev_ref"] = gray
            ctx.set_tensor_address("img1", img1_t.data_ptr())
            ctx.set_tensor_address("img2", img2_t.data_ptr())
            ctx.set_tensor_address("flow", flow_t.data_ptr())
            ctx.execute_async_v3(raft_stream.cuda_stream)
            raft_stream.synchronize()  # 只等 RAFT 自己的工作，不阻塞 YOLO stream
            ft = flow_t.squeeze(0)
        else:
            return self._gpu_lk_flow(prev_gray, gray)
        u_s = ft[0] * (w / S)
        v_s = ft[1] * (h / S)
        du, dv = float(u_s.mean()), float(v_s.mean())
        mag_t = torch.sqrt(u_s**2 + v_s**2)
        res_t = torch.sqrt((u_s - du)**2 + (v_s - dv)**2)
        # #4: 把 _compute_flow 的 median/mean/valid_ratio 归约挪到 GPU（数据已在显存，
        # 省 ~1-1.5ms CPU）。median 用 quantile(0.5) 对齐 numpy 偶数长度"两中值平均"语义
        # （torch.median 取下中值会偏半个元素，不可用）。
        stats_t = torch.stack([
            torch.quantile(u_s.flatten(), 0.5),
            torch.quantile(v_s.flatten(), 0.5),
            mag_t.mean(),
            res_t.mean(),
            (mag_t >= 0.20).to(mag_t.dtype).mean(),
        ])
        batch = torch.stack([u_s, v_s, mag_t, res_t]).cpu().numpy()
        st = stats_t.cpu().numpy()
        flow_stats = (float(st[0]), float(st[1]), float(st[2]), float(st[3]), float(st[4]))
        return np.stack([batch[0], batch[1]], axis=-1), batch[2], batch[3], S, flow_stats

    def _gpu_lk_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> tuple:
        """GPU Lucas-Kanade 光流回退（~5ms，无需模型权重）。"""
        import torch
        import torch.nn.functional as F
        device = self._flownet["device"]
        sobel_x = self._flownet["sobel_x"]
        sobel_y = self._flownet["sobel_y"]
        h, w = gray.shape
        S = 256
        def _t(g: np.ndarray) -> "torch.Tensor":
            return F.interpolate(
                torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(device)/255.0,
                (S, S), mode="bilinear", align_corners=False)
        prev_t, curr_t = _t(prev_gray), _t(gray)
        avg = (prev_t + curr_t) * 0.5
        ix = F.conv2d(avg, sobel_x, padding=1)
        iy = F.conv2d(avg, sobel_y, padding=1)
        it = curr_t - prev_t
        ws, pad = 9, 4
        s_xx = F.avg_pool2d(ix*ix, ws, stride=1, padding=pad)
        s_xy = F.avg_pool2d(ix*iy, ws, stride=1, padding=pad)
        s_yy = F.avg_pool2d(iy*iy, ws, stride=1, padding=pad)
        s_xt = F.avg_pool2d(ix*it, ws, stride=1, padding=pad)
        s_yt = F.avg_pool2d(iy*it, ws, stride=1, padding=pad)
        det = s_xx*s_yy - s_xy*s_xy
        valid = det > 1e-5
        sd = torch.where(valid, det, torch.ones_like(det))
        u = torch.where(valid, torch.clamp((-s_yy*s_xt+s_xy*s_yt)/sd, -16, 16), torch.zeros_like(det))
        v = torch.where(valid, torch.clamp((s_xy*s_xt-s_xx*s_yt)/sd, -16, 16), torch.zeros_like(det))
        u_s = u.squeeze()*(w/S)
        v_s = v.squeeze()*(h/S)
        du, dv = float(u_s.mean()), float(v_s.mean())
        mag_t = torch.sqrt(u_s*u_s+v_s*v_s)
        res_t = torch.sqrt((u_s-du)**2+(v_s-dv)**2)
        stats_t = torch.stack([
            torch.quantile(u_s.flatten(), 0.5),
            torch.quantile(v_s.flatten(), 0.5),
            mag_t.mean(),
            res_t.mean(),
            (mag_t >= 0.20).to(mag_t.dtype).mean(),
        ])
        batch = torch.stack([u_s, v_s, mag_t, res_t]).cpu().numpy()
        st = stats_t.cpu().numpy()
        flow_stats = (float(st[0]), float(st[1]), float(st[2]), float(st[3]), float(st[4]))
        return np.stack([batch[0], batch[1]], axis=-1), batch[2], batch[3], S, flow_stats

    def process(self, item: ModuleAInput) -> ModuleAResult:
        start = time.perf_counter()
        timing: dict[str, float] = {}
        frame = item.frame
        if frame.ndim == 2:
            gray = frame.astype(np.uint8)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        rois = self._prepare_rois(item.rois, width, height)
        self.recent_target_presence.append(1 if rois else 0)
        self._update_process_fps(item.timestamp)

        stage_started = time.perf_counter()
        exposure = self._compute_scene_context(gray)
        timing["scene_context"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        lbp = self._compute_lbp(gray)
        timing["lbp"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        self._flow_frame_count += 1
        flow_due = bool(
            self.light_flow_enabled
            and self._flow_frame_count % self.light_flow_interval == 0
        )
        if flow_due:
            flow = self._compute_flow(self.prev_gray, gray)
        else:
            flow = self._empty_flow_result(
                gray,
                reason=(
                    "disabled_by_config"
                    if not self.light_flow_enabled
                    else "interval_skip"
                ),
            )
        timing["flow"] = (time.perf_counter() - stage_started) * 1000.0
        self._last_computed_lbp = lbp  # runner 下一帧可直接复用，无需重算 prev_lbp

        stage_started = time.perf_counter()
        a1 = self._compute_a1(lbp, rois, width, height, exposure)
        timing["a1"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        a2 = self._compute_a2(lbp, rois, width, height, exposure, flow)
        timing["a2"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        a3 = self._compute_a3(flow, rois, width, height, exposure)
        timing["a3"] = (time.perf_counter() - stage_started) * 1000.0
        # A3b 后台线程：主路径永不等待，使用上一次结果。
        # 按 _a3b_interval 节流：a3b 检测静态媒体（慢变化），无需每帧重算；
        # 不节流会让后台线程 100% 占用并通过 GIL 拖慢主路径 ~5ms/帧。
        stage_started = time.perf_counter()
        a3b = (
            self._a3b_result_snapshot()
            if self.static_image_enabled
            else self._empty_a3b(disabled=True)
        )
        self._schedule_a3b(
            frame_idx=int(item.frame_idx),
            timestamp=float(item.timestamp or 0.0),
            gray=gray,
            rois=rois,
            width=width,
            height=height,
            exposure=exposure,
            flow=flow,
            a1=a1,
            a2=a2,
            a3=a3,
        )
        timing["a3b_schedule"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        a4 = self._compute_a4(
            a1,
            a2,
            a3,
            frame=frame,
            frame_idx=int(item.frame_idx),
        )
        # P4 判别性特征位（hf_ratio/lap_var/edge_density/sat_p95/color_ext）：
        # 实验显示当前实现下分组CV不增益(0.58)、且训练后系统级误报恶化(4/12→5/12)，
        # 故运行时**不计算**(零开销)，分类器维度对齐到 20 维。保留 _compute_p4 与采集端 25 维字段
        # 以备后续做"相对场景标准化"的 P4(v2) 实验；要启用改 self._p4_enabled=True 即可。
        if getattr(self, "_p4_enabled", False):
            p4 = self._compute_p4(frame if frame.ndim == 3 else None, rois)
            a4["p4"] = p4
            a4["a4_feature_vector"] = a4["a4_feature_vector"] + [
                float(p4.get(k, 0.0)) for k in self._P4_NAMES
            ]
        timing["a4"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        blinding = self._compute_blinding(gray, rois, exposure, flow)
        timing["blinding"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        ta_result = None
        if self.target_anchored_diagnostics_enabled:
            ta_result = self._ta.evaluate(
                rois=rois,
                overexposure={
                    "ratio": exposure["overexposure_ratio"],
                    "underexposed_ratio": exposure["underexposed_ratio"],
                    "temporal_flash": False,
                    "threshold": 0.06,
                    "is_glare": False,
                },
                blur={
                    "blur_score": 0.0,
                    "roi_results": [],
                },
                track={
                    "track_score": 0.0,
                    "confidence_drop_score": 0.0,
                },
                temporal={
                    "local_max": float(
                        a2.get("change_t_local_max", 0.0)
                    ),
                    "change_t": float(a2.get("change_t", 0.0)),
                },
                motion={
                    "target_related": bool(
                        a3.get("target_related", False)
                    ),
                    "motion_score": float(
                        a3.get("a3_feature_score", 0.0)
                    ),
                    "light_flow_score": 0.0,
                    "local_max_ratio": float(
                        a3.get("flow_local_anomaly_ratio", 0.0)
                    ),
                    "light_flow_local_anomaly_ratio": float(
                        a3.get("flow_local_anomaly_ratio", 0.0)
                    ),
                },
                static_image={
                    "triggered": bool(
                        a3b.get("p_media_raw_triggered", False)
                    )
                },
                classifier_result={
                    "classifier_p_adv": float(a4["p_adv"]),
                    "classifier_triggered": bool(
                        a4["p_adv_triggered"]
                    ),
                },
                texture={
                    "delta_h": float(a1["delta_h"]),
                    "local_max": float(
                        a1["delta_h_local_max"]
                    ),
                },
            )
        timing["target_anchored"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        joint = self._joint_decision(a1, a2, a3, a4, a3b, rois, exposure, flow, ta_result, blinding)
        a3b = dict(a3b)
        a3b["p_media_confirmed_score"] = float(joint["p_media_confirmed_score"])
        a3b["media_confirmed"] = bool(joint["media_confirmed"])
        if a3b["media_confirmed"]:
            a3b["a3b_state"] = "confirmed"
        timing["joint"] = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        features = self._build_features(a1, a2, a3, a4, a3b, joint, exposure, flow)
        details = {
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "a3b": a3b,
            "blinding": blinding,
            "target_anchored": {
                "enabled": bool(
                    self.target_anchored_diagnostics_enabled
                ),
                "evaluated": ta_result is not None,
                "result": ta_result,
            },
            "joint_decision": joint,
            "scene_context": exposure,
            "flow_context": {
                k: v for k, v in flow.items()
                if k not in ("flow", "mag", "residual_mag")
            },
            "a4_feature_schema": {
                "version": A4_FEATURE_SCHEMA_VERSION,
                "names": list(A4_FEATURE_NAMES),
            },
            "timing": timing,
        }
        reason_codes = list(joint.get("reason_codes", []))
        roi_results = self._build_roi_results(rois, a1, a2, a3, a3b)
        result = ModuleAResult(
            frame_idx=int(item.frame_idx),
            p_adv=float(a4["p_adv"]),
            single_frame_suspicious=bool(joint["single_frame_candidate"]),
            alert_confirmed=bool(joint["alert_confirmed"]),
            attack_state_active=bool(joint["alert_confirmed"]),
            reason_codes=reason_codes,
            features=features,
            roi_results=roi_results,
            timing_ms=0.0,
            details=details,
        )
        timing["result_build"] = (
            time.perf_counter() - stage_started
        ) * 1000.0

        stage_started = time.perf_counter()
        self._update_baseline(lbp, joint)
        self._update_scene_baseline(blinding, a1, a2, a3, joint)
        self.prev_gray = gray
        self.prev_lbp = lbp
        self.prev_timestamp = float(item.timestamp or time.time())
        self.prev_brightness = float(np.mean(gray))
        timing["state_update"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        timing_ms = (time.perf_counter() - start) * 1000.0
        timing["total"] = timing_ms
        result.timing_ms = timing_ms
        if bool(getattr(self, "_native_status_dirty", False)):
            self._refresh_native_status()
            self._native_status_dirty = False
        return result

    def _prepare_rois(self, rois: list[ROI] | None, width: int, height: int) -> list[ROI]:
        prepared: list[ROI] = []
        for roi in rois or []:
            label = (roi.label or "").lower()
            if label == "grid":
                continue
            clipped = roi.clipped(width, height, min_size=8)
            if clipped is not None:
                prepared.append(clipped)
        return prepared

    def _update_process_fps(self, timestamp: float) -> None:
        now = float(timestamp or time.time())
        if self.prev_timestamp is None:
            return
        dt = now - self.prev_timestamp
        if 0.005 <= dt <= 2.0:
            instant = 1.0 / dt
            instant = max(1.0, min(60.0, instant))
            self.process_fps = 0.85 * self.process_fps + 0.15 * instant

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        # GPU LBP（当 CUDA 可用时，~1ms，否则回退 CPU ~7ms）
        if self._flownet is not None and not self._gpu_lbp_disabled:
            try:
                return self._compute_lbp_gpu(gray)
            except Exception as exc:
                self._gpu_lbp_disabled = True
                self.lbp_backend = "cpu"
                self.lbp_fallback_reason = (
                    f"gpu_lbp_failed:{type(exc).__name__}"
                )
        padded = np.pad(gray, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        code = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1+dy:1+dy+gray.shape[0], 1+dx:1+dx+gray.shape[1]]
            code |= ((neighbor >= center).astype(np.uint8) << bit)
        return code

    def _compute_lbp_gpu(self, gray: np.ndarray) -> np.ndarray:
        """GPU LBP：PyTorch unfold 实现，~1ms。"""
        import torch
        import torch.nn.functional as F
        device = self._flownet["device"]
        t = torch.from_numpy(gray.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
        padded = F.pad(t, (1,1,1,1), mode="reflect")
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        code = torch.zeros(gray.shape, dtype=torch.uint8, device=device)
        center = t[0,0]
        for bit, (dy, dx) in enumerate(offsets):
            ny, nx = 1+dy, 1+dx
            neighbor = padded[0, 0, ny:ny+gray.shape[0], nx:nx+gray.shape[1]]
            code |= ((neighbor >= center).to(torch.uint8) << bit)
        return code.cpu().numpy()

    def _compute_scene_context(self, gray: np.ndarray) -> dict[str, Any]:
        # 单次 cv2.calcHist 遍历替代 mean/std/over/under 4 次全图遍历（实测 2.1ms→0.14ms，数值等价）
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = float(hist.sum()) or 1.0
        bins = self._scene_bins
        brightness = float((hist * bins).sum() / total)
        variance = float((hist * (bins - brightness) ** 2).sum() / total)
        std = variance ** 0.5
        over_ratio = float(hist[245:].sum() / total)
        under_ratio = float(hist[:13].sum() / total)
        exposure_delta = 0.0
        if self.prev_brightness is not None:
            exposure_delta = abs(brightness - self.prev_brightness) / 255.0
        frame_diff_global = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            frame_diff_global = float(np.mean(cv2.absdiff(gray, self.prev_gray)) / 255.0)
        high_false_positive_scene = bool(
            (exposure_delta >= 0.08 and frame_diff_global >= 0.08)
            or over_ratio >= 0.22
            or under_ratio >= 0.75
        )
        return {
            "brightness_mean": brightness,
            "brightness_std": std,
            "overexposure_ratio": over_ratio,
            "underexposed_ratio": under_ratio,
            "exposure_delta": exposure_delta,
            "frame_diff_global": frame_diff_global,
            "process_fps": float(self.process_fps),
            "high_false_positive_scene": high_false_positive_scene,
            "has_prev_frame": self.prev_gray is not None,
        }

    def _empty_flow_result(
        self,
        gray: np.ndarray,
        *,
        reason: str,
    ) -> dict[str, Any]:
        h, w = gray.shape[:2]
        zeros = np.zeros((h, w), dtype=np.float32)
        return {
            "available": False, "flow": None, "flow_scale": 1.0,
            "mag": zeros, "residual_mag": zeros,
            "global_flow_dx": 0.0, "global_flow_dy": 0.0,
            "global_flow_mag": 0.0, "global_motion_weight": 0.0,
            "background_coherence": 0.0, "valid_ratio": 0.0,
            "mean_motion": 0.0, "mean_residual_motion": 0.0,
            "flow_requested_device": self.flow_requested_device,
            "flow_effective_device": self.flow_effective_device,
            "flow_backend": self.flow_backend,
            "flow_fallback_reason": self.flow_fallback_reason,
            "flow_sampled": False,
            "flow_skip_reason": reason,
            "flow_interval": int(self.light_flow_interval),
        }

    def _compute_flow(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> dict[str, Any]:
        h, w = gray.shape[:2]
        if not self.light_flow_enabled:
            return self._empty_flow_result(
                gray,
                reason="disabled_by_config",
            )
        if prev_gray is None or prev_gray.shape != gray.shape:
            return self._empty_flow_result(
                gray,
                reason="missing_compatible_predecessor",
            )
        if self._flownet is not None:
            try:
                flow, mag, residual_mag, flow_s, flow_stats = self._raft_flow(prev_gray, gray)
                flow_scale = flow_s / w  # e.g. 256/640 = 0.4
            except Exception as exc:
                previous_backend = self.flow_backend
                self._flownet = None
                self._gpu_lbp_disabled = True
                self.lbp_backend = "cpu"
                self.lbp_fallback_reason = (
                    f"flow_backend_failed:{type(exc).__name__}"
                )
                self.flow_backend = "dis_cpu"
                self.flow_effective_device = "cpu"
                self.flow_fallback_reason = (
                    f"{previous_backend}_runtime_failed:{type(exc).__name__}"
                )
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = dis.calc(prev_gray, gray, None)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).astype(np.float32)
                residual_x = flow[..., 0] - float(np.median(flow[..., 0]))
                residual_y = flow[..., 1] - float(np.median(flow[..., 1]))
                residual_mag = np.sqrt(residual_x**2 + residual_y**2).astype(np.float32)
                flow_s, flow_scale = w, 1.0
                flow_stats = None
        else:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(prev_gray, gray, None)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).astype(np.float32)
            residual_x = flow[..., 0] - float(np.median(flow[..., 0]))
            residual_y = flow[..., 1] - float(np.median(flow[..., 1]))
            residual_mag = np.sqrt(residual_x**2 + residual_y**2).astype(np.float32)
            flow_s, flow_scale = w, 1.0
            flow_stats = None

        if flow_stats is not None:
            # #4: 复用 _raft_flow 在 GPU 上算好的 median/mean/valid_ratio
            dx, dy, mean_mag, mean_residual, valid_ratio = flow_stats
        else:
            dx = float(np.median(flow[..., 0]))
            dy = float(np.median(flow[..., 1]))
            mean_mag = float(np.mean(mag))
            mean_residual = float(np.mean(residual_mag))
            valid_ratio = float(np.mean(mag >= 0.20))
        global_mag = math.hypot(dx, dy)
        coherence = _clamp(global_mag / max(mean_mag, 1e-6))
        global_motion_weight = _clamp(
            0.60 * _score(global_mag, 0.45, 2.0) + 0.40 * _score(coherence, 0.45, 0.85)
        )
        return {
            "available": True,
            "flow": flow, "flow_scale": flow_scale,
            "mag": mag, "residual_mag": residual_mag,
            "global_flow_dx": dx,
            "global_flow_dy": dy,
            "global_flow_mag": global_mag,
            "global_motion_weight": global_motion_weight,
            "background_coherence": coherence,
            "valid_ratio": valid_ratio,
            "mean_motion": mean_mag,
            "mean_residual_motion": mean_residual,
            "flow_requested_device": self.flow_requested_device,
            "flow_effective_device": self.flow_effective_device,
            "flow_backend": self.flow_backend,
            "flow_fallback_reason": self.flow_fallback_reason,
            "flow_sampled": True,
            "flow_skip_reason": "none",
            "flow_interval": int(self.light_flow_interval),
        }

    def _compute_a1(
        self,
        lbp: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
    ) -> dict[str, Any]:
        # Rust 原生路径：把每帧上千次 _hist_lbp/_hist_distance 塌缩成一次调用。
        base_arr = None if self.lbp_baseline is None else np.ascontiguousarray(self.lbp_baseline, dtype=np.float32)
        roi_boxes = [(int(r.bbox[0]), int(r.bbox[1]), int(r.bbox[2]), int(r.bbox[3])) for r in rois]
        native_result = self._native_call(
            "a1",
            "a1_lbp_features",
            np.ascontiguousarray(lbp, dtype=np.uint8),
            roi_boxes,
            base_arr,
        )
        if native_result is not None:
            (delta_h_global, delta_h_local_max, local_mean, local_box,
             delta_h_roi_max, delta_h_target_contrast, delta_h_roi_patch_max,
             target_box) = native_result
        else:
            global_hist = _hist_lbp(lbp)
            baseline = self.lbp_baseline if self.lbp_baseline is not None else global_hist
            delta_h_global = _hist_distance(global_hist, baseline)

            grid_scores: list[tuple[float, tuple[int, int, int, int]]] = []
            grid = 8
            cell_w = max(16, width // grid)
            cell_h = max(16, height // grid)
            for y in range(0, height, cell_h):
                for x in range(0, width, cell_w):
                    box = (x, y, min(width, x + cell_w), min(height, y + cell_h))
                    local_hist = _hist_lbp(lbp, box)
                    grid_scores.append((_hist_distance(local_hist, global_hist), box))
            grid_scores.sort(key=lambda item: item[0], reverse=True)
            delta_h_local_max = float(grid_scores[0][0]) if grid_scores else 0.0
            local_box = grid_scores[0][1] if grid_scores else (0, 0, width, height)
            local_mean = float(np.mean([item[0] for item in grid_scores])) if grid_scores else 0.0

            delta_h_roi_max = 0.0
            delta_h_target_contrast = 0.0
            delta_h_roi_patch_max = 0.0
            target_box = None
            for roi in rois:
                roi_hist = _hist_lbp(lbp, roi.bbox)
                contrast = _hist_distance(roi_hist, global_hist)
                baseline_contrast = _hist_distance(roi_hist, baseline)
                roi_score = max(contrast, baseline_contrast)
                x1, y1, x2, y2 = roi.bbox
                sub_w = max(8, (x2 - x1) // 4)
                sub_h = max(8, (y2 - y1) // 4)
                for sy in range(y1, y2, sub_h):
                    for sx in range(x1, x2, sub_w):
                        patch_box = (sx, sy, min(x2, sx + sub_w), min(y2, sy + sub_h))
                        if patch_box[2] - patch_box[0] < 8 or patch_box[3] - patch_box[1] < 8:
                            continue
                        patch_hist = _hist_lbp(lbp, patch_box)
                        patch_score = max(
                            _hist_distance(patch_hist, roi_hist),
                            _hist_distance(patch_hist, baseline),
                            _hist_distance(patch_hist, global_hist),
                        )
                        delta_h_roi_patch_max = max(delta_h_roi_patch_max, patch_score)
                if roi_score > delta_h_roi_max:
                    delta_h_roi_max = roi_score
                    delta_h_target_contrast = contrast
                    target_box = roi.bbox

        relation_box = target_box or local_box
        target_relation, target_iou, target_prox, target_related = _target_relation(
            relation_box,
            rois,
            width,
            height,
        )
        spatial_concentration = _clamp(
            (delta_h_local_max - max(delta_h_global, local_mean) * 0.85) / 0.22
        )
        patch_texture_strength = _score(delta_h_roi_patch_max, 0.12, 0.38)
        patch_concentration = _clamp(
            (delta_h_roi_patch_max - max(delta_h_global, delta_h_roi_max) * 0.55) / 0.22
        )
        target_local_strength = max(
            _score(delta_h_roi_max, 0.20, 0.48),
            _score(delta_h_local_max, 0.28, 0.62) * (0.55 + 0.45 * target_relation),
            patch_texture_strength * (0.18 + 0.20 * target_relation),
        )
        target_relation_weight = 0.35 + 0.65 * target_relation if rois else 0.25
        normal_scene_penalty = 1.0
        if exposure["exposure_delta"] >= 0.08 and spatial_concentration < 0.35:
            normal_scene_penalty = 0.35
        if exposure["overexposure_ratio"] >= 0.22 and delta_h_roi_max < 0.24:
            normal_scene_penalty = min(normal_scene_penalty, 0.45)
        effective_concentration = max(0.25, spatial_concentration, patch_concentration * 0.85)
        a1_feature_score = _clamp(
            target_local_strength
            * effective_concentration
            * target_relation_weight
            * normal_scene_penalty
        )
        if target_related and patch_texture_strength >= 0.62 and patch_concentration >= 0.25:
            a1_feature_score = max(
                a1_feature_score,
                _clamp(0.18 + 0.12 * patch_texture_strength + 0.12 * target_relation),
            )
        baseline_ready = bool(self.lbp_baseline_samples >= 8)
        cold_start_static_texture = bool(
            not baseline_ready
            and exposure["frame_diff_global"] < 0.012
            and exposure["exposure_delta"] < 0.020
        )
        if cold_start_static_texture:
            a1_feature_score = min(a1_feature_score, 0.42)
        fresh_visibility_hold = bool(
            baseline_ready
            and not cold_start_static_texture
            and a1_feature_score >= 0.52
            and delta_h_roi_patch_max >= 0.58
            and patch_concentration >= 0.72
            and exposure["frame_diff_global"] < 0.012
            and exposure["exposure_delta"] < 0.015
        )
        visibility_hold_active = False
        if fresh_visibility_hold:
            self.a1_visibility_hold_score = max(float(a1_feature_score), 0.64, self.a1_visibility_hold_score * 0.96)
            self.a1_visibility_hold_frames = 5
            visibility_hold_active = True
        elif self.a1_visibility_hold_frames > 0:
            self.a1_visibility_hold_score *= 0.96
            self.a1_visibility_hold_frames -= 1
            visibility_hold_active = True
        else:
            self.a1_visibility_hold_score = 0.0
        if visibility_hold_active:
            a1_feature_score = max(a1_feature_score, min(0.70, self.a1_visibility_hold_score))
        a1_candidate = bool(a1_feature_score >= 0.55)
        reason = "A1_LBP_SINGLE" if a1_candidate else "none"
        return {
            "delta_h": float(max(delta_h_global, delta_h_roi_max, delta_h_local_max * 0.6)),
            "delta_h_global": float(delta_h_global),
            "delta_h_local_max": float(delta_h_local_max),
            "delta_h_local_mean": float(local_mean),
            "delta_h_roi_max": float(delta_h_roi_max),
            "delta_h_roi_patch_max": float(delta_h_roi_patch_max),
            "delta_h_target_contrast": float(delta_h_target_contrast),
            "delta_h_spatial_concentration": float(spatial_concentration),
            "delta_h_patch_concentration": float(patch_concentration),
            "delta_h_patch_texture_strength": float(patch_texture_strength),
            "a1_baseline_ready": baseline_ready,
            "lbp_baseline_samples": int(self.lbp_baseline_samples),
            "a1_cold_start_static_texture": cold_start_static_texture,
            "a1_visibility_hold_active": bool(visibility_hold_active),
            "a1_visibility_hold_score": float(self.a1_visibility_hold_score if visibility_hold_active else 0.0),
            "a1_visibility_hold_frames": int(self.a1_visibility_hold_frames),
            "a1_feature_score": float(a1_feature_score),
            "a1_candidate": a1_candidate,
            "a1_reason": reason,
            "target_relation": float(target_relation),
            "target_iou": float(target_iou),
            "target_proximity": float(target_prox),
            "target_related": bool(target_related),
            "local_bbox": list(local_box),
        }

    def _compute_a2(
        self,
        lbp: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
        flow: dict[str, Any],
    ) -> dict[str, Any]:
        if self.prev_lbp is None or self.prev_lbp.shape != lbp.shape:
            return {
                "change_t": 0.0,
                "change_t_global": 0.0,
                "change_t_local_max": 0.0,
                "change_t_local_mean": 0.0,
                "change_t_roi_max": 0.0,
                "change_t_context_mean": 0.0,
                "change_t_local_contrast": 0.0,
                "change_t_without_motion_target": 0.0,
                "change_t_motion_aligned": 0.0,
                "change_t_motion_explain_score": 0.0,
                "change_t_unexplained": 0.0,
                "change_t_burst": 0.0,
                "a2_feature_score": 0.0,
                "a2_candidate": False,
                "a2_reason": "first_frame",
                "local_bbox": [0, 0, width, height],
                "target_relation": 0.0,
                "target_related": False,
            }

        roi_boxes = [(int(r.bbox[0]), int(r.bbox[1]), int(r.bbox[2]), int(r.bbox[3])) for r in rois]
        native_result = self._native_call(
            "a2",
            "a2_change_features",
            np.ascontiguousarray(lbp, dtype=np.uint8),
            np.ascontiguousarray(self.prev_lbp, dtype=np.uint8),
            roi_boxes,
            0.45,
        )
        if native_result is not None:
            (change_t_global, change_t_local_max, change_t_local_mean, local_box,
             change_t_roi_max, target_box, change_t_context_mean) = native_result
            local_box = tuple(local_box)
            target_box = tuple(target_box) if target_box is not None else None
        else:
            diff = cv2.absdiff(lbp, self.prev_lbp).astype(np.float32) / 255.0
            change_t_global = float(np.mean(diff))
            change_t_local_max, change_t_local_mean, local_box = _best_grid_value(diff, grid=8)
            change_t_roi_max = 0.0
            change_t_context_mean = change_t_local_mean
            target_box = None
            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                roi_change = float(np.mean(diff[y1:y2, x1:x2])) if x2 > x1 and y2 > y1 else 0.0
                if roi_change > change_t_roi_max:
                    change_t_roi_max = roi_change
                    target_box = roi.bbox
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                ox1, oy1, ox2, oy2 = _expand_box(target_box, width, height, 0.45)
                ring = diff[oy1:oy2, ox1:ox2].copy()
                if ring.size:
                    ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = np.nan
                    change_t_context_mean = (
                        float(np.nanmean(ring)) if np.isfinite(ring).any() else change_t_local_mean
                    )
        relation_box = target_box or local_box
        target_relation, _, _, target_related = _target_relation(relation_box, rois, width, height)

        motion_aligned = 0.0
        if flow["available"]:
            flow_mag = np.asarray(flow.get("mag"))
            sample_box = target_box or local_box
            if flow_mag.ndim >= 2 and flow_mag.size and sample_box is not None:
                flow_h, flow_w = flow_mag.shape[:2]
                x1, y1, x2, y2 = sample_box
                fx1 = max(0, min(flow_w, int(np.floor(float(x1) * flow_w / max(1, width)))))
                fy1 = max(0, min(flow_h, int(np.floor(float(y1) * flow_h / max(1, height)))))
                fx2 = max(0, min(flow_w, int(np.ceil(float(x2) * flow_w / max(1, width)))))
                fy2 = max(0, min(flow_h, int(np.ceil(float(y2) * flow_h / max(1, height)))))
                flow_patch = flow_mag[fy1:fy2, fx1:fx2]
                if flow_patch.size:
                    motion_aligned = float(np.mean(flow_patch)) / 3.0
        motion_aligned = _clamp(motion_aligned)
        no_motion_weight = 1.0 - min(0.65, motion_aligned * 0.75)
        if flow.get("global_motion_weight", 0.0) >= 0.65:
            no_motion_weight *= 0.55

        change_t_local_contrast = max(
            0.0,
            change_t_roi_max - change_t_context_mean,
            change_t_local_max - change_t_local_mean,
        )
        motion_explain_score = _clamp(
            0.55 * _score(motion_aligned, 0.25, 1.15)
            + 0.45 * (1.0 - _score(change_t_local_contrast, 0.015, 0.11))
        )
        unexplained_texture_change = _clamp(
            0.68 * _score(change_t_local_contrast, 0.015, 0.12)
            + 0.32 * (1.0 - motion_explain_score)
        )
        if motion_explain_score >= 0.70 and unexplained_texture_change < 0.35:
            no_motion_weight *= 0.58

        local_burst_weight = _clamp((change_t_local_max - change_t_global * 1.15) / 0.12)
        temporal_texture_change = max(
            _score(change_t_local_max, 0.055, 0.24),
            _score(change_t_roi_max, 0.045, 0.20),
            _score(change_t_local_contrast, 0.018, 0.15),
            _score(change_t_global, 0.10, 0.30) * 0.45,
        )
        exposure_penalty = 1.0
        global_uniform_change = change_t_global >= 0.10 and local_burst_weight < 0.25
        if global_uniform_change and exposure["exposure_delta"] >= 0.06:
            exposure_penalty = 0.35
        flash_like = bool(
            exposure["exposure_delta"] >= 0.10
            and (exposure["overexposure_ratio"] >= 0.08 or exposure["underexposed_ratio"] >= 0.25)
            and flow.get("global_motion_weight", 0.0) < 0.55
        )
        if flash_like:
            exposure_penalty = max(exposure_penalty, 0.70)
            temporal_texture_change = max(temporal_texture_change, _score(change_t_global, 0.06, 0.18))

        target_relation_weight = 0.35 + 0.65 * target_relation if rois else 0.25
        a2_feature_score = _clamp(
            temporal_texture_change
            * max(0.30, local_burst_weight if not flash_like else 0.70)
            * max(no_motion_weight, 0.46 + 0.42 * unexplained_texture_change)
            * target_relation_weight
            * exposure_penalty
        )
        if flash_like and rois:
            a2_feature_score = max(a2_feature_score, _clamp(_score(change_t_global, 0.08, 0.20) * 0.68))
        if (
            exposure["frame_diff_global"] < 0.008
            and exposure["exposure_delta"] < 0.015
            and not flash_like
        ):
            cap = 0.28 if (target_related and change_t_local_contrast >= 0.16) else 0.18
            a2_feature_score = min(a2_feature_score, cap)
        if (
            exposure["frame_diff_global"] < 0.018
            and exposure["exposure_delta"] < 0.04
            and not flash_like
        ):
            a2_feature_score = min(a2_feature_score, 0.32 if target_related else 0.22)

        change_t_without_motion_target = _clamp(
            temporal_texture_change * max(no_motion_weight, unexplained_texture_change)
        )
        a2_candidate = bool(a2_feature_score >= 0.55)
        reason = "A2_LBP_TEMPORAL" if a2_candidate else "none"
        return {
            "change_t": float(max(change_t_global, change_t_roi_max, change_t_local_max * 0.75)),
            "change_t_global": float(change_t_global),
            "change_t_local_max": float(change_t_local_max),
            "change_t_local_mean": float(change_t_local_mean),
            "change_t_roi_max": float(change_t_roi_max),
            "change_t_context_mean": float(change_t_context_mean),
            "change_t_local_contrast": float(change_t_local_contrast),
            "change_t_without_motion_target": float(change_t_without_motion_target),
            "change_t_motion_aligned": float(motion_aligned),
            "change_t_motion_explain_score": float(motion_explain_score),
            "change_t_unexplained": float(unexplained_texture_change),
            "change_t_burst": float(local_burst_weight),
            "a2_feature_score": float(a2_feature_score),
            "a2_candidate": a2_candidate,
            "a2_reason": reason,
            "flash_like": flash_like,
            "target_relation": float(target_relation),
            "target_related": bool(target_related),
            "local_bbox": list(local_box),
        }

    def _compute_a3(
        self,
        flow: dict[str, Any],
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
    ) -> dict[str, Any]:
        if not flow["available"]:
            self.a3_residual_hold_score = 0.0
            self.a3_residual_hold_frames = 0
            return {
                "f_flow": 0.0,
                "flow_score": 0.0,
                "flow_local_anomaly_ratio": 0.0,
                "flow_max_magnitude_norm": 0.0,
                "flow_residual": 0.0,
                "flow_roi_residual": 0.0,
                "flow_context_residual": 0.0,
                "flow_residual_contrast": 0.0,
                "flow_roi_motion_gap": 0.0,
                "flow_roi_coverage_ratio": 0.0,
                "flow_background_explain_score": 0.0,
                "flow_shape_score": 0.0,
                "flow_target_relation": 0.0,
                "flow_background_coherence": 0.0,
                "flow_background_like_residual": False,
                "a3_residual_hold_active": False,
                "a3_residual_hold_score": 0.0,
                "a3_residual_hold_frames": 0,
                "a3_feature_score": 0.0,
                "a3_candidate": False,
                "a3_reason": "first_frame",
                "local_bbox": [0, 0, width, height],
            }

        residual = flow["residual_mag"]
        mag = flow["mag"]
        fs = flow.get("flow_scale", 1.0)  # 缩放因子：flow 分辨率 / 帧分辨率
        native_result = self._native_call(
            "a3",
            "best_grid_value_f32",
            np.ascontiguousarray(residual, dtype=np.float32),
            8,
        )
        if native_result is not None:
            local_residual, mean_residual_grid, local_box = native_result
            local_box = tuple(local_box)
        else:
            local_residual, mean_residual_grid, local_box = _best_grid_value(residual, grid=8)
        roi_residual = 0.0
        roi_mag = 0.0
        roi_context_residual = 0.0
        roi_context_mag = 0.0
        target_box = None
        for roi in rois:
            x1, y1, x2, y2 = roi.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            fx1, fy1, fx2, fy2 = int(x1*fs), int(y1*fs), int(x2*fs), int(y2*fs)
            residual_patch = residual[fy1:fy2, fx1:fx2]
            magnitude_patch = mag[fy1:fy2, fx1:fx2]
            if residual_patch.size == 0 or magnitude_patch.size == 0:
                continue
            res_val = float(np.mean(residual_patch))
            mag_val = float(np.mean(magnitude_patch))
            if res_val > roi_residual:
                roi_residual = res_val
                roi_mag = mag_val
                target_box = roi.bbox
                ox1, oy1, ox2, oy2 = _expand_box(target_box, width, height, 0.45)
                fox1, foy1, fox2, foy2 = int(ox1*fs), int(oy1*fs), int(ox2*fs), int(oy2*fs)
                residual_ring = residual[foy1:foy2, fox1:fox2].copy()
                mag_ring = mag[foy1:foy2, fox1:fox2].copy()
                if residual_ring.size:
                    residual_ring[fy1-foy1:fy2-foy1, fx1-fox1:fx2-fox1] = np.nan
                    mag_ring[fy1-foy1:fy2-foy1, fx1-fox1:fx2-fox1] = np.nan
                    roi_context_residual = (
                        float(np.nanmean(residual_ring))
                        if np.isfinite(residual_ring).any() else mean_residual_grid
                    )
                    roi_context_mag = (
                        float(np.nanmean(mag_ring))
                        if np.isfinite(mag_ring).any() else float(np.mean(mag))
                    )

        relation_box = target_box or local_box
        target_relation, _, _, target_related = _target_relation(relation_box, rois, width, height)
        residual_threshold = max(0.45, float(np.mean(residual)) + float(np.std(residual)) * 1.2)
        flow_local_anomaly_ratio = float(np.mean(residual >= residual_threshold))
        # Fraction of anomalous-flow pixels that fall inside YOLO ROI boxes.
        # High value → motion is explained by detected targets (real person walking),
        # not by adversarial artifacts outside/between detection regions.
        roi_coverage_ratio = 0.0
        if rois and flow_local_anomaly_ratio > 0.0:
            _high = residual >= residual_threshold
            _rmask = np.zeros(residual.shape, dtype=bool)
            for _r in rois:
                _x1, _y1, _x2, _y2 = int(_r.bbox[0]*fs), int(_r.bbox[1]*fs), int(_r.bbox[2]*fs), int(_r.bbox[3]*fs)
                _rmask[_y1:_y2, _x1:_x2] = True
            _n = float(np.sum(_high))
            if _n > 0.0:
                roi_coverage_ratio = float(np.sum(_high & _rmask)) / _n
        flow_max_magnitude_norm = _clamp(float(np.percentile(mag, 95)) / 6.0)
        roi_residual_contrast = max(0.0, roi_residual - roi_context_residual)
        roi_motion_gap = abs(roi_mag - roi_context_mag)
        flow_residual_score = max(
            _score(local_residual, 0.65, 3.2),
            _score(roi_residual, 0.50, 2.7),
            _score(roi_residual_contrast, 0.20, 1.35),
        )
        flow_shape_score = _clamp(
            max(
                _score(local_residual, mean_residual_grid * 1.4 + 0.20, mean_residual_grid * 2.4 + 0.90),
                _score(roi_residual_contrast, 0.18, 1.15),
                _score(roi_motion_gap, 0.35, 2.4) * 0.82,
            )
            * (1.0 - min(0.75, flow_local_anomaly_ratio))
        )
        camera_weight = float(flow.get("global_motion_weight", 0.0))
        a3_feature_score = _clamp(
            flow_residual_score
            * max(0.25, flow_shape_score)
            * (0.35 + 0.65 * target_relation if rois else 0.20)
            * (1.0 - 0.78 * camera_weight)
        )
        # Boost only when anomalous flow is NOT explained by YOLO detections
        # (low roi_coverage_ratio → flow is outside/between target boxes → adversarial).
        # High coverage means the flow is accounted for by real moving people — do not boost.
        if roi_mag >= 4.0 and roi_residual >= 1.4 and target_related and roi_coverage_ratio < 0.45:
            a3_feature_score = max(a3_feature_score, 0.62)
        if target_related and roi_residual_contrast >= 0.35 and roi_motion_gap >= 0.40 and roi_coverage_ratio < 0.45:
            a3_feature_score = max(
                a3_feature_score,
                _clamp(0.42 + 0.30 * _score(roi_residual_contrast, 0.35, 1.40)),
            )
        if exposure["exposure_delta"] >= 0.10 and camera_weight < 0.35 and not target_related:
            a3_feature_score *= 0.45
        normal_target_motion = bool(
            target_related
            and exposure["exposure_delta"] < 0.06
            and (
                # Mild target motion: small anomaly ratio and low contrast vs context
                (flow_local_anomaly_ratio < 0.22 and roi_residual_contrast < 1.20 and roi_motion_gap < 1.20)
                # Strong but target-explained: ≥50 % of anomalous-flow pixels inside YOLO boxes
                # means the flow is accounted for by detected targets, not adversarial artifacts.
                or roi_coverage_ratio >= 0.50
            )
        )
        if normal_target_motion:
            a3_feature_score = min(a3_feature_score, 0.22)
        background_motion_explain_score = _clamp(
            0.42 * _score(roi_context_residual, 0.35, 1.40)
            + 0.28 * _score(roi_context_mag, 0.45, 1.80)
            + 0.30 * (1.0 - target_relation)
        )
        background_like_residual = bool(
            (not target_related or target_relation < 0.20)
            and exposure["frame_diff_global"] < 0.045
            and exposure["exposure_delta"] < 0.035
            and camera_weight < 0.35
            and background_motion_explain_score >= 0.45
        )
        if background_like_residual:
            a3_feature_score = min(a3_feature_score, 0.40)
        fresh_residual_hold = bool(
            a3_feature_score >= 0.86
            and roi_residual_contrast >= 1.20
            and roi_motion_gap >= 1.20
            and not background_like_residual
        )
        hold_active = False
        if fresh_residual_hold:
            self.a3_residual_hold_score = max(float(a3_feature_score), self.a3_residual_hold_score * 0.94)
            self.a3_residual_hold_frames = 4
            hold_active = True
        elif self.a3_residual_hold_frames > 0:
            self.a3_residual_hold_score *= 0.94
            self.a3_residual_hold_frames -= 1
            a3_feature_score = max(a3_feature_score, min(0.91, self.a3_residual_hold_score))
            hold_active = True
        else:
            self.a3_residual_hold_score = 0.0
        a3_feature_score = _clamp(a3_feature_score)
        a3_candidate = bool(a3_feature_score >= 0.55)
        reason = "A3_FLOW_ARTIFACT" if a3_candidate else "none"
        return {
            "f_flow": float(max(flow_residual_score, a3_feature_score)),
            "flow_score": float(a3_feature_score),
            "flow_local_anomaly_ratio": float(flow_local_anomaly_ratio),
            "flow_max_magnitude_norm": float(flow_max_magnitude_norm),
            "flow_residual": float(max(local_residual, roi_residual)),
            "flow_roi_residual": float(roi_residual),
            "flow_context_residual": float(roi_context_residual),
            "flow_residual_contrast": float(roi_residual_contrast),
            "flow_roi_motion_gap": float(roi_motion_gap),
            "flow_roi_coverage_ratio": float(roi_coverage_ratio),
            "flow_background_explain_score": float(background_motion_explain_score),
            "flow_background_like_residual": bool(background_like_residual),
            "a3_residual_hold_active": bool(hold_active),
            "a3_residual_hold_score": float(self.a3_residual_hold_score if hold_active else 0.0),
            "a3_residual_hold_frames": int(self.a3_residual_hold_frames),
            "flow_shape_score": float(flow_shape_score),
            "flow_target_relation": float(target_relation),
            "flow_background_coherence": float(flow.get("background_coherence", 0.0)),
            "flow_global_motion_weight": float(camera_weight),
            "a3_feature_score": float(a3_feature_score),
            "a3_candidate": a3_candidate,
            "a3_reason": reason,
            "target_related": bool(target_related),
            "local_bbox": list(local_box),
        }

    def _compute_a4(
        self,
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        a3b: dict[str, Any] | None = None,
        frame: np.ndarray | None = None,
        frame_idx: int | None = None,
    ) -> dict[str, Any]:
        s1 = float(a1["a1_feature_score"])
        s2 = float(a2["a2_feature_score"])
        s3 = float(a3["a3_feature_score"])

        # A4 is a synchronous physical-attack classifier. Keep the optional
        # argument for call compatibility, but never consume asynchronous A3b
        # cache values in its feature schema.
        _ = a3b
        a4_feature_vector: list[float] = [
            # A1 组
            float(a1["delta_h"]), float(a1["delta_h_roi_max"]),
            float(a1["delta_h_local_max"]), float(a1["delta_h_target_contrast"]),
            s1,
            # A2 组
            float(a2["change_t"]), float(a2["change_t_roi_max"]),
            float(a2["change_t_local_max"]), float(a2["change_t_without_motion_target"]),
            s2,
            # A3 组（f_flow 即特征向量的代表标量，见设计稿 §3）
            float(a3["f_flow"]), float(a3["flow_local_anomaly_ratio"]),
            float(a3["flow_residual"]), float(a3["flow_shape_score"]),
            float(a3["flow_target_relation"]), s3,
        ]
        patch_interval = max(
            1,
            int(getattr(self, "_a4_patch_feature_interval", 1)),
        )
        cached_patch_features = getattr(
            self,
            "_a4_patch_feature_cache",
            None,
        )
        patch_features_reused = bool(
            frame is not None
            and frame_idx is not None
            and cached_patch_features is not None
            and int(frame_idx) % patch_interval != 0
        )
        if patch_features_reused:
            patch_features = cached_patch_features
        else:
            patch_features = extract_a4_patch_features(frame)
            if frame is not None:
                self._a4_patch_feature_cache = patch_features
        patch_baseline_vectors = getattr(
            self,
            "_a4_patch_baseline_vectors",
            None,
        )
        if patch_baseline_vectors is None:
            patch_baseline_vectors = deque(maxlen=12)
            self._a4_patch_baseline_vectors = patch_baseline_vectors
        if frame is not None and len(patch_baseline_vectors) < 12:
            patch_baseline_vectors.append(tuple(patch_features))
        patch_baseline_ready = len(patch_baseline_vectors) >= 12
        if patch_baseline_vectors:
            patch_baseline = np.median(
                np.asarray(patch_baseline_vectors, dtype=np.float32),
                axis=0,
            )
            patch_delta_features = tuple(
                float(current - baseline)
                for current, baseline in zip(
                    patch_features,
                    patch_baseline,
                    strict=True,
                )
            )
        else:
            patch_delta_features = (0.0,) * len(patch_features)
        a4_feature_vector.extend(float(value) for value in patch_features)
        a4_feature_vector.extend(patch_delta_features)

        classifier_used = False
        classifier_p_adv: float | None = None
        if (
            self._classifier is not None
            and self.a4_classifier_loaded
            and not self._a4_classifier_runtime_disabled
        ):
            try:
                expected = int(
                    getattr(
                        self._classifier,
                        "n_features_in_",
                        len(a4_feature_vector),
                    )
                    or len(a4_feature_vector)
                )
                actual = len(a4_feature_vector)
                if expected != actual:
                    raise ValueError(
                        "feature_schema_mismatch:"
                        f"expected={expected},actual={actual}"
                    )
                if hasattr(self._classifier, "feature_importances_"):
                    importance_count = int(
                        np.asarray(
                            self._classifier.feature_importances_
                        ).size
                    )
                    if importance_count != actual:
                        raise ValueError(
                            "feature_schema_mismatch:"
                            f"importances={importance_count},actual={actual}"
                        )
                classifier_p_adv = float(
                    self._classifier.predict_proba(
                        [a4_feature_vector]
                    )[0][1]
                )
                if not math.isfinite(classifier_p_adv):
                    raise ValueError(
                        f"non_finite_probability:{classifier_p_adv}"
                    )
                classifier_used = True
                self.a4_classifier_error = None
                self.a4_classifier_fallback_reason = "none"
            except Exception as exc:
                self._a4_classifier_runtime_disabled = True
                self.a4_classifier_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.a4_classifier_fallback_reason = (
                    "feature_schema_mismatch"
                    if "feature_schema_mismatch" in str(exc)
                    else "predict_failed"
                )

        # 贡献归因只覆盖前 16 个 A1/A2/A3 基础特征。后 80 个 patch/delta
        # 特征属于 A4 救援证据，不反算为单个基础模块贡献。降级路径使用固定
        # 经验乘数，避免错误 artifact 继续影响 dominant input。
        if classifier_used and hasattr(self._classifier, "feature_importances_"):
            fi = self._classifier.feature_importances_
            # 只聚合基础特征区间；当前生产 schema 总计 96 维。
            w1 = s1 * float(np.sum(fi[0:5]))    # A1 (5维): ~0.328
            w2 = s2 * float(np.sum(fi[5:10]))   # A2 (5维): ~0.297
            w3 = s3 * float(np.sum(fi[10:16]))  # A3 (6维): ~0.244
        else:
            w1, w2, w3 = s1 * 1.00, s2 * 1.08, s3 * 1.12
        weighted = {
            "A1_LBP_SINGLE": w1,
            "A2_LBP_TEMPORAL": w2,
            "A3_FLOW_ARTIFACT": w3,
        }
        total = sum(weighted.values())
        contributions = {
            "a1_contribution": 0.0 if total <= 1e-6 else w1 / total,
            "a2_contribution": 0.0 if total <= 1e-6 else w2 / total,
            "a3_contribution": 0.0 if total <= 1e-6 else w3 / total,
        }
        top_name, top_value = max(weighted.items(), key=lambda item: item[1])
        sorted_vals = sorted(weighted.values(), reverse=True)
        margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
        dominant = top_name if margin >= 0.08 or top_value >= 0.70 else "A4_MIXED"
        max_score = max(s1, s2, s3)
        sorted_scores = sorted([s1, s2, s3], reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        third_score = sorted_scores[2] if len(sorted_scores) > 2 else 0.0
        synergy = min(0.40, (s1 + s2 + s3 - max_score) * 0.40)
        multi_evidence = _clamp(
            0.55 * _score(second_score, 0.22, 0.55)
            + 0.25 * _score(third_score, 0.10, 0.40)
            + 0.20 * _score(synergy, 0.06, 0.22)
        )

        rule_p_adv = self._rule_p_adv(
            max_score,
            second_score,
            third_score,
            multi_evidence,
        )
        if classifier_used and classifier_p_adv is not None:
            p_adv_raw = max(rule_p_adv, classifier_p_adv)
        else:
            p_adv_raw = rule_p_adv

        p_adv_calibrated = _clamp(p_adv_raw)
        p_adv = p_adv_calibrated
        decision_threshold = float(
            self.a4_classifier_decision_threshold
            if classifier_used
            else self.theta_adv
        )
        classifier_triggered = bool(
            classifier_used
            and classifier_p_adv is not None
            and patch_baseline_ready
            and classifier_p_adv >= decision_threshold
        )
        rule_triggered = bool(rule_p_adv >= self.theta_adv)
        fused_triggered = bool(rule_triggered or classifier_triggered)
        if classifier_triggered:
            dominant = "A4_PATCH_CLASSIFIER"
        return {
            "p_adv_raw": float(p_adv_raw),
            "p_adv_calibrated": float(p_adv_calibrated),
            "p_adv": float(p_adv),
            "p_adv_triggered": fused_triggered,
            "a4_rule_triggered": rule_triggered,
            "a1_contribution": float(contributions["a1_contribution"]),
            "a2_contribution": float(contributions["a2_contribution"]),
            "a3_contribution": float(contributions["a3_contribution"]),
            "dominant_adv_input": dominant,
            "a4_second_feature_score": float(second_score),
            "a4_third_feature_score": float(third_score),
            "a4_multi_evidence": float(multi_evidence),
            "a4_synergy": float(synergy),
            "a4_feature_vector": a4_feature_vector,
            "a4_patch_feature_count": len(patch_features),
            "a4_patch_delta_feature_count": len(patch_delta_features),
            "a4_patch_feature_interval": patch_interval,
            "a4_patch_features_reused": patch_features_reused,
            "a4_patch_baseline_samples": len(patch_baseline_vectors),
            "a4_patch_baseline_ready": patch_baseline_ready,
            "a4_classifier_used": bool(classifier_used),
            "a4_classifier_p_adv": (
                None
                if classifier_p_adv is None
                else float(classifier_p_adv)
            ),
            "a4_classifier_triggered": classifier_triggered,
            "a4_rule_p_adv": float(rule_p_adv),
            "a4_classifier_configured": bool(
                self.a4_classifier_configured
            ),
            "a4_classifier_loaded": bool(self.a4_classifier_loaded),
            "a4_classifier_error": self.a4_classifier_error,
            "a4_classifier_fallback_reason": (
                self.a4_classifier_fallback_reason
            ),
            "a4_classifier_path": self.a4_classifier_path,
            "a4_classifier_resolved_path": (
                self.a4_classifier_resolved_path
            ),
            "a4_classifier_metadata": dict(
                self.a4_classifier_metadata
            ),
            "a4_feature_schema_version": A4_FEATURE_SCHEMA_VERSION,
            "a4_feature_names": list(A4_FEATURE_NAMES),
            "a4_async_a3b_features_used": False,
            "theta_adv": float(self.theta_adv),
            "a4_decision_threshold": decision_threshold,
            "a4_decision_threshold_source": (
                "classifier_metadata"
                if classifier_used
                else "runtime_rule_config"
            ),
        }

    @staticmethod
    def _rule_p_adv(max_score: float, second_score: float, third_score: float, multi_evidence: float) -> float:
        """Return rule evidence used beside XGBoost and as its safe fallback.

        Production A3 prefers RAFT TensorRT, then GPU Lucas-Kanade, then DIS
        CPU. A4 keeps this rule score active even when the 96-feature XGBoost
        rescue classifier is loaded; a missing or invalid classifier therefore
        degrades visibly without disabling physical-attack scoring.
        """
        linear = (
            3.05 * max_score
            + 1.18 * second_score
            + 0.42 * third_score
            + 0.58 * multi_evidence
            - 1.52
        )
        return 1.0 / (1.0 + math.exp(-linear))

    @staticmethod
    def _pctl(values: deque, q: float) -> float:
        """从滚动窗口取分位数作为"本场景正常参考值"。窗口空时返回 0。"""
        if not values:
            return 0.0
        return float(np.percentile(np.fromiter(values, dtype=np.float64), q))

    _P4_NAMES = ["p4_hf_ratio", "p4_lap_var", "p4_edge_density", "p4_sat_p95", "p4_color_ext"]

    def _compute_p4(self, frame: np.ndarray | None, rois: list[ROI]) -> dict[str, float]:
        """P4 判别性特征：区分"对抗纹理/遮挡/强光"与"自然工人纹理"。
        在最高置信目标 ROI(无则中心区)上算高频能量比/锐度/边缘密度/饱和度/红黄极端度。
        加结构型攻击(patch/occluder/glare)用合成的平滑/纯色/高饱和内容覆盖自然纹理 →
        ROI 高频能量比与锐度显著低于自然工人纹理 → 分开"弱攻击 vs 干净纹理突发"的关键轴
        （实验单特征 AUC：hf_ratio 0.99 / lap_var 0.98）。ROI 统一缩放到 128 方形，尺度无关且快。"""
        z = {k: 0.0 for k in self._P4_NAMES}
        if frame is None or frame.ndim != 3:
            return z
        h, w = frame.shape[:2]
        box = None
        bestc = -1.0
        for r in rois:
            c = float(getattr(r, "confidence", 0.0) or 0.0)
            if c > bestc:
                bestc = c
                box = r.bbox
        if box is None:
            x1, y1, x2, y2 = int(w * 0.35), int(h * 0.2), int(w * 0.65), int(h * 0.75)
        else:
            x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return z
        roi = cv2.resize(frame[y1:y2, x1:x2], (128, 128), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gf = g.astype(np.float32)
        # 高频能量比（FFT，radial mask 128 缓存）
        if getattr(self, "_p4_radmask", None) is None:
            yy, xx = np.ogrid[:128, :128]
            rad = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2)
            self._p4_radmask = rad > (0.3 * rad.max())
        F = np.abs(np.fft.fftshift(np.fft.fft2(gf)))
        hf_ratio = float(F[self._p4_radmask].sum() / (F.sum() + 1e-9))
        lap_var = float(cv2.Laplacian(gf, cv2.CV_32F).var())
        edge_density = float((cv2.Canny(g, 80, 160) > 0).mean())
        sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
        sat_p95 = float(np.percentile(sat, 95))
        b = roi[:, :, 0].astype(np.float32)
        gc = roi[:, :, 1].astype(np.float32)
        rr = roi[:, :, 2].astype(np.float32)
        redish = float(((rr > 120) & (rr - b > 40) & (rr - gc > 10)).mean())
        yellowish = float(((rr > 120) & (gc > 120) & (b < 110)).mean())
        return {
            "p4_hf_ratio": hf_ratio,
            "p4_lap_var": lap_var,
            "p4_edge_density": edge_density,
            "p4_sat_p95": sat_p95,
            "p4_color_ext": max(redish, yellowish),
        }

    def _compute_blinding(
        self,
        gray: np.ndarray,
        rois: list[ROI],
        exposure: dict[str, Any],
        flow: dict[str, Any],
    ) -> dict[str, Any]:
        """支路B：致盲/去信号型攻击检测（motion_blur / visibility / glare 致盲）。

        核心：不看绝对阈值，看"相对本场景近 N 帧自身基线的退化"——
        清晰度(拉普拉斯方差)、对比度(亮度std)、YOLO目标置信度强度 三者**骤降**，
        叠加曾有目标(recent_target_presence)而当前漏检 = 致盲证据。
        这样常年模糊/低对比的场景不会误报，只有突发退化才告警。
        """
        # 1) 逐帧清晰度(拉普拉斯方差)：优先 Rust 单遍融合(免中间float32数组+二次遍历)，回退 cv2
        native_result = self._native_call(
            "blind",
            "blinding_laplacian_var",
            np.ascontiguousarray(gray, dtype=np.uint8),
        )
        if native_result is not None:
            sharpness = float(native_result)
        else:
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            sharpness = float(lap.var())
        contrast = float(exposure["brightness_std"])
        det_strength = float(sum(float(getattr(r, "confidence", 0.0) or 0.0) for r in rois))

        # 帧间清晰度相对上一帧的骤降(冷启动期 + ramp 期都有用)，及强过曝致盲
        prev_sharp = self._prev_sharp
        self._prev_sharp = sharpness  # 每帧更新(无论后续走哪个分支)
        sharp_drop_short = _clamp((prev_sharp - sharpness) / prev_sharp) if prev_sharp > 1e-6 else 0.0
        glare_blind0 = _score(float(exposure["overexposure_ratio"]), 0.16, 0.55)
        recent_present = (sum(self.recent_target_presence) - (1 if rois else 0)) >= 3

        ready = bool(len(self._sb_sharp) >= self._scene_baseline_min)
        if not self._blind_enabled:
            return {
                "p_blind": 0.0, "p_blind_triggered": False, "blind_ready": ready,
                "sharpness": sharpness, "contrast": contrast, "det_strength": det_strength,
                "sharp_drop": 0.0, "contrast_drop": 0.0, "det_drop": 0.0,
                "glare_blind": 0.0, "target_loss": 0.0, "blind_type": "none",
                "sharp_drop_short": float(sharp_drop_short),
                "blind_independent_support": False,
            }
        if not ready:
            # P1 冷启动绝对兜底：场景基线未就绪时(如视频开头即攻击)，靠"曾有目标却骤然漏检
            # + (帧间清晰度骤降 或 强过曝)"判定致盲——目标丢失是强证据但需退化佐证防误报。
            now_lost = len(rois) == 0
            degrade = max(sharp_drop_short, glare_blind0)
            cold_independent_support = bool(
                glare_blind0 >= 0.30
                or (
                    recent_present
                    and now_lost
                    and sharp_drop_short >= 0.12
                )
            )
            if recent_present and now_lost:
                p_cold = _clamp(0.45 + 0.55 * degrade)
            else:
                p_cold = _clamp(0.70 * glare_blind0)
            if not cold_independent_support:
                p_cold = min(p_cold, 0.40)
            btype = "cold_glare" if glare_blind0 >= sharp_drop_short else "cold_blur"
            return {
                "p_blind": float(p_cold), "p_blind_triggered": bool(p_cold >= self.theta_blind),
                "blind_ready": False, "sharpness": sharpness, "contrast": contrast,
                "det_strength": det_strength, "sharp_drop": float(sharp_drop_short),
                "contrast_drop": 0.0, "det_drop": 0.0, "glare_blind": float(glare_blind0),
                "target_loss": float(recent_present and now_lost), "blind_type": btype,
                "sharp_drop_short": float(sharp_drop_short),
                "low_motion_target_loss_support": False,
                "blind_independent_support": bool(
                    cold_independent_support
                ),
            }

        # 2) 本场景参考值（分位数，鲁棒于个别异常帧）
        ref_sharp = self._pctl(self._sb_sharp, 70)
        ref_contrast = self._pctl(self._sb_contrast, 70)
        ref_det = self._pctl(self._sb_detstr, 70)

        sharp_drop = _clamp((ref_sharp - sharpness) / ref_sharp) if ref_sharp > 1e-6 else 0.0
        contrast_drop = _clamp((ref_contrast - contrast) / ref_contrast) if ref_contrast > 1e-6 else 0.0
        # 致盲性强光：当前过曝相对基线（基线过曝用 0 起底）
        glare_blind = _score(float(exposure["overexposure_ratio"]), 0.16, 0.55)

        # 目标丢失：本场景曾稳定有目标(ref_det>0) 而当前置信度强度骤降
        recent_targets = sum(self.recent_target_presence) >= 2
        det_drop = _clamp((ref_det - det_strength) / ref_det) if ref_det > 1e-3 else 0.0
        target_loss = det_drop if recent_targets else 0.0

        # 3) 融合：必须"退化"且"目标受影响"才算致盲（避免对清晰画面的正常运动误报）。
        degrade = max(sharp_drop, contrast_drop, glare_blind, 0.8 * sharp_drop_short)
        evidence = max(target_loss, 0.5 * glare_blind)  # 强光本身就是致盲证据，权重减半
        p_blind = _clamp(degrade * (0.35 + 0.65 * evidence))

        # 退化但目标未受影响（YOLO 仍检出）→ 多半是正常快动/对焦，强抑制
        if target_loss < 0.25 and glare_blind < 0.30:
            p_blind = min(p_blind, 0.40)

        if glare_blind >= max(sharp_drop, contrast_drop):
            blind_type = "glare_blind"
        elif sharp_drop >= contrast_drop:
            blind_type = "motion_blur"
        else:
            blind_type = "visibility"

        # A motion-blur score can be large when ordinary worker motion, helmet
        # removal, or a head turn causes the detector confidence to disappear.
        # The independent low-motion branch captures the opposite physical
        # pattern: targets disappear while the whole scene remains stable and
        # sharpness stays below its own baseline for several frames.  This is
        # the characteristic signal in true blur/visibility degradation, and
        # it does not treat ordinary high-motion target exit as blur evidence.
        # Normalize Laplacian energy by the frame luminance variance.  This
        # separates a real loss of high-frequency detail from a naturally
        # high-detail scene whose target detector temporarily drops out:
        # both can have a large relative sharpness delta, but only the former
        # has little residual edge energy for its remaining contrast.
        blur_detail_ratio = sharpness / max(contrast * contrast, 1e-6)
        motion_blur_scene_degradation_support = bool(
            blur_detail_ratio <= 0.25
            and float(
                exposure.get("underexposed_ratio", 0.0)
            ) < 0.10
            and float(
                exposure.get("frame_diff_global", 0.0)
            ) <= 0.015
            and float(
                flow.get("global_motion_weight", 0.0)
            ) <= 0.45
        )
        low_motion_target_loss_support = bool(
            blind_type == "motion_blur"
            and sharp_drop >= 0.18
            and target_loss >= 0.50
            and motion_blur_scene_degradation_support
        )
        if low_motion_target_loss_support:
            low_motion_blind_score = _clamp(
                self.theta_blind
                + 0.20 * _score(sharp_drop, 0.18, 0.40)
                + 0.15 * _score(target_loss, 0.50, 1.0)
            )
            p_blind = max(p_blind, low_motion_blind_score)

        motion_blur_visual_degradation_support = bool(
            motion_blur_scene_degradation_support
            and (
                contrast_drop >= 0.18
                or sharp_drop >= 0.85
                or sharp_drop_short >= 0.12
            )
        )
        motion_blur_independent_support = bool(
            blind_type != "motion_blur"
            or low_motion_target_loss_support
            or motion_blur_visual_degradation_support
            or float(exposure.get("exposure_delta", 0.0)) >= 0.010
            or float(exposure.get("overexposure_ratio", 0.0)) >= 0.10
        )
        if blind_type == "motion_blur" and not motion_blur_independent_support:
            p_blind = min(p_blind, 0.40)

        return {
            "p_blind": float(p_blind),
            "p_blind_triggered": bool(p_blind >= self.theta_blind),
            "blind_ready": True,
            "sharpness": sharpness, "contrast": contrast, "det_strength": det_strength,
            "ref_sharpness": ref_sharp, "ref_contrast": ref_contrast, "ref_det": ref_det,
            "sharp_drop": float(sharp_drop), "contrast_drop": float(contrast_drop),
            "det_drop": float(det_drop), "glare_blind": float(glare_blind),
            "target_loss": float(target_loss), "blind_type": blind_type,
            "sharp_drop_short": float(sharp_drop_short),
            "blur_detail_ratio": float(blur_detail_ratio),
            "motion_blur_scene_degradation_support": bool(
                motion_blur_scene_degradation_support
            ),
            "low_motion_target_loss_support": bool(
                low_motion_target_loss_support
            ),
            "blind_independent_support": bool(motion_blur_independent_support),
        }

    @staticmethod
    def _moire_score(gray: np.ndarray) -> float:
        """摩尔纹频域突出度：屏幕翻拍的像素栅格混叠在中段频谱形成孤立周期峰，自然图像是
        平滑 1/f 谱无孤立峰。返回 中段最强峰/中段均值。**诊断量，未接入触发**——
        实验仅 2 个非理想"屏幕在场景"视频，AUC≈0.73(弱)，待真翻拍数据验证后再决定是否启用。
        在 A3b 后台线程计算，零热路径开销。"""
        try:
            g = cv2.resize(gray, (256, 256)).astype(np.float32)
            g -= g.mean()
            F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
            yy, xx = np.ogrid[:256, :256]
            rad = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
            vals = F[(rad > 30) & (rad < 110)]
            if vals.size == 0:
                return 0.0
            return float(vals.max() / (vals.mean() + 1e-9))
        except Exception:
            return 0.0

    def _compute_a3b(
        self,
        gray: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
        flow: dict[str, Any],
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
    ) -> dict[str, Any]:
        moire = self._moire_score(gray)  # 诊断量(未接入触发)
        candidates = self._extract_media_candidates(gray, rois, width, height)
        best = candidates[0] if candidates else None
        track_state = self._update_media_track(best["bbox"] if best else None)
        l2 = self._media_l2_validation(best, flow, width, height)

        if best is None:
            raw_scores = {
                "edge": 0.0,
                "rect": 0.0,
                "area": 0.0,
                "plane": 0.0,
                "track": 0.0,
                "warp_residual": 0.0,
                "flow_gap": 0.0,
                "inside_motion": 0.0,
                "outside_motion": 0.0,
                "yolo_context": 0.0,
            }
            p_media_raw = 0.0
            p_media_type = "normal"
            bbox = None
            target_related = False
            strong_evidence = False
        else:
            bbox = best["bbox"]
            target_related = bool(best["target_related"])
            track_score = track_state["track_score"]
            display_frame_score = float(best.get("display_frame_score", 0.0))
            boundary_score = float(best.get("boundary_score", 0.0))
            area_ratio = float(best.get("candidate_area_ratio", 0.0))
            plane_score = _clamp(
                0.30 * best["rect_score"]
                + 0.20 * best["edge_score"]
                + 0.18 * track_score
                + 0.16 * l2["homography_inlier_ratio"]
                + 0.16 * max(display_frame_score, boundary_score)
            )
            flow_gap_score = _score(l2["flow_gap"], 0.25, 1.6)
            warp_score = _score(l2["warp_residual"], 0.04, 0.28)
            yolo_context = max(best["target_iou"], best["target_proximity"])
            no_target_screen_context = _clamp(
                0.55 * display_frame_score
                + 0.25 * _score(area_ratio, 0.035, 0.20)
                + 0.20 * best["rect_score"]
            )
            media_context = max(yolo_context, no_target_screen_context)
            screen_replay_score = _clamp(
                0.30 * plane_score
                + 0.26 * max(flow_gap_score, warp_score)
                + 0.22 * display_frame_score
                + 0.10 * best["edge_score"]
                + 0.12 * media_context
            )
            paper_photo_score = _clamp(
                0.38 * plane_score
                + 0.20 * (1.0 - _score(l2["inside_motion"], 0.15, 1.0))
                + 0.18 * track_score
                + 0.14 * display_frame_score
                + 0.10 * media_context
            )
            static_image_score = _clamp(
                0.32 * plane_score
                + 0.18 * best["rect_score"]
                + 0.18 * track_score
                + 0.17 * display_frame_score
                + 0.15 * media_context
            )
            p_media_raw_unadjusted = max(screen_replay_score, paper_photo_score, static_image_score)
            raw_reliability = 1.0
            if area_ratio < 0.085 and display_frame_score < 0.36:
                raw_reliability = min(raw_reliability, 0.68)
            if (
                target_related
                and area_ratio < 0.070
                and best["target_iou"] >= 0.25
                and (
                    boundary_score < 0.08
                    or display_frame_score < 0.58
                    or best.get("border_contrast_score", 0.0) < 0.70
                )
            ):
                raw_reliability = min(raw_reliability, 0.70)
            if (
                area_ratio < 0.060
                and best["target_iou"] >= 0.65
                and boundary_score < 0.12
            ):
                raw_reliability = min(raw_reliability, 0.66)
            if (
                not target_related
                and display_frame_score < 0.58
                and best.get("border_contrast_score", 0.0) < 0.70
                and boundary_score < 0.36
            ):
                raw_reliability = min(raw_reliability, 0.74)
            if (
                not target_related
                and best["edge_score"] < 0.28
                and l2["flow_gap"] < 0.75
                and display_frame_score < 0.76
            ):
                raw_reliability = min(raw_reliability, 0.78)
            if (
                not target_related
                and l2["warp_residual"] >= 0.12
                and l2["flow_gap"] < 0.90
                and display_frame_score < 0.76
                and best["edge_score"] < 0.50
            ):
                raw_reliability = min(raw_reliability, 0.82)
            p_media_raw = _clamp(p_media_raw_unadjusted * raw_reliability)
            # Gate: genuine replay/paper attacks have at least one of: flow_gap (screen
            # static vs moving background), warp_residual (perspective from phone tilt),
            # or visible display frame. Background structures (walls, windows) have none
            # of these → cap at sub-threshold to suppress false positives.
            has_media_signal = (
                l2["flow_gap"] >= 0.35          # strong inside/outside motion contrast
                or l2["warp_residual"] >= 0.08  # perspective distortion (angled device)
                or display_frame_score >= 0.28  # visible screen frame
                or (l2["flow_gap"] >= 0.20 and boundary_score >= 0.35)
            )
            if not has_media_signal:
                p_media_raw = min(p_media_raw, 0.22)
            if screen_replay_score >= max(paper_photo_score, static_image_score) and screen_replay_score >= 0.45:
                p_media_type = "screen_replay"
            elif paper_photo_score >= max(screen_replay_score, static_image_score) and paper_photo_score >= 0.45:
                p_media_type = "paper_photo"
            elif static_image_score >= 0.42:
                p_media_type = "static_image_spoof"
            elif p_media_raw >= 0.35:
                p_media_type = "flat_media_candidate"
            else:
                p_media_type = "normal"
            screen_like_evidence = bool(
                display_frame_score >= 0.52
                and best["rect_score"] >= 0.45
                and area_ratio >= 0.012
            )
            strong_evidence = bool(
                (
                    plane_score >= 0.55
                    and track_score >= 0.45
                    and (target_related or flow_gap_score >= 0.55 or warp_score >= 0.55)
                )
                or (
                    screen_like_evidence
                    and track_score >= 0.38
                    and p_media_raw >= self.theta_media_raw
                )
            )
            raw_scores = {
                "edge": float(best["edge_score"]),
                "rect": float(best["rect_score"]),
                "area": float(best["area_score"]),
                "area_ratio": float(area_ratio),
                "plane": float(plane_score),
                "track": float(track_score),
                "warp_residual": float(l2["warp_residual"]),
                "flow_gap": float(l2["flow_gap"]),
                "inside_motion": float(l2["inside_motion"]),
                "outside_motion": float(l2["outside_motion"]),
                "inner_outer_motion_ratio": float(l2["inner_outer_motion_ratio"]),
                "homography_inlier_ratio": float(l2["homography_inlier_ratio"]),
                "screen_replay": float(screen_replay_score),
                "paper_photo": float(paper_photo_score),
                "static_image": float(static_image_score),
                "raw_unadjusted": float(p_media_raw_unadjusted),
                "raw_reliability": float(raw_reliability),
                "yolo_context": float(yolo_context),
                "target_iou": float(best["target_iou"]),
                "target_proximity": float(best["target_proximity"]),
                "display_frame": float(display_frame_score),
                "boundary": float(boundary_score),
                "border_contrast": float(best.get("border_contrast_score", 0.0)),
                "inner_texture": float(best.get("inner_texture_score", 0.0)),
                "candidate_score": float(best.get("candidate_score", 0.0)),
                "source_score": float(best.get("source_score", 0.0)),
            }

        policy = self._apply_media_policy(
            p_media_raw=p_media_raw,
            p_media_type=p_media_type,
            bbox=bbox,
            target_related=target_related,
            strong_evidence=strong_evidence,
            scores=raw_scores,
            flow=flow,
            exposure=exposure,
            a1=a1,
            a2=a2,
            a3=a3,
            width=width,
            height=height,
        )
        p_media_policy = float(policy["p_media_policy"])
        p_media_triggered = bool(p_media_policy >= self.theta_media)
        p_media_raw_triggered = bool(p_media_raw >= self.theta_media_raw)
        return {
            "media_candidates": candidates[:5],
            "track_id": 1 if self.media_track is not None else None,
            "stable_count": int(track_state["stable_count"]),
            "track_score": float(track_state["track_score"]),
            "bbox_jitter": float(track_state["bbox_jitter"]),
            "scale_jitter": float(track_state["scale_jitter"]),
            "candidate_lifetime": int(track_state["candidate_lifetime"]),
            "plane_score": float(raw_scores.get("plane", 0.0)),
            "warp_residual": float(raw_scores.get("warp_residual", 0.0)),
            "flow_gap": float(raw_scores.get("flow_gap", 0.0)),
            "inside_motion": float(raw_scores.get("inside_motion", 0.0)),
            "outside_motion": float(raw_scores.get("outside_motion", 0.0)),
            "inner_outer_motion_ratio": float(raw_scores.get("inner_outer_motion_ratio", 0.0)),
            "homography_inlier_ratio": float(raw_scores.get("homography_inlier_ratio", 0.0)),
            "p_media": float(p_media_policy),
            "p_media_raw": float(p_media_raw),
            "p_media_raw_triggered": p_media_raw_triggered,
            "p_media_policy": float(p_media_policy),
            "p_media_confirmed_score": 0.0,
            "media_confirmed": False,
            "p_media_triggered": p_media_triggered,
            "p_media_type": p_media_type,
            "p_media_bbox": None if bbox is None else [int(v) for v in bbox],
            "p_media_target_related": bool(target_related),
            "p_media_scores": raw_scores,
            "p_media_strong_evidence": bool(strong_evidence),
            "a3b_moire": float(moire),
            "p_media_background_static_suppressed": bool(policy["background_static_suppressed"]),
            "a3b_display_score": float(p_media_policy),
            "suppressed_reason": policy["suppressed_reason"],
            "score_cap": float(policy["score_cap"]),
            "media_candidate_allowed": bool(policy["media_candidate_allowed"]),
            "a3b_state": policy["a3b_state"],
        }

    def _extract_media_candidates(
        self,
        gray: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 60, 160)
        edge_mask = edges > 0
        # 原生路径需要 uint8 连续视图（C-contiguous）；候选几何先集中，
        # 随后整帧一次调用 a3b_boxes_stats，避免逐框 Python→Rust 往返。
        edge_mask_u8 = np.ascontiguousarray(edge_mask, dtype=np.uint8)
        frame_area = float(width * height)
        candidates: list[dict[str, Any]] = []
        pending_candidate_specs: list[dict[str, Any]] = []
        collecting_candidate_specs = True
        native_result_unset = object()

        def _odd(value: float) -> int:
            int_value = max(3, int(round(float(value))))
            return int_value if int_value % 2 == 1 else int_value + 1

        def _border_values(patch: np.ndarray, border: int) -> np.ndarray:
            if patch.size == 0:
                return np.asarray([], dtype=patch.dtype)
            h, w = patch.shape[:2]
            border = max(1, min(border, max(1, h // 2), max(1, w // 2)))
            return np.concatenate(
                [
                    patch[:border, :].reshape(-1),
                    patch[h - border:h, :].reshape(-1),
                    patch[:, :border].reshape(-1),
                    patch[:, w - border:w].reshape(-1),
                ]
            )

        def _add_candidate(
            box: tuple[int, int, int, int] | list[float],
            *,
            source: str,
            contour_area: float | None = None,
            quad_score_hint: float | None = None,
            native_result_override: Any = native_result_unset,
        ) -> None:
            # 全局候选上限：复杂画面会产生数百个候选，每个都做昂贵的逐框统计
            # （边界/纹理/IoU），不设上限会使单轮飙到 200-500ms 并抢 GIL 拖垮主路径。
            if not collecting_candidate_specs and len(candidates) >= 64:
                return
            clipped = _clip_box(box, width, height, min_size=18)
            if clipped is None:
                return
            x1, y1, x2, y2 = clipped
            bw = x2 - x1
            bh = y2 - y1
            box_area = max(1.0, float((x2 - x1) * (y2 - y1)))
            area_ratio = box_area / max(1.0, frame_area)
            if area_ratio < 0.003 or area_ratio > 0.82:
                return
            rect_fill = _clamp(float(contour_area) / box_area) if contour_area is not None else 0.58
            quad_score = float(quad_score_hint) if quad_score_hint is not None else 0.65
            aspect = (x2 - x1) / max(1.0, y2 - y1)
            aspect_score = 1.0 if 0.34 <= aspect <= 3.20 else (0.62 if 0.24 <= aspect <= 4.20 else 0.35)
            if collecting_candidate_specs:
                pending_candidate_specs.append(
                    {
                        "box": clipped,
                        "source": source,
                        "contour_area": contour_area,
                        "quad_score_hint": quad_score_hint,
                    }
                )
                return
            local_edges = edge_mask[y1:y2, x1:x2]
            native_result = (
                self._native_call(
                    "a3b",
                    "a3b_one_box_stats",
                    edge_mask_u8,
                    gray,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
                if native_result_override is native_result_unset
                else native_result_override
            )
            if native_result is not None:
                (edge_density, border_edge_density, inner_edge_density,
                 border_mean, inner_mean, gray_std) = native_result
                edge_score = _clamp(edge_density / 0.16)
                boundary_score = _clamp(
                    0.70 * (border_edge_density / 0.22)
                    + 0.30 * _score(border_edge_density - inner_edge_density, 0.02, 0.18)
                )
                border_contrast_score = _score(abs(border_mean - inner_mean) / 255.0, 0.025, 0.16)
                inner_texture_score = _score(gray_std / 255.0, 0.04, 0.18)
            else:
                edge_density = float(np.mean(local_edges)) if local_edges.size else 0.0
                edge_score = _clamp(edge_density / 0.16)
                border = max(2, int(min(bw, bh) * 0.035))
                border_edges = _border_values(local_edges.astype(np.uint8), border)
                border_edge_density = float(np.mean(border_edges)) if border_edges.size else 0.0
                if bw > 2 * border + 2 and bh > 2 * border + 2:
                    inner_edges = local_edges[border:bh - border, border:bw - border]
                    inner_edge_density = float(np.mean(inner_edges)) if inner_edges.size else edge_density
                else:
                    inner_edge_density = edge_density
                boundary_score = _clamp(
                    0.70 * (border_edge_density / 0.22)
                    + 0.30 * _score(border_edge_density - inner_edge_density, 0.02, 0.18)
                )
                local_gray = gray[y1:y2, x1:x2].astype(np.float32)
                gray_border = _border_values(local_gray, border)
                if bw > 2 * border + 2 and bh > 2 * border + 2:
                    inner_gray = local_gray[border:bh - border, border:bw - border]
                else:
                    inner_gray = local_gray
                border_mean = float(np.mean(gray_border)) if gray_border.size else 0.0
                inner_mean = float(np.mean(inner_gray)) if inner_gray.size else border_mean
                border_contrast_score = _score(abs(border_mean - inner_mean) / 255.0, 0.025, 0.16)
                inner_texture_score = _score(float(np.std(local_gray)) / 255.0, 0.04, 0.18)
            rect_score = _clamp(
                0.30 * rect_fill
                + 0.26 * quad_score
                + 0.18 * aspect_score
                + 0.26 * boundary_score
            )
            area_score = _clamp(_score(area_ratio, 0.006, 0.26))
            _relation, target_iou, target_prox, _target_related = _target_relation(clipped, rois, width, height)
            anchored_target_prox = target_prox if target_iou >= 0.08 else target_prox * 0.35
            target_related = bool(
                target_iou >= 0.08
                or (
                    target_iou >= 0.05
                    and target_prox >= 0.70
                    and area_ratio >= 0.12
                )
            )
            yolo_context_score = max(_score(target_iou, 0.02, 0.35), anchored_target_prox * 0.85)
            display_frame_score = _clamp(
                0.36 * boundary_score
                + 0.30 * border_contrast_score
                + 0.20 * rect_score
                + 0.14 * inner_texture_score
            )
            source_score = {
                "edge_contour": 0.50,
                "closed_plane": 0.70,
                "projection_rect": 0.78,
            }.get(source, 0.50)
            slender_penalty = 0.10 if min(bw, bh) / max(1.0, max(bw, bh)) < 0.23 and area_ratio < 0.045 else 0.0
            score = _clamp(
                0.20 * edge_score
                + 0.22 * rect_score
                + 0.16 * area_score
                + 0.18 * yolo_context_score
                + 0.20 * display_frame_score
                + 0.04 * source_score
                - slender_penalty
            )
            if score < 0.22 and display_frame_score < 0.45:
                return
            candidates.append(
                {
                    "candidate_bbox": [int(v) for v in clipped],
                    "bbox": clipped,
                    "candidate_source": source,
                    "candidate_edge_score": float(edge_score),
                    "candidate_rect_score": float(rect_score),
                    "candidate_area_ratio": float(area_ratio),
                    "candidate_target_iou": float(target_iou),
                    "candidate_target_proximity": float(anchored_target_prox),
                    "edge_score": float(edge_score),
                    "rect_score": float(rect_score),
                    "area_score": float(area_score),
                    "boundary_score": float(boundary_score),
                    "display_frame_score": float(display_frame_score),
                    "border_contrast_score": float(border_contrast_score),
                    "inner_texture_score": float(inner_texture_score),
                    "source_score": float(source_score),
                    "target_iou": float(target_iou),
                    "target_proximity": float(anchored_target_prox),
                    "target_related": bool(target_related),
                    "candidate_score": float(score),
                }
            )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 按面积降序，只处理最大的 top-K 个轮廓：媒体边框必然是大轮廓，
        # 小轮廓是纹理噪声。复杂画面轮廓可达上千个，每个候选 ~1ms，不设上限会使
        # a3b 单轮飙到 500ms+ 并持续抢 GIL 拖垮主检测路径。
        _min_area = 0.0015 * frame_area
        _sized = [(float(cv2.contourArea(c)), c) for c in contours]
        _sized = [(a, c) for a, c in _sized if a >= _min_area]
        _sized.sort(key=lambda t: t[0], reverse=True)
        for area, contour in _sized[:40]:
            x, y, bw, bh = cv2.boundingRect(contour)
            peri = float(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True) if peri > 0 else contour
            quad_score = 1.0 if len(approx) == 4 else _clamp(1.0 - abs(len(approx) - 4) / 8.0)
            _add_candidate((x, y, x + bw, y + bh), source="edge_contour", contour_area=area, quad_score_hint=quad_score)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_odd(min(width, height) * 0.018), _odd(min(width, height) * 0.018)),
        )
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_odd(min(width, height) * 0.012), _odd(min(width, height) * 0.012)),
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        closed = cv2.dilate(closed, dilate_kernel, iterations=1)
        closed_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in closed_contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)
            box_area = float(max(1, bw * bh))
            if box_area / max(1.0, frame_area) < 0.012:
                continue
            peri = float(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, 0.025 * peri, True) if peri > 0 else contour
            quad_score = 1.0 if len(approx) == 4 else _clamp(1.0 - abs(len(approx) - 4) / 10.0)
            _add_candidate((x, y, x + bw, y + bh), source="closed_plane", contour_area=area, quad_score_hint=quad_score)

        def _smooth(values: np.ndarray, kernel_size: int) -> np.ndarray:
            kernel_size = max(3, int(kernel_size))
            kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
            return np.convolve(values.astype(np.float32), kernel, mode="same")

        def _peak_lines(values: np.ndarray, limit: int) -> list[tuple[int, float]]:
            min_gap = max(8, int(min(width, height) * 0.035))
            return _projection_peak_lines(values, limit, min_gap)

        sobel_x = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        col_strength = _smooth(np.mean(edge_mask, axis=0) + np.mean(sobel_x, axis=0) / 255.0, max(9, width // 80))
        row_strength = _smooth(np.mean(edge_mask, axis=1) + np.mean(sobel_y, axis=1) / 255.0, max(9, height // 80))
        x_lines = _peak_lines(col_strength, 9)
        y_lines = _peak_lines(row_strength, 9)
        for xi, (x1, xs1) in enumerate(x_lines):
            for x2, xs2 in x_lines[xi + 1:]:
                bw = x2 - x1
                if bw < width * 0.08 or bw > width * 0.96:
                    continue
                for yi, (y1, ys1) in enumerate(y_lines):
                    for y2, ys2 in y_lines[yi + 1:]:
                        bh = y2 - y1
                        if bh < height * 0.08 or bh > height * 0.96:
                            continue
                        line_strength = min(xs1, xs2, ys1, ys2)
                        if line_strength < 0.20:
                            continue
                        _add_candidate(
                            (x1, y1, x2, y2),
                            source="projection_rect",
                            contour_area=None,
                            quad_score_hint=_clamp(0.55 + 0.45 * line_strength),
                        )

        collecting_candidate_specs = False
        pending_candidate_specs = pending_candidate_specs[:64]
        batch_native_results: list[Any] | None = None
        if pending_candidate_specs:
            raw_batch_results = self._native_call(
                "a3b",
                "a3b_boxes_stats",
                edge_mask_u8,
                gray,
                tuple(
                    tuple(int(value) for value in spec["box"])
                    for spec in pending_candidate_specs
                ),
            )
            if raw_batch_results is not None:
                normalized_batch_results = list(raw_batch_results)
                if len(normalized_batch_results) == len(
                    pending_candidate_specs
                ):
                    batch_native_results = normalized_batch_results
                else:
                    self.native_fallback_counts["a3b"] += 1
                    self.native_last_error = (
                        "a3b:a3b_boxes_stats:result_length_mismatch:"
                        f"expected={len(pending_candidate_specs)}:"
                        f"actual={len(normalized_batch_results)}"
                    )
                    self._native_status_dirty = True

        for index, spec in enumerate(pending_candidate_specs):
            native_result = (
                batch_native_results[index]
                if batch_native_results is not None
                else None
            )
            _add_candidate(
                spec["box"],
                source=str(spec["source"]),
                contour_area=spec["contour_area"],
                quad_score_hint=spec["quad_score_hint"],
                native_result_override=native_result,
            )

        def _selection_score(item: dict[str, Any]) -> float:
            box = item["bbox"]
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            aspect_frag = min(bw, bh) / max(1.0, max(bw, bh))
            fragment_penalty = 0.12 if aspect_frag < 0.23 and item["candidate_area_ratio"] < 0.045 else 0.0
            return float(
                item["candidate_score"]
                + 0.12 * item["display_frame_score"]
                + 0.08 * _score(item["candidate_area_ratio"], 0.012, 0.18)
                + (0.05 if item["target_related"] else 0.0)
                + 0.04 * item["source_score"]
                - fragment_penalty
            )

        candidates.sort(key=_selection_score, reverse=True)
        deduped: list[dict[str, Any]] = []
        for candidate in candidates:
            area = _bbox_area(candidate["bbox"])
            keep = True
            for existing in deduped:
                iou = _bbox_iou(candidate["bbox"], existing["bbox"])
                ex_area = _bbox_area(existing["bbox"])
                x1 = max(candidate["bbox"][0], existing["bbox"][0])
                y1 = max(candidate["bbox"][1], existing["bbox"][1])
                x2 = min(candidate["bbox"][2], existing["bbox"][2])
                y2 = min(candidate["bbox"][3], existing["bbox"][3])
                inter = _bbox_area((x1, y1, x2, y2))
                containment = inter / max(1.0, min(area, ex_area))
                if iou >= 0.72 or containment >= 0.90:
                    keep = False
                    break
            if keep:
                deduped.append(candidate)
            if len(deduped) >= 16:
                break
        candidates = deduped
        candidates.sort(
            key=lambda item: (
                _selection_score(item),
                item["display_frame_score"],
                item["candidate_area_ratio"],
            ),
            reverse=True,
        )
        return candidates

    def _update_media_track(self, bbox: tuple[int, int, int, int] | None) -> dict[str, Any]:
        if bbox is None:
            if self.media_track is not None:
                self.media_track.miss_count += 1
                if self.media_track.miss_count > 3:
                    self.media_track = None
            return {
                "stable_count": 0 if self.media_track is None else self.media_track.stable_count,
                "track_score": 0.0,
                "bbox_jitter": 1.0,
                "scale_jitter": 1.0,
                "candidate_lifetime": 0 if self.media_track is None else self.media_track.lifetime,
            }
        area = _bbox_area(bbox)
        if self.media_track is None:
            self.media_track = _MediaTrack(bbox=bbox, last_area=area)
            return {
                "stable_count": 1,
                "track_score": 0.25,
                "bbox_jitter": 0.0,
                "scale_jitter": 0.0,
                "candidate_lifetime": 1,
            }
        iou = _bbox_iou(bbox, self.media_track.bbox)
        old_area = max(1.0, self.media_track.last_area)
        scale_jitter = abs(area - old_area) / old_area
        bbox_jitter = 1.0 - iou
        if iou >= 0.35:
            self.media_track.stable_count += 1
            self.media_track.lifetime += 1
            self.media_track.miss_count = 0
        else:
            self.media_track.stable_count = 1
            self.media_track.lifetime = 1
            self.media_track.miss_count = 0
        self.media_track.bbox = bbox
        self.media_track.last_area = area
        track_score = _clamp(
            0.55 * _score(self.media_track.stable_count, 1.0, 5.0)
            + 0.25 * iou
            + 0.20 * (1.0 - min(1.0, scale_jitter))
        )
        return {
            "stable_count": int(self.media_track.stable_count),
            "track_score": float(track_score),
            "bbox_jitter": float(bbox_jitter),
            "scale_jitter": float(scale_jitter),
            "candidate_lifetime": int(self.media_track.lifetime),
        }

    def _media_l2_validation(
        self,
        candidate: dict[str, Any] | None,
        flow: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, float]:
        if candidate is None or not flow["available"]:
            return {
                "plane_score": 0.0,
                "warp_residual": 0.0,
                "flow_gap": 0.0,
                "inside_motion": 0.0,
                "outside_motion": 0.0,
                "inner_outer_motion_ratio": 0.0,
                "homography_inlier_ratio": 0.0,
            }
        box = candidate["bbox"]
        x1, y1, x2, y2 = box
        mag = flow["mag"]
        fs = flow.get("flow_scale", 1.0)
        fx1, fy1, fx2, fy2 = int(x1*fs), int(y1*fs), int(x2*fs), int(y2*fs)
        inside = mag[fy1:fy2, fx1:fx2]
        inside_motion = float(np.mean(inside)) if inside.size else 0.0
        outer_box = _expand_box(box, width, height, 0.35)
        ox1, oy1, ox2, oy2 = outer_box
        fox1, foy1, fox2, foy2 = int(ox1*fs), int(oy1*fs), int(ox2*fs), int(oy2*fs)
        ring = mag[foy1:foy2, fox1:fox2].copy()
        if ring.size:
            ring[fy1-foy1:fy2-foy1, fx1-fox1:fx2-fox1] = np.nan
            outside_motion = float(np.nanmean(ring)) if np.isfinite(ring).any() else 0.0
        else:
            outside_motion = 0.0
        flow_gap = abs(inside_motion - outside_motion)
        inner_outer_ratio = inside_motion / max(0.05, outside_motion)
        # Compatibility field name retained for consumers. The rebuilt path
        # does not run KLT/RANSAC homography here; this is a plane-likeness
        # heuristic from rectangle, edge, and inside/outside flow evidence.
        homography_inlier_ratio = _clamp(
            0.50 * candidate["rect_score"]
            + 0.25 * candidate["edge_score"]
            # 算法修复：flow_gap 作正向单边证据（差异越大越可疑），
            # 移除原来的 (1-flow_gap/3) 反向项，消除"静止背景=高分"的误报根源
            + 0.25 * _clamp((flow_gap - 0.30) / 0.70)
        )
        residual_patch = flow["residual_mag"][fy1:fy2, fx1:fx2]
        warp_residual = float(np.mean(residual_patch) / 5.0) if residual_patch.size else 0.0
        return {
            "plane_score": float(homography_inlier_ratio),
            "warp_residual": float(warp_residual),
            "flow_gap": float(flow_gap),
            "inside_motion": float(inside_motion),
            "outside_motion": float(outside_motion),
            "inner_outer_motion_ratio": float(inner_outer_ratio),
            "homography_inlier_ratio": float(homography_inlier_ratio),
        }

    def _apply_media_policy(
        self,
        *,
        p_media_raw: float,
        p_media_type: str,
        bbox: tuple[int, int, int, int] | None,
        target_related: bool,
        strong_evidence: bool,
        scores: dict[str, float],
        flow: dict[str, Any],
        exposure: dict[str, Any],
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        suppressed_reason = "none"
        score_cap = 1.0
        background_static_suppressed = False
        if bbox is None:
            score_cap = 0.0
            suppressed_reason = "no_media_candidate"
        else:
            x1, y1, x2, y2 = bbox
            area_ratio = _bbox_area(bbox) / max(1.0, width * height)
            candidate_area_ratio = max(float(area_ratio), float(scores.get("area_ratio", 0.0)))
            display_frame = float(scores.get("display_frame", 0.0))
            source_score = float(scores.get("source_score", 0.0))
            screen_like_evidence = bool(
                p_media_raw >= self.theta_media_raw
                and p_media_type in ("screen_replay", "paper_photo", "static_image_spoof", "flat_media_candidate")
                and display_frame >= 0.52
                and scores.get("rect", 0.0) >= 0.45
                and candidate_area_ratio >= 0.018
                and (
                    strong_evidence
                    or source_score >= 0.70
                    or candidate_area_ratio >= 0.040
                    or scores.get("flow_gap", 0.0) >= 1.0
                    or scores.get("warp_residual", 0.0) >= 0.24
                )
                and not (
                    scores.get("target_iou", 0.0) >= 0.55
                    and candidate_area_ratio < 0.040
                )
            )
            # The authoritative A3b clip contains a clearly bounded phone
            # display whose internal person detections make the display plane
            # look "target-near".  Camera motion can also make the same stable
            # display look like a background window.  Do not let those scene
            # heuristics veto a physically strong display plane: the edge band
            # deliberately excludes the saturated high-edge natural structures
            # seen in the fixed outdoor normal clip.
            robust_display_plane = bool(
                strong_evidence
                and p_media_raw >= self.theta_media_raw
                and display_frame >= 0.70
                and scores.get("border_contrast", 0.0) >= 0.80
                and 0.40 <= scores.get("edge", 0.0) <= 0.62
                and scores.get("rect", 0.0) >= 0.60
                and source_score >= 0.70
                and candidate_area_ratio >= 0.08
            )
            touches_edge = x1 <= 4 or y1 <= 4 or x2 >= width - 4 or y2 >= height - 4
            large_background_plane = bool(
                not target_related
                and area_ratio >= 0.18
                and not screen_like_evidence
            )
            camera_translation = bool(
                flow.get("global_motion_weight", 0.0) >= 0.70
                and scores.get("yolo_context", 0.0) < 0.25
                and not screen_like_evidence
            )
            border_or_letterbox = bool(
                touches_edge
                and not target_related
                and area_ratio >= 0.08
                and not screen_like_evidence
            )
            weak_background = bool(
                not target_related
                and not strong_evidence
                and not screen_like_evidence
                and scores.get("flow_gap", 0.0) < 0.45
                and scores.get("warp_residual", 0.0) < 0.10
            )
            weak_background_plane = bool(
                not target_related
                and not screen_like_evidence
                and scores.get("yolo_context", 0.0) < 0.10
                and scores.get("edge", 0.0) < 0.45
                and not (
                    scores.get("flow_gap", 0.0) >= 1.0
                    or scores.get("warp_residual", 0.0) >= 0.24
                )
            )
            unanchored_small_media_plane = bool(
                not target_related
                and not screen_like_evidence
                and scores.get("yolo_context", 0.0) < 0.15
                and candidate_area_ratio < 0.020
            )
            proximity_only_background = bool(
                not screen_like_evidence
                and scores.get("target_iou", 0.0) < 0.02
                and scores.get("flow_gap", 0.0) < 0.45
                and scores.get("warp_residual", 0.0) < 0.10
            )
            tiny_media_fragment = bool(
                candidate_area_ratio < (0.025 if not target_related else 0.018)
                and scores.get("target_iou", 0.0) < 0.30
                and (display_frame < 0.62 or not target_related)
            )
            target_attached_patch = bool(
                target_related
                and not screen_like_evidence
                and (
                    (
                        scores.get("plane", 0.0) < 0.35
                        and max(a1["a1_feature_score"], a2["a2_feature_score"], a3["a3_feature_score"]) >= 0.55
                    )
                    or (
                        scores.get("target_iou", 0.0) >= 0.50
                        and candidate_area_ratio < 0.030
                        and scores.get("flow_gap", 0.0) < 0.50
                    )
                )
            )
            target_attached_small_occluder = bool(
                target_related
                and not screen_like_evidence
                and scores.get("target_iou", 0.0) >= 0.35
                and candidate_area_ratio < 0.030
            )
            proximity_only_target_edge = bool(
                target_related
                and not screen_like_evidence
                and scores.get("target_iou", 0.0) < 0.08
                and scores.get("target_proximity", 0.0) >= 0.50
                and candidate_area_ratio < 0.030
                and scores.get("flow_gap", 0.0) < 0.60
                and scores.get("warp_residual", 0.0) < 0.15
            )
            glare_or_texture_adv = bool(
                target_related
                and not screen_like_evidence
                and max(a1["a1_feature_score"], a2["a2_feature_score"], a3["a3_feature_score"]) >= 0.62
                and scores.get("target_iou", 0.0) >= 0.65
                and scores.get("flow_gap", 0.0) < 1.20
                and scores.get("warp_residual", 0.0) < 0.24
            )
            competing_adv_evidence = bool(
                not screen_like_evidence
                and max(a1["a1_feature_score"], a2["a2_feature_score"], a3["a3_feature_score"]) >= 0.70
                and scores.get("flow_gap", 0.0) < 1.0
                and scores.get("warp_residual", 0.0) < 0.24
            )
            natural_scene_texture = bool(
                scores.get("edge", 0.0) >= 0.84
                and scores.get("target_iou", 0.0) < 0.12
                and scores.get("flow_gap", 0.0) < 0.45
                and scores.get("warp_residual", 0.0) < 0.065
                and not (
                    candidate_area_ratio >= 0.30
                    and display_frame >= 0.66
                    and scores.get("edge", 0.0) < 0.55
                )
            )
            target_near_small_scene_plane = bool(
                target_related
                and (
                    (
                        candidate_area_ratio < 0.18
                        and scores.get("target_iou", 0.0) < 0.12
                    )
                    or (
                        candidate_area_ratio < 0.10
                        and scores.get("target_iou", 0.0) < 0.30
                    )
                    or (
                        p_media_type == "screen_replay"
                        and candidate_area_ratio < 0.06
                        and scores.get("target_iou", 0.0) < 0.38
                        and scores.get("track", 0.0) < 0.55
                    )
                )
            )
            target_attached_small_screen_fragment = bool(
                target_related
                and candidate_area_ratio < 0.070
                and scores.get("target_iou", 0.0) >= 0.25
                and (
                    scores.get("border_contrast", 0.0) < 0.65
                    or scores.get("boundary", 0.0) < 0.08
                    or display_frame < 0.58
                    or source_score <= 0.55
                    or scores.get("track", 0.0) < 0.25
                )
            )
            target_attached_weak_display_plane = bool(
                target_related
                and candidate_area_ratio < 0.120
                and scores.get("target_iou", 0.0) >= 0.25
                and display_frame < 0.56
                and scores.get("boundary", 0.0) < 0.18
            )
            target_proximity_scene_plane = bool(
                target_related
                and scores.get("target_iou", 0.0) < 0.25
                and scores.get("target_proximity", 0.0) >= 0.50
                and 0.055 <= candidate_area_ratio < 0.240
                and display_frame < 0.64
                and scores.get("edge", 0.0) < 0.45
                and scores.get("flow_gap", 0.0) < 0.55
                and scores.get("warp_residual", 0.0) < 0.14
                and not (
                    display_frame >= 0.70
                    and scores.get("border_contrast", 0.0) >= 0.80
                    and scores.get("edge", 0.0) >= 0.50
                )
            )
            target_adjacent_low_edge_scene_plane = bool(
                target_related
                and candidate_area_ratio < 0.200
                and display_frame < 0.56
                and scores.get("edge", 0.0) < 0.46
                and scores.get("boundary", 0.0) < 0.22
                and scores.get("flow_gap", 0.0) < 0.75
                and scores.get("warp_residual", 0.0) < 0.17
                and not (
                    display_frame >= 0.64
                    and scores.get("border_contrast", 0.0) >= 0.80
                    and scores.get("edge", 0.0) >= 0.50
                )
            )
            unanchored_high_motion_fragment = bool(
                not target_related
                and candidate_area_ratio < 0.040
                and (
                    scores.get("flow_gap", 0.0) >= 1.20
                    or scores.get("warp_residual", 0.0) >= 0.35
                )
            )
            low_display_target_plane = bool(
                target_related
                and candidate_area_ratio >= 0.035
                and display_frame < 0.50
                and scores.get("border_contrast", 0.0) < 0.70
                and not (
                    candidate_area_ratio >= 0.30
                    and scores.get("border_contrast", 0.0) >= 0.85
                )
            )
            adv_explained_glare_or_texture_plane = bool(
                max(a1["a1_feature_score"], a2["a2_feature_score"]) >= 0.78
                and (a1["a1_feature_score"] + a2["a2_feature_score"]) >= 1.45
                and p_media_type in ("paper_photo", "screen_replay", "flat_media_candidate")
                and display_frame < 0.64
                and scores.get("boundary", 0.0) < 0.22
            )
            low_edge_glare_texture_plane = bool(
                target_related
                and p_media_type in ("paper_photo", "screen_replay", "flat_media_candidate")
                and max(a1["a1_feature_score"], a2["a2_feature_score"]) >= 0.60
                and display_frame < 0.60
                and scores.get("boundary", 0.0) < 0.18
                and scores.get("edge", 0.0) < 0.24
                and scores.get("flow_gap", 0.0) < 0.40
                and scores.get("warp_residual", 0.0) < 0.12
            )
            edge_attached_motion_plane = bool(
                touches_edge
                and target_related
                and p_media_type == "screen_replay"
                and candidate_area_ratio >= 0.20
                and scores.get("boundary", 0.0) < 0.15
                and display_frame < 0.60
            )
            background_window_camera_motion = bool(
                not target_related
                and candidate_area_ratio >= 0.12
                and display_frame >= 0.53
                and scores.get("inner_outer_motion_ratio", 0.0) < 0.90
            )
            background_transient_motion_plane = bool(
                not target_related
                and scores.get("warp_residual", 0.0) >= 0.16
                and not (
                    candidate_area_ratio >= 0.30
                    and display_frame >= 0.66
                    and scores.get("edge", 0.0) < 0.55
                    and scores.get("inner_outer_motion_ratio", 0.0) >= 1.0
                )
            )
            framewide_visibility_plane = bool(
                touches_edge
                and candidate_area_ratio >= 0.50
                and display_frame < 0.35
                and scores.get("border_contrast", 0.0) < 0.10
            )
            no_target_high_edge_scene_motion = bool(
                not target_related
                and candidate_area_ratio < 0.25
                and scores.get("edge", 0.0) >= 0.58
                and scores.get("warp_residual", 0.0) >= 0.12
                and display_frame < 0.75
            )
            if border_or_letterbox:
                score_cap = min(score_cap, 0.30)
                suppressed_reason = "border_or_letterbox"
            if camera_translation:
                score_cap = min(score_cap, 0.38)
                suppressed_reason = "camera_translation_edge"
            if large_background_plane:
                score_cap = min(score_cap, 0.35)
                suppressed_reason = "background_large_plane"
                background_static_suppressed = True
            if weak_background:
                score_cap = min(score_cap, 0.22)
                suppressed_reason = "background_static_weak_evidence"
                background_static_suppressed = True
            if weak_background_plane:
                score_cap = min(score_cap, 0.25)
                suppressed_reason = "background_plane_without_target_context"
                background_static_suppressed = True
            if unanchored_small_media_plane:
                score_cap = min(score_cap, 0.48)
                suppressed_reason = "unanchored_small_media_plane"
                background_static_suppressed = True
            if proximity_only_background:
                score_cap = min(score_cap, 0.22)
                suppressed_reason = "proximity_only_background_edge"
                background_static_suppressed = True
            if natural_scene_texture:
                score_cap = min(score_cap, 0.34)
                suppressed_reason = "natural_scene_texture_plane"
                background_static_suppressed = True
            if target_near_small_scene_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_near_small_scene_plane"
            if target_attached_small_screen_fragment:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_attached_small_screen_fragment_prefers_A1_A2_A3"
            if target_attached_weak_display_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_attached_weak_display_plane_prefers_A1_A2_A3"
            if target_proximity_scene_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_proximity_scene_plane_prefers_A1_A2_A3"
            if target_adjacent_low_edge_scene_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_adjacent_low_edge_scene_plane_prefers_A1_A2_A3"
            if unanchored_high_motion_fragment:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "unanchored_high_motion_fragment"
                background_static_suppressed = True
            if low_display_target_plane:
                score_cap = min(score_cap, 0.46)
                suppressed_reason = "low_display_target_plane_prefers_A1_A2_A3"
            if adv_explained_glare_or_texture_plane:
                score_cap = min(score_cap, 0.46)
                suppressed_reason = "adv_explained_glare_or_texture_plane"
            if low_edge_glare_texture_plane:
                score_cap = min(score_cap, 0.46)
                suppressed_reason = "low_edge_glare_texture_plane_prefers_A1_A2"
            if edge_attached_motion_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "edge_attached_motion_plane"
            if background_window_camera_motion:
                score_cap = min(score_cap, 0.38)
                suppressed_reason = "background_window_camera_motion"
                background_static_suppressed = True
            if background_transient_motion_plane:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "background_transient_motion_plane"
                background_static_suppressed = True
            if framewide_visibility_plane:
                score_cap = min(score_cap, 0.34)
                suppressed_reason = "framewide_visibility_plane"
                background_static_suppressed = True
            if no_target_high_edge_scene_motion:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "no_target_high_edge_scene_motion"
                background_static_suppressed = True
            if tiny_media_fragment:
                score_cap = min(score_cap, 0.30)
                suppressed_reason = "tiny_media_fragment"
                background_static_suppressed = True
            if target_attached_patch:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_attached_patch_prefers_A1_A2_A3"
            if target_attached_small_occluder:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "target_attached_small_occluder_prefers_A1_A2_A3"
            if proximity_only_target_edge:
                score_cap = min(score_cap, 0.42)
                suppressed_reason = "proximity_only_target_edge"
            if glare_or_texture_adv:
                score_cap = min(score_cap, 0.46)
                suppressed_reason = "glare_or_texture_adv_prefers_A1_A2"
            if competing_adv_evidence:
                score_cap = min(score_cap, 0.46)
                suppressed_reason = "base_adv_evidence_prefers_A1_A2_A3"
            if exposure.get("high_false_positive_scene", False) and not target_related:
                score_cap = min(score_cap, 0.38)
                suppressed_reason = "global_exposure_scene_media_suppressed"
            if screen_like_evidence and suppressed_reason in (
                "background_large_plane",
                "background_static_weak_evidence",
                "background_plane_without_target_context",
                "unanchored_small_media_plane",
                "proximity_only_background_edge",
                "base_adv_evidence_prefers_A1_A2_A3",
                "global_exposure_scene_media_suppressed",
            ):
                score_cap = max(score_cap, 0.72)
                suppressed_reason = "none"
                background_static_suppressed = False
            if robust_display_plane and suppressed_reason in (
                "target_near_small_scene_plane",
                "background_window_camera_motion",
                "background_transient_motion_plane",
            ):
                score_cap = max(score_cap, 0.72)
                suppressed_reason = "none"
                background_static_suppressed = False
            # 纯静止背景最终防线（对应生产代码 _merge_static_image 的 score_cap=0.08 逻辑）
            # 放在 screen_like_evidence 覆写之后，不可被绕过。
            # 判定依据：no target-related + 无运动差异 + 无翻拍物理证据。
            # 不加 yolo_context 条件 —— not target_related 已保证无真实目标重叠，
            # proximity-only 的接近度不能阻止纯静止背景被压制。
            if (
                not target_related
                and not strong_evidence
                and scores.get("flow_gap", 0.0) < 0.25
                and scores.get("warp_residual", 0.0) < 0.10
            ):
                score_cap = min(score_cap, 0.08)
                background_static_suppressed = True
                suppressed_reason = "pure_static_background"
        # 方案二：消除 p_media 在抑制上限处的"死平线"（如 framewide_visibility_plane 的 0.34）。
        # - 当 score_cap >= theta_media（未把决策压到阈下）：保持精确 min，触发区一字不动。
        # - 当 score_cap < theta_media（已被抑制到阈值以下、本就不可能触发）：用 cap*tanh(raw/cap)
        #   做连续软饱和——随 raw 单调变化、始终 < cap < 阈值，去掉平线但不改变任何触发/告警决策。
        _cap = float(score_cap)
        _raw = float(p_media_raw)
        if _cap >= self.theta_media:
            p_media_policy = min(_raw, _cap)
        elif _cap > 1e-6:
            p_media_policy = float(_cap * math.tanh(_raw / _cap))
        else:
            p_media_policy = min(_raw, _cap)
        media_candidate_allowed = bool(
            p_media_policy >= self.theta_media
            and suppressed_reason in ("none",)
        )
        a3b_state = "normal"
        if p_media_raw >= self.theta_media_raw and not media_candidate_allowed:
            a3b_state = "suppressed"
        if media_candidate_allowed:
            a3b_state = "candidate"
        return {
            "p_media_policy": float(p_media_policy),
            "suppressed_reason": suppressed_reason,
            "score_cap": float(score_cap),
            "media_candidate_allowed": bool(media_candidate_allowed),
            "background_static_suppressed": bool(background_static_suppressed),
            "a3b_state": a3b_state,
        }

    def _a3b_result_source_frame_units(
        self,
        a3b: dict[str, Any],
    ) -> int:
        """Convert one new A3b result into source-frame-equivalent coverage.

        ``a3b_result_seq`` remains the dedupe key.  Temporal progress is derived
        from the result's source lineage so static-image interval, worker delay,
        and a small number of dropped detector frames do not silently redefine
        ``media_run`` or the confirmation window.
        """

        interval_units = max(
            1,
            int(a3b.get("a3b_source_interval_frames", 1) or 1),
        )
        source_frame_idx: int | None = None
        try:
            raw_frame_idx = a3b.get("a3b_source_frame_idx")
            if raw_frame_idx is not None and not isinstance(
                raw_frame_idx,
                bool,
            ):
                source_frame_idx = int(raw_frame_idx)
        except (TypeError, ValueError):
            source_frame_idx = None

        source_timestamp: float | None = None
        try:
            raw_timestamp = a3b.get("a3b_source_timestamp")
            if raw_timestamp is not None:
                candidate_timestamp = float(raw_timestamp)
                if math.isfinite(candidate_timestamp):
                    source_timestamp = candidate_timestamp
        except (TypeError, ValueError):
            source_timestamp = None

        units = interval_units
        previous_frame_idx = self._a3b_last_consumed_source_frame_idx
        if (
            source_frame_idx is not None
            and previous_frame_idx is not None
            and source_frame_idx > previous_frame_idx
        ):
            units = max(units, source_frame_idx - previous_frame_idx)

        source_fps = 0.0
        try:
            source_fps = float(a3b.get("a3b_source_fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            source_fps = 0.0
        previous_timestamp = self._a3b_last_consumed_source_timestamp
        if (
            source_timestamp is not None
            and previous_timestamp is not None
            and source_timestamp > previous_timestamp
            and math.isfinite(source_fps)
            and source_fps > 0.0
        ):
            timestamp_units = int(
                round((source_timestamp - previous_timestamp) * source_fps)
            )
            units = max(units, timestamp_units)

        self._a3b_last_consumed_source_frame_idx = source_frame_idx
        self._a3b_last_consumed_source_timestamp = source_timestamp
        return max(1, int(units))

    def _joint_decision(
        self,
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        a4: dict[str, Any],
        a3b: dict[str, Any],
        rois: list[ROI],
        exposure: dict[str, Any],
        flow: dict[str, Any],
        ta_result: dict[str, Any] | None = None,
        blinding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        blinding = blinding or {}
        adv_threshold = float(
            a4.get("a4_decision_threshold", self.theta_adv)
        )
        classifier_adv_rescue_requested = bool(
            a4.get("a4_classifier_used", False)
            and a4.get("a4_patch_baseline_ready", False)
            and a4.get("a4_classifier_triggered", False)
        )
        classifier_adv_rescue_dark_scene_blocked = bool(
            classifier_adv_rescue_requested
            and float(exposure.get("underexposed_ratio", 0.0))
            >= self._a4_classifier_rescue_underexposed_max
        )
        classifier_adv_rescue = bool(
            classifier_adv_rescue_requested
            and not classifier_adv_rescue_dark_scene_blocked
        )
        window_frames = int(math.ceil(float(self.process_fps) * 0.5))
        window_frames = int(max(3, min(8, window_frames)))
        adv_hit_required = int(math.ceil(window_frames * (0.67 if exposure["high_false_positive_scene"] else 0.60)))
        media_scores = a3b.get("p_media_scores", {})
        media_screen_window_evidence = bool(
            a3b.get("p_media_strong_evidence", False)
            and float(a3b.get("p_media_policy", 0.0)) >= self.theta_media
            and str(a3b.get("p_media_type", "normal")) in (
                "screen_replay",
                "paper_photo",
                "static_image_spoof",
                "flat_media_candidate",
            )
            and float(media_scores.get("display_frame", 0.0)) >= 0.52
            and float(media_scores.get("area_ratio", 0.0)) >= 0.018
        )
        robust_media_evidence = bool(
            float(a3b.get("p_media_policy", 0.0)) >= self.theta_media
            and float(media_scores.get("display_frame", 0.0)) >= 0.64
            and float(media_scores.get("area_ratio", 0.0)) >= 0.10
            and (
                float(media_scores.get("border_contrast", 0.0)) >= 0.75
                or float(media_scores.get("boundary", 0.0)) >= 0.25
            )
        )
        media_hit_required = (
            max(2, int(math.ceil(window_frames * 0.50)))
            if (
                a3b["p_media_target_related"]
                and a3b["p_media_strong_evidence"]
            ) or media_screen_window_evidence
            else int(math.ceil(window_frames * 0.67))
        )

        max_feature = max(
            float(a1["a1_feature_score"]),
            float(a2["a2_feature_score"]),
            float(a3["a3_feature_score"]),
        )
        target_related_feature = bool(
            a1.get("target_related", False)
            or a2.get("target_related", False)
            or a3.get("target_related", False)
        )
        recent_targets = sum(self.recent_target_presence) >= 2
        no_target_fallback = bool(
            not rois
            and recent_targets
            and max_feature >= 0.86
            and (
                exposure["overexposure_ratio"] >= 0.25
                or exposure["underexposed_ratio"] >= 0.75
                or a3["flow_residual"] >= 2.8
            )
        )
        dominant_adv = str(a4.get("dominant_adv_input", "A4_MIXED"))
        unsupported_a3_motion = bool(
            dominant_adv == "A3_FLOW_ARTIFACT"
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.18
            and float(a3.get("flow_max_magnitude_norm", 0.0)) < 0.95
            and max(float(a1["a1_feature_score"]), float(a2["a2_feature_score"])) < 0.55
            and exposure["exposure_delta"] < 0.08
        )
        unsupported_a2_motion = bool(
            dominant_adv == "A2_LBP_TEMPORAL"
            and float(a1["a1_feature_score"]) < 0.30
            and float(a3["a3_feature_score"]) < 0.45
            and exposure["exposure_delta"] < 0.08
            and not bool(a2.get("flash_like", False))
        )
        normal_motion_texture_change = bool(
            float(a2.get("change_t_global", 0.0)) >= 0.22
            and float(a2.get("change_t_local_max", 0.0)) < 0.60
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.12
            and exposure["frame_diff_global"] < 0.06
            and exposure["overexposure_ratio"] < 0.08
            and exposure["underexposed_ratio"] < 0.20
        )
        nonlocal_a1_a3_scene_spike = bool(
            dominant_adv in ("A1_LBP_SINGLE", "A3_FLOW_ARTIFACT")
            and float(a1["a1_feature_score"]) >= 0.55
            and float(a3["a3_feature_score"]) >= 0.40
            and float(a2["a2_feature_score"]) < 0.45
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.18
            and float(a3.get("flow_max_magnitude_norm", 0.0)) < 0.75
            and exposure["frame_diff_global"] < 0.04
            and exposure["overexposure_ratio"] < 0.02
            and exposure["underexposed_ratio"] < 0.20
        )
        a3b_reason = str(a3b.get("suppressed_reason", "none"))
        a3b_indicates_physical_patch = a3b_reason in {
            "target_attached_patch_prefers_A1_A2_A3",
            "target_attached_small_occluder_prefers_A1_A2_A3",
            "target_attached_small_screen_fragment_prefers_A1_A2_A3",
            "target_attached_weak_display_plane_prefers_A1_A2_A3",
            "target_proximity_scene_plane_prefers_A1_A2_A3",
            "target_adjacent_low_edge_scene_plane_prefers_A1_A2_A3",
            "low_display_target_plane_prefers_A1_A2_A3",
            "adv_explained_glare_or_texture_plane",
        }
        media_display = float(media_scores.get("display_frame", 0.0))
        media_boundary = float(media_scores.get("boundary", 0.0))
        media_area_ratio = float(media_scores.get("area_ratio", 0.0))
        visibility_texture_probe = False
        visibility_texture_rescue = bool(
            bool(a1.get("a1_visibility_hold_active", False))
            and bool(a4.get("p_adv_triggered", False))
            and float(a1["a1_feature_score"]) >= 0.55
            and float(a1.get("delta_h_roi_patch_max", 0.0)) >= 0.58
            and float(a1.get("delta_h_patch_concentration", 0.0)) >= 0.72
            and exposure["frame_diff_global"] < 0.012
            and exposure["exposure_delta"] < 0.015
            and a3b_reason in {
                "framewide_visibility_plane",
                "low_display_target_plane_prefers_A1_A2_A3",
                "target_attached_patch_prefers_A1_A2_A3",
                "target_attached_small_screen_fragment_prefers_A1_A2_A3",
                "base_adv_evidence_prefers_A1_A2_A3",
                "tiny_media_fragment",
            }
        )
        physical_patch_rescue_signal = bool(
            a3b_indicates_physical_patch
            and (
                media_boundary < 0.18
                or media_display < 0.58
                or (
                    a3b_reason == "low_display_target_plane_prefers_A1_A2_A3"
                    and media_display < 0.52
                    and media_area_ratio >= 0.06
                )
            )
        )
        low_motion_background_like_adv = bool(
            exposure["frame_diff_global"] < 0.018
            and exposure["exposure_delta"] < 0.04
            and max_feature <= 0.60
            and a3b_reason in {
                "target_near_small_scene_plane",
                "background_window_camera_motion",
                "background_transient_motion_plane",
                "natural_scene_texture_plane",
                "background_plane_without_target_context",
                "target_attached_small_screen_fragment_prefers_A1_A2_A3",
                "target_attached_weak_display_plane_prefers_A1_A2_A3",
                "target_attached_small_occluder_prefers_A1_A2_A3",
                "target_proximity_scene_plane_prefers_A1_A2_A3",
                "target_adjacent_low_edge_scene_plane_prefers_A1_A2_A3",
                "low_display_target_plane_prefers_A1_A2_A3",
            }
            and not physical_patch_rescue_signal
        )
        baseline_ready = bool(self.lbp_baseline_samples >= 8)
        physical_patch_rescue_evidence = bool(
            exposure["frame_diff_global"] >= 0.010
            or exposure["exposure_delta"] >= 0.020
            or float(a1["a1_feature_score"]) >= 0.60
            or float(a2["a2_feature_score"]) >= 0.55
            or float(a3.get("flow_residual_contrast", 0.0)) >= 1.20
            or bool(a3.get("a3_residual_hold_active", False))
            or visibility_texture_rescue
        )
        adv_multi_evidence_rescue = bool(
            physical_patch_rescue_signal
            and bool(a4.get("p_adv_triggered", False))
            and max_feature >= 0.40
            and float(a4.get("a4_multi_evidence", 0.0)) >= 0.34
            and physical_patch_rescue_evidence
        )
        a3_residual_fallback_floor = 0.74 if bool(a3.get("a3_residual_hold_active", False)) else 0.86
        a3_residual_fallback = bool(
            float(a3["a3_feature_score"]) >= a3_residual_fallback_floor
            and (
                float(a3.get("flow_residual_contrast", 0.0)) >= 1.20
                or bool(a3.get("a3_residual_hold_active", False))
            )
            and physical_patch_rescue_evidence
            and a3b_reason in {
                "base_adv_evidence_prefers_A1_A2_A3",
                "target_attached_patch_prefers_A1_A2_A3",
                "target_attached_small_occluder_prefers_A1_A2_A3",
                "target_attached_weak_display_plane_prefers_A1_A2_A3",
                "target_proximity_scene_plane_prefers_A1_A2_A3",
                "target_adjacent_low_edge_scene_plane_prefers_A1_A2_A3",
                "low_display_target_plane_prefers_A1_A2_A3",
            }
        )
        a3_only_background_motion = bool(
            dominant_adv == "A3_FLOW_ARTIFACT"
            and float(a1["a1_feature_score"]) < 0.25
            and float(a2["a2_feature_score"]) <= 0.34
            and float(a3["a3_feature_score"]) >= 0.80
            and exposure["exposure_delta"] < 0.006
            and a3b_reason in {
                "background_window_camera_motion",
                "edge_attached_motion_plane",
                "background_transient_motion_plane",
                "framewide_visibility_plane",
            }
        )
        cold_start_low_motion_adv = bool(
            not baseline_ready
            and exposure["frame_diff_global"] < 0.012
            and exposure["exposure_delta"] < 0.020
            and not adv_multi_evidence_rescue
        )
        stationary_texture_only_adv = bool(
            exposure["frame_diff_global"] < 0.008
            and exposure["exposure_delta"] < 0.015
            and max_feature < 0.58
            and float(a2["a2_feature_score"]) <= 0.30
            and float(a3["a3_feature_score"]) <= 0.45
            and not adv_multi_evidence_rescue
            and not visibility_texture_rescue
        )
        background_plane_adv = bool(
            a3b_reason in {
                "background_transient_motion_plane",
                "background_window_camera_motion",
                "background_plane_without_target_context",
                "natural_scene_texture_plane",
            }
            and not bool(a3b.get("p_media_target_related", False))
            and media_area_ratio >= 0.10
            and media_display < 0.76
            and max_feature < 0.90
            and not robust_media_evidence
            and not adv_multi_evidence_rescue
        )
        structural_adv_evidence_rescue = False
        # 场景自适应基线抑制（P0）：用 z-score 对**运动场景也生效**，压制干净高能工地场景误报。
        # 当前最强特征分在本场景滚动分布的正常带内(z<2 且不远超 p80)即视为场景固有高能纹理/运动，
        # 抑制支路A候选；真实攻击会显著超出本场景分布(z≫2)不被误抑。adv 多证据/可见性救援时不抑。
        scene_baseline_normal = False
        if self._scene_baseline_enabled and len(self._sb_maxfeat) >= self._scene_baseline_min:
            arr = np.fromiter(self._sb_maxfeat, dtype=np.float64)
            mu = float(arr.mean())
            sd = float(arr.std())
            ref_p80 = float(np.percentile(arr, 80))
            z = (max_feature - mu) / max(sd, 0.04)
            static_normal = bool(
                max_feature <= ref_p80 + 0.06
                and exposure["frame_diff_global"] < 0.06
                and exposure["exposure_delta"] < 0.05
            )
            moving_normal = bool(z < 2.0 and max_feature <= ref_p80 + 0.10)
            scene_baseline_normal = bool(
                (static_normal or moving_normal)
                and not adv_multi_evidence_rescue
                and not structural_adv_evidence_rescue
                and not visibility_texture_rescue
                and not classifier_adv_rescue
            )
        adv_candidate_allowed = bool(
            (
                bool(a4.get("p_adv_triggered", False))
                or structural_adv_evidence_rescue
            )
            and (max_feature >= 0.48 or adv_multi_evidence_rescue or visibility_texture_rescue)
            and (
                target_related_feature
                or no_target_fallback
                or a3_residual_fallback
                or adv_multi_evidence_rescue
                or visibility_texture_rescue
            )
            and not (
                flow.get("global_motion_weight", 0.0) >= 0.82
                and max_feature < 0.82
            )
            and not (unsupported_a3_motion and not a3_residual_fallback)
            and not unsupported_a2_motion
            and not normal_motion_texture_change
            and not nonlocal_a1_a3_scene_spike
            and not low_motion_background_like_adv
            and not cold_start_low_motion_adv
            and not stationary_texture_only_adv
            and not background_plane_adv
            and not a3_only_background_motion
            and not scene_baseline_normal
        )
        rule_adv_candidate_allowed = bool(adv_candidate_allowed)
        if classifier_adv_rescue:
            adv_candidate_allowed = True
        localized_patch_context = bool(
            a3b_reason in {
                "target_attached_patch_prefers_A1_A2_A3",
                "target_attached_small_occluder_prefers_A1_A2_A3",
                "target_attached_small_screen_fragment_prefers_A1_A2_A3",
                "target_attached_weak_display_plane_prefers_A1_A2_A3",
            }
        )
        localized_a1_attack_support = bool(
            float(a1["a1_feature_score"]) >= 0.78
            and float(
                a1.get("delta_h_roi_patch_max", 0.0)
            ) >= 0.55
            and float(
                a1.get("delta_h_patch_concentration", 0.0)
            ) >= 0.70
            and localized_patch_context
        )
        glare_attack_support = bool(
            float(
                exposure.get("overexposure_ratio", 0.0)
            ) >= 0.10
            and float(
                exposure.get("underexposed_ratio", 0.0)
            ) < 0.10
            and max(
                float(a1["a1_feature_score"]),
                float(a2["a2_feature_score"]),
            ) >= 0.78
        )
        photometric_attack_support = bool(
            glare_attack_support
            or (
                bool(a2.get("flash_like", False))
                and float(a2["a2_feature_score"]) >= 0.55
                and (
                    float(
                        exposure.get("exposure_delta", 0.0)
                    ) >= 0.020
                    or float(
                        exposure.get("overexposure_ratio", 0.0)
                    ) >= 0.10
                )
            )
        )
        a3_independent_attack_support = bool(
            localized_a1_attack_support
            or photometric_attack_support
            or a3_residual_fallback
            or no_target_fallback
        )
        normal_articulated_target_motion = bool(
            target_related_feature
            and (
                float(a1["a1_feature_score"]) >= 0.78
                or float(a2["a2_feature_score"]) >= 0.55
            )
            and not localized_a1_attack_support
            and not photometric_attack_support
            and exposure["exposure_delta"] < 0.020
            and exposure["overexposure_ratio"] < 0.08
            and exposure["underexposed_ratio"] < 0.30
            and (
                float(
                    a3.get("flow_background_explain_score", 0.0)
                ) >= 0.65
                or (
                    # Fixed-camera outdoor footage produces strong A2/A3
                    # responses while people walk across the ROI.  The motion
                    # is nevertheless temporally aligned with the target and
                    # exposure-stable, so it must not be treated as a patch.
                    float(a1["a1_feature_score"]) <= 0.45
                    and float(a2["a2_feature_score"]) >= 0.78
                    and float(
                        a2.get("change_t_motion_aligned", 0.0)
                    ) >= 0.44
                    and float(
                        a2.get("change_t_motion_explain_score", 0.0)
                    ) >= 0.12
                    and float(
                        a3.get("flow_roi_coverage_ratio", 0.0)
                    ) >= 0.05
                    and exposure["frame_diff_global"] >= 0.012
                )
            )
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.18
            and (
                float(a3.get("flow_roi_coverage_ratio", 0.0)) >= 0.15
                or (
                    float(a1["a1_feature_score"]) < 0.55
                    and float(a2["a2_feature_score"]) >= 0.55
                    and float(
                        a2.get("change_t_motion_aligned", 0.0)
                    ) >= 0.80
                    and float(
                        a2.get(
                            "change_t_motion_explain_score",
                            0.0,
                        )
                    ) >= 0.30
                    and float(
                        a3.get("flow_residual_contrast", 0.0)
                    ) < 1.20
                )
            )
        )
        normal_high_contrast_target_texture_motion = bool(
            localized_a1_attack_support
            and not photometric_attack_support
            and exposure["underexposed_ratio"] >= 0.30
            and float(a2["a2_feature_score"]) >= 0.75
            and float(
                a2.get("change_t_motion_aligned", 0.0)
            ) >= 0.80
            and float(
                a2.get("change_t_motion_explain_score", 0.0)
            ) >= 0.30
            and exposure["frame_diff_global"] >= 0.020
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.18
            and float(a3.get("flow_roi_coverage_ratio", 0.0)) < 0.10
        )
        normal_roi_flow_target_motion = bool(
            dominant_adv == "A3_FLOW_ARTIFACT"
            and bool(a3.get("target_related", False))
            and not localized_a1_attack_support
            and not photometric_attack_support
            and float(a1["a1_feature_score"]) <= 0.45
            and float(a2["a2_feature_score"]) <= 0.32
            and exposure["frame_diff_global"] < 0.012
            and exposure["exposure_delta"] < 0.010
            and float(a2.get("change_t_roi_max", 0.0)) <= 0.22
            and float(
                a2.get("change_t_motion_aligned", 0.0)
            ) >= 0.30
            and float(
                a2.get("change_t_motion_explain_score", 0.0)
            ) >= 0.05
            and float(a3.get("flow_local_anomaly_ratio", 0.0)) < 0.13
            and float(a3.get("flow_max_magnitude_norm", 0.0)) < 0.70
            and float(a3.get("flow_roi_coverage_ratio", 0.0)) >= 0.14
        )
        normal_target_motion_exclusion = bool(
            normal_articulated_target_motion
            or normal_high_contrast_target_texture_motion
            or normal_roi_flow_target_motion
            or (
                target_related_feature
                and float(
                    a3.get("flow_roi_coverage_ratio", 0.0)
                ) >= 0.15
                and not a3_independent_attack_support
            )
        )
        if normal_target_motion_exclusion and not classifier_adv_rescue:
            adv_candidate_allowed = False
            rule_adv_candidate_allowed = False
        adv_explicitly_suppressed = bool(
            normal_target_motion_exclusion and not classifier_adv_rescue
        )
        media_candidate_allowed = bool(
            a3b["media_candidate_allowed"]
            and not normal_motion_texture_change
        )
        effective_rule_triggered = bool(
            a4.get("a4_rule_triggered", False)
            or (
                a4.get("p_adv_triggered", False)
                and not a4.get("a4_classifier_triggered", False)
            )
        )
        rule_adv_single_frame_candidate = bool(
            rule_adv_candidate_allowed
            and (
                effective_rule_triggered
                or structural_adv_evidence_rescue
            )
        )
        adv_single_frame_candidate = bool(
            rule_adv_single_frame_candidate
            or classifier_adv_rescue
        )
        adv_physical_support = bool(
            classifier_adv_rescue
            or (
                not normal_target_motion_exclusion
                and (target_related_feature or no_target_fallback)
                and a3_independent_attack_support
            )
        )
        # 候选连续性桥接(2026-06-30 修 adv_patch 漏报,见 docs/技术.算法/2026-06-30-adv_patch召回根因-*)：
        # 抑制门在持续攻击段偶发翻假会把连续候选打散,使 N-of-M 确认失败。桥接只允许跨越分数/阈值
        # 的短暂断点：本帧仍须有特征与目标/退化上下文，且不能命中任何明确的正常场景/背景运动抑制。
        # 因此它不会在 scene_baseline_normal、normal_target_motion_exclusion 等门已经否决当前帧时，
        # 仅凭 raw p_adv 重新制造 candidate。remaining 表示还可容忍的 raw-trigger 断点次数。
        adv_candidate_bridged = False
        adv_candidate_bridge_eligible = False
        adv_candidate_bridge_explicit_suppression = bool(
            not classifier_adv_rescue
            and (
                (
                    flow.get("global_motion_weight", 0.0) >= 0.82
                    and max_feature < 0.82
                )
                or (unsupported_a3_motion and not a3_residual_fallback)
                or unsupported_a2_motion
                or normal_motion_texture_change
                or nonlocal_a1_a3_scene_spike
                or low_motion_background_like_adv
                or cold_start_low_motion_adv
                or stationary_texture_only_adv
                or background_plane_adv
                or a3_only_background_motion
                or scene_baseline_normal
                or normal_target_motion_exclusion
            )
        )
        adv_candidate_bridge_recent_physical_support = bool(
            self._adv_cand_bridge_has_physical_support
        )
        adv_candidate_bridge_independent_support = bool(
            adv_physical_support
            or adv_candidate_bridge_recent_physical_support
        )
        adv_explicitly_suppressed = bool(
            adv_candidate_bridge_explicit_suppression
        )
        if classifier_adv_rescue:
            adv_explicit_suppression_reason = "none"
        elif normal_roi_flow_target_motion:
            adv_explicit_suppression_reason = (
                "normal_roi_flow_target_motion"
            )
        elif normal_high_contrast_target_texture_motion:
            adv_explicit_suppression_reason = (
                "normal_high_contrast_target_texture_motion"
            )
        elif normal_articulated_target_motion:
            adv_explicit_suppression_reason = (
                "normal_articulated_target_motion"
            )
        elif normal_target_motion_exclusion:
            adv_explicit_suppression_reason = (
                "normal_target_motion_exclusion"
            )
        elif scene_baseline_normal:
            adv_explicit_suppression_reason = (
                "scene_baseline_normal"
            )
        elif normal_motion_texture_change:
            adv_explicit_suppression_reason = (
                "normal_motion_texture_change"
            )
        elif low_motion_background_like_adv:
            adv_explicit_suppression_reason = (
                "low_motion_background_like_adv"
            )
        elif unsupported_a3_motion:
            adv_explicit_suppression_reason = (
                "unsupported_a3_motion"
            )
        elif unsupported_a2_motion:
            adv_explicit_suppression_reason = (
                "unsupported_a2_motion"
            )
        else:
            adv_explicit_suppression_reason = (
                "adv_candidate_policy_suppressed"
                if adv_explicitly_suppressed
                else "none"
            )
        adv_candidate_bridge_blocked = bool(
            adv_candidate_bridge_explicit_suppression
        )
        adv_candidate_bridge_support = bool(
            max_feature >= 0.40
            and (
                target_related_feature
                or no_target_fallback
                or a3_residual_fallback
                or adv_multi_evidence_rescue
                or visibility_texture_rescue
            )
        )
        if adv_single_frame_candidate:
            self._adv_cand_bridge_has_physical_support = bool(
                self._adv_cand_bridge_has_physical_support
                and self._adv_cand_bridge_remaining > 0
                or adv_physical_support
            )
            self._adv_cand_bridge_remaining = self._adv_cand_bridge_frames
        elif (
            self._adv_cand_bridge_frames > 0
            and self._adv_cand_bridge_remaining > 0
            and bool(a4["p_adv_triggered"])
        ):
            self._adv_cand_bridge_remaining -= 1
            adv_candidate_bridge_eligible = bool(
                a4["p_adv_triggered"]
                and (
                    adv_candidate_bridge_support
                    or adv_candidate_bridge_recent_physical_support
                )
                and not adv_candidate_bridge_blocked
            )
            if adv_candidate_bridge_eligible:
                adv_single_frame_candidate = True
                adv_candidate_bridged = True
        # 支路B 致盲候选：p_blind 触发即候选（_compute_blinding 内部已做"曾有目标+退化"门控）
        blind_independent_support = bool(
            blinding.get("blind_independent_support", False)
        )
        blind_explicitly_suppressed = bool(
            blinding.get("p_blind_triggered", False)
            and not blind_independent_support
        )
        blind_single_frame_candidate = bool(
            blinding.get("p_blind_triggered", False)
            and blind_independent_support
        )
        # A3b 独立触发: demo 默认禁用(背景结构墙/门框误报)。config rebuilt_a3b_independent_trigger
        # 开启时恢复候选, 经收紧门(edge/border_contrast 数据方案)+ N-of-M 确认, 找回画中画翻拍检测。
        _ms = a3b.get("p_media_scores", {}) or {}
        _edge = float(_ms.get("edge", 0.0))
        _bc = float(_ms.get("border_contrast", 0.0))
        _cand = float(_ms.get("candidate_score", 0.0))
        media_result_seq = int(a3b.get("a3b_result_seq", 0) or 0)
        media_result_fresh = bool(
            a3b.get("a3b_result_fresh", False)
        )
        media_result_is_new = bool(
            media_result_fresh
            and media_result_seq
            > self._a3b_last_consumed_result_seq
        )
        media_result_consumed = bool(
            self._a3b_independent_trigger
            and media_result_is_new
        )
        media_tighten_candidate_pass = bool(_cand >= self._a3b_gate_candidate_min)
        media_tighten_edge_pass = bool(
            self._a3b_gate_edge_min <= _edge <= self._a3b_gate_edge_max
        )
        media_tighten_border_pass = bool(
            _bc >= self._a3b_gate_border_contrast_min
        )
        media_tighten_robust_display_pass = bool(
            media_candidate_allowed
            and bool(a3b.get("p_media_strong_evidence", False))
            and float(a3b.get("p_media_policy", 0.0)) >= self.theta_media
            and _cand >= max(0.55, self._a3b_gate_candidate_min - 0.15)
            and 0.40 <= _edge <= max(0.62, self._a3b_gate_edge_max)
            and _bc >= self._a3b_gate_border_contrast_min
            and float(_ms.get("display_frame", 0.0)) >= 0.70
            and float(_ms.get("area_ratio", 0.0)) >= 0.08
            and float(_ms.get("rect", 0.0)) >= 0.60
            and float(_ms.get("source_score", 0.0)) >= 0.70
        )
        media_bbox = a3b.get("p_media_bbox")
        media_tighten_aspect_ratio = 0.0
        if (
            isinstance(media_bbox, (list, tuple))
            and len(media_bbox) == 4
        ):
            try:
                media_bbox_width = max(
                    0.0,
                    float(media_bbox[2]) - float(media_bbox[0]),
                )
                media_bbox_height = max(
                    0.0,
                    float(media_bbox[3]) - float(media_bbox[1]),
                )
                if media_bbox_height > 0.0:
                    media_tighten_aspect_ratio = (
                        media_bbox_width / media_bbox_height
                    )
            except (TypeError, ValueError):
                media_tighten_aspect_ratio = 0.0
        media_tighten_aspect_pass = bool(
            self._a3b_gate_aspect_ratio_min
            <= media_tighten_aspect_ratio
            <= self._a3b_gate_aspect_ratio_max
        )
        media_gate_ok = bool(
            media_candidate_allowed
            and (
                not self._a3b_independent_trigger
                or media_tighten_aspect_pass
            )
        )
        if self._a3b_independent_trigger and self._a3b_tighten_gate and media_gate_ok:
            media_gate_ok = bool(
                (
                    media_tighten_candidate_pass
                    and media_tighten_edge_pass
                    and media_tighten_border_pass
                    and media_tighten_aspect_pass
                )
                or (
                    media_tighten_robust_display_pass
                    and media_tighten_aspect_pass
                )
            )
        # 持续段确认只消费新的成功后台结果。主循环重复读取同一缓存 seq
        # 可以维持现有确认/hold，但不得伪装成新的独立 A3b 证据。
        media_source_frame_units = 0
        if media_result_consumed:
            media_source_frame_units = (
                self._a3b_result_source_frame_units(a3b)
            )
            self._a3b_last_consumed_result_seq = media_result_seq
            if media_gate_ok:
                if self._media_run_gap > self._a3b_media_run_gap_tol:
                    self._media_run = 0
                self._media_run += media_source_frame_units
                self._media_run_gap = 0
            elif self._media_run > 0:
                self._media_run_gap += media_source_frame_units
                if self._media_run_gap > self._a3b_media_run_gap_tol:
                    self._media_run = 0
                    self._media_run_gap = 0
        media_single_frame_candidate = bool(
            self._a3b_independent_trigger
            and media_result_fresh
            and media_gate_ok
            and self._media_run >= self._a3b_media_run_floor
        )
        single_frame_candidate = bool(adv_single_frame_candidate or blind_single_frame_candidate)
        if rule_adv_single_frame_candidate:
            self.adv_hits.append(1)
            self.adv_scores.append(float(a4["p_adv"]))
            self.adv_support_hits.append(1 if adv_physical_support else 0)
        else:
            self.adv_hits.append(0)
            self.adv_scores.append(0.0)
            self.adv_support_hits.append(0)
        if classifier_adv_rescue_dark_scene_blocked:
            self.classifier_adv_hits.clear()
        self.classifier_adv_hits.append(
            1 if classifier_adv_rescue else 0
        )
        if blind_single_frame_candidate:
            self.blind_hits.append(1)
            self.blind_scores.append(float(blinding.get("p_blind", 0.0)))
        else:
            self.blind_hits.append(0)
            self.blind_scores.append(0.0)
        if media_result_consumed:
            media_vote_units = min(
                self.max_history,
                max(1, media_source_frame_units),
            )
            for _ in range(media_vote_units):
                if media_single_frame_candidate:
                    self.media_hits.append(1)
                    self.media_scores.append(
                        float(a3b["p_media_policy"])
                    )
                else:
                    self.media_hits.append(0)
                    self.media_scores.append(0.0)
        rule_adv_count = sum(list(self.adv_hits)[-window_frames:])
        classifier_adv_count = sum(
            list(self.classifier_adv_hits)[
                -self.a4_classifier_alarm_window:
            ]
        )
        adv_count = max(rule_adv_count, classifier_adv_count)
        adv_support_count = sum(list(self.adv_support_hits)[-window_frames:])
        media_count = sum(list(self.media_hits)[-window_frames:])
        blind_count = sum(list(self.blind_hits)[-window_frames:])
        rule_adv_confirmed = bool(
            rule_adv_count >= adv_hit_required
            and adv_support_count >= 1
        )
        classifier_adv_confirmed = bool(
            classifier_adv_count
            >= self.a4_classifier_alarm_required_hits
        )
        adv_confirmed = bool(
            rule_adv_confirmed or classifier_adv_confirmed
        )
        media_confirmed = bool(
            media_result_fresh
            and media_gate_ok
            and media_count >= media_hit_required
        )
        # 支路B 确认：致盲攻击持续多帧（与高误报场景一致用更高占比），需曾有目标语境
        blind_confirm_ratio = max(
            self._blind_confirm_ratio,
            0.67 if exposure["high_false_positive_scene"] else 0.0,
        )
        blind_hit_required = max(
            3,
            int(math.ceil(window_frames * blind_confirm_ratio)),
        )
        blind_confirmed = bool(blind_count >= blind_hit_required)
        p_media_confirmed_score = (
            float(max(list(self.media_scores)[-window_frames:] or [0.0]))
            if media_confirmed else 0.0
        )
        adv_primary_preferred = bool(
            adv_confirmed
            and a4["p_adv"] >= max(adv_threshold, 0.70)
            and (
                max(float(a1["a1_feature_score"]), float(a2["a2_feature_score"])) >= 0.78
                or (
                    float(a1["a1_feature_score"]) + float(a2["a2_feature_score"]) >= 1.35
                    and a3b_reason in {
                        "glare_or_texture_adv_prefers_A1_A2",
                        "base_adv_evidence_prefers_A1_A2_A3",
                        "adv_explained_glare_or_texture_plane",
                    }
                )
                or (
                    a3_residual_fallback
                    and float(a3["a3_feature_score"]) >= 0.80
                    and bool(a3.get("a3_residual_hold_active", False))
                )
            )
            and not robust_media_evidence
        )

        # --- 持续对抗升级(场景自适应 + fps归一化, 2026-06-30 重写): 见 __init__ 注释 ---
        # 追踪原始 p_adv>=theta_adv 的**当前连续段长度** _adv_run(每帧都算, 即使本帧
        # adv_candidate_allowed 被 scene_baseline_normal 等否决——这正是捕获"持续攻击"的关键)。
        # 容忍段内 1 帧掉落(补丁下偶发翻假), 掉落 2 帧则视为段结束。段结束且从未升级时, 把该段
        # 长度学入 _benign_run_ref(本场景良性突发尺度, 带慢衰减); 已升级的段不污染参考(冻结)。
        raw_adv_trigger = bool(a4.get("p_adv_triggered", False))
        # 无目标纯背景静止帧(pure_static_background 及 background_*_suppressed 家族, 且无 target_related):
        # p_adv 高属 XGBoost 对静止纹理的伪响应, 逐帧门已否决其为候选; 不计入持续段, 防止凭空升级。
        adv_static_bg_no_evidence = bool(
            self._sustained_adv_exclude_static_bg
            and not target_related_feature
            and bool(a3b.get("p_media_background_static_suppressed", False))
        )
        # recent_target 门(demo blind-branch 原则): 最近 8 帧至少 N 帧有目标, 兼顾"补丁瞬时遮挡
        # 丢目标但遮挡前在场"与"空场景无目标不凭空升级"。adv_physical_support 降为默认关的收紧旋钮。
        recent_target_present = (
            sum(self.recent_target_presence) >= self._sustained_adv_recent_target_min
        )
        sustained_adv_has_independent_support = bool(
            adv_physical_support
        )
        sustained_adv_support_requirement_satisfied = bool(
            sustained_adv_has_independent_support
            or not self._sustained_adv_require_physical_support
        )
        sustained_adv_scene_allowed = bool(
            not adv_explicitly_suppressed
        )
        sustained_hit = bool(
            raw_adv_trigger
            and (target_related_feature or not self._sustained_adv_require_target)
            and not adv_static_bg_no_evidence
            and recent_target_present
            # sustained escalation 只能救援“仍有有效候选或独立物理证据”的持续段。
            # 若逐帧策略已明确判为 scene baseline normal，且本帧也没有 physical
            # support，则不得仅凭 raw p_adv 推翻抑制并升级为 confirmed。
            and sustained_adv_support_requirement_satisfied
            and sustained_adv_scene_allowed
        )
        if sustained_hit:
            self._adv_run += 1
            self._adv_run_gap = 0
        elif self._adv_run > 0 and self._adv_run_gap < 1:
            # 段内容忍单帧间隙: 保持不增长, 不结束。
            self._adv_run_gap += 1
        else:
            # 段结束: 未升级的良性段更新场景良性突发参考(慢衰减取 max)。
            if self._adv_run > 0 and not self._adv_run_escalated:
                self._benign_run_ref = max(
                    float(self._adv_run),
                    self._benign_run_ref * self._sustained_adv_benign_decay,
                )
            self._adv_run = 0
            self._adv_run_gap = 0
            self._adv_run_escalated = False
        # 时间下限(帧) = round(process_fps * sustained_seconds); fps 归一化, 物理持续先验。
        sustained_adv_floor = int(max(1, round(float(self.process_fps) * self._sustained_adv_seconds)))
        # 场景自适应门槛 = run_mult * 本场景良性突发尺度。
        sustained_adv_run_bar = float(self._sustained_adv_run_mult) * float(self._benign_run_ref)
        sustained_adv_escalated = bool(
            self._sustained_adv_enabled
            and self._adv_run >= sustained_adv_floor
            and float(self._adv_run) >= sustained_adv_run_bar
        )
        if sustained_adv_escalated:
            # 标记本段已升级 → 段结束时不污染良性突发参考(冻结)。
            self._adv_run_escalated = True

        # --- 致盲持续升级(2026-07-12 修完全致盲 glare 008/016 漏检)---
        # 长记忆锁: 本场景一旦近窗出现过 >=N 帧有目标即永久置位(致盲后近窗滑窗会凑不满, 故用锁)。
        if sum(self.recent_target_presence) >= self._blind_sustained_established_min:
            self._blind_target_established = True
        blind_no_target = len(rois) == 0
        blind_degrade_evidence = (
            blind_independent_support
            and (
                float(blinding.get("sharp_drop", 0.0)) >= self._blind_sustained_degrade_min
                or float(blinding.get("glare_blind", 0.0)) >= self._blind_sustained_degrade_min
            )
        )
        blind_run_hit = bool(
            self._blind_sustained_enabled
            and self._blind_target_established
            and blind_no_target
            and blind_degrade_evidence
        )
        if blind_run_hit:
            self._blind_run += 1
            self._blind_run_gap = 0
        elif self._blind_run > 0 and self._blind_run_gap < 1:
            self._blind_run_gap += 1  # 容忍单帧间隙
        else:
            self._blind_run = 0
            self._blind_run_gap = 0
        blind_sustained_escalated = bool(
            self._blind_sustained_enabled
            and self._blind_run >= self._blind_sustained_floor
        )

        global_suppressed = False
        global_suppressed_reason = "none"
        alert_confirmed = False
        primary_channel = "none"
        alert_confirmation_source = "none"
        current_adv_overrides_stale_media = bool(
            adv_confirmed
            and adv_single_frame_candidate
            and not media_single_frame_candidate
            and not robust_media_evidence
        )
        if (
            (adv_primary_preferred or current_adv_overrides_stale_media)
            and adv_confirmed
            and not adv_explicitly_suppressed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "adv"
            alert_confirmation_source = "adv_temporal"
        elif media_confirmed and not global_suppressed:
            alert_confirmed = True
            primary_channel = "media"
            alert_confirmation_source = "media_temporal"
        elif (
            adv_confirmed
            and not adv_explicitly_suppressed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "adv"
            alert_confirmation_source = "adv_temporal"
        elif (
            blind_confirmed
            and not blind_explicitly_suppressed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "blind"
            alert_confirmation_source = "blind_temporal"

        # 持续对抗升级: 逐帧候选被 scene_baseline_normal 打散、上面各支未确认时, 若长窗内原始
        # p_adv 持续占优则强制升级为 adv 确认(绕过逐帧门, 不改动 adv_candidate_allowed 本身)。
        if (
            not alert_confirmed
            and sustained_adv_escalated
            and not adv_explicitly_suppressed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "adv"
            alert_confirmation_source = "adv_sustained"

        # 致盲持续升级: 完全致盲(持续丢目标)时 adv/blind 逐帧通道双失效, 用"确立后持续缺席+退化佐证"
        # 的连续段作证据强制升级为 blind 确认(绕过逐帧 p_blind N-of-M)。修 glare 008/016 完全致盲漏检。
        if (
            not alert_confirmed
            and blind_sustained_escalated
            and not blind_explicitly_suppressed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "blind"
            alert_confirmation_source = "blind_sustained"

        # --- 报警保持窗口（2026-06-30 行为调优，修复"风机出现时断警告"）---
        # 一旦确认，维持 _alert_hold_frames 帧；期间逐帧候选短暂不足也保持 ATTACK，
        # 期间再次确认则刷新保持。global_suppressed（明确抑制）时不保持。
        alert_held = False
        alert_hold_refresh_signal = False
        alert_hold_refresh_source = "none"
        alert_hold_blocked_reason = "none"
        if (
            self._alert_hold_channel == "adv"
            and adv_explicitly_suppressed
        ):
            alert_hold_blocked_reason = (
                adv_explicit_suppression_reason
            )
        elif (
            self._alert_hold_channel == "blind"
            and blind_explicitly_suppressed
        ):
            alert_hold_blocked_reason = (
                "blind_independent_support_missing"
            )
        alert_hold_window_frames = (
            self._a3b_alert_hold_frames
            if primary_channel == "media"
            else self._alert_hold_frames
        )
        if alert_confirmed:
            self._alert_hold_remaining = alert_hold_window_frames
            self._alert_hold_channel = primary_channel
        elif alert_hold_blocked_reason != "none":
            self._alert_hold_remaining = 0
            self._alert_hold_channel = "none"
        elif (
            self._alert_hold_remaining > 0
            and not global_suppressed
        ):
            # 只允许原确认通道的当前有效 candidate 刷新保持窗。旧逻辑仅凭
            # raw p_adv>=theta_adv 就续命，会让 blind/media 告警被无关 adv 分数
            # 延长，也会在 adv_candidate 已消失或被抑制后无限保持。
            if self._alert_hold_channel == "adv":
                alert_hold_refresh_signal = bool(
                    adv_single_frame_candidate
                )
                alert_hold_refresh_source = "adv_candidate"
            elif self._alert_hold_channel == "blind":
                alert_hold_refresh_signal = bool(
                    blind_single_frame_candidate
                )
                alert_hold_refresh_source = "blind_candidate"
            elif self._alert_hold_channel == "media":
                alert_hold_refresh_signal = bool(
                    media_single_frame_candidate
                )
                alert_hold_refresh_source = "media_candidate"
            if (
                self._alert_hold_refresh_on_padv
                and alert_hold_refresh_signal
            ):
                self._alert_hold_remaining = (
                    self._a3b_alert_hold_frames
                    if self._alert_hold_channel == "media"
                    else self._alert_hold_frames
                )
            else:
                self._alert_hold_remaining -= 1
            alert_confirmed = True
            alert_held = True
            primary_channel = self._alert_hold_channel if self._alert_hold_channel != "none" else "adv"
            alert_confirmation_source = f"{primary_channel}_hold"

        adv_score_over_threshold = bool(
            a4.get("p_adv_triggered", False)
        )
        if adv_single_frame_candidate:
            adv_confirmation_blocked_reason = "none"
        elif not adv_score_over_threshold:
            adv_confirmation_blocked_reason = (
                "score_below_threshold"
            )
        elif adv_explicitly_suppressed:
            adv_confirmation_blocked_reason = (
                adv_explicit_suppression_reason
            )
        else:
            adv_confirmation_blocked_reason = (
                "candidate_policy_rejected"
            )

        if media_single_frame_candidate and adv_single_frame_candidate:
            candidate_source = "BOTH"
        elif media_single_frame_candidate:
            candidate_source = "A3B_MEDIA"
        elif adv_single_frame_candidate:
            candidate_source = "A4_ADV"
        elif blind_single_frame_candidate:
            candidate_source = "B_BLIND"
        else:
            candidate_source = "NONE"

        dominant_input = a4["dominant_adv_input"]
        if primary_channel == "blind":
            dominant_input = "B_BLIND_" + str(blinding.get("blind_type", "none")).upper()
        elif primary_channel == "media" or (candidate_source == "A3B_MEDIA" and not adv_single_frame_candidate):
            dominant_input = "A3B_MEDIA"
        elif candidate_source == "BOTH" and a3b["p_media_policy"] >= a4["p_adv"] + 0.08:
            dominant_input = "A3B_MEDIA"

        public_reason, debug_reason, reason_codes = self._build_reasons(
            dominant_input,
            a1,
            a2,
            a3,
            a4,
            a3b,
            adv_candidate_allowed,
            media_candidate_allowed,
        )
        alert_level = "confirmed" if alert_confirmed else ("candidate" if single_frame_candidate else "normal")
        if blind_single_frame_candidate:
            reason_codes = list(reason_codes) + [f"B_BLIND_{str(blinding.get('blind_type','none')).upper()}"]
        confirm_count = max(int(adv_count), int(media_count))
        confirm_window = {
            "window_frames": int(window_frames),
            "adv_hit_required": int(adv_hit_required),
            "adv_support_count": int(adv_support_count),
            "rule_adv_count": int(rule_adv_count),
            "rule_adv_confirmed": bool(rule_adv_confirmed),
            "classifier_adv_window": int(
                self.a4_classifier_alarm_window
            ),
            "classifier_adv_hit_required": int(
                self.a4_classifier_alarm_required_hits
            ),
            "classifier_adv_count": int(classifier_adv_count),
            "classifier_adv_confirmed": bool(
                classifier_adv_confirmed
            ),
            "media_hit_required": int(media_hit_required),
            "blind_hit_required": int(blind_hit_required),
            "blind_confirm_ratio": float(blind_confirm_ratio),
            "adv_count": int(adv_count),
            "media_count": int(media_count),
            "blind_count": int(blind_count),
            "alert_held": bool(alert_held),
            "alert_hold_remaining": int(self._alert_hold_remaining),
            "alert_hold_window_frames": int(
                self._a3b_alert_hold_frames
                if self._alert_hold_channel == "media"
                else self._alert_hold_frames
            ),
        }
        return {
            "adv_single_frame_candidate": bool(adv_single_frame_candidate),
            "rule_adv_single_frame_candidate": bool(
                rule_adv_single_frame_candidate
            ),
            "adv_score_over_threshold": bool(
                adv_score_over_threshold
            ),
            "adv_confirmation_blocked_reason": (
                adv_confirmation_blocked_reason
            ),
            "adv_explicitly_suppressed": bool(
                adv_explicitly_suppressed
            ),
            "adv_explicit_suppression_reason": (
                adv_explicit_suppression_reason
            ),
            "localized_a1_attack_support": bool(
                localized_a1_attack_support
            ),
            "localized_patch_context": bool(
                localized_patch_context
            ),
            "photometric_attack_support": bool(
                photometric_attack_support
            ),
            "glare_attack_support": bool(
                glare_attack_support
            ),
            "adv_candidate_bridged": bool(adv_candidate_bridged),
            "adv_candidate_bridge_eligible": bool(
                adv_candidate_bridge_eligible
            ),
            "adv_candidate_bridge_support": bool(
                adv_candidate_bridge_support
            ),
            "adv_candidate_bridge_blocked": bool(
                adv_candidate_bridge_blocked
            ),
            "adv_candidate_bridge_recent_physical_support": bool(
                adv_candidate_bridge_recent_physical_support
            ),
            "adv_candidate_bridge_independent_support": bool(
                adv_candidate_bridge_independent_support
            ),
            "adv_candidate_bridge_explicit_suppression": bool(
                adv_candidate_bridge_explicit_suppression
            ),
            "adv_candidate_bridge_remaining": int(
                self._adv_cand_bridge_remaining
            ),
            "media_single_frame_candidate": bool(media_single_frame_candidate),
            "single_frame_candidate": bool(single_frame_candidate),
            "candidate_source": candidate_source,
            "adv_candidate_allowed": bool(adv_candidate_allowed),
            "classifier_adv_rescue": bool(classifier_adv_rescue),
            "classifier_adv_rescue_requested": bool(
                classifier_adv_rescue_requested
            ),
            "classifier_adv_rescue_dark_scene_blocked": bool(
                classifier_adv_rescue_dark_scene_blocked
            ),
            "classifier_adv_rescue_underexposed_max": float(
                self._a4_classifier_rescue_underexposed_max
            ),
            "normal_target_motion_exclusion": bool(
                normal_target_motion_exclusion
            ),
            "normal_articulated_target_motion": bool(
                normal_articulated_target_motion
            ),
            "normal_high_contrast_target_texture_motion": bool(
                normal_high_contrast_target_texture_motion
            ),
            "normal_roi_flow_target_motion": bool(
                normal_roi_flow_target_motion
            ),
            "a3_independent_attack_support": bool(
                a3_independent_attack_support
            ),
            "media_candidate_allowed": bool(media_candidate_allowed),
            "media_tighten_gate_enabled": bool(self._a3b_tighten_gate),
            "media_tighten_candidate_score": float(_cand),
            "media_tighten_edge": float(_edge),
            "media_tighten_border_contrast": float(_bc),
            "media_tighten_aspect_ratio": float(
                media_tighten_aspect_ratio
            ),
            "media_tighten_candidate_pass": bool(media_tighten_candidate_pass),
            "media_tighten_edge_pass": bool(media_tighten_edge_pass),
            "media_tighten_border_pass": bool(media_tighten_border_pass),
            "media_tighten_robust_display_pass": bool(
                media_tighten_robust_display_pass
            ),
            "media_tighten_aspect_pass": bool(
                media_tighten_aspect_pass
            ),
            "media_gate_ok": bool(media_gate_ok),
            "media_result_seq": int(media_result_seq),
            "media_result_fresh": bool(media_result_fresh),
            "media_result_consumed": bool(media_result_consumed),
            "media_source_frame_units": int(
                media_source_frame_units
            ),
            "media_last_consumed_result_seq": int(
                self._a3b_last_consumed_result_seq
            ),
            "media_run": int(self._media_run),
            "media_run_gap": int(self._media_run_gap),
            "media_run_floor": int(self._a3b_media_run_floor),
            "media_count": int(media_count),
            "media_hit_required": int(media_hit_required),
            "adv_confirmed": bool(adv_confirmed),
            "media_confirmed": bool(media_confirmed),
            "blind_confirmed": bool(blind_confirmed),
            "blind_single_frame_candidate": bool(blind_single_frame_candidate),
            "blind_independent_support": bool(blind_independent_support),
            "blind_explicitly_suppressed": bool(
                blind_explicitly_suppressed
            ),
            "blind_degrade_evidence": bool(blind_degrade_evidence),
            "blind_sustained_run": int(self._blind_run),
            "blind_sustained_floor": int(self._blind_sustained_floor),
            "blind_sustained_escalated": bool(
                blind_sustained_escalated
            ),
            "p_blind": float(blinding.get("p_blind", 0.0)),
            "blind_type": str(blinding.get("blind_type", "none")),
            "scene_baseline_normal": bool(scene_baseline_normal),
            # 诊断用只读字段(暴露运动抑制门状态, 不改判定逻辑):
            "normal_motion_texture_change": bool(normal_motion_texture_change),
            "nonlocal_a1_a3_scene_spike": bool(nonlocal_a1_a3_scene_spike),
            "low_motion_background_like_adv": bool(low_motion_background_like_adv),
            "target_related_feature": bool(target_related_feature),
            "adv_physical_support": bool(adv_physical_support),
            "sustained_adv_run": int(self._adv_run),
            "sustained_adv_floor": int(sustained_adv_floor),
            "sustained_adv_run_bar": float(sustained_adv_run_bar),
            "sustained_adv_benign_ref": float(self._benign_run_ref),
            "sustained_adv_seconds": float(self._sustained_adv_seconds),
            "sustained_adv_escalated": bool(sustained_adv_escalated),
            "sustained_adv_has_independent_support": bool(
                sustained_adv_has_independent_support
            ),
            "sustained_adv_support_requirement_satisfied": bool(
                sustained_adv_support_requirement_satisfied
            ),
            "sustained_adv_requires_physical_support": bool(
                self._sustained_adv_require_physical_support
            ),
            "sustained_adv_scene_allowed": bool(
                sustained_adv_scene_allowed
            ),
            "p_media_confirmed_score": float(p_media_confirmed_score),
            "alert_confirmed": bool(alert_confirmed),
            "alert_level": alert_level,
            "alert_confirmation_source": (
                alert_confirmation_source
            ),
            "primary_channel": primary_channel,
            "dominant_input": dominant_input,
            "public_reason": public_reason,
            "debug_reason": debug_reason,
            "suppressed_reason": global_suppressed_reason if global_suppressed else a3b.get("suppressed_reason", "none"),
            "confirm_window": confirm_window,
            "confirm_count": int(confirm_count),
            "adv_primary_preferred": bool(adv_primary_preferred),
            "current_adv_overrides_stale_media": bool(current_adv_overrides_stale_media),
            "robust_media_evidence": bool(robust_media_evidence),
            "cold_start_low_motion_adv": bool(cold_start_low_motion_adv),
            "stationary_texture_only_adv": bool(stationary_texture_only_adv),
            "background_plane_adv": bool(background_plane_adv),
            "adv_multi_evidence_rescue": bool(adv_multi_evidence_rescue),
            "structural_adv_evidence_rescue": bool(structural_adv_evidence_rescue),
            "visibility_texture_probe": bool(visibility_texture_probe),
            "visibility_texture_rescue": bool(visibility_texture_rescue),
            "physical_patch_rescue_evidence": bool(physical_patch_rescue_evidence),
            "a3_residual_fallback": bool(a3_residual_fallback),
            "unsupported_a3_motion": bool(unsupported_a3_motion),
            "a3_only_background_motion": bool(a3_only_background_motion),
            "reason_codes": reason_codes,
            "alert_hold_refresh_signal": bool(
                alert_hold_refresh_signal
            ),
            "alert_hold_refresh_source": alert_hold_refresh_source,
            "alert_hold_blocked_reason": (
                alert_hold_blocked_reason
            ),
            "process_fps": float(self.process_fps),
        }

    def _build_reasons(
        self,
        dominant_input: str,
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        a4: dict[str, Any],
        a3b: dict[str, Any],
        adv_candidate_allowed: bool,
        media_candidate_allowed: bool,
    ) -> tuple[str, str, list[str]]:
        if dominant_input == "A3B_MEDIA":
            public = f"media candidate: {a3b['p_media_type']}"
            code = "static_media_spoof"
        elif dominant_input == "A1_LBP_SINGLE":
            public = "single-frame LBP texture anomaly"
            code = "a1_lbp_single"
        elif dominant_input == "A2_LBP_TEMPORAL":
            public = "temporal LBP inconsistency"
            code = "a2_lbp_temporal"
        elif dominant_input == "A3_FLOW_ARTIFACT":
            public = "local optical-flow artifact"
            code = "a3_flow_artifact"
        elif dominant_input == "A4_MIXED":
            public = "mixed A1/A2/A3 evidence"
            code = "a4_mixed"
        else:
            public = "normal"
            code = "normal"
        debug = (
            f"A1={a1['a1_feature_score']:.3f}, "
            f"A2={a2['a2_feature_score']:.3f}, "
            f"A3={a3['a3_feature_score']:.3f}, "
            f"p_adv={a4['p_adv']:.3f}, "
            f"p_media_raw={a3b['p_media_raw']:.3f}, "
            f"p_media_policy={a3b['p_media_policy']:.3f}, "
            f"adv_allowed={int(adv_candidate_allowed)}, "
            f"media_allowed={int(media_candidate_allowed)}, "
            f"suppressed={a3b.get('suppressed_reason', 'none')}"
        )
        reason_codes = [] if code == "normal" else [code]
        return public, debug, reason_codes

    def _build_features(
        self,
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        a4: dict[str, Any],
        a3b: dict[str, Any],
        joint: dict[str, Any],
        exposure: dict[str, Any],
        flow: dict[str, Any],
    ) -> dict[str, Any]:
        self.a1_display_score = self._display_ema(self.a1_display_score, a1["a1_feature_score"])
        self.a2_display_score = self._display_ema(self.a2_display_score, a2["a2_feature_score"])
        self.a3_display_score = self._display_ema(self.a3_display_score, a3["a3_feature_score"])
        a3b_state_score = max(a3b["p_media_policy"], joint["p_media_confirmed_score"])
        self.a3b_display_score = self._display_ema(self.a3b_display_score, a3b_state_score)
        self.a4_display_score = self._display_ema(self.a4_display_score, a4["p_adv"])
        primary_raw = max(
            a1["a1_feature_score"],
            a2["a2_feature_score"],
            a3["a3_feature_score"],
            a3b_state_score,
            a4["p_adv"] if joint["adv_single_frame_candidate"] else 0.0,
        )
        self.primary_display_score = self._display_ema(self.primary_display_score, primary_raw)

        features: dict[str, Any] = {}
        features.update(a1)
        features.update(a2)
        features.update(a3)
        features.update(a4)
        features.update({
            k: v for k, v in a3b.items()
            if k != "media_candidates"
        })
        features.update({
            "p_media_confirmed_score": float(joint["p_media_confirmed_score"]),
            "media_confirmed": bool(joint["media_confirmed"]),
            "adv_single_frame_candidate": bool(joint["adv_single_frame_candidate"]),
            "adv_score_over_threshold": bool(
                joint.get("adv_score_over_threshold", False)
            ),
            "adv_confirmation_blocked_reason": str(
                joint.get(
                    "adv_confirmation_blocked_reason",
                    "none",
                )
            ),
            "adv_explicitly_suppressed": bool(
                joint.get("adv_explicitly_suppressed", False)
            ),
            "a3_independent_attack_support": bool(
                joint.get("a3_independent_attack_support", False)
            ),
            "media_single_frame_candidate": bool(joint["media_single_frame_candidate"]),
            "single_frame_candidate": bool(joint["single_frame_candidate"]),
            "candidate_source": joint["candidate_source"],
            "adv_candidate_allowed": bool(joint["adv_candidate_allowed"]),
            "media_candidate_allowed": bool(joint["media_candidate_allowed"]),
            "adv_primary_preferred": bool(joint.get("adv_primary_preferred", False)),
            "robust_media_evidence": bool(joint.get("robust_media_evidence", False)),
            "cold_start_low_motion_adv": bool(joint.get("cold_start_low_motion_adv", False)),
            "stationary_texture_only_adv": bool(joint.get("stationary_texture_only_adv", False)),
            "background_plane_adv": bool(joint.get("background_plane_adv", False)),
            "adv_multi_evidence_rescue": bool(joint.get("adv_multi_evidence_rescue", False)),
            "visibility_texture_probe": bool(joint.get("visibility_texture_probe", False)),
            "visibility_texture_rescue": bool(joint.get("visibility_texture_rescue", False)),
            "physical_patch_rescue_evidence": bool(joint.get("physical_patch_rescue_evidence", False)),
            "a3_residual_fallback": bool(joint.get("a3_residual_fallback", False)),
            "unsupported_a3_motion": bool(joint.get("unsupported_a3_motion", False)),
            "a3_only_background_motion": bool(joint.get("a3_only_background_motion", False)),
            "alert_confirmed": bool(joint["alert_confirmed"]),
            "alert_confirmation_source": str(
                joint.get("alert_confirmation_source", "none")
            ),
            "alert_hold_blocked_reason": str(
                joint.get("alert_hold_blocked_reason", "none")
            ),
            "dominant_input": joint["dominant_input"],
            "primary_channel": joint["primary_channel"],
            "p_blind": float(joint.get("p_blind", 0.0)),
            "blind_type": str(joint.get("blind_type", "none")),
            "blind_confirmed": bool(joint.get("blind_confirmed", False)),
            "public_reason": joint["public_reason"],
            "debug_reason": joint["debug_reason"],
            "suppressed_reason": joint["suppressed_reason"],
            "confirm_window": joint["confirm_window"],
            "confirm_count": int(joint["confirm_count"]),
            "alert_level": joint["alert_level"],
            "process_fps": float(self.process_fps),
            "window_frames": int(joint["confirm_window"]["window_frames"]),
            "a1_display_score": float(self.a1_display_score),
            "a2_display_score": float(self.a2_display_score),
            "a3_display_score": float(self.a3_display_score),
            "a3b_display_score": float(self.a3b_display_score),
            "a4_display_score": float(self.a4_display_score),
            "primary_display_score": float(self.primary_display_score),
            "overexposure_ratio": float(exposure["overexposure_ratio"]),
            "underexposed_ratio": float(exposure["underexposed_ratio"]),
            "exposure_delta": float(exposure["exposure_delta"]),
            "frame_diff_global": float(exposure["frame_diff_global"]),
            "global_motion_weight": float(flow.get("global_motion_weight", 0.0)),
            "static_image_score": float(a3b["p_media_policy"]),
            "static_image_triggered": bool(a3b["p_media_triggered"]),
        })
        # Compatibility names consumed by the existing dashboard.
        features["a1_score"] = float(a1["a1_feature_score"])
        features["a2_score"] = float(a2["a2_feature_score"])
        features["a3_score"] = float(a3["a3_feature_score"])
        features["blur_score"] = float(a3["a3_feature_score"])
        features["light_flow_score"] = float(a3["flow_score"])
        return features

    def _display_ema(self, previous: float, current: float) -> float:
        current = float(current)
        if current >= previous:
            return current
        return previous * 0.82 + current * 0.18

    def _build_roi_results(
        self,
        rois: list[ROI],
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        a3b: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for roi in rois:
            results.append(
                {
                    "roi": roi.to_dict(),
                    "a1_feature_score": float(a1["a1_feature_score"]),
                    "a2_feature_score": float(a2["a2_feature_score"]),
                    "a3_feature_score": float(a3["a3_feature_score"]),
                    "p_media_policy": float(a3b["p_media_policy"]),
                }
            )
        return results

    def _update_baseline(self, lbp: np.ndarray, joint: dict[str, Any]) -> None:
        hist = _hist_lbp(lbp)
        if self.lbp_baseline is None:
            self.lbp_baseline = hist
            self.lbp_baseline_samples = 1
            return
        bootstrap = self.lbp_baseline_samples < self._bootstrap_frames
        # 问题3修复：bootstrap 阶段强制更新基线，防止攻击帧污染初始基线。
        # 超出 bootstrap 后，只在非告警帧更新（原逻辑）。
        should_update = bootstrap or (not joint["alert_confirmed"] and not joint["single_frame_candidate"])
        if should_update:
            alpha = 0.25 if bootstrap else (0.03 if self.lbp_baseline_samples >= 20 else 0.12)
            self.lbp_baseline = (1.0 - alpha) * self.lbp_baseline + alpha * hist
            total = float(self.lbp_baseline.sum())
            if total > 0.0:
                self.lbp_baseline = self.lbp_baseline / total
            self.lbp_baseline_samples += 1

    def _update_scene_baseline(
        self,
        blinding: dict[str, Any],
        a1: dict[str, Any],
        a2: dict[str, Any],
        a3: dict[str, Any],
        joint: dict[str, Any],
    ) -> None:
        """更新场景自适应基线（清晰度/对比度/目标置信度强度/最大特征分）。
        攻击候选/确认帧均冻结更新：避免攻击帧污染基线后失去"相对退化/离群"的判定能力
        （实测只冻结确认帧会让 glare 等攻击的高分帧在确认前被基线吸收→反被 z-score 抑制→漏检）。"""
        if not (self._scene_baseline_enabled or self._blind_enabled):
            return
        if bool(joint.get("alert_confirmed", False)):
            return
        if bool(joint.get("single_frame_candidate", False)):
            # C1: A1 单支饱和(A2/A3 无佐证)的未确认候选不冻结基线,避免多人交叉纹理跑飞;
            # 带 A2/A3 佐证的候选仍冻结(glare/patch 攻击同时抬 A2/A3,照旧冻结+检出)。
            a2s = float(a2["a2_feature_score"])
            a3s = float(a3["a3_feature_score"])
            multi_evidence = a2s >= self._sb_carveout_a2 or a3s >= self._sb_carveout_a3
            if (not self._sb_a1only_carveout) or multi_evidence:
                return
        # 致盲疑似期冻结: 曾确立目标 + 当前完全丢目标 + 有退化佐证 → 不更新基线, 阻断致盲帧污染,
        # 使 sharp_drop/det_drop 维持高位, p_blind 不再单调塌陷。退化佐证门控防合法人离场误冻结。
        if self._blind_suspect_freeze_baseline:
            recent_target_established = (
                sum(self.recent_target_presence) >= self._blind_suspect_recent_target_min
            )
            no_current_target = (self.recent_target_presence[-1] == 0) if self.recent_target_presence else True
            degrade_evidence = (
                bool(blinding.get("blind_independent_support", True))
                and (
                    float(blinding.get("sharp_drop", 0.0)) >= self._blind_suspect_degrade_min
                    or float(blinding.get("glare_blind", 0.0)) >= self._blind_suspect_degrade_min
                )
            )
            if recent_target_established and no_current_target and degrade_evidence:
                return
        self._sb_sharp.append(float(blinding.get("sharpness", 0.0)))
        self._sb_contrast.append(float(blinding.get("contrast", 0.0)))
        self._sb_detstr.append(float(blinding.get("det_strength", 0.0)))
        self._sb_maxfeat.append(max(
            float(a1["a1_feature_score"]),
            float(a2["a2_feature_score"]),
            float(a3["a3_feature_score"]),
        ))

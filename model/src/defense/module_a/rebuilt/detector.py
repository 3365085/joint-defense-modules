from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..types import ROI, ModuleAInput, ModuleAResult
from .target_anchored import TargetAnchoredAnalyzer

# 可选 Rust 原生加速：A1 LBP 直方图聚合（数值与 Python 等价，已对拍验证）。
# 导入失败则自动回退纯 Python 实现，功能不受影响。
try:
    import module_a_native as _NATIVE
except Exception:
    _NATIVE = None

from pathlib import Path as _Path


def _resolve_rebuilt_data_dir() -> _Path:
    """Resolve the data directory holding a4_classifier.pkl and raft_small_* assets.

    Candidate order (first existing wins):
      1. ``MODULE_A_REBUILT_DATA_DIR`` env var
      2. ``defense/module_a/rebuilt/data`` (bundled next to this package)
      3. ``model/data`` (main project data dir)
      4. ``<repo>/rebuilt_demo/data`` (read-only demo source; reused in place)
    Returns candidate 2 as the writable default when none exist yet (so the
    RAFT engine build can create it).
    """
    import os

    here = _Path(__file__).resolve()
    candidates: list[_Path] = []
    env_dir = os.environ.get("MODULE_A_REBUILT_DATA_DIR")
    if env_dir:
        candidates.append(_Path(env_dir))
    candidates.append(here.parent / "data")              # defense/module_a/rebuilt/data
    candidates.append(here.parents[4] / "data")          # model/data
    candidates.append(here.parents[5] / "rebuilt_demo" / "data")  # repo/rebuilt_demo/data (read-only)
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return here.parent / "data"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


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
    """Design-aligned rebuilt Module A detector.

    This demo implementation intentionally does not inherit the production
    Module A detector. It keeps the experiment scoped to rebuilt_demo while
    preserving the public ModuleAInput / ModuleAResult contract.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        module_config = (config or {}).get("module_a", config or {})
        self.frame_size = int(module_config.get("frame_size", 640))
        self.theta_adv = float(module_config.get("rebuilt_theta_adv", 0.65))
        self.theta_media = float(module_config.get("rebuilt_theta_media", 0.55))
        self.theta_media_raw = float(module_config.get("rebuilt_theta_media_raw", 0.50))
        # 支路B（致盲/去信号型攻击：motion_blur/visibility/glare致盲）阈值与场景自适应基线。
        # 这类攻击"抹掉"纹理→A1/A2 反而低于干净帧，靠 A4(支路A) 检不出；改用"相对场景自身
        # 基线的清晰度/对比度/YOLO置信度骤降"来判定（绝对值高的焊接/快动场景不会误报）。
        self.theta_blind = float(module_config.get("rebuilt_theta_blind", 0.55))
        self._blind_enabled = bool(module_config.get("rebuilt_blind_branch", True))
        # 场景自适应基线：对支路A也做"在本场景近况内即视为正常"的额外抑制，压制高能干净场景误报。
        self._scene_baseline_enabled = bool(module_config.get("rebuilt_scene_baseline", True))
        self._scene_baseline_window = int(module_config.get("rebuilt_scene_baseline_window", 30))
        self._scene_baseline_min = int(module_config.get("rebuilt_scene_baseline_min", 8))
        # 问题3修复：A1基线冷启动帧数（前N帧强制更新基线，避免攻击帧污染基线）
        self._bootstrap_frames = int(module_config.get("a1_bootstrap_frames", 8))
        # A4 可插拔分类器：demo 默认接 rebuilt_demo/data/a4_classifier.pkl（用当前检测器重采
        # 特征+分组CV调参重训）。config 可覆盖；缺失/失败自动回退手工规则。不碰主项目配置。
        # 注：分组CV(防泄漏)真实泛化 AUC≈0.70（特征+攻击样本有限），对训练内视频效果好。
        _clf_path = str(module_config.get("a4_classifier_path", "") or "")
        if not _clf_path:
            _bundled = _resolve_rebuilt_data_dir() / "a4_classifier.pkl"
            if _bundled.exists():
                _clf_path = str(_bundled)
        self._classifier = self._load_classifier(_clf_path)
        self._flownet = self._load_flownet()
        self.max_history = 8
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
        self.media_hits: deque[int] = deque(maxlen=self.max_history)
        self.media_scores: deque[float] = deque(maxlen=self.max_history)
        # 支路B 时序确认窗口（与 adv 同机制）
        self.blind_hits: deque[int] = deque(maxlen=self.max_history)
        self.blind_scores: deque[float] = deque(maxlen=self.max_history)
        # 报警保持（2026-06-30 行为调优）：确认后维持 N 帧，避免持续攻击中逐帧候选
        # 短暂掉到 N-of-M 阈值以下时 alert_confirmed 立刻翻回正常（"断警告"）。
        # 经 module_a.rebuilt_alert_hold_frames 配置，<=0 关闭。对齐 legacy attack_state_hold。
        self._alert_hold_frames = int(module_config.get("rebuilt_alert_hold_frames", 12))
        self._alert_hold_remaining = 0
        self._alert_hold_channel = "none"
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
        self.media_track: _MediaTrack | None = None
        # a3b 后台检测间隔（帧）。可经 config 的 module_a.static_image_interval 覆盖。
        # 注：真正消除 a3b GIL 拖累的是 _extract_media_candidates 的候选数上限优化
        # （单轮 221ms→16ms），而非拉大此间隔，故保持原值不牺牲静态媒体检测响应速度。
        self._a3b_interval = int(module_config.get("static_image_interval", 4))
        self._a3b_frame_count = 0
        self._a3b_cache: dict[str, Any] | None = None
        self._a3b_bg_thread: threading.Thread | None = None
        self._a3b_bg_result: dict[str, Any] | None = None
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
        self.adv_hits.clear()
        self.adv_scores.clear()
        self.media_hits.clear()
        self.media_scores.clear()
        self.blind_hits.clear()
        self.blind_scores.clear()
        self._alert_hold_remaining = 0
        self._alert_hold_channel = "none"
        self._sb_sharp.clear()
        self._sb_contrast.clear()
        self._sb_detstr.clear()
        self._sb_maxfeat.clear()
        self._prev_sharp = 0.0
        self.media_track = None
        self._a3b_frame_count = 0
        self._a3b_cache = None
        self._a3b_bg_thread = None
        self._a3b_bg_result = None
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

    def _run_a3b_bg(self, gray, rois, width, height, exposure, flow, a1, a2, a3):
        try:
            self._a3b_bg_result = self._compute_a3b(gray, rois, width, height, exposure, flow, a1, a2, a3)
        except Exception:
            pass

    @staticmethod
    def _empty_a3b() -> dict[str, Any]:
        return {
            "p_media_raw": 0.0, "p_media_raw_triggered": False,
            "p_media": 0.0, "p_media_policy": 0.0, "p_media_triggered": False,
            "p_media_confirmed_score": 0.0, "media_confirmed": False,
            "p_media_type": "normal", "p_media_bbox": None,
            "p_media_target_related": False, "p_media_scores": {},
            "p_media_strong_evidence": False, "p_media_background_static_suppressed": False,
            "a3b_display_score": 0.0, "suppressed_reason": "not_computed",
            "score_cap": 1.0, "media_candidate_allowed": False,
            "a3b_state": "idle", "a3b_moire": 0.0,
        }

    @staticmethod
    def _load_classifier(path: str) -> Any:
        """加载 sklearn pickle 分类器（predict_proba 接口）。路径为空或加载失败时返回 None。"""
        if not path:
            return None
        try:
            import pickle
            from pathlib import Path
            with open(Path(path), "rb") as fh:
                clf = pickle.load(fh)
            if not hasattr(clf, "predict_proba"):
                return None
            return clf
        except Exception:
            return None

    @staticmethod
    def _load_flownet() -> Any:
        """优先级: RAFT-TRT FP16 (~2ms) → GPU LK (~5ms) → DIS-CPU。"""
        try:
            import torch
            if not torch.cuda.is_available():
                print("[FlowNet] CUDA unavailable, DIS fallback", flush=True)
                return None
            device = torch.device("cuda:0")
            result = ModuleADetector._try_load_raft_trt(device)
            if result is not None:
                return result
            return ModuleADetector._load_gpu_lk(device)
        except Exception as e:
            print(f"[FlowNet] init failed: {e}", flush=True)
            return None

    @staticmethod
    def _try_load_raft_trt(device: Any) -> Any:
        """尝试加载/构建 RAFT-small TRT FP16 引擎，失败返回 None。"""
        import torch, tensorrt as trt, os, shutil, tempfile
        from pathlib import Path
        data_dir = _resolve_rebuilt_data_dir()
        engine_path = data_dir / "raft_small_fp16_256.engine"
        onnx_path = data_dir / "raft_small_256.onnx"
        os.makedirs(data_dir, exist_ok=True)
        if not engine_path.exists():
            if not ModuleADetector._build_raft_trt_engine(onnx_path, engine_path):
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
        raft_stream = torch.cuda.Stream()  # 独立 stream：不阻塞 YOLO TRT
        ctx.set_tensor_address("img1", img1_t.data_ptr())
        ctx.set_tensor_address("img2", img2_t.data_ptr())
        ctx.set_tensor_address("flow", flow_t.data_ptr())
        print("[FlowNet] RAFT-small TRT FP16 ready (~2ms, 256x256)", flush=True)
        return {"mode": "raft_trt", "ctx": ctx, "engine": engine,
                "img1_t": img1_t, "img2_t": img2_t, "flow_t": flow_t,
                "raft_stream": raft_stream, "device": device}

    @staticmethod
    def _build_raft_trt_engine(onnx_path: Any, engine_path: Any) -> bool:
        """从 ONNX 构建 RAFT TRT FP16 引擎，成功返回 True。"""
        import torch, torch.nn as nn, tensorrt as trt, shutil, os, tempfile
        from pathlib import Path
        try:
            if not Path(onnx_path).exists():
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                class _W(nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, a, b): return self.m(a, b, num_flow_updates=4)[-1]
                d = torch.device("cuda:0")
                w = _W(raft_small(weights=Raft_Small_Weights.DEFAULT).to(d).eval()).to(d).eval()
                dummy = torch.zeros(1, 3, 256, 256, device=d)
                with torch.no_grad():
                    torch.onnx.export(w, (dummy, dummy), str(onnx_path),
                                      input_names=["img1", "img2"], output_names=["flow"],
                                      opset_version=16, do_constant_folding=True,
                                      export_params=True, dynamo=False)
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                tmp = f.name
            shutil.copy2(str(onnx_path), tmp)
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(tmp):
                os.remove(tmp); return False
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
            eb = builder.build_serialized_network(network, config)
            os.remove(tmp)
            if eb is None: return False
            with open(engine_path, "wb") as f: f.write(memoryview(eb))
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
        import torch, torch.nn.functional as F
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
        u_s = u.squeeze()*(w/S); v_s = v.squeeze()*(h/S)
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
        frame = item.frame
        if frame.ndim == 2:
            gray = frame.astype(np.uint8)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        rois = self._prepare_rois(item.rois, width, height)
        self.recent_target_presence.append(1 if rois else 0)
        self._update_process_fps(item.timestamp)

        exposure = self._compute_scene_context(gray)
        lbp = self._compute_lbp(gray)
        flow = self._compute_flow(self.prev_gray, gray)
        self._last_computed_lbp = lbp  # runner 下一帧可直接复用，无需重算 prev_lbp

        a1 = self._compute_a1(lbp, rois, width, height, exposure)
        a2 = self._compute_a2(lbp, rois, width, height, exposure, flow)
        a3 = self._compute_a3(flow, rois, width, height, exposure)
        # A3b 后台线程：主路径永不等待，使用上一次结果。
        # 按 _a3b_interval 节流：a3b 检测静态媒体（慢变化），无需每帧重算；
        # 不节流会让后台线程 100% 占用并通过 GIL 拖慢主路径 ~5ms/帧。
        a3b = self._a3b_bg_result if self._a3b_bg_result is not None else self._empty_a3b()
        self._a3b_frame_count += 1
        if (self._a3b_frame_count >= self._a3b_interval
                and (self._a3b_bg_thread is None or not self._a3b_bg_thread.is_alive())):
            self._a3b_frame_count = 0
            self._a3b_bg_thread = threading.Thread(
                target=self._run_a3b_bg,
                args=(gray.copy(), list(rois), width, height, dict(exposure), dict(flow), dict(a1), dict(a2), dict(a3)),
                daemon=True,
            )
            self._a3b_bg_thread.start()
        a4 = self._compute_a4(a1, a2, a3, a3b)
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
        blinding = self._compute_blinding(gray, rois, exposure, flow)
        ta_result = self._ta.evaluate(
            rois=rois,
            overexposure={
                "ratio": exposure["overexposure_ratio"],
                "underexposed_ratio": exposure["underexposed_ratio"],
                "temporal_flash": False,  # 由 _joint_decision 独立处理，不双重判断
                "threshold": 0.06,
                "is_glare": False,  # 由 _joint_decision 独立处理，TA 只做目标锚定
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
                "local_max": float(a2.get("change_t_local_max", 0.0)),
                "change_t": float(a2.get("change_t", 0.0)),
            },
            motion={
                "target_related": bool(a3.get("target_related", False)),
                "motion_score": float(a3.get("a3_feature_score", 0.0)),
                "light_flow_score": 0.0,
                "local_max_ratio": float(a3.get("flow_local_anomaly_ratio", 0.0)),
                "light_flow_local_anomaly_ratio": float(a3.get("flow_local_anomaly_ratio", 0.0)),
            },
            static_image={"triggered": bool(a3b.get("p_media_raw_triggered", False))},
            classifier_result={
                "classifier_p_adv": float(a4["p_adv"]),
                "classifier_triggered": bool(a4["p_adv_triggered"]),
            },
            texture={
                "delta_h": float(a1["delta_h"]),
                "local_max": float(a1["delta_h_local_max"]),
            },
        )
        joint = self._joint_decision(a1, a2, a3, a4, a3b, rois, exposure, flow, ta_result, blinding)
        a3b = dict(a3b)
        a3b["p_media_confirmed_score"] = float(joint["p_media_confirmed_score"])
        a3b["media_confirmed"] = bool(joint["media_confirmed"])
        if a3b["media_confirmed"]:
            a3b["a3b_state"] = "confirmed"

        features = self._build_features(a1, a2, a3, a4, a3b, joint, exposure, flow)
        details = {
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "a3b": a3b,
            "blinding": blinding,
            "joint_decision": joint,
            "scene_context": exposure,
            "flow_context": {
                k: v for k, v in flow.items()
                if k not in ("flow", "mag", "residual_mag")
            },
        }
        reason_codes = list(joint.get("reason_codes", []))
        timing_ms = (time.perf_counter() - start) * 1000.0
        result = ModuleAResult(
            frame_idx=int(item.frame_idx),
            p_adv=float(a4["p_adv"]),
            single_frame_suspicious=bool(joint["single_frame_candidate"]),
            alert_confirmed=bool(joint["alert_confirmed"]),
            attack_state_active=bool(joint["alert_confirmed"]),
            reason_codes=reason_codes,
            features=features,
            roi_results=self._build_roi_results(rois, a1, a2, a3, a3b),
            timing_ms=timing_ms,
            details=details,
        )

        self._update_baseline(lbp, joint)
        self._update_scene_baseline(blinding, a1, a2, a3, joint)
        self.prev_gray = gray
        self.prev_lbp = lbp
        self.prev_timestamp = float(item.timestamp or time.time())
        self.prev_brightness = float(np.mean(gray))
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
        if self._flownet is not None:
            try:
                return self._compute_lbp_gpu(gray)
            except Exception:
                pass
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
        import torch, torch.nn.functional as F
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

    def _compute_flow(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> dict[str, Any]:
        h, w = gray.shape[:2]
        if prev_gray is None or prev_gray.shape != gray.shape:
            zeros = np.zeros((h, w), dtype=np.float32)
            return {
                "available": False, "flow": None, "flow_scale": 1.0,
                "mag": zeros, "residual_mag": zeros,
                "global_flow_dx": 0.0, "global_flow_dy": 0.0,
                "global_flow_mag": 0.0, "global_motion_weight": 0.0,
                "background_coherence": 0.0, "valid_ratio": 0.0,
                "mean_motion": 0.0, "mean_residual_motion": 0.0,
            }
        if self._flownet is not None:
            try:
                flow, mag, residual_mag, flow_s, flow_stats = self._raft_flow(prev_gray, gray)
                flow_scale = flow_s / w  # e.g. 256/640 = 0.4
            except Exception:
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
        }

    def _compute_a1(
        self,
        lbp: np.ndarray,
        rois: list[ROI],
        width: int,
        height: int,
        exposure: dict[str, Any],
    ) -> dict[str, Any]:
        if _NATIVE is not None:
            # Rust 原生路径：把每帧上千次 _hist_lbp/_hist_distance 塌缩成一次调用。
            base_arr = None if self.lbp_baseline is None else np.ascontiguousarray(self.lbp_baseline, dtype=np.float32)
            roi_boxes = [(int(r.bbox[0]), int(r.bbox[1]), int(r.bbox[2]), int(r.bbox[3])) for r in rois]
            (delta_h_global, delta_h_local_max, local_mean, local_box,
             delta_h_roi_max, delta_h_target_contrast, delta_h_roi_patch_max,
             target_box) = _NATIVE.a1_lbp_features(
                np.ascontiguousarray(lbp, dtype=np.uint8), roi_boxes, base_arr)
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

        if _NATIVE is not None:
            roi_boxes = [(int(r.bbox[0]), int(r.bbox[1]), int(r.bbox[2]), int(r.bbox[3])) for r in rois]
            (change_t_global, change_t_local_max, change_t_local_mean, local_box,
             change_t_roi_max, target_box, change_t_context_mean) = _NATIVE.a2_change_features(
                np.ascontiguousarray(lbp, dtype=np.uint8),
                np.ascontiguousarray(self.prev_lbp, dtype=np.uint8),
                roi_boxes, 0.45,
            )
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
        if flow["available"] and target_box is not None:
            x1, y1, x2, y2 = target_box
            motion_aligned = float(np.mean(flow["mag"][y1:y2, x1:x2])) / 3.0
        elif flow["available"]:
            x1, y1, x2, y2 = local_box
            motion_aligned = float(np.mean(flow["mag"][y1:y2, x1:x2])) / 3.0
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
        if _NATIVE is not None:
            local_residual, mean_residual_grid, local_box = _NATIVE.best_grid_value_f32(
                np.ascontiguousarray(residual, dtype=np.float32), 8)
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
            res_val = float(np.mean(residual[fy1:fy2, fx1:fx2]))
            mag_val = float(np.mean(mag[fy1:fy2, fx1:fx2]))
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
    ) -> dict[str, Any]:
        s1 = float(a1["a1_feature_score"])
        s2 = float(a2["a2_feature_score"])
        s3 = float(a3["a3_feature_score"])
        a3b = a3b or {}
        a3b_scores = a3b.get("p_media_scores", {}) or {}

        # 按设计稿 §7.2 构建 16 维特征向量（A1×5 + A2×5 + A3×6）
        # 问题2修复：不再只传三个标量，而是传完整特征组
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
            # A3b 组（4维）
            float(a3b.get("p_media_raw", 0.0)),
            float(a3b_scores.get("flow_gap", 0.0)),
            float(a3b_scores.get("warp_residual", 0.0)),
            float(a3b_scores.get("display", 0.0)),
        ]

        # 贡献权重：优先从分类器 feature_importances_ 按模块分组求和，再乘当前帧分数
        # 无分类器时用固定经验乘数（1.00/1.08/1.12）回退
        if self._classifier is not None and hasattr(self._classifier, "feature_importances_"):
            fi = self._classifier.feature_importances_  # shape (20,)
            # 用各模块的全局重要性之和作为乘数，比固定 1.00/1.08/1.12 更有数据依据
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

        if self._classifier is not None:
            try:
                # 分类器可能是旧版(20维) 也可能是含P4的新版(25维)；按其期望维度裁/补零。
                expected = int(getattr(self._classifier, "n_features_in_", len(a4_feature_vector)) or len(a4_feature_vector))
                if expected == len(a4_feature_vector):
                    fv = a4_feature_vector
                elif expected < len(a4_feature_vector):
                    fv = a4_feature_vector[:expected]
                else:
                    fv = a4_feature_vector + [0.0] * (expected - len(a4_feature_vector))
                p_adv_raw = float(self._classifier.predict_proba([fv])[0][1])
            except Exception:
                p_adv_raw = self._rule_p_adv(max_score, second_score, third_score, multi_evidence)
        else:
            p_adv_raw = self._rule_p_adv(max_score, second_score, third_score, multi_evidence)

        p_adv_calibrated = _clamp(p_adv_raw)
        p_adv = p_adv_calibrated
        return {
            "p_adv_raw": float(p_adv_raw),
            "p_adv_calibrated": float(p_adv_calibrated),
            "p_adv": float(p_adv),
            "p_adv_triggered": bool(p_adv >= self.theta_adv),
            "a1_contribution": float(contributions["a1_contribution"]),
            "a2_contribution": float(contributions["a2_contribution"]),
            "a3_contribution": float(contributions["a3_contribution"]),
            "dominant_adv_input": dominant,
            "a4_second_feature_score": float(second_score),
            "a4_third_feature_score": float(third_score),
            "a4_multi_evidence": float(multi_evidence),
            "a4_synergy": float(synergy),
            "a4_feature_vector": a4_feature_vector,
            "a4_classifier_used": self._classifier is not None,
            "theta_adv": float(self.theta_adv),
        }

    @staticmethod
    def _rule_p_adv(max_score: float, second_score: float, third_score: float, multi_evidence: float) -> float:
        """规则融合回退，待有标注数据后用 XGBoost/MLP 替换（见设计稿 §7.1）。
        A3 使用 Farneback 稠密光流（非 FlowNetC-S）；FlowNetC-S 需要模型文件，
        部署后可替换 _compute_flow() 并保持 a3_feature_score 接口不变。
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
        if _NATIVE is not None and hasattr(_NATIVE, "blinding_laplacian_var"):
            sharpness = float(_NATIVE.blinding_laplacian_var(np.ascontiguousarray(gray, dtype=np.uint8)))
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
            }
        if not ready:
            # P1 冷启动绝对兜底：场景基线未就绪时(如视频开头即攻击)，靠"曾有目标却骤然漏检
            # + (帧间清晰度骤降 或 强过曝)"判定致盲——目标丢失是强证据但需退化佐证防误报。
            now_lost = len(rois) == 0
            degrade = max(sharp_drop_short, glare_blind0)
            if recent_present and now_lost:
                p_cold = _clamp(0.45 + 0.55 * degrade)
            else:
                p_cold = _clamp(0.70 * glare_blind0)
            btype = "cold_glare" if glare_blind0 >= sharp_drop_short else "cold_blur"
            return {
                "p_blind": float(p_cold), "p_blind_triggered": bool(p_cold >= self.theta_blind),
                "blind_ready": False, "sharpness": sharpness, "contrast": contrast,
                "det_strength": det_strength, "sharp_drop": float(sharp_drop_short),
                "contrast_drop": 0.0, "det_drop": 0.0, "glare_blind": float(glare_blind0),
                "target_loss": float(recent_present and now_lost), "blind_type": btype,
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

        return {
            "p_blind": float(p_blind),
            "p_blind_triggered": bool(p_blind >= self.theta_blind),
            "blind_ready": True,
            "sharpness": sharpness, "contrast": contrast, "det_strength": det_strength,
            "ref_sharpness": ref_sharp, "ref_contrast": ref_contrast, "ref_det": ref_det,
            "sharp_drop": float(sharp_drop), "contrast_drop": float(contrast_drop),
            "det_drop": float(det_drop), "glare_blind": float(glare_blind),
            "target_loss": float(target_loss), "blind_type": blind_type,
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
        # 原生路径需要 uint8 连续视图（C-contiguous）；只算一次，供逐候选 a3b_one_box_stats 复用。
        edge_mask_u8 = np.ascontiguousarray(edge_mask, dtype=np.uint8) if _NATIVE is not None else None
        frame_area = float(width * height)
        candidates: list[dict[str, Any]] = []

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
        ) -> None:
            # 全局候选上限：复杂画面会产生数百个候选，每个都做昂贵的逐框统计
            # （边界/纹理/IoU），不设上限会使单轮飙到 200-500ms 并抢 GIL 拖垮主路径。
            if len(candidates) >= 64:
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
            local_edges = edge_mask[y1:y2, x1:x2]
            if _NATIVE is not None:
                (edge_density, border_edge_density, inner_edge_density,
                 border_mean, inner_mean, gray_std) = _NATIVE.a3b_one_box_stats(
                    edge_mask_u8, gray, int(x1), int(y1), int(x2), int(y2))
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
            if values.size == 0:
                return []
            vmax = float(np.max(values))
            if vmax <= 1e-6:
                return []
            norm = values.astype(np.float32) / vmax
            threshold = max(0.18, float(np.percentile(norm, 88)) * 0.82)
            groups: list[tuple[int, float]] = []
            start: int | None = None
            for idx, value in enumerate(norm):
                if value >= threshold and start is None:
                    start = idx
                elif value < threshold and start is not None:
                    end = idx
                    segment = norm[start:end]
                    weights = segment + 1e-4
                    center = int(round(float(np.average(np.arange(start, end), weights=weights))))
                    groups.append((center, float(np.max(segment))))
                    start = None
            if start is not None:
                end = len(norm)
                segment = norm[start:end]
                weights = segment + 1e-4
                center = int(round(float(np.average(np.arange(start, end), weights=weights))))
                groups.append((center, float(np.max(segment))))
            groups.sort(key=lambda item: item[1], reverse=True)
            selected: list[tuple[int, float]] = []
            min_gap = max(8, int(min(width, height) * 0.035))
            for center, strength in groups:
                if all(abs(center - old_center) >= min_gap for old_center, _ in selected):
                    selected.append((center, strength))
                if len(selected) >= limit:
                    break
            selected.sort(key=lambda item: item[0])
            return selected

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
            and a4["p_adv"] >= self.theta_adv
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
            and a4["p_adv"] >= self.theta_adv
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
        ta_suspicious = bool(ta_result["suspicious"]) if ta_result is not None else True
        ta_classifier_bonus = bool(ta_result.get("classifier_bonus", False)) if ta_result is not None else False
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
                and not visibility_texture_rescue
            )
        adv_candidate_allowed = bool(
            a4["p_adv"] >= self.theta_adv
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
        media_candidate_allowed = bool(
            a3b["media_candidate_allowed"]
            and not normal_motion_texture_change
        )
        adv_single_frame_candidate = bool(a4["p_adv_triggered"] and adv_candidate_allowed)
        # 支路B 致盲候选：p_blind 触发即候选（_compute_blinding 内部已做"曾有目标+退化"门控）
        blind_single_frame_candidate = bool(blinding.get("p_blind_triggered", False))
        # A3b is disabled as independent trigger — background structures (walls, doorways)
        # produce false positives. A3b features should be added to XGBoost instead.
        # Only re-enable when moiré pattern detection is added to A3b.
        media_single_frame_candidate = False
        single_frame_candidate = bool(adv_single_frame_candidate or blind_single_frame_candidate)
        if adv_single_frame_candidate:
            self.adv_hits.append(1)
            self.adv_scores.append(float(a4["p_adv"]))
        else:
            self.adv_hits.append(0)
            self.adv_scores.append(0.0)
        if blind_single_frame_candidate:
            self.blind_hits.append(1)
            self.blind_scores.append(float(blinding.get("p_blind", 0.0)))
        else:
            self.blind_hits.append(0)
            self.blind_scores.append(0.0)
        if media_single_frame_candidate:
            self.media_hits.append(1)
            self.media_scores.append(float(a3b["p_media_policy"]))
        else:
            self.media_hits.append(0)
            self.media_scores.append(0.0)
        adv_count = sum(list(self.adv_hits)[-window_frames:])
        media_count = sum(list(self.media_hits)[-window_frames:])
        blind_count = sum(list(self.blind_hits)[-window_frames:])
        adv_confirmed = bool(adv_count >= adv_hit_required)
        media_confirmed = bool(media_count >= media_hit_required)
        # 支路B 确认：致盲攻击持续多帧（与高误报场景一致用更高占比），需曾有目标语境
        blind_hit_required = int(math.ceil(window_frames * (0.67 if exposure["high_false_positive_scene"] else 0.60)))
        blind_confirmed = bool(blind_count >= blind_hit_required)
        p_media_confirmed_score = (
            float(max(list(self.media_scores)[-window_frames:] or [0.0]))
            if media_confirmed else 0.0
        )
        adv_primary_preferred = bool(
            adv_confirmed
            and a4["p_adv"] >= max(self.theta_adv, 0.70)
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

        global_suppressed = False
        global_suppressed_reason = "none"
        alert_confirmed = False
        primary_channel = "none"
        current_adv_overrides_stale_media = bool(
            adv_confirmed
            and adv_single_frame_candidate
            and not media_single_frame_candidate
            and not robust_media_evidence
        )
        if (
            (adv_primary_preferred or current_adv_overrides_stale_media)
            and adv_confirmed
            and not global_suppressed
        ):
            alert_confirmed = True
            primary_channel = "adv"
        elif media_confirmed and not global_suppressed:
            alert_confirmed = True
            primary_channel = "media"
        elif adv_confirmed and not global_suppressed:
            alert_confirmed = True
            primary_channel = "adv"
        elif blind_confirmed and not global_suppressed:
            alert_confirmed = True
            primary_channel = "blind"

        # --- 报警保持窗口（2026-06-30 行为调优，修复"风机出现时断警告"）---
        # 一旦确认，维持 _alert_hold_frames 帧；期间逐帧候选短暂不足也保持 ATTACK，
        # 期间再次确认则刷新保持。global_suppressed（明确抑制）时不保持。
        alert_held = False
        if alert_confirmed:
            self._alert_hold_remaining = self._alert_hold_frames
            self._alert_hold_channel = primary_channel
        elif (
            self._alert_hold_frames > 0
            and self._alert_hold_remaining > 0
            and not global_suppressed
        ):
            self._alert_hold_remaining -= 1
            alert_confirmed = True
            alert_held = True
            primary_channel = self._alert_hold_channel if self._alert_hold_channel != "none" else "adv"

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
            "media_hit_required": int(media_hit_required),
            "blind_hit_required": int(blind_hit_required),
            "adv_count": int(adv_count),
            "media_count": int(media_count),
            "blind_count": int(blind_count),
            "alert_held": bool(alert_held),
            "alert_hold_remaining": int(self._alert_hold_remaining),
        }
        return {
            "adv_single_frame_candidate": bool(adv_single_frame_candidate),
            "media_single_frame_candidate": bool(media_single_frame_candidate),
            "single_frame_candidate": bool(single_frame_candidate),
            "candidate_source": candidate_source,
            "adv_candidate_allowed": bool(adv_candidate_allowed),
            "media_candidate_allowed": bool(media_candidate_allowed),
            "adv_confirmed": bool(adv_confirmed),
            "media_confirmed": bool(media_confirmed),
            "blind_confirmed": bool(blind_confirmed),
            "blind_single_frame_candidate": bool(blind_single_frame_candidate),
            "p_blind": float(blinding.get("p_blind", 0.0)),
            "blind_type": str(blinding.get("blind_type", "none")),
            "scene_baseline_normal": bool(scene_baseline_normal),
            "p_media_confirmed_score": float(p_media_confirmed_score),
            "alert_confirmed": bool(alert_confirmed),
            "alert_level": alert_level,
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
            "visibility_texture_probe": bool(visibility_texture_probe),
            "visibility_texture_rescue": bool(visibility_texture_rescue),
            "physical_patch_rescue_evidence": bool(physical_patch_rescue_evidence),
            "a3_residual_fallback": bool(a3_residual_fallback),
            "unsupported_a3_motion": bool(unsupported_a3_motion),
            "a3_only_background_motion": bool(a3_only_background_motion),
            "reason_codes": reason_codes,
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
        if bool(joint.get("single_frame_candidate", False)) or bool(joint.get("alert_confirmed", False)):
            return
        self._sb_sharp.append(float(blinding.get("sharpness", 0.0)))
        self._sb_contrast.append(float(blinding.get("contrast", 0.0)))
        self._sb_detstr.append(float(blinding.get("det_strength", 0.0)))
        self._sb_maxfeat.append(max(
            float(a1["a1_feature_score"]),
            float(a2["a2_feature_score"]),
            float(a3["a3_feature_score"]),
        ))

# A3b 边缘 NPU 重设计（2026-05-13 第二轮）

## 当前 A3b 在 NPU 上为什么不达标

### NPU 三大禁忌
1. **CPU 专属算子**：cv2.Canny / cv2.findContours / cv2.approxPolyDP /
   cv2.boundingRect / cv2.warpPerspective / cv2.findHomography / SIFT / RANSAC
2. **动态 Python 循环**：candidates 数量未知，contours 数量未知 → 不能 JIT
3. **CPU↔NPU 来回**：每帧至少 1 次 `.cpu().numpy()` + 返回

### 现有 A3b 热点来源
- `extract_edge_candidates`：Canny + contours + bounding rect + polygon approx，**全是 CPU** — 2-3 ms/调用
- `candidate_track` 与 `homography_verifier`：IoU 跟踪 + cv2.findHomography — **全是 CPU**
- `static_media/detector.py` 大循环里：`.mean().item()` × N + cv2.rectangle mask 等

**结论**：即便我们在 RTX 5060 跑到 9-11 ms，NPU 上至少 ×3 因为大部分是 CPU 算子
串行在 ARM 核上。边缘 NPU 上想达到 15 ms/帧必须**彻底换成纯张量流水**。

---

## 第一轮探索的教训（prototype 的结果 2026-05-13 08:45）

我试了 4 个"纯替代信号"：moire FFT / planar flow gap / color saturation /
gradient orientation。跑 7 clip 后发现：

```
clip                    moire  flow   color  gradient
adv_patch               0.000  0.469  0.448  0.726
clean_baseline          0.061  0.571  0.317  0.743    ← 干净场景 flow 最高！
glare_attacked          0.000  0.258  0.645  0.798
motion_blur             0.014  0.388  0.290  0.812
occlusion               0.098  0.320  0.298  0.810
screen_spoof            0.027  0.376  0.018  0.740    ← 屏幕翻拍色彩反而最低！
visibility_degradation  0.037  0.339  0.233  0.806
```

四个信号都**没有良好的 separation**。原因：

1. **Moire 假设不成立**：`samples/screen_spoof_attacked.mp4` 是 2618 帧长视频，
   经过 H.264 编码把高频 moire 磨平了。信号在 0.03 级别。
2. **Flow-gap 反向**：干净仓库 clip 里人走动是正常场景中最大的 flow_gap 来源。
3. **Color**：glare 场景最高（强光让屏幕 saturation 反而低）。
4. **Gradient orientation**：几乎所有场景都 ~0.8，因为 640×640 里建筑线条占多数。

**结论**：不能简单换算法。现有 A3b 的好处在于它**已经把 YOLO ROI + 运动对比**
这些 NPU 友好的信号用好了。真正不 NPU 友好的只是 L0 候选提取和 L2 homography。

---

## 修正方案：留存好部分，替换坏部分

现有 A3b 由两条路径构成：
1. **Legacy patch-track 路径**（在 `detector.py` 大循环里）：基于 YOLO ROI +
   patch 相似度 + 运动对比 + 屏幕边缘统计。**全是 torch，NPU 友好**。
2. **A3+ 路径**（L0→L1→L2→L3）：Canny → contours → IoU tracks →
   findHomography。**CPU 专属，不 NPU 友好**。

现实：从已训练的 `module_a_static_media_classifier_v1.json` 看（16 维特征输入），
最重要的权重全在"motion / contrast / screen_context_*"这些 Legacy 提供的信号上
（`best_context_motion` 1.00，`best_roi_motion` 在隐层 0.63），L0/L2 的贡献
（通过 `best_static_image_score` 这种汇总进入）反而不是主导。

所以**真正该做的事**：

### 方案 A：增加 `static_image_backend: legacy_yolo_only` 模式

- 完全跳过 `extract_edge_candidates` 与 `run_l2_homography`（它们都 cv2）。
- 只保留 Legacy patch-track 路径（全 torch）。
- `p_media`/`p_media_triggered` 置 None（这两个只在 L0/L1/L2 存在时才有语义）。
- `static_image_triggered` 仍然走 Legacy 路径的触发逻辑。

预期：在 RTX 上 A3b p95 从 9-11 ms → **~1-2 ms**；在 NPU 上所有算子能直接跑。

### 方案 B：补充一个独立的 NPU 友好的频域/色彩信号（作为辅助）

虽然 prototype 的信号 separation 差，但**融合**到已有 MLP 里可能有增益。
这是训练问题，不是算法问题。今晚先不动分类器。

---

## 今晚选择做方案 A

### 代码位置
- 新增 `module_a.static_image_backend` 配置（默认 `legacy` 保持向后兼容）
- `GPUStaticMediaSpoofDetector.compute`：根据 backend 跳过 L0/L1/L2
- 保留既有 Legacy 所有触发逻辑、不影响 adv_patch / screen_spoof 检测率

### 预期测量
- A3b wallclock p95：9-11 ms → **~1-2 ms**
- 7 clip smoke：alert_frames 偏差 < 5%，总体仍然全绿
- NPU portability：A3b 调用链里不再有 cv2 调用（通过 grep 验证）

### 风险缓解
- 默认仍是 `legacy` 全功能，`legacy_yolo_only` 是 opt-in
- 单元测试覆盖两种 backend 都能处理输入

---

## 未来（合并后）可进一步做的事

1. **方案 B 落地**：用手工合成的屏幕翻拍数据 + 实采对抗补丁，重训分类器把
   moire / flow-gap 信号纳入。
2. **INT8 量化**：整个 ModuleA 用 `torch.ao.quantization` 或 ONNX Runtime 的
   动态量化，边缘 NPU 常用。
3. **TorchScript / ONNX export**：把 ModuleA 主 forward 图导出成静态图，
   然后 RKNN-toolkit2 / QAT 跑 INT8。



---

## 实施结果（2026-05-13 10:40）

### 已完成
- `candidate_extraction_torch.py`：纯 torch L0 候选提取器（Sobel + density grid + prefix-sum 矩形枚举）
- `feature_builder.py`：`torch_native` 后端分发 + `run_l2_torch_native` 平面性替代
- `detector.py`：`torch_native` 模式跳过 Legacy YOLO-ROI 循环，纯走 A3+ cascade
- `config.py`：`backend` 支持 `legacy | legacy_yolo_only | torch_native`

### 测量结果

| 指标 | legacy | torch_native |
|---|---|---|
| screen_spoof alerts | 549 | **1536** ✅ |
| adv_patch alerts | 504 | 需要 Legacy loop → 0 ❌ |
| clean_baseline alerts | 0 | **3248** ❌ |

### 根因分析

`torch_native` 模式下 clean_baseline 出现大量 FP 的原因：
1. torch 候选提取器在低阈值（0.03/0.08）下对建筑边缘过于敏感
2. 这些候选被 CandidateTrackManager 持续追踪，track_score 快速达到 1.0
3. `_compute_p_media_decision` 的 `static_image_score` 公式中 `track_score` 权重 0.25 + `low_residual` 0.15 + `low_flow_gap` 0.15 = 0.55，加上 edge_score 0.20×0.6 = 0.12，总分 0.67 > 0.65 阈值
4. 虽然加了 `has_motion_evidence` 门控（plane_score >= 0.15 OR flow_gap >= 0.3），但 `run_l2_torch_native` 在静态场景下 flow_gap 可以达到 0.3+（因为 YOLO 检测到的人在走动，ROI 内外 diff 不同）

### 结论

`torch_native` 后端需要更多调参工作才能在 clean 场景上达到 0 FP。这不是今晚能完成的——需要：
1. 在真实 NPU 硬件上跑 QAT 微调阈值
2. 或者重训 `_compute_p_media_decision` 的权重（需要标注数据）
3. 或者把 adv_patch 检测也移到 A3+ cascade（目前只有 Legacy loop 能检测 adv_patch）

**当前状态**：`torch_native` 标记为 experimental，默认 `legacy` 不变。代码已就位，合并后在 NPU 硬件上继续调参。

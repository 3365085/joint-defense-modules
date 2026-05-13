# A3b 纯 torch 候选提取器（第三轮方案）

## 上一轮教训

`legacy_yolo_only` 直接跳过 A3+ cascade 后：
- `adv_patch`: 504 → 0 alerts ❌
- `screen_spoof`: 549 → 9 alerts ❌

结论：**A3+ cascade 确实在做关键工作**，不能简单去掉。

## 真正需要的：把 cv2 Canny + contours + homography 换成 torch kernel

### 1. Canny → Sobel + NMS 阈值（torch）

Canny 的核心步骤：高斯滤波、梯度、非极大抑制、双阈值滞后。用 torch 等价实现：

```python
def torch_canny(gray, low=0.1, high=0.3):
    # gray: (1, 1, H, W) in [0, 1]
    # 高斯平滑（5×5 Gaussian conv）
    blurred = F.conv2d(gray, gauss_kernel, padding=2)
    # Sobel 梯度
    gx = F.conv2d(blurred, sobel_x, padding=1)
    gy = F.conv2d(blurred, sobel_y, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)
    # 角度量化到 4 个方向（0/45/90/135）
    angle = torch.atan2(gy, gx)
    # 简化 NMS：对每个方向用 3×3 dilate 比较
    # ... (可以用 F.max_pool2d 近似实现)
    # 双阈值：直接两个 mask
    strong = mag > high
    weak = (mag > low) & (mag <= high)
    # 滞后连接：对 strong dilate 几次再 AND weak
    return (strong | (weak & F.max_pool2d(strong, 3, stride=1, padding=1)))
```

这里简化 NMS 可以让边缘稍粗但语义相同。NPU 友好（conv + pool + element-wise）。

### 2. findContours → connected components via torch label

NPU 真正不支持的是 `cv2.findContours`，它是基于 Suzuki-Abe 算法的连通域追踪。
在 NPU 上做连通域很难。但我们其实**不需要精确的连通域**——A3+ 只要候选矩形
bounding box。可以用更粗糙但可向量化的方法：

**方案**：把边缘图做 **max_pool2d 收缩到 8×8 网格**，每个 cell 存是否有足够
边缘密度，然后用**水平扫描 + 垂直扫描**找连续高密度块作为 bbox 候选。

```python
def extract_bboxes_vectorized(edges, grid=16):
    # edges: (1, 1, H, W) binary
    # 降采样到 grid×grid 密度图
    grid_map = F.adaptive_avg_pool2d(edges.float(), (grid, grid)) > 0.2  # (1,1,grid,grid)
    # 水平扫描：每行连续 True 的最长区段
    # 垂直扫描：类似
    # 交叉 → 候选 bbox
    ...
```

这不如 cv2.findContours 精确，但能识别矩形占主导的区域。对屏幕翻拍 / 对抗
补丁足够。

### 3. findHomography → 光流残差（已有方案）

之前提过：用 ROI 内外的 `GPULightOpticalFlowDetector` 残差比替代单应性检验。
这个改动相对独立，可以单独做。

---

## 工程风险评估

这是**算法级重写**，不是简单优化：

- 实现量：约 200-300 行新 torch 代码 × 2（candidate + homography）
- 验证难度：需要大量 A/B 测试确认 adv_patch + screen_spoof 检测率不降
- 时间成本：~4-6 小时深度调参

**今晚决定**：把基础设施 + 契约铺好（`backend` 配置 + 编辑/单元测试），
**真正的 torch 实现在合并后的 Phase B 做**，因为：

1. 风险：仓促重写可能回退 detection rate
2. 需要训练：NPU kernel 的阈值需要在真实 NPU 上用 QAT 微调
3. 现有实现配合 `l0_interval=5` + batched CPU transfer 在 RTX 上 **已经是 9-11 ms**
4. 真正到 NPU 时候，建议评估：
   - RK3588 的 cv2 是否有硬件加速 Canny（有的 Mali GPU 支持 OpenCL 版 Canny）
   - 是否能用 CPU 线程做 L0（在 NPU 并行推理主模型时）

## 今晚能做的具体事

### ✅ 已做
- 新增 `static_image_backend: legacy | legacy_yolo_only` 配置，合并后可根据
  NPU 能力选择
- `探索/A3b_edge_npu_redesign.md` + `A3b_torch_native_candidate.md` 描述两种
  切换策略 + 后续 torch 替代实现草图
- `tests/run_samples_smoke_edge.py` 测量 `legacy_yolo_only` 的检测回落
  （做对比基线）

### 🚧 下一阶段（合并后）
- 实现 `torch_canny` + `extract_bboxes_vectorized`
- 实现 `planar_flow_homography` 替代 cv2.findHomography
- 逐特征 A/B 对比 detection rate
- RKNN export 验证

---

## 结论

今晚的探索揭露了一个关键事实：**A3+ cascade 对 adv_patch / screen_spoof 的
检测是必需的**，不是多余负担。所以优化方向应该是**把 cv2 换成 torch**，
而不是**简单跳过 A3+**。

这是一个需要较大工程量的重写，不适合今晚一次完成。今晚把契约层（`backend`
config）铺好，让合并后的人（或我自己）能顺着这条路走下去。

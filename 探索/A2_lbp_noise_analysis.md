# A2 LBP 时域纹理噪声分析 + 改进方案

## 背景

`clean_baseline.mp4`（4548 帧仓库巡检真实视频）里：
- `local_temporal_texture_change`：**3582 次**
- `temporal_texture_change`：**644 次**

虽然 3/5 告警状态机挡住了，最终 `alert_frames=0`，但 reason code 级统计会给下游 evidence / event 审计造成噪音；也说明 A2 时域纹理对真实场景的抖动 sensitivity 太高。

## 原理

### 当前实现 (`temporal_texture.py`)

```python
diff = |curr_lbp - prev_lbp| / 255.0       # 全图 LBP 码差
change_t = diff.mean()                     # 全局均值
local = F.adaptive_avg_pool2d(diff, 16×16) # 16×16 grid
local_max = local.max()
```

### Rule fusion 阈值（`module_a_baseline.yaml`）

```yaml
temporal_trigger: 0.03         # change_t >= 0.03 → 触发 temporal_texture_change
local_temporal_trigger: 0.045  # local_max >= 0.045 → 触发 local_temporal_texture_change
```

## 为什么会噪声高

1. **LBP 是 8bit 离散码**：即使像素值只变 1 级灰度（真实视频 JPEG 噪声、码率抖动），LBP 编码可能从 `0b00011111` 跳到 `0b00100000`，差异 `31/255=0.12`，远超阈值。
2. **grid_size=16 在 640×640 上每格 40×40 像素**：摄像头焦距远的场景里小人头可能只占 10×10，一个小运动就让该格 diff 均值拉满。
3. **阈值 0.045 (即约 11/255) 本身就很低**：主要是为了对抗 adv_patch 的细纹动作。

## 改进方案

### 方案 A：双尺度 LBP 平均 (radius=1 + radius=3)

`radius=3` 覆盖范围大、对局部噪声容忍但易漏检小 trigger。`radius=1` 覆盖范围小、对细节敏感但对 sensor noise 敏感。把两者加权平均：`delta = 0.3 * r1 + 0.7 * r3`。

优点：抗噪声 + 保留细节。
缺点：增加一次 LBP 编码（+~0.5 ms GPU）。

### 方案 B：在 LBP diff 前先做一次 3×3 spatial median

对 LBP 码图先滤一次中值滤波再做时域差分。能有效抑制零散椒盐噪声。

优点：实现便宜（一次 `F.max_pool` approximation）。
缺点：对 adv_patch 的检测率可能略降。

### 方案 C：增加 "最小持续" 阈值

要求 `change_t >= temporal_trigger` 在 N=2 连续帧都成立才 emit。把单帧抖动吃掉。

实现：`GPUTemporalTextureAnalyzer` 内部记一个 `_last_triggered` 窗口。
优点：几乎没性能代价，直接减掉孤立假阳性。
缺点：对 glare / 快速 adv_patch 会晚一帧响应。

### 方案 D：软阈值 + 背景 EMA

维护 `change_t` 的长期 EMA（~30 帧），触发条件改为 `change_t > EMA * 2.0 AND change_t > floor`。每个场景的基线不同，自适应下去。

优点：对低码率 / 高码率视频都适应。
缺点：训练/热启期需要冷启窗口。

## 选定方案

**优先 C + D 组合**：方案 C 便宜、对所有路径都有效；方案 D 自适应。保留 reason code 语义不变。

方案 A 需要动 LBP 本身（同时影响 `summarize` 的 `delta_h`），改动面大暂缓。

## 验证指标

- clean_baseline 上 `local_temporal_texture_change` 次数 < 500（目标 <1000，今天首轮）
- 7 clip smoke 必须全绿
- 单元测试全过
- A2 p95 延迟不超过当前值 + 5%

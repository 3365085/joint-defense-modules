# A3b 媒体 ROI 与 PPE 业务证据协作复核

## 背景

用户在“视频中出现干扰视频”场景中指出：屏幕/媒体内容里的人员、头部、安全帽不应直接当作真实 PPE 业务证据；同时中间橙色马甲目标的 `head/helmet` 较稳定，但 `person` 输出存在跳变。

本轮按 API 复跑同一视频，使用 `http://localhost:7860/api/start` 启动三类模型：

- 视频：`素材/视频中出现干扰视频/5e145bf778577e75118502e263d00c41.mp4`
- 模型：`model/baseline_training/runs/baseline_yolov8_three_put/best.pt`
- 类别：`helmet, head, person`
- profile：`desktop_rtx`
- A3b：`static_image_enabled=true`, `a3b_sensitivity=balanced`

## API 事实

修复前 `run_id=5`：

- `/api/overlay` 共 116 条记录。
- `ppe_confirmed` 首次出现在 frame 3 / `t=0.1s`，原因 `bare_head_without_matched_helmet`。
- A3b 首次 `suspect` 出现在 frame 37 / `t=1.233s`，来源 `observed_window`。
- `person` 计数在 1 到 3 之间跳变；早期和中段存在多个重叠 person track。
- `/api/overlay` 原本没有 `a3b_bbox` 字段，只有 `/api/status` 末帧能看到 `a3b_bbox`。

修复后 `run_id=1`：

- `/api/overlay` 共 116 条记录。
- 从 frame 32 / `t=1.067s` 起，overlay 出现 `a3b_bbox=[200,40,526,415]`，`a3b_p_media=0.55`，PPE 记录变为 `ppe_reason=source_auth_media_roi_suppressed`。
- 103 条 overlay 记录显示 `ppe_source_auth_media_suppressed=true`。
- 末帧 `/api/status` 显示 `ppe_person_count=0`、`ppe_head_count=0`、`ppe_helmet_count=0`，`ppe_source_auth_media_suppressed_count=3`，`ppe_source_auth_media_bbox` 与 `a3b_bbox` 对齐。
- 仍有前 12 条记录在 A3b 尚未给出 `p_media_bbox` 前触发 PPE warning/confirmed；这是当前 A3b 初始观测窗口内的因果盲区，不能用尚未产生的媒体 ROI 做同帧抑制。

## 代码链路依据

- `model/src/defense/runtime/frame_processor.py`
  - `process()` 先从 Module A `info` 中提取 `details.module_a_features.static_media`。
  - `evaluate_ppe_business()` 接收 `p_media_bbox` 与 source-auth 抑制激活状态。
  - `_build_status()` 输出 `a3b_bbox`、`a3b_p_media` 和 PPE source-auth 抑制统计字段。
- `model/src/defense/runtime/ppe_business.py`
  - 仅当 A3b/media ROI 已存在且当前 PPE 检测框落入该 ROI 时，过滤 `person/head/helmet` 业务证据。
  - 当当前帧所有 PPE 证据都被媒体 ROI 抑制时，重置 PPE 时序状态和 display tracker，避免旧框/旧告警继续滞留。
- `model/src/defense/runtime/overlay_records.py`
  - `/api/overlay` 增加 additive 字段：`a3b_bbox`、`a3b_p_media`、`ppe_source_auth_media_suppressed*`。
- `model/src/defense/module_a/ppe_postprocess.py`
  - `person` 仍只作为上下文参与 head/helmet 误报抑制，`candidate` 仍由 `head_count > 0` 驱动，没有把 `person` 改成 PPE 违规目标。

## 当前判断

1. `person/head/helmet` 三类已经接入检测和状态输出。`person` 是协作上下文，不直接触发 PPE 违规。
2. `person` 不稳定主要不是类别映射问题，而是同一目标附近多个 person 框没有进一步做 person-level 去重，`kept_person_indices` 当前会保留所有达到阈值的 person 框。此项属于计数/后处理口径优化，待实验确认，不在本次最小修复中合并。
3. 屏幕/媒体内 PPE 误触发的根因是 PPE business 在修复前先于 A3b 状态汇总运行，拿不到 `p_media_bbox`，因此屏幕内 `head/person/helmet` 会被当作真实业务证据。
4. 本次修复将 A3b/media ROI 回流到 PPE business，同帧抑制 ROI 内 PPE 证据，并保持 Web API 字段兼容。

## 影响范围

- 不改变模型权重。
- 不改变类别语义。
- 不改变 head/helmet/person 的基础阈值。
- 不增加主检测路径 GPU 推理。
- 不删除或重命名既有 Web API 字段，只增加 source-auth/PPE 协作观测字段。

## 待实验确认

- 前 1.07 秒 A3b 尚未产生 `p_media_bbox`，PPE 仍可能短暂确认。若产品要求“媒体场景首秒也绝不出现 PPE 业务告警”，需要单独设计 source-auth 优先/初始观测窗口策略，这会改变告警时序语义，不能作为本次最小修复静默合入。
- person 重复框导致 `person_count` 跳变，可后续评估 person-only NMS 或 person context 去重，但需要结合真实视频视觉验收确认不会削弱 head/helmet 抑制效果。


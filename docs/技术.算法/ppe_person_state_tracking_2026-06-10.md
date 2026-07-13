# PPE 人物状态记忆机制记录（2026-06-10）

## 问题背景

完整视频第 4 秒附近存在多人重叠后分离的场景。上一轮检测框跟踪已经改善拖框和部分反转，但仍有两个问题：

1. 多人重叠后，分离出来的单个人恢复显示偏慢；
2. 个别人物的 `head` / `helmet` 显示状态会发生短时反转。

这些问题从 `overlay_full.json` 看并不是模型完全没有检出，而是显示跟踪层只按框级 track 工作，缺少“这个人是谁、这个人的 PPE 状态是什么”的记忆。

## 当前判断

本轮在 `PPEDisplayTracker` 内新增人物状态记忆层，不增加额外 GPU 推理，不修改模型权重，不改变 `head` / `helmet` 的 PPE 主证据语义。

机制目标：

1. 给可靠 `person` track 建立 `person_state_id`；
2. 将附近的 `head` / `helmet` track 绑定到人物状态；
3. 人物短时遮挡或重叠时保持其最近稳定 PPE 状态；
4. 分离后新出现的头/帽框继承原人物状态，减少重新冷启动等待；
5. 人物贴边离开时进入 `edge_pending`，等待若干帧后再清理人物状态，避免刚接触边界就丢状态；
6. 不同人物状态下的接近头/帽框，在 shadow 去重时优先保留，降低多人重叠时误删。

## 代码链路依据

主要改动位置：

- `model/src/defense/module_a/postprocess/ppe_tracking.py`
  - `StableTrack` 增加 `person_state_id`、`person_state_status`、`ppe_state_label`、`hold_after_person_state` 等诊断字段；
  - `PPEDisplayTracker` 增加人物状态池、状态匹配、遮挡标记、贴边延迟清理、PPE 状态确认和重复状态合并；
  - `_remove_shadow_tracks()` 的 overlap-safe 判断增加不同 `person_state_id` 的保留证据；
  - `_render_tracks()` 与低上下文过滤支持 `hold_after_person_state`。
- `model/src/defense/runtime/frame_processor.py`
  - YOLOv8 质量模式默认启用人物状态机制；
  - 暴露可配置参数：`person_state_hold_frames`、`person_state_edge_hold_frames`、`person_state_match_distance`、`person_state_occlusion_iou`、`person_state_confirm_frames`、`person_state_render_miss_grace`、`person_state_min_person_confidence`。
- `model/src/defense/runtime/ppe_business.py`
  - 业务显示过滤允许 `hold_after_person_state` 的 `head` / `helmet` track 保留，避免 tracker 已保持但最终输出又被裁掉。
- `model/tests/test_ppe_display_tracker_person_state.py`
  - 覆盖单帧 `head/helmet` 反转抑制、遮挡期间状态保持、贴边延迟清理。

## 影响范围

影响的是显示跟踪和 overlay 输出，不影响模型推理本体，不新增额外 detector 推理，不改变模型准入/净化流程。

新增字段会进入 `ppe_tracks` 和 `ppe_tracking_diagnostics.person_state`，旧消费者可以忽略这些字段。

## 性能判断

该机制主要是少量框之间的 CPU 匹配和状态维护，复杂度接近 `O(n^2)`，但 n 是当前帧人物/头/帽框数量，远小于 YOLO 推理成本。按当前视频规模，预计额外开销约 `0.1ms` 到 `2ms/frame`，通常不会成为瓶颈。

风险主要不是性能，而是错绑和旧状态滞留。因此本轮加入：

1. `person_state_min_person_confidence`：低置信度 person 不建立长期状态；
2. `person_state_hold_frames`：非贴边遮挡有最大保持帧；
3. `person_state_edge_hold_frames`：贴边离开单独延迟清理；
4. `person_state_confirm_frames`：人物 PPE 状态切换需要连续确认；
5. 重复 person state 合并，避免低阈值下同一人生成多个身份。

## 验证结果

测试：

- `pixi run cmd /C "cd /D model && set PYTHONPATH=src&& python -m pytest tests/test_ppe_display_tracker_person_state.py tests/test_ppe_display_tracker_label_stability.py tests/test_frame_processor_status_contract.py tests/test_video_diagnostic_contract.py tests/test_video_defense_pipeline_reuse.py -q"`
- 结果：`15 passed`

全段结果：

- 结果视频：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v1-full-video/result_overlay_full.mp4`
- overlay JSON：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v1-full-video/overlay_full.json`
- summary：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v1-full-video/overlay_summary_full.json`
- report：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v1-full-video/overlay_report_full.json`
- csv：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v1-full-video/overlay_rows_full.csv`

统计观察：

- `person_state` 峰值：10；
- `hold_after_person_state` 出现在 602 个帧上；
- 第 225-275 帧窗口内最大 person-state hold 数：1；
- 粗略同轨迹 `head/helmet` 反转：13；
- 粗略空间反转：51。

## v2 复验记录

用户要求重新生成完整结果视频后，使用当前源码重新导出全段 overlay 并重新渲染结果视频。本轮仍隐藏 `person` 显示框，只保留 `head` / `helmet` 作为视觉验收主显示目标。

产物：

- 结果视频：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/result_overlay_full.mp4`
- overlay JSON：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/overlay_full.json`
- summary：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/overlay_summary_full.json`
- report：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/overlay_report_full.json`
- csv：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/overlay_rows_full.csv`
- 视频渲染摘要：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v2-full-video/result_overlay_full.summary.json`

验收帧证据：

- `acceptance_frames_225_3s`：frame 225-406，共 182 张 PNG，3840x2160，覆盖第 4 秒附近多人重叠和分离场景；
- `acceptance_frames_690_3s`：frame 690-871，共 182 张 PNG，3840x2160，覆盖中段近景多人场景；
- `acceptance_frames_1130_3s`：frame 1130-1311，共 182 张 PNG，3840x2160，覆盖右侧售货机区域。

复验结果：

- overlay 记录数：1555；
- 结果视频帧数：1555；
- 结果视频分辨率：3840x2160；
- 抽检 frame 240、270、720、1160：未见明显旧框滞留；第 4 秒多人重叠分离后，分离人物能较快恢复独立头框；抽检帧内未见明显 `head` / `helmet` 状态反转。

## v3 helmet 可信门槛修复

用户复验 v2 后指出一个关键业务事实：完整视频最后 5 秒没有任何人戴帽，且全程只有右侧外卖小哥戴帽。v2 中最后段白衣人经过柱子后被长期显示为 `helmet`，这不是旧框保持，而是 raw detector 在遮挡/小目标场景下偶发吐出低可信小 `helmet` 框后，人物 PPE 状态被过早写成 `helmet`。

本轮 v3 在人物状态机制上补充 `helmet` 可信门槛：

1. `StableTrack` 输出新增 `ppe_state_trusted`，用于区分“检测到过某个 PPE 标签”和“该 PPE 状态可用于显示/继承”；
2. track 侧新增 `helmet_seen_streak`、`helmet_seen_conf_sum`、`helmet_seen_max_area_ratio`，记录连续帽子证据；
3. person state 侧新增 `ppe_trusted`、`helmet_evidence_streak`、`helmet_evidence_conf_sum`、`helmet_evidence_max_area_ratio`；
4. `head` 默认可作为可信无帽状态；`helmet` 必须满足连续性、置信度和面积证据，才能覆盖人物状态并进入显示继承；
5. `_apply_person_state_memory_to_tracks()` 和 `_extend_nearby_ppe_tracks_from_state()` 不再用未 trusted 的 `helmet` 覆盖显示标签；
6. `_filter_low_context_display_tracks()` 会过滤绑定人物状态但未 trusted 的小 `helmet`，除非该框已由时序证据明确提升。

v3 验证结论：

- 结果目录：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v3-helmet-trust-full-video/`
- 结果视频：`result_overlay_full.mp4`，1555 帧，3840x2160；
- overlay JSON：`overlay_full.json`；
- summary/report/csv：`overlay_summary_full.json`、`overlay_report_full.json`、`overlay_rows_full.csv`；
- 最后 5 秒验收帧：`acceptance_frames_1250_last5_no_helmet`，frame 1250-1554，共 305 张 PNG；
- 外卖小哥正例验收帧：`acceptance_frames_470_3s_takeout_helmet`，frame 470-651，共 182 张 PNG；
- 第 4 秒多人重叠验收帧：`acceptance_frames_225_3s`，frame 225-406，共 182 张 PNG。

overlay 硬门禁：

- frame 1250-1554：raw detector 仍有 70 帧出现过 `helmet` 原始候选，但显示输出 `ppe_tracks` 中 `helmet` 为 0 帧，`ppe_helmet_count/ppe_effective_helmet_count` 为 0 帧；
- frame 220-575：外卖小哥 `helmet` 显示保留 331 帧；
- 最后段白衣人主要落在 `person_state_id=1600`。该人物在 frame 1397-1400 曾出现未可信 `helmet` 状态，`ppe_state_trusted=False`，未进入显示；后续状态转为可信 `head` 并保持到末尾。

测试：

- `pixi run cmd /C "cd /D model && set PYTHONPATH=src&& python -m pytest tests/test_ppe_display_tracker_person_state.py tests/test_ppe_display_tracker_label_stability.py tests/test_frame_processor_status_contract.py tests/test_video_diagnostic_contract.py tests/test_video_defense_pipeline_reuse.py -q"`
- 结果：`17 passed`

结论：v3 已把“出现过低可信 helmet 候选”和“这个人应显示为 helmet”拆开。最后 5 秒无帽场景现在不会显示 helmet；外卖小哥作为唯一戴帽正例仍然保留 helmet。

## 后续建议

## 已知残留：多人重叠分离后的恢复延迟

用户复验 v3 后认为整体检测框效果已经可接受，但第 4 秒附近多人重叠后再分离的场景仍存在“反应有点慢”的残留。当前判断是：

1. 这不是模型完全漏检，也不是最后白衣人那类 `helmet` 状态污染问题；
2. 主要发生在多人贴近、遮挡和再次分离时，person state 与 head/helmet track 需要重新建立更清晰的绑定关系；
3. 现有机制为了避免旧框滞留和错误状态继承，对分离后的新小框仍保留了一定确认门槛，因此会牺牲少量即时响应；
4. 下一轮若继续优化，应优先做“分离加速”而不是继续放宽全局阈值：例如只在 person state 已存在且多人中心距离快速增大时，缩短新 head track 的确认时间，或在遮挡解除后的 3-5 帧内提高新框接管优先级；
5. 这类优化需要再次用第 4 秒 frame 225-406 区间做独立视觉验收，避免把已经解决的最后 5 秒误 helmet 问题重新放出来。

本轮暂不继续调参，原因是用户已经确认“框的问题暂时就这样，效果挺不错”，提交前只保留该残留记录。

1. 用户先验收 `result_overlay_full.mp4`，重点看第 4 秒附近、多人重叠分离、右侧售货机区域；
2. 若仍有个别旧框滞留，优先微调 `person_state_render_miss_grace` 和 `person_state_hold_frames`；
3. 若仍有个别状态错绑，优先提高 `person_state_min_person_confidence` 或降低 `person_state_match_distance`；
4. 若用户确认效果，再整理提交；提交后按规则运行 `codegraph init -i`。
## v4 head anchor：手部遮挡分离误识别为 head

用户指出新的视觉问题：人的手从画面上经过人头，分离时手也被识别并显示成 `head`。当前判断是：这类现象通常不是业务计数逻辑问题，而是 YOLO raw detector 在遮挡/手臂经过头部时把手部纹理短暂判成 `head`，随后显示 tracking 在 person-state 记忆和低上下文过滤中没有足够强的“头部位置锚点”约束，导致离开真实头部区域的 `head` 候选仍可能被渲染。

代码链路依据：

1. `_ppe_person_state_score()` 原本只要求 `head/helmet` 与 person 上半身区域有一定 IoU、包含度或中心距离关系；上半身区域覆盖到肩膀和手臂，无法单独排除“手在头附近经过后分离”的情况；
2. `_filter_low_context_display_tracks()` 对 `helmet` 有低置信度上下文过滤和 trusted 约束，但对 `head` 主要只有置信度门槛；高置信度手部误检可能直接进入显示；
3. person-state 机制已经能记住一个人的 PPE 状态，但 v3 只对 `helmet` trusted 做了严格门控，还没有把最近可信 `head/helmet` 位置作为后续 `head` 候选的几何锚点。

本轮 v4 修改：

1. person state 新增 `ppe_box`，在 `_update_person_ppe_state()` 接受当前 `head/helmet` 状态时记录最近可信 PPE 框；
2. 新增 `_person_head_zone_box()`，用 person box 顶部约 34% 作为头部候选区域，而不是继续使用上半身 55% 的宽松区域；
3. 新增 `_head_track_supported_by_person_state()` 和 `_head_track_supported_by_any_person_state()`：`head` 候选必须同时落在对应人物的头部区域，并且在 state 已有 `ppe_box` 锚点后，需要与该锚点保持 IoU、包含度或中心距离关系；
4. `_ppe_person_state_score()` 在给 `head` 绑定 person state 前先检查 head-anchor 支撑，避免手部候选污染 person PPE 状态；
5. `_filter_low_context_display_tracks()` 对 `head` 增加 `head_not_supported_by_person_head_anchor` drop reason，不再让离开头部锚点的高置信度 `head` 候选直接显示。

验证结果：

- 聚焦回归测试：`pixi run cmd /C "cd /D model && set PYTHONPATH=src&& python -m pytest -q tests/test_ppe_display_tracker_person_state.py tests/test_ppe_display_tracking.py tests/test_ppe_business.py"`，结果 `32 passed`；
- v4 固定镜头全段产物：`model/runs/visual_acceptance/2026-06-10-project-yolov8-person-state-v4-head-anchor-full-video/`；
- 结果视频：`result_overlay_full.mp4`，1555 帧，3840x2160；
- summary/report/csv：`overlay_summary_full.json`、`overlay_report_full.json`、`overlay_rows_full.csv`；
- 头部锚点检查摘要：`head_anchor_check_summary.json`；
- 最后 5 秒 frame 1250-1554：raw detector 仍有 70 帧 `helmet` 候选，但显示层 `helmet` 为 0 帧；
- 外卖小哥 frame 220-575：显示层 `helmet` 保留 329 帧，相比 v3 的 331 帧少 2 帧，未发现正例大幅回退；
- 新增 head-anchor 过滤在固定镜头全段触发 858 次，说明该规则实际压制了离开人物头部锚点的 `head` 候选。

影响范围：

本轮不修改模型权重、类别语义或检测阈值，也不新增 GPU 推理；只在显示 tracking 层加强 `head` 的几何支撑约束。潜在风险是个别真实头部在剧烈弯腰、快速侧移或 person box 质量很差时可能被短暂压掉，所以 v4 必须以用户人工视频验收为准，不能仅凭 agent 判断宣称通过。

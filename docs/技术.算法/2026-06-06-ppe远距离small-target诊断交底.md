# PPE 远距离 small-target 诊断交底

日期：2026-06-06

## 背景

用户提出远距离场景中可能出现“检测出 person 或人体区域，但没有稳定 head/helmet”的问题。本交底只做代码链路与候选实验设计，不把 person-only 直接改成未戴帽证据，不增加实时主路径 GPU 推理。

## 已从代码确认的证据链

当前链路为：

`detector backend -> DetectionFrameResult -> FrameProcessor.process -> evaluate_ppe_business -> ppe raw summary -> PPEDisplayTracker -> SafetyHelmetState -> business filter -> status/overlay -> preview render`

关键层：

- detector 输出 `boxes/classes/confidences/names`，`FrameProcessor.process()` 在 640 输入上调用 `pipeline.process_frame()`。
- ROI 重检只在非实时且预算允许时触发，realtime 文件/相机场景不会额外跑主路径 detector。
- `summarize_ppe_from_detections()` 生成 PPE raw summary，负责 label 归一、raw/effective/weak/promoted/count/reason，并把小目标、低置信、缺少上下文的 head/helmet 写入 `helmet_fp_suppression`。
- `PPEDisplayTracker` 执行 incoming、merge、assign、age/prune、shadow removal、low-context filter、render gate。
- temporal weak promotion 不增加推理，只靠连续弱证据 streak/avg confidence/context 将弱 head/helmet 提升进 PPE counts。
- `evaluate_ppe_business()` 在 tracker 后应用 temporal evidence、helmet mutex、`SafetyHelmetState`，最后 `_filter_tracks_for_ppe_counts()` 做业务显示过滤。
- status 的 `ppe_head_count/ppe_helmet_count` 来自最终可见 tracks，raw count 另有 `ppe_raw_*` 与 `raw_class_counts` 字段，因此 UI count 不能直接等同 detector raw。

## 可能断点

远距离 head/helmet 缺失不是单点问题，可能发生在以下层：

1. raw detector 层：head/helmet 根本未出框，或低于 detector/业务阈值。
2. low confidence 层：候选低于 `candidate_min_confidence`，或没有 person 上半身上下文，只能作为 weak evidence。
3. postprocess suppression 层：小 helmet/head 可能被标为 `small_low_conf_helmet`、`helmet_without_person_context`、`small_low_conf_head`、`small_no_context_head`、`edge_isolated_head`。
4. temporal weak promotion 层：远距离抖动、断检、head/helmet 标签互跳会导致 streak 清零。
5. stable label 层：小目标标签切换更容易被防抖压住。
6. business filter 层：raw/track 有证据，但 business counts 不允许时最终显示会被过滤。
7. render/overlay gate 层：file realtime 默认 `max_render_misses=2`，弱证据默认不 eligible，因此容易闪断。
8. person context 层：person 是上下文/定位约束，不是“检测到人即未戴帽”的证据。
9. source-auth media suppression 层：A3b media ROI 激活时可能抑制 ROI 内 PPE detections 并 reset temporal PPE。

## 候选实验

优先做 text-only 诊断，不直接改实时行为：

1. 每帧导出 funnel：`raw_class_counts -> ppe_raw_class_counts -> ppe_class_counts -> ppe_tracks -> overlay ppe_tracks`。
2. 记录 suppression reason 直方图：小目标低置信、缺上下文、head/helmet overlap、source-auth media ROI。
3. person-conditioned 指标：每个 person 的 area ratio、上半身 ROI、最近 head/helmet 距离、IoU、containment、最近候选置信度。
4. 远距分桶：按 person/head/helmet area ratio 统计 raw 出框率、weak 率、promotion 成功率、render 保留率。
5. temporal 指标：weak streak、avg confidence、promotion 失败原因、label switch 次数、misses 分布。
6. gate attribution：business filter drops、render gate drops、shadow decisions、overlay hold/interpolation/drop source。
7. person-only 序列：连续 person-only uncertain 的帧数/时长，以及后续是否出现 head/helmet。
8. 离线 ROI 评估：ROI 触发次数、redetect 置信度提升、类别变化、NMS 丢弃数量、额外 inference ms。

## 工程约束

- 不把 person-only 改成未戴帽告警。
- 不在 realtime 主路径增加额外 GPU inference。
- ROI redetect 只适合非实时/离线或明确性能预算下的后续专项。
- display-only 实验只能解决“显示消失”，不能解决 raw detector 根本没出框。
- temporal-only 实验需要同时统计误报风险，不能只看 promoted 增量。

## 结论

下一步 small-target 工作应先补 text-only funnel 诊断，把“raw 没有”与“raw 有但被 weak/suppression/business/render/overlay 吃掉”分开。分清断点后，再选择 detector/数据问题、temporal 参数问题、显示保持问题，或离线 ROI 能否作为后续优化方向。

## 当前 Web API 证据

本轮已新增 `ppe_small_target_diagnostics`，并通过同一目标视频跑 Web API text-only 证据。

证据目录：

`runtime_evidence/ppe_box_tuning_20260605_215454/small_target_funnel_current`

主要产物：

- `small_target_overlay.json`
- `small_target_status_polls.json`
- `small_target_summary.json`
- `small_target_funnel_by_frame.csv`

运行条件：

- profile：`desktop_rtx`
- realtime：`true`
- feature options：`static_image_enabled=true`，`a3b_sensitivity=high`
- display options：`show_boxes=true`，`show_person_boxes=false`
- B 模块：`test_bypass_model_security=true`

关键结果：

| 指标 | count |
| --- | ---: |
| overlay records | 308 |
| diagnostic records | 308 |
| raw head nonzero records | 288 |
| raw helmet nonzero records | 88 |
| raw person nonzero records | 1 |
| visible head nonzero records | 284 |
| visible helmet nonzero records | 33 |
| person-only raw records | 0 |

suppression reason：

| reason | count |
| --- | ---: |
| small_no_context_head | 487 |
| edge_isolated_head | 202 |
| small_low_conf_head | 188 |
| small_low_conf_helmet | 38 |
| helmet_without_person_context | 28 |

当前判断：

1. 这段目标视频不适合验证“稳定 person 有框但 head/helmet 缺失”的行为优化，因为 raw person 只有 1 条记录，person-only raw 为 0。
2. 这段视频适合验证 small head/helmet 证据在 postprocess/weak/suppression 层的流失：head weak 总量 877，helmet weak 总量 66。
3. 当前不能把 person-only 相关策略作为可采纳行为候选；需要专门找或生成“person 稳定、head/helmet 远距离弱/缺失”的素材后再跑。
4. 当前可执行候选应保持为 text-only 诊断：按 suppression reason 和 area/context 分桶，先定位是 detector/raw 缺失、上下文不足、temporal promotion 失败，还是 display/render gate。

## 补充素材复验：单人仓库巡检视频

为避免只依赖一段目标视频，本轮又选择单人仓库巡检素材复跑同一 Web API text-only funnel。

素材：

`D:\联合防御模块\素材\真实视频\12_监控视角_仓库巡检\015_clean_baseline_single_worker_normal_6f9897da7479.mp4`

证据目录：

`runtime_evidence/ppe_box_tuning_20260605_215454/small_target_funnel_single_worker_current`

主要产物：

- `small_target_overlay.json`
- `small_target_status_polls.json`
- `small_target_summary.json`
- `small_target_funnel_by_frame.csv`

关键结果：

| 指标 | count |
| --- | ---: |
| overlay records | 940 |
| diagnostic records | 940 |
| raw helmet nonzero records | 757 |
| ppe raw helmet nonzero records | 747 |
| visible helmet nonzero records | 735 |
| person-only raw records | 0 |
| person-only ppe raw records | 0 |
| person context without head/helmet records | 0 |

reason 与 suppression：

| 项 | count |
| --- | ---: |
| helmet_evidence_present | 714 |
| no_ppe_evidence_detected | 205 |
| temporal_weak_helmet_promoted | 21 |
| small_low_conf_helmet | 21 |
| helmet_without_person_context | 17 |

补充判断：

1. 单人仓库素材同样没有形成“稳定 person 有框但 head/helmet 缺失”的验证条件，person-only raw 与 person context without head/helmet 均为 0。
2. 该素材确认了 helmet 弱证据与上下文缺失会被 funnel 记录出来，但不能支撑 person-conditioned 行为调优。
3. small-target 行为修改仍应暂缓；当前可合入价值是诊断字段和证据流程，而不是新的默认检测策略。

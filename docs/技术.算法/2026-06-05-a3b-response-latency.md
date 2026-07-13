# A3b 时序响应延迟记录

日期：2026-06-05

## 背景

用户反馈 Web 端 A3b 反应时间约 3 秒，希望在保持 A3b 时序检测特性的前提下进一步加快。

## 当前判断

Web 端检测线程已经使用 latest-only backpressure，预览线程与检测线程解耦；本轮 API 基准显示主要瓶颈不是算力排队，而是 A3b 的时序确认口径与门控。

基准样本：

- 视频：`D:\联合防御模块\素材\视频中出现干扰视频\VID20260512200916.mp4`
- 模型：`D:\联合防御模块\purification_lab\seven_experiment_archive\oga_semantic_vest\poisoned_best.pt`
- profile：`desktop_rtx`
- A3b 灵敏度：`balanced`

观测摘要：

- 平均 `processing_ms` 约 28.6 ms，P95 约 39.0 ms。
- 平均 `a3b_static_media_ms` 约 4.2 ms，P95 约 11.1 ms。
- `dropped_detection_frames` 最大为 1。
- `wall_s - source_time_s` 平均约 0.92 s。
- A3b 首次 observed 分数过阈值出现在源时钟约 7.55 s，但后续未形成 confirmed/suspect trigger，主要原因是 `border_suppressed` 与 `no_candidate_or_screen_cue`。

## 代码链路依据

- `src/defense/runtime/backend_pipeline.py`：检测队列为 latest-only，不会长期积压旧帧。
- `src/defense/runtime/runner.py`：预览线程 `_preview_render_loop` 独立于检测线程，文件实时预览按视频时钟选择 overlay。
- `src/defense/runtime/frame_processor.py`：A3b 软确认由 `A3BSoftTriggerState` 生成 `a3b_state`、`a3b_triggered`、`a3b_debug`。
- `src/defense/runtime/a3b_soft_trigger.py`：默认 observed-only warning 需要窗口命中和 track/score 阈值，未通过质量门时只进入 suspect/observing，不给 confirmed confidence。
- `src/defense/runtime/config.py`：Web 前端的 A3b 灵敏度通过 feature options 映射为阈值、静态媒体检测间隔和窗口命中数。

## 结论

在不改模型权重、不改 PPE 类别语义、不改变默认 balanced 口径的前提下，可以把 `sensitive` / `high` 灵敏度明确作为快速响应档：

- `sensitive`：observed-only 窗口命中从 3 次降为 2 次。
- `high`：confirmed 窗口命中与 observed-only 窗口命中均为 2 次，静态媒体检测间隔仍为 2 帧。

这属于显式行为调优档位，不影响默认 balanced 的保守确认策略。

## 后续建议

若用户仍希望默认档也缩短到 2 次命中，需要单独做误报回归，覆盖真实视频、强光、边界遮挡、手机随手录制和干扰视频样本。当前尚未从代码中确认 balanced 默认降窗不会增加误报，需实验确认。

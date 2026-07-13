# 2026-05-27 head/helmet-only 时序候选探索

## 背景

当前 `model/run_model` 下多数组合模型只输出 `helmet/head`，不输出 `person`。因此 PPE 检测不能依赖 `person` 作为主上下文，只能以 `head` 和 `helmet` 的逐帧证据、短时时序稳定和业务状态机来克服远距离、小目标、手机录屏类素材中的断框问题。

## 代码链路依据

- 推理阈值入口：`model/src/defense/module_a/backends/detector_backend.py` 的 `UltralyticsDetectorBackend.predict()` 使用 `self.confidence` 传给 YOLO `conf` 参数。低于该阈值的框不会进入后处理。
- PPE 单帧语义：`model/src/defense/module_a/ppe_postprocess.py` 的 `summarize_ppe_from_detections()` 使用 `PPEPostprocessConfig.min_confidence=0.25` 统计正式 `head/helmet` 证据。
- 时序显示：`model/src/defense/module_a/postprocess/ppe_tracking.py` 的 `PPEDisplayTracker.update()`、`apply_temporal_evidence()` 负责弱证据晋升、短时持框和显示稳定。
- 业务输出：`model/src/defense/runtime/ppe_business.py` 的 `evaluate_ppe_business()` 串联单帧 PPE、`PPEDisplayTracker` 和 `SafetyHelmetState`。

## 当前判断

“连续空窗”不是单纯显示层问题。对于 2 类模型，部分视频帧中 YOLO raw detection 本身连续为空；如果只在显示层延长持框，会把旧框拖太久，容易出现旧框滞留和误导性显示。

更合理的方向是“双阈值”：

- 候选阈值低一些，例如 `0.18`，让低置信 head/helmet 进入时序候选。
- 业务阈值保持 `0.25`，低置信候选不能直接触发 PPE 告警。
- 候选必须经过连续时序证据晋升后才参与稳定显示或业务解释。
- 持框上限控制在短窗口内，例如 12 帧；超过该窗口仍无 raw detection 则宁可短暂无框，也不显示旧框。

## 实验结果

实验模型：

- `model/run_model/oda_sig_multiperiod/oda_sig_multiperiod_purified.pt`
- 类别表：`{0: helmet, 1: head}`

实验视频：

- `素材/视频中出现干扰视频/5e145bf778577e75118502e263d00c41.mp4`
- `素材/视频中出现干扰视频/d6415677a016dbf211e665f648e75607.mp4`
- `素材/真实视频/12_监控视角_仓库巡检/015_clean_baseline_single_worker_normal_6f9897da7479.mp4`
- `素材/真实视频/07_上下文素材_工地远景无稳定目标/pexels_1197802_construction_site.mp4`

输出路径：

- `model/runs/ppe_temporal_grid_head_helmet_20260527/summary.json`
- `model/runs/ppe_temporal_grid_head_helmet_20260527/summary_two_threshold.json`
- `model/runs/ppe_temporal_grid_head_helmet_20260527/val_threshold_summary.json`
- `model/runs/ppe_temporal_grid_head_helmet_20260527/visual_two_thr018_hold12_5e145/side_by_side_debug.mp4`

关键结果：

- 当前策略在 `5e145...mp4` 上：`stable_zero_frames=31`，最长稳定断窗 `15` 帧。
- 单纯降低 YOLO 阈值到 `0.20` 并持框 12 帧：最长断窗降到 `7` 帧，但远景正常视频出现少量候选显示风险。
- 双阈值 `candidate=0.18, business=0.25, hold=12`：`5e145...mp4` 上最长稳定断窗降到 `4` 帧；远景正常视频未产生稳定显示框。
- `d641...mp4` 仍然无法明显恢复，说明该样本的 raw head/helmet 证据过弱，仅靠时序候选无法可靠补齐。
- 500 张验证图片阈值扫描显示：`0.18` 相比 `0.25` 提高 head/helmet 召回，但引入更多 FP；因此它适合作为“候选阈值”，不适合直接作为业务阈值。

## 光流方案判断

光流和当前方法有重叠：都试图在 raw detection 缺失期间估计目标连续位置。区别是当前 `PPEDisplayTracker` 已有低成本的速度外推和短时持框，而光流会额外计算图像纹理运动。

暂不建议上全帧光流：

- 全帧光流会增加实时路径 CPU/GPU 压力。
- 手机录屏、屏幕反光、线缆和背景纹理可能产生错误跟随。
- 对于超过约半秒的 raw detection 空窗，光流也不能证明目标仍是 helmet/head，只能证明某块纹理在移动。

可预留的轻量方案是 ROI 级 KLT/LK 光流：只在已有稳定 head/helmet track 短暂丢失时，对该小 ROI 做 3 到 6 帧跟踪；超过窗口仍无 YOLO 证据立即停止。该方案需要单独做实时性测试，当前未落生产。

## 建议

下一步优先实现可配置的 head/helmet 双阈值候选链路：

- 在推理端允许 `candidate_confidence` 低于正式 `confidence`。
- 在 PPE 后处理中把低置信 head/helmet 标记为 `temporal_candidate`，不直接计入 `head_count/helmet_count`。
- 在 `PPEDisplayTracker` 中要求连续命中和平均置信度达标后才晋升。
- 默认仅对 `head_helmet_only` 模型开启，保留有 `person` 模型的现有路径。
- 保留 `business_min_confidence=0.25`，避免低置信候选直接影响告警。

未能从本轮实验确认：

- 30 多个分类视频全量网格的最佳全局参数。
- ROI 光流在当前机器实时路径中的真实延迟。
- `d641...mp4` 是否需要模型训练/输入增强才能恢复，而非纯算法补偿。

## 本轮已落地实现

本轮已按上述方向做了窄范围实现，默认只影响 head/helmet-only 模型：

- `model/src/defense/module_a/backends/detector_backend.py`
  - `UltralyticsDetectorBackend` 和 `YoloV5DetectorBackend` 增加 `candidate_confidence`。
  - `_prediction_confidence()` 只有在类别表为 head/helmet-only 且无 `person` 时，才把推理阈值降到候选阈值；有 `person` 的模型继续使用正式 `confidence`。
- `model/src/defense/module_a/ppe_postprocess.py`
  - `PPEPostprocessConfig` 增加 `candidate_min_confidence`。
  - `summarize_ppe_from_detections()` 对低于业务阈值、高于候选阈值的 head/helmet 标记 `low_conf_temporal_candidate`，只进入弱证据/时序链路，不计入正式 `head_count/helmet_count`。
- `model/src/defense/runtime/ppe_business.py`
  - `evaluate_ppe_business()` 支持传入 `postprocess_config`。
- `model/src/defense/runtime/frame_processor.py`
  - 从运行配置构造 PPE 后处理配置，传给生产业务链路。
- `model/src/defense/runtime/ppe_state.py`
  - 增加高置信裸头快响分支：正式 `head` 且最高置信度达到阈值时，2 帧即可确认；低置信候选和 temporal promotion 不走快响。
- `model/configs/module_a_runtime.yaml`
  - `ppe_tracking.business_min_confidence: 0.25`
  - `ppe_tracking.temporal_candidate_min_confidence: 0.18`
  - `ppe_tracking.fast_alert_trigger_count: 2`
  - `ppe_tracking.fast_alert_min_head_confidence: 0.65`
- `model/tools/render_ppe_debug_video.py`
  - 支持 `--business-confidence`，用于复现双阈值可视化验证。

## 实现后验证

- 回归测试：
  - `pixi run python -m pytest -q tests/test_detector_backend_candidate_confidence.py tests/test_ppe_postprocess.py tests/test_ppe_business.py tests/test_ppe_display_tracking.py tests/test_runtime_config_feature_options.py`
  - 结果：`40 passed`
  - `pixi run python -m pytest -q tests/test_model_security_runtime.py tests/test_model_security_bypass_and_metrics.py tests/test_monitor_engine_shutdown.py tests/test_runtime_config_feature_options.py tests/test_web_prebuffer_contract.py tests/test_detector_backend_candidate_confidence.py tests/test_ppe_business.py tests/test_ppe_display_tracking.py tests/test_ppe_postprocess.py`
  - 结果：`95 passed`
  - `pixi run python -m compileall -q src tests tools`
  - 结果：通过
- 检测视频：
  - 命令：`pixi run python tools\render_ppe_debug_video.py --video "D:\联合防御模块\素材\视频中出现干扰视频\5e145bf778577e75118502e263d00c41.mp4" --model "D:\联合防御模块\model\run_model\oda_sig_multiperiod\oda_sig_multiperiod_purified.pt" --out-dir "D:\联合防御模块\model\runs\ppe_temporal_candidate_impl_20260527\two_thr018_hold12_5e145" --family ultralytics --backend pytorch --device cuda:0 --confidence 0.18 --business-confidence 0.25 --image-size 640 --max-render-misses 12 --max-frames 180`
  - 输出视频：`model/runs/ppe_temporal_candidate_impl_20260527/two_thr018_hold12_5e145/side_by_side_debug.mp4`
  - 指标：`stable_zero_frames=18`，`stable_longest_zero_after_first=4`，`raw_gap_recovered_frames=32`
  - 连续 3 秒抽帧：`model/runs/ppe_temporal_candidate_impl_20260527/two_thr018_hold12_5e145/frame_review_2s_5s`
  - 已查看 `contact_sheet_1.jpg`、`contact_sheet_2.jpg`、`contact_sheet_3.jpg`，未观察到明显旧框漂移；仍可看到 raw 低阈值偶发墙面/线缆假框，但未被直接变成稳定业务框。
- 速度测试：
  - 输出：`model/runs/ppe_temporal_candidate_impl_20260527/speed_benchmark.json`
  - `5e145...mp4` + 2 类模型：基线 `73.39 FPS`，双阈值 `74.09 FPS`；平均推理 `10.36ms -> 10.42ms`。
  - 输出：`model/runs/ppe_temporal_candidate_impl_20260527/speed_benchmark_warehouse.json`
  - `015_clean...mp4` + 2 类模型：基线 `68.77 FPS`，双阈值 `66.29 FPS`；平均推理 `9.78ms -> 10.15ms`。
  - `015_clean...mp4` + 3 类模型：配置候选阈值后仍保持正式阈值 `0.25`，`82.26 FPS -> 81.32 FPS`，说明带 `person` 的模型不会被 head/helmet-only 候选阈值牵连。
- 报警响应测试：
  - 强裸头 `head conf=0.86`：确认从原默认第 3 帧提前到第 2 帧，`confirmed_source=fast_head`。
  - 弱小目标 `head conf=0.31`：仍需先 temporal promotion，再按普通窗口确认，首次确认第 5 帧，避免弱证据快速误报。
  - 真实视频输出：`model/runs/ppe_fast_alert_impl_20260527/two_thr018_hold12_5e145/side_by_side_debug.mp4`
  - 真实视频指标：`first_warning_frame=3`，`first_confirmed_frame=3`，`confirmed_frames=118`，`warning_frames=151`。
  - 连续 3 秒抽帧：`model/runs/ppe_fast_alert_impl_20260527/two_thr018_hold12_5e145/frame_review_2s_5s`，已查看 `contact_sheet_1.jpg` 到 `contact_sheet_3.jpg`，未观察到新增拖框。

## 当前边界

双阈值时序候选能显著缩短 2 类模型在可恢复样本上的断框，但不能凭空恢复完全没有 head/helmet 证据的长空窗。对于 `d641...mp4` 这类 raw 证据过少的样本，后续应优先评估训练增强、输入尺度、模型族差异或 ROI 级轻量光流，而不是继续无限延长旧框显示。

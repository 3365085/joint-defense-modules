# Web 隐式自定义模型覆盖导致性能下降的复核记录

## 问题背景

用户在 Web 端实跑视频时观察到约 12 FPS，怀疑服务未重启到最新代码。

## 核验结论

本次问题不是旧服务或旧代码未重启。运行中的 Web 进程启动时间晚于 Module A 核心代码修改时间；再次重启后问题仍可复现。

实际 `/api/status` 显示：

- profile：`desktop_rtx`
- 实际 backend：`pytorch`
- 实际模型：`baseline_training/runs/baseline_yolov8_three_put/best.pt`
- 检测吞吐：约 12～16 FPS

但 `desktop_rtx` 的默认配置应为：

- backend：`tensorrt`
- 模型：`baseline_training/runs/classmate_maskbd_v4/best.engine`

代码链路确认：

1. Web 主页面从浏览器保存的运行模型配置构造 `custom_model`；
2. 模型安全测试绕过只跳过 B 模块准入，不改变 `custom_model`；
3. 当历史配置仍将自定义 PT 标记为启用时，`MonitorEngine -> PipelineCache -> load_runtime_config` 会按设计用该 PT 覆盖 profile 默认模型；
4. 因此服务重启不会清除浏览器保存的自定义模型选择，仍会继续运行 PyTorch。

这不是 Module A 隔帧或 latest-only 逻辑把视频限制为 12 个总帧。目标视频实际读取到 `279/280` 帧；约 12 的数值是检测吞吐 FPS。

## 已执行修复

- 在模型安全页将“使用自定义模型”恢复为关闭，保留测试绕过，仅使其跳过 B 模块准入。
- 主监控页新增“下次启动使用的运行模型”可见摘要。
- 当自定义模型启用时，主监控页明确提示它会覆盖 profile 默认模型，并提供“下次使用默认模型”按钮。
- 增加配置回归测试，保证历史自定义模型路径在 `enabled=false` 时不能覆盖 `desktop_rtx` 的 TensorRT backend/artifact。
- 增加 Web/FastAPI 回归测试，保证安全测试绕过只透传模型选择，不自行改写 profile 或模型。

## 真实 Web 复跑

视频：

`D:\联合防御模块\素材\视频中出现干扰视频\5e145bf778577e75118502e263d00c41.mp4`

恢复默认模型后：

- backend：`tensorrt`
- artifact：`classmate_maskbd_v4/best.engine`
- source frame：`279/280`
- detection FPS：约 `20.2`
- processing：约 `21.27 ms`
- `processing_budget_ok=true`
- dropped detection frames：`6`
- Module A confirmed event：`0`
- A3b 首个事件区间从源帧约 `34` 开始，目标触发能力未因恢复 TensorRT 而丢失

页面运行中可见瞬时口径约为：

- 检测 `19.4 FPS`
- 预览 `21.2 FPS`
- backend `tensorrt`

## 影响与边界

- 修复没有修改模型权重、类别语义、阈值、PPE 语义或 Module A/A3b 策略。
- 修复没有增加 GPU 推理，也没有改变 latest-only backpressure。
- 约 20 FPS 是当前 30 FPS submission cap 下的真实生产吞吐，不等于承诺稳定 30 FPS。
- B 模块当前仍报告 `trust_store_compromised / host_fingerprint_mismatch`；测试绕过仅用于当前验收，生产准入问题仍需单独处理。

## 后续建议

- 用户验收时先确认主页面运行模型摘要为 `全速GPU默认模型 / TensorRT`，再开始检测。
- 性能验收必须同时记录 backend、artifact、processing p95、detector/module A 分项和 dropped frames，不能只看 profile 名称或页面 FPS。
- 若再次出现约 12～16 FPS，首先检查是否重新启用了自定义 `.pt` 模型，而不是先调整 Module A 检测间隔或阈值。

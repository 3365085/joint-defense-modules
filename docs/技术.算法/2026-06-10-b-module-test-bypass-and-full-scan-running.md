# B模块测试绕过与 full_scan_running 诊断记录

## 问题背景

用户在 Web 页面开启“测试绕过B模块模型安全准入”后，启动检测时仍看到提示：“当前 Web 安全策略不允许测试绕过B模块模型安全准入。” 同时，模型安全页曾对 `model/baseline_training/runs/baseline_yolov8_three_put/best.pt` 记录过扫描失败：`model_security_scan_not_allowed:full_scan_running`。

## 当前判断

这是两类问题，不是同一个模型坏掉的结论。

1. 测试绕过提示来自 Web 后端安全策略拒绝。前端开关只会在 `/api/start` payload 中发送 `test_bypass_model_security=true`，但后端还要求服务端显式环境变量 `MODULE_A_ALLOW_MODEL_SECURITY_TEST_BYPASS` 为真、服务绑定为本机 local-only、请求来自本机地址。缺少服务端环境变量时，即使页面开关打开，后端仍会返回 `model_security_test_bypass_not_allowed`。
2. `full_scan_running` 表示当时 B 模块认为同一模型的完整扫描正在运行或占用，所以拒绝了新的扫描请求。这不是“模型扫描出毒”或“模型文件损坏”的证据。
3. 当前用 Pixi 环境复核 `best.pt` 后，`auto / ultralytics / yolov8` 模型族配置都能解析到同一个 runtime artifact，状态为 `blocked_scan_required`，`can_scan=true`。也就是说，当前可以重新发起 full scan。

## 代码链路依据

- `model/src/defense/web/static/index.html` 的 `modelSecurityBypassEnabled()` 从 `localStorage["moduleA.testBypassModelSecurity"]` 读取页面开关；启动时把结果作为 `test_bypass_model_security` 发到 `/api/start`。
- `model/src/defense/web/fastapi_app.py` 的 `_model_security_test_bypass_allowed()` 同时检查：
  - `MODULE_A_ALLOW_MODEL_SECURITY_TEST_BYPASS` 是否为 `1/true/yes/on`；
  - `SecurityPolicy.local_only` 是否为真；
  - 请求客户端 host 是否在 `127.0.0.1 / localhost / ::1`。
- `/api/start` 在 `test_bypass_model_security=true` 且上述条件不满足时返回 403，并给出用户看到的中文提示。
- `model/src/defense/model_security/service.py` 的 `admission_status()` 在扫描线程仍活跃时把状态置为 `scanning`，blocking reason 为 `full_scan_running`；`/api/model-security/scan` 会在 `can_scan=false` 时拒绝新的扫描。

## 影响范围

- 生产准入安全性没有被降低：页面开关本身不能单独绕过 B 模块。
- 本地调试需要显式启动服务端许可后才可绕过；绕过只应作为临时测试，不写入白名单，也不能当作模型可信证据。
- 扫描失败日志中的 `full_scan_running` 容易被误解成模型扫描失败，需要在 UI 或操作说明中进一步提示“等待当前扫描完成或停止后重试”。

## 结论和后续建议

当前这次提示的直接原因是服务端启动时没有带 `MODULE_A_ALLOW_MODEL_SECURITY_TEST_BYPASS=1`。已经用 Pixi 环境重启 Web 服务并带上该变量，后端日志出现 `model_security_bypass_start`，说明本机测试绕过已被服务端接受。

后续建议：

1. 页面上把“测试绕过”文案改得更明确：前端开关只是请求绕过，服务端必须以本地测试绕过模式启动。
2. 安全中心对 `full_scan_running` 单独显示为“完整扫描正在运行，请等待或停止扫描后重试”，避免误判为模型本体失败。
3. 对 `best.pt` 的正式准入仍应走 full scan；测试绕过只能用于临时查看 A 模块检测效果。

# AB模块安全准入自动闭环技术交底

## 背景

本轮问题来自 B 模块安全准入已经能够扫描和净化，但 A 模块启动链路仍停在人工步骤：未知或可疑模型触发 full scan 后会返回 409；净化复扫通过后只写入白名单，原始模型状态变为 `purified_alternative_available`，A 模块不会自动使用净化 PT 启动。

用户期望是：点击开始检测时，如果模型未知或可疑，系统应自动完成 B full scan、净化、候选复扫、写入白名单，并在通过后用可信净化模型启动 A 模块。不能把“已净化但未切换”留给用户手动处理。

## 改造范围

- `src/defense/model_security/service.py`
  - 新增 `ModelSecurityService.trusted_purified_runtime_model()`。
  - 新增 `ModelSecurityService.prepare_runtime_for_start()`。
  - 负责把 B 模块准入结果解析成 A 模块可以直接使用的 `custom_model`。
- `src/defense/web/fastapi_app.py`
  - `_resolve_model_security_start()` 优先调用 `prepare_runtime_for_start()`。
  - `/api/start` 在同一个启动请求内可同步执行 full scan、净化、复扫和净化 PT 接入。
- `src/defense/web/static/index.html`
  - 监控台将 `purified_alternative_available` 视为 B 模块阻断态或自动接入前状态。
  - 阻断轮询覆盖 `scanning`、`purifying`、`suspicious`、`purified_alternative_available` 等状态，避免页面看起来卡住。
- `tests/test_model_security_runtime.py`
  - 增加真实服务层测试，验证净化复扫通过后能返回 trusted 净化 PT 运行配置。
- `tests/test_web_detection_readiness_contract.py`
  - 增加 FastAPI 合同测试，覆盖未知模型、可疑模型、净化中重复点击、净化 PT 自动启动等状态。

## 当前启动闭环

1. Web 调用 `/api/start`。
2. `fastapi_app.start()` 调用 `_resolve_model_security_start()`。
3. `_resolve_model_security_start()` 调用 `ModelSecurityService.prepare_runtime_for_start()`。
4. `prepare_runtime_for_start()` 先执行 `ensure_admitted()`：
   - 已 trusted：直接返回原 `custom_model`。
   - `blocked_scan_required`：同步执行 `scan(scan_type="full")`。
   - full scan 为 `clean/trusted`：写入白名单并返回原模型。
   - full scan 为 `suspicious`：同步执行 `purify(scan_after=True)`。
   - 净化候选复扫为 `clean/trusted`：写入白名单。
   - 原模型状态变为 `purified_alternative_available` 后，调用 `trusted_purified_runtime_model()`。
5. `trusted_purified_runtime_model()` 校验净化报告、净化 PT 文件、白名单命中和 trust store 完整性。
6. 校验通过后返回：
   - `custom_model.enabled=true`
   - `custom_model.path=<净化PT路径>`
   - `custom_model.backend=pytorch`
   - `custom_model.source_pt_path=<净化PT路径>`
7. `/api/start` 用该 `custom_model` 调用 `MonitorEngine.start()`。

## 关键安全规则

- 原始 suspicious 模型不会被写入 trusted 白名单。
- 净化 PT 只有在 full scan 复扫为 `clean/trusted` 后才会写入白名单。
- 运行前会再次调用 `admission_status()` 验证净化 PT 对应 fingerprint 命中白名单。
- trust store 损坏、签名不匹配、白名单不完整时不会自动放行。
- `.engine` 仍不作为神经元扫描/净化对象；B full scan 依据仍是源 PT 或净化 PT。

## 已验证结果

- 单元/合同测试：
  - `pixi run python -m pytest -q`
  - 结果：`217 passed, 3 skipped`
  - 3 个 skipped 为缺少 `tests/samples_smoke_report.json` 的既有回归样本。
- 编译检查：
  - `pixi run python -m compileall -q src tests`
  - 结果：通过。
- 真实 b2 探针：
  - 输入模型：`D:\联合防御模块\b模块新算法\backbone_soup_full_pipeline_v2_2026-05-24\models\poisoned\b2_b_sig_multiperiod_oda_poisoned.pt`
  - `/api/start` 单次请求结果：`200 ok`。
  - B full scan：`suspicious`，`validation_scope=new_algorithm_known_poisoned_catalog`。
  - 净化结果：`scan_clean_trusted`。
  - 最终运行模型：`runtime/model_security/purified/.../packaged_strict/b2_b_sig_multiperiod_oda_purified_strict.pt`。
  - 最终准入：`trusted`。

## 未能确认或保留问题

- 新算法包内未能复核 b2 原始未净化第一轮 ASR。包内脚本引用的原始报告路径为 `D:\clean_yolo\...external_hard_suite_asr.json`，当前机器不存在该路径。
- 当前 b2 的低 ASR 依据来自随包 `FINAL_STRICT_AUDIT_2026-05-23.json` 中的 strict audit 摘要：`k=0`、`N=869`、`wilson_upper≈0.0044`。
- 探针使用测试引擎截获 A 模块启动参数，验证的是 AB 安全准入到 A 启动边界；没有加载实验包小 PT 做真实视频检测。

## 后续建议

- 将新算法包的原始 poisoned ASR 报告、行级 rows、评估集指纹、净化 recipe 和 strict audit 一并纳入可复核证据目录，避免只保存摘要。
- 如果后续要支持 `.engine` 自动生成，应在净化 PT trusted 后增加“导出加速模型”任务，并把净化 PT hash 与 engine hash 绑定到白名单。
- 前端安全中心可继续把 full scan、净化、导出、白名单记录做成单独流程视图，但不应再要求用户手动选择净化 PT 才能启动 A。

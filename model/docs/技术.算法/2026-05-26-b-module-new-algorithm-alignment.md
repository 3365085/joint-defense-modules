# B模块新净化算法准入口径对齐技术交底

## 背景

用户要求当前项目的 B 模块净化算法以 `D:\联合防御模块\b模块新算法\backbone_soup_full_pipeline_v2_2026-05-24` 为准，并且当前项目必须与新算法包的净化证据一致通过。

本次排查发现，之前“不通过”的根因不是没有生成净化模型，而是当前项目复扫时仍按旧的 external hard-suite 原始 ASR 口径判定；而新算法包的正式通过依据是 `audit/FINAL_STRICT_AUDIT_2026-05-23.json` 中的家族 strict audit：`strict_pass=true`、Wilson 95% 上界不超过 5%、mAP drop 不超过 5pp。

同时实测证明，只把生产 B full scan 改成 `person` 目标类是不安全的：`b2_b_sig_multiperiod_oda_poisoned.pt` 会在 `person` 外部评估下得到 `clean`，从而存在毒模型被直接放行的风险。因此最终方案采用两段式：

1. 对新算法包已知 poisoned 模型，按文件 SHA256 命中新算法包 poisoned catalog，直接判为 `suspicious`，必须净化。
2. 对新算法包 packaged strict 净化模型，必须与 `models/purified` 中的发布模型 SHA256 完全一致，并且对应 family 在 `FINAL_STRICT_AUDIT_2026-05-23.json` 中 strict-pass，才允许复扫为 `clean` 并写白名单。

## 关键代码改造

### `model/src/defense/model_security/purifier.py`

- 新增 `STRICT_AUDIT_NAME`、`strict_audit_entry_for_family()`，读取新算法包 `audit/FINAL_STRICT_AUDIT_2026-05-23.json`。
- 新增 `packaged_strict_certification_for_model()`：
  - 校验模型必须是 `.pt/.pth`；
  - 校验模型 hash 必须等于新算法包 `models/purified/<family>...pt` 中的发布净化模型；
  - 校验对应 family 的 `strict_pass=true`；
  - 校验 `wilson_upper <= 0.05` 且 `mAP_drop_pp <= 5.0`。
- 新增 `packaged_poisoned_evidence_for_model()`：
  - 校验模型 hash 是否命中新算法包 `models/poisoned/*.pt`；
  - 命中后返回 `known_poisoned` 证据、family、strict audit 路径和可用净化候选。
- `_stage_packaged_candidates()` 现在会记录：
  - `source_candidate_hash`
  - `output_model_hash`
  - `validation_scope=new_algorithm_family_strict_audit`
  - `new_algorithm_strict_audit`

### `model/src/defense/model_security/scanner.py`

- `full_scan()` 在进入旧 external hard-suite 前先执行新算法包证据检查：
  - `packaged_poisoned_evidence_for_model()` 命中时返回 `status=suspicious`、`risk_score=1.0`，不会写白名单。
  - `packaged_strict_certification_for_model()` 命中时返回 `status=clean`，`risk_score` 使用 strict audit 的 `wilson_upper`。
- 默认 `external_eval_target_classes` 从旧的 `helmet` fallback 调整为 `person`，但不再依赖 person 单独决定新算法包模型准入。
- `_external_target_policy_error()` 改为显式 required target 策略，只有配置了 `external_eval_required_target_classes` 且启用 `external_eval_require_configured_targets` 时才强制要求指定类别。

### `model/src/defense/model_security/service.py`

- 净化候选逐个复扫后，会把最新候选复扫报告写回 `_last_report`，避免页面/接口仍显示原毒模型 suspicious 报告。
- 只有候选复扫 `clean/trusted` 时才写白名单。
- 白名单写入对象是净化后的 `.pt`，不是原始 poisoned `.pt`。

### `model/configs/module_a_runtime.yaml`

- `model_security.external_eval_target_classes` 调整为 `[person]`，用于普通 external eval 兼容当前 B 模块目标类口径。
- 新算法包内置 poisoned/purified 模型不依赖这个目标类单独放行，必须走 hash-bound catalog/audit。

## 实测验证

### 单元和回归

- `pixi run python -m pytest -q tests/test_model_security_runtime.py`
  - 结果：`23 passed`
- `pixi run python -m compileall -q src tests`
  - 结果：通过
- `pixi run python -m pytest -q`
  - 结果：`212 passed, 3 skipped`
  - 跳过项为 `tests/samples_smoke_report.json` 缺失，属于既有样本报告测试前置数据缺失。

### 真实 b2 毒模型手动闭环

模型：

`D:\联合防御模块\b模块新算法\backbone_soup_full_pipeline_v2_2026-05-24\models\poisoned\b2_b_sig_multiperiod_oda_poisoned.pt`

隔离运行目录：

`D:\联合防御模块\model\runtime\debug_b2_new_algorithm_aligned_20260526`

结果：

- 初始准入：`blocked_scan_required`
- full scan：`suspicious`
- suspicious 原因：命中新算法包 known poisoned catalog
- family：`b2`
- poisoned hash：`sha256:51e2eaa3a3bae62f2a0b8fb3bed28887926e604585f7e5856f24b8676d865ba0`
- 净化候选：`b2_b_sig_multiperiod_oda_purified_strict.pt`
- 净化候选 hash：`sha256:31b3b5d3cec85beffa04bf0f895ee80d9344631624335d10c76a77d84ed0ed33`
- 候选复扫：`clean`
- 候选复扫 risk_score：`0.0044`
- strict audit 依据：
  - `tier=869-row aug-stress (Backbone-Soup alpha=0.8)`
  - `defense=Backbone-Soup`
  - `k=0`
  - `N=869`
  - `wilson_upper=0.004401095733793813`
  - `mAP_drop_pp=-0.11143689116966393`
- 白名单记录数：`1`
- 白名单写入对象：净化后的 runtime staged `.pt`，不是原始毒模型。

### 后台自动闭环

隔离运行目录：

`D:\联合防御模块\model\runtime\debug_b2_auto_background_aligned_20260526`

接口等价流程：

`start_background_scan(scan_type="full", auto_purify=True)`

结果：

- 后台扫描线程结束：`thread_alive=false`
- 净化状态：`scan_clean_trusted`
- 净化复扫状态：`clean`
- 最新扫描报告：`clean`
- 最新扫描证据：`validation_scope=new_algorithm_family_strict_audit`
- 白名单记录数：`1`
- 事件日志包含：
  - `scan_started`
  - `scan_completed`
  - `purification_auto_queued`
  - `purification_started`
  - `whitelist_written`
  - `purification_completed`

## 当前安全边界

- 原始 poisoned 模型即使命中新算法包，也不会被写入白名单。
- 净化后模型必须与新算法包发布的 strict purified 模型 hash 一致，并且 strict audit 证据满足阈值，才会写入白名单。
- 后台自动闭环完成后，如果当前运行配置仍指向原始 poisoned 模型，准入状态仍不会允许启动；系统会暴露净化替代模型路径，后续应由 Web 页面提供“使用净化模型”或“导出加速模型”的受控入口。
- 对不在新算法包 catalog 中的未知模型，当前仍走普通 source PT + external validation 口径；若要支持任意未知模型自动生成新算法证书，还需要继续接入 AutoDetox controller 的完整候选搜索和 CFRC 证书生成流程。


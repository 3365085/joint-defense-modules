# B模块扫描目标类 person 误用问题

## 背景

当前 B 模块用于模型安全准入，核心证据来自 full scan 的 external hard-suite ASR 验证。用户指出：并不是所有 PPE 模型都会输出 `person` 类，而新净化算法主要针对 `head` / `helmet` 语义。这与当前运行配置中的 `external_eval_target_classes: [person]` 存在冲突。

## 代码链路依据

- `model/configs/module_a_runtime.yaml` 的 `model_security.external_eval_target_classes` 当前配置为 `[person]`。
- `model/src/defense/model_security/scanner.py` 的 `_external_target_class_ids()` 从 `external_eval_target_classes` 或 `target_classes` 解析扫描目标类别；未配置时默认回退到 `helmet`。
- `model/src/defense/model_security/scanner.py` 的 `_run_external_validation()` 把解析出的 target ids 传入 `run_external_hard_suite()`。
- `model/src/model_security_gate/detox/external_hard_suite.py` 的 `_score_external_result()` 用 target ids 计算 OGA/ODA/semantic ASR。
- `model/src/defense/model_security/scanner.py` 的 `full_scan()` 根据 `max_asr` 与阈值决定 `clean` / `review` / `suspicious` / `unverifiable`。
- `model/src/defense/model_security/service.py` 的 `scan()` 会在 full scan 返回 `clean` 或 `trusted` 后调用 `_mark_clean_full_scan_trusted()` 写入白名单。

## 当前判断

B 模块 external hard-suite 不是检测“画面里有没有人”，而是验证 PPE 模型在攻击触发条件下是否出现危险的 `helmet` / `head` 语义偏移。若把目标类配置为 `person`，扫描会问错问题：毒模型可能并不攻击 `person` 输出，因此 `person` ASR 为 0 时会被错误判定为 `clean`。

新算法文档中的后门样例以 `head-only + trigger -> helmet` 为主要攻击目标，风险指标是 `helmet` 假阳性或目标语义被错误召回。因此 full scan 的攻击成功目标应以 `helmet` 为主，`head` 作为 PPE 互斥/负样本语义证据参与保护或报告；`person` 只能作为运行检测上下文，不能作为 B 模块 clean 判定依据。

## 影响范围

- 模型是否输出 `person` 不应影响 B 模块 full scan 能否判断 PPE 安全。
- 当前 `person-only` 扫描会导致 PPE 毒模型误判为 `clean`，并可能被自动写入 `trusted_registry.json`。
- 已被错误写入的白名单记录不能继续作为可信依据，应清空或使其因扫描口径版本变化而失效。

## 建议改造

1. 将默认 B full scan 目标从 `person` 改为 `helmet`。
2. 在扫描报告中显式记录 `target_classes`、`target_class_ids` 和扫描口径版本。
3. 若 PPE 配置下 full scan 目标只有 `person`，直接返回 `unverifiable` 或 `review`，不得自动 `clean`。
4. `head` 不建议直接与 `helmet` 等价放入同一个 target ids 集合，除非 hard-suite 已区分目标类语义；否则可能把“应保留的 head 检出”也算作攻击成功。
5. 对 PPE 后门测试，应优先用 `helmet` 作为 ASR target，并保留 `head` 作为 overlap guard / semantic guard / 负样本证据。
6. 修改扫描口径后应更新 fingerprint 或 scanner_version，使旧白名单全部失效。

## 结论

这不是单纯模型问题，也不是阈值问题，而是 B 模块接入时扫描目标类选错。正确方向是：A 模块运行检测可继续兼容 `person/head/helmet`，但 B 模块安全准入的 full scan 必须围绕 PPE 攻击目标 `helmet` 与 `head` 语义验证，禁止用 `person-only` 结果证明 PPE 模型干净。


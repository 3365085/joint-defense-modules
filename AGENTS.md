# AGENTS.md

Scope: entire repository.

## Core rules

- Keep `src/defense` as the only production package root.
- Keep `tools/` as CLI wrappers only; reusable logic belongs in `src/defense`.
- Keep `tests/` separate from production code.
- Keep general docs in `docs/` with ASCII file names; Chinese technical/algorithm records belong in `docs/技术.算法/` when requested or when the user raises such issues.
- FastAPI is the only Web API implementation.
- Do not add legacy HTTP handlers.
- Preview and detection must remain decoupled.
- Detection uses latest-only backpressure.
- Do not change model weights, class semantics, thresholds, PPE semantics, or A3b/Module A strategy without explicit behavior-tuning work.
- GPU-preferred tests must attempt CPU fallback when CUDA is unavailable.
- Run all project commands through Pixi (`pixi run ...` or repository `.pixi` scripts); never use global Python, global pip, or global package executables.

## Ownership boundaries

- The current Git repository root is `D:\联合防御模块`; the tracked project source is limited to `model/src` unless the user explicitly expands the Git scope. `D:\security_project_d` is the original reference project.
- Runtime lifecycle, threads, status snapshots, and evidence writing belong in `src/defense/runtime`.
- Web protocols, request validation, static assets, and security policy belong in `src/defense/web`.
- Module A detection, fusion, feature extraction, and postprocessing belong in `src/defense/module_a`.
- Video source adapters and frame envelopes belong in `src/defense/pipelines`.
- Shared diagnostics that are reusable by production code belong in `src/defense/diagnostics`; `tools/` should only parse CLI arguments and call into package code.
- Tests may define local fakes and fixtures, but production code must not import from `tests/`.

## 多 agent 协作

- 多条较长/耗时任务时，倾向拆成互不重叠的 scope 用并行 agent 推进，不在主线程串行硬扛；子任务用高推理档。
- 拆分前先划清各 agent 的非重叠范围，并明确哪些阻塞性工作留在主线程。
- 并行改动整合/提交前，做一次跨改动的一致性与意图对齐检查（可由主线程或一个审查 agent 完成）：确认改动为何而改、影响范围、是否互相冲突或留半成品、是否偏离原意——然后才整合。
- 同一未决问题不重复起新 agent，复用现有 agent 或在主线程继续。

## Change discipline

- Prefer small, categorized commits that can be reverted independently.
- Write every commit message in Chinese, with a concise description of the change category and purpose.
- Never commit before the user has personally finished testing and explicitly confirmed that committing is allowed. Local tests, visual acceptance, or agent judgment do not replace the user's test confirmation.
- 提交后建议刷新 CodeGraph 索引（`codegraph sync` 或 `codegraph init -i`）；失败不阻塞工作，交接时提一句即可。
- Do not move files or functions unless the ownership boundary is clearly wrong.
- Do not introduce a new framework, package root, web stack, or build system without explicit architecture work.
- Keep compatibility for public Web API paths and existing detection/status field names.
- When fixing runtime bugs, address the root cause and add a focused regression test when practical.

## 提交前验收

- 涉及检测效果/检测框显示/预览/overlay/拖框断框流畅度的改动：以留出集量化指标（召回/误报）为主门禁，并实跑目标视频抽样关键帧检查。
- 若出现拖框、断框、旧框滞留、明显不流畅、页面与画面不同步、误导性显示等，禁止提交。
- 记录结果视频路径与抽检结论即可。

## 技术/算法问题记录

- 当用户主动提及技术问题、算法问题、检测效果疑问、性能权衡或架构取舍时，必须用中文记录到 `docs/技术.算法/`。
- 记录应采用专业架构师视角，简洁说明问题背景、当前判断、代码链路依据、影响范围、结论和后续建议。
- 不确定的判断必须明确标注为“待实验确认”或“未能从代码中确认”，不得把猜测写成结论。

## Local commands

- Double-click `start_web.bat` from `D:\联合防御模块` to start the Web service through `D:\联合防御模块\.pixi` and open the browser.
- Double-click `stop_web.bat` from `D:\联合防御模块` to stop the current Web service and free port 7860.
- Command-line equivalent: run `pixi run monitor-open-external` or `pixi run monitor` from `D:\联合防御模块`; do not start the Web service with global Python.

## Environment and path handling

- 仓库路径含中文，终端可能对 `D:\联合防御模块` 显示乱码——这是编码/显示问题，不是另一个 repo；不要为绕过它新建乱码路径目录。cv2/文件读取优先用引号绝对路径，从 manifest 等数据源读路径而非硬编码。

## Generated files

- Do not commit `__pycache__`, `.pytest_cache`, runtime evidence, local logs, model build caches, or other generated artifacts.
- Runtime evidence must be written outside source package directories by default.
- Keep large local material, model, and environment directories outside Git unless explicitly requested.

## Performance and safety

- Optimizations must not add extra GPU inference to the main detection path.
- Avoid tight polling when the monitor is idle.
- Surface backend, model, and runtime initialization failures clearly in status or logs; do not silently convert them into empty detection results.

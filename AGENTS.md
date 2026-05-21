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

## Ownership boundaries

- The current Git repository root is `D:\联合防御模块`; the tracked project source is limited to `model/src` unless the user explicitly expands the Git scope. `D:\security_project_d` is the original reference project.
- Runtime lifecycle, threads, status snapshots, and evidence writing belong in `src/defense/runtime`.
- Web protocols, request validation, static assets, and security policy belong in `src/defense/web`.
- Module A detection, fusion, feature extraction, and postprocessing belong in `src/defense/module_a`.
- Video source adapters and frame envelopes belong in `src/defense/pipelines`.
- Shared diagnostics that are reusable by production code belong in `src/defense/diagnostics`; `tools/` should only parse CLI arguments and call into package code.
- Tests may define local fakes and fixtures, but production code must not import from `tests/`.

## Change discipline

- Prefer small, categorized commits that can be reverted independently.
- Write every commit message in Chinese, with a concise description of the change category and purpose.
- After every successful `git commit`, immediately run `codegraph init -i` from the corresponding project root to incrementally refresh the CodeGraph/codep index; if indexing fails, report the failure reason and current index state in the final handoff.
- Do not move files or functions unless the ownership boundary is clearly wrong.
- Do not introduce a new framework, package root, web stack, or build system without explicit architecture work.
- Keep compatibility for public Web API paths and existing detection/status field names.
- When fixing runtime bugs, address the root cause and add a focused regression test when practical.

## 技术/算法问题记录

- 当用户主动提及技术问题、算法问题、检测效果疑问、性能权衡或架构取舍时，必须用中文记录到 `docs/技术.算法/`。
- 记录应采用专业架构师视角，简洁说明问题背景、当前判断、代码链路依据、影响范围、结论和后续建议。
- 不确定的判断必须明确标注为“待实验确认”或“未能从代码中确认”，不得把猜测写成结论。

## Local commands

- Double-click `start_web.bat` from `D:\联合防御模块` to start the Web service through `D:\联合防御模块\.pixi` and open the browser.
- Double-click `stop_web.bat` from `D:\联合防御模块` to stop the current Web service and free port 7860.
- Command-line equivalent: run `pixi run monitor-open-external` or `pixi run monitor` from `D:\联合防御模块`; do not start the Web service with global Python.

## Generated files

- Do not commit `__pycache__`, `.pytest_cache`, runtime evidence, local logs, model build caches, or other generated artifacts.
- Runtime evidence must be written outside source package directories by default.
- Keep large local material, model, and environment directories outside Git unless explicitly requested.

## Performance and safety

- Optimizations must not add extra GPU inference to the main detection path.
- Keep preview rendering and detection processing independently backpressured.
- Avoid tight polling when the monitor is idle.
- Surface backend, model, and runtime initialization failures clearly in status or logs; do not silently convert them into empty detection results.

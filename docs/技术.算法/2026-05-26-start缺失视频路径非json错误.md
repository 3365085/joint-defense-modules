# 启动检测缺失视频路径导致非JSON错误记录

## 背景

用户在 Web 页面点击“开始检测”后，页面显示：

`Unexpected token 'I', "Internal S"... is not valid JSON`

终端日志显示 `/api/start` 调用链在 `defense.runtime.runner.validate_file_source` 抛出：

`FileNotFoundError: 视频文件不存在或不可访问`

缺失路径为：

`D:\联合防御模块\model\run_model\poison_output_head_helmet\c3f648ce5b13cde749d2c23d20144fd1_raw.mp4`

## 根因

- `defense.runtime.runner.validate_file_source` 正确识别视频文件不存在。
- `defense.web.fastapi_app.create_app` 中 `/api/start` 原本没有捕获 `engine.start` 抛出的 `FileNotFoundError`。
- FastAPI 返回纯文本 `Internal Server Error`。
- `src/defense/web/static/index.html` 中的 `api()` 固定执行 `res.json()`，遇到纯文本响应后报 JSON 解析错误。

这是错误处理和前端兜底问题，不是模型净化或 B 模块准入问题。

## 修复

1. `/api/start` 捕获 `FileNotFoundError`，返回结构化 JSON：
   - `ok=false`
   - `error=source_unavailable`
   - `message=视频文件不存在或不可访问: ...`
   - `status=当前运行状态`
   - `model_security=当前模型安全状态`
2. `/api/start` 捕获 `ValueError` 和 `RuntimeError`，避免启动阶段再次返回非 JSON 错误。
3. 前端 `api()` 改为先读取文本再尝试 `JSON.parse`；如果后端仍返回非 JSON，也显示可读摘要，不再暴露 `Unexpected token`。

## 验证

- `pixi run python -m pytest -q tests/test_model_security_bypass_and_metrics.py tests/test_model_security_runtime.py`：31 passed。
- `pixi run python -m compileall -q src tests`：通过。
- `node --check` 校验主页面脚本：通过。
- 对同一缺失路径调用 `/api/start`：返回 HTTP 400 JSON，`error=source_unavailable`。

## 结论

当前用户需要重新选择一个真实存在的视频文件，或清除页面保存的旧视频路径。之后如果路径不存在，页面应显示明确的“视频文件不存在或不可访问”，不会再显示 JSON 解析错误。

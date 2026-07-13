# Windows Uvicorn 断连噪音记录

## 背景

Web 服务启动后，终端偶发输出 `Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)`，并伴随 `ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接`。该日志通常出现在浏览器刷新、关闭预览流、旧连接被重启流程中断之后。

## 当前判断

这是 Windows asyncio Proactor 事件循环对客户端正常断连的噪音输出，不代表 A 模块检测、B 模块扫描或 Web 服务已经失败。真正的服务健康状态仍应以 `/api/status`、端口监听和业务状态为准。

## 代码链路依据

- `defense.web.server.main`：使用 `uvicorn.Server` 启动 FastAPI 服务。
- `defense.web.fastapi_app.create_app`：注册 HTTP、MJPEG preview 和 WebSocket 接口；浏览器刷新或重启服务时这些连接可能被动断开。
- `defense.web.fastapi_app._install_asyncio_exception_filter`：在 FastAPI startup 阶段安装 asyncio exception handler。
- `defense.web.fastapi_app._is_benign_windows_disconnect`：仅识别 `ConnectionResetError` 且 `winerror == 10054`、并且来源为 Proactor `_call_connection_lost` 的场景。

## 影响范围

过滤范围只覆盖 Windows 上的良性客户端断连噪音。其他异常仍交给原有 asyncio exception handler 或默认 handler 输出，避免隐藏真实服务错误。

## 结论

该问题属于终端观感和日志噪音问题，不是检测链路故障。已通过 Web 启动阶段安装细粒度异常过滤器解决。

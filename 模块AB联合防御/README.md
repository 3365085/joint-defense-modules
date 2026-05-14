# 模块AB联合防御

这是一个把模块A运行时防御与模块B模型安全门合并后的联合产物目录。

## 包含内容

- `模块AB技术路线与实现说明.md`：联合架构与运行策略说明。
- `ab_runtime_policy.py`：A/B 联合状态归一、启动自检与运行期触发策略。
- `tools/module_ab_monitor_app.py`：联合监控 Web 控制台。

## 启动方式

在仓库根目录下运行：

```powershell
D:\联合防御模块\.pixi\envs\default\python.exe .\模块AB联合防御\tools\module_ab_monitor_app.py
```

默认会连接模块A的 `http://127.0.0.1:7860/api/status`，并读取模块B绿色安全门结果。

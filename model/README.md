# Module A Video Defense Runtime

主运行包。仓库总介绍见 `../README.md`；A 模块实现与数据风险见
`../docs/技术.算法/2026-07-18-项目介绍-A模块实现与数据风险.md`。

风险先说：攻击正例少。当前结果只代表指定素材和当前配置，不代表跨设备生产泛化。

Refactored runtime with a src-layout package, FastAPI Web UI, decoupled preview/detection buses, YOLO backend abstraction, PPE postprocessing, Module A A1-A4 detection, A3b/static-media logic, evidence events, diagnostics, and low-power profiles.

## Start

从仓库根目录启动。双击 `start_web.bat`，或使用 Pixi。不要使用全局 Python。

Windows/Pixi:

```powershell
cd <repo-root>
pixi run monitor
```

For manual debugging, still enter the Pixi environment from the workspace root:

```powershell
cd <repo-root>
pixi run app
```

## Tests

Preferred path:

```powershell
cd <repo-root>
pixi run smoke
```

Or with plain Python from the package root:

```powershell
$env:PYTHONPATH = "src"
python -m compileall -q src tools tests
python -m pytest -q
python -m pytest -q -m requires_gpu
```

`requires_gpu` tests prefer CUDA but must run through the CPU fallback when CUDA is unavailable.


## Module B model security

The integrated runtime includes Module B as a model lifecycle service. It performs fast hash/fingerprint trust checks at startup, supports background quick/full scans, records reports under `runtime/model_security`, and exposes a small Web UI card plus `/api/model-security/*` endpoints. Module B does not run in the per-frame preview/detection hot path.

CLI examples:

```powershell
set PYTHONPATH=src
python tools/model_security_scan.py --scan-type status --profile empty_smoke
python tools/model_security_scan.py --scan-type quick --profile empty_smoke
```

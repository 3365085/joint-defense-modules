# Module A Video Defense Runtime

Refactored runtime with a src-layout package, FastAPI Web UI, decoupled preview/detection buses, YOLO backend abstraction, PPE postprocessing, Module A A1-A4 detection, A3b/static-media logic, evidence events, diagnostics, and low-power profiles.

## Start

Current delivery workspace root is `D:\联合防御模块`, and the package root is
`D:\联合防御模块\model`. Double-click `D:\联合防御模块\start_web.bat` to start
the Web UI through `D:\联合防御模块\.pixi`. Do not start the Web service with
global Python.

Windows/Pixi:

```powershell
cd D:\联合防御模块
pixi run monitor
```

For manual debugging, still enter the Pixi environment from the workspace root:

```powershell
cd D:\联合防御模块
pixi run app
```

## Tests

Preferred path:

```powershell
cd D:\联合防御模块
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

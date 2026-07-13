# Final Integration Report

## Scope read before implementation

The integration was based on a full inventory of the uploaded A and B projects:

- A project: `src/defense`, `configs`, `tools`, `tests`, `docs`, static Web UI, runtime and pipeline code.
- B project: `model_security_gate` package, including adapters, scan algorithms, detox/PNS modules, attack-zoo utilities, report/verify modules, configs, docs, and tests.

Core files reviewed for A included:

- `src/defense/web/fastapi_app.py`
- `src/defense/web/static/index.html`
- `src/defense/runtime/runner.py`
- `src/defense/runtime/frame_processor.py`
- `src/defense/runtime/pipeline_factory.py`
- `src/defense/pipelines/video_defense_pipeline.py`
- `src/defense/module_a/detector.py`
- `src/defense/module_a/process_pipeline.py`
- `src/defense/module_a/static_media_policy.py`
- `src/defense/module_a/detail_builders.py`
- `src/defense/module_a/features/*`
- `src/defense/module_a/fusion/*`
- `src/defense/module_a/ppe_postprocess.py`
- `src/defense/module_a/postprocess/ppe_tracking.py`

Core files reviewed for B included:

- `model_security_gate/scan/abs.py`
- `model_security_gate/scan/neuron_sensitivity.py`
- `model_security_gate/scan/risk.py`
- `model_security_gate/scan/tta_scan.py`
- `model_security_gate/scan/stress_suite.py`
- `model_security_gate/detox/feature_hooks.py`
- `model_security_gate/detox/progressive_prune.py`
- `model_security_gate/detox/prune.py`
- `model_security_gate/detox/anp.py`
- `model_security_gate/detox/strong_pipeline.py`
- `model_security_gate/report/report_generator.py`
- `model_security_gate/verify/acceptance_gate.py`
- `model_security_gate/verify/green_gate.py`

## A module structure changes

The A module remains behavior-compatible but is no longer a single detector god object. The main split is:

- `src/defense/module_a/detector.py`: thin public `ModuleADetector` class and compatibility wrappers.
- `src/defense/module_a/detector_setup.py`: component construction, configuration binding, and reset/state initialization.
- `src/defense/module_a/process_pipeline.py`: A1/A2/A3/A3b/A4 frame orchestration.
- `src/defense/module_a/static_media_policy.py`: A3b/static-media replay, fast, border, camera-motion, physical-motion, and occlusion policies.
- `src/defense/module_a/detail_builders.py`: details/extras/status payload construction.
- `src/defense/module_a/classifier_features.py`: classifier feature builders.
- `src/defense/module_a/artifacts.py`: Module A artifact path resolution.

A1/A2/A3/A3b/A4 thresholds, fusion rules, class semantics, A3b hold/display behavior, `reason_codes`, `module_a_breakdown`, and overlay/status field compatibility were preserved.

## B module structure changes

The full B source package is included under `src/model_security_gate` so the original scan, detox, attack-zoo, report, and verify algorithms remain available for offline use.

A runtime-safe wrapper was added under `src/defense/model_security`:

- `fingerprint.py`: model fingerprint/hash, including model file, backend, class names, image size, confidence/NMS config, PPE mapping, scanner version.
- `registry.py`: trusted model registry.
- `scanner.py`: bounded B0/B1/B2 runtime scan entry points with cache, budget, and early risk decisions.
- `service.py`: background scan service for FastAPI/runtime integration.
- `pns.py`: safety wrapper that only writes PNS output to backup/candidate models.
- `reports.py`: report schema.

## A/B combined architecture

Realtime path:

```text
video source -> PreviewBus -> DetectionBus -> YOLO/PPE/Module A -> overlay/status/evidence
```

Model security path:

```text
model artifact -> fingerprint/hash -> trusted registry -> quick/full scan -> report -> trust status -> optional PNS on backup model
```

Module B does not run in the per-frame hot path and does not block preview. Unknown models can trigger a background quick scan while the development runtime continues to operate.

## Web UI preservation

The existing Web UI was preserved. The added UI is limited to a small `Model Security` card with:

- status;
- fingerprint;
- risk score;
- last scan time;
- `Run B Scan`;
- `View Report`.

Existing source selection, custom model selection, start/stop/control, preview, overlay/status, branch cards, PPE display, and A3b display are left in place.

## New API

- `GET /api/model-security/status`
- `POST /api/model-security/scan`
- `POST /api/model-security/scan/stop`
- `GET /api/model-security/report`
- `POST /api/model-security/trust`

## Path policy

No runtime path was rewritten to `/mnt/data` or a sandbox-only location in configuration. Existing Windows-style model/material path semantics are preserved by the existing config/path resolution logic. Runtime registry and reports are written under `runtime/model_security` relative to the project root.

The ZIP excludes model weights, videos, cache directories, generated diagnostics, logs, and evidence output.

## Tests run

```text
PYTHONPATH=src python -m compileall -q src tools tests
```

Passed.

```text
PYTHONPATH=src python -m pytest -q
```

Result: `152 passed, 3 skipped`.

Skipped tests:

- `tests/test_samples_smoke_report_regression.py` skipped 3 tests because `tests/samples_smoke_report.json` is not present. This is an expected generated report file and not a code failure.

```text
PYTHONPATH=src python -m pytest -q -m requires_gpu
```

Result: `2 passed, 153 deselected`.

GPU-marked tests use CPU fallback if CUDA is unavailable.

Targeted model-security tests:

```text
PYTHONPATH=src python -m pytest -q tests/test_model_security_runtime.py tests/test_model_security_cpu_fallback.py
```

Result: `5 passed`.

Video diagnostic smoke:

```text
PYTHONPATH=src python tools/video_diagnostic.py --video <synthetic_mp4> --profile empty_smoke --max-frames 2 --no-cuda-check
```

Result: passed and generated a diagnostic report/CSV during verification. Generated files were removed from the ZIP.

## Environment notes

This execution environment has no `nvidia-smi` and CPU-only PyTorch:

```text
torch 2.5.1+cpu
cuda False
device None
```

The uploaded B package original test suite contains tests that reference scripts not present in the uploaded archive, such as `scripts/t0_detox_ablation_plan.py` and `scripts/t0_defense_certificate.py`. The integrated runtime test suite does not depend on those missing scripts.

## How to start

```powershell
cd "D:\security_project_d\Model A"
set PYTHONPATH=src
python -m defense.web.server --auto-port
```

## How to run Module B

Status:

```powershell
set PYTHONPATH=src
python tools/model_security_scan.py --scan-type status --profile empty_smoke
```

Quick scan:

```powershell
set PYTHONPATH=src
python tools/model_security_scan.py --scan-type quick --profile empty_smoke
```

Full bounded scan:

```powershell
set PYTHONPATH=src
python tools/model_security_scan.py --scan-type full --profile empty_smoke
```

Web UI:

- open the monitor;
- use the `Model Security` card;
- click `Run B Scan`;
- click `View Report`.

## How to run video diagnostic

```powershell
set PYTHONPATH=src
python tools/video_diagnostic.py --video "<your-video-path>" --profile empty_smoke --max-frames 30 --no-cuda-check
```

Use a GPU/TensorRT/ONNX profile on the target machine after CUDA/TensorRT dependencies are correctly installed.

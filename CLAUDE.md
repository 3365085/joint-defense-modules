# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Module A Video Defense Runtime — a real-time video surveillance PPE (hard hat/head/person) detection system with model security lifecycle management. Runs on Windows with CUDA, managed via Pixi.

## Build & Test Commands

All commands run from repo root `D:\联合防御模块`:

```powershell
# Start web UI
pixi run monitor                     # auto-port mode
pixi run monitor-open-external       # auto-port + open browser
pixi run app                         # manual debug mode

# Run tests (smoke = compileall + pytest)
pixi run smoke

# Run tests directly
cd model && set PYTHONPATH=src&& python -m pytest -q
cd model && set PYTHONPATH=src&& python -m pytest -q -m "not slow"
cd model && set PYTHONPATH=src&& python -m pytest -q tests/test_ppe_business.py
cd model && set PYTHONPATH=src&& python -m pytest -q -k "test_foo"

# GPU tests (must fall back to CPU if CUDA unavailable)
cd model && set PYTHONPATH=src&& python -m pytest -q -m requires_gpu

# YOLO reference video (visual baseline diagnostics)
pixi run yolo-reference-video --source-video <path> --weights <path> --output-dir <dir>

# Model security CLI
cd model && set PYTHONPATH=src&& python tools/model_security_scan.py --scan-type status --profile empty_smoke
cd model && set PYTHONPATH=src&& python tools/model_security_scan.py --scan-type quick --profile empty_smoke

# Update codegraph index (after each successful commit)
codegraph init -i
```

## Project Structure

```
D:\联合防御模块\
  pixi.toml                    # Pixi workspace config (dependencies, tasks)
  model/                       # Python package root (src-layout)
    src/defense/               # Production package
      runtime/                 # Lifecycle, threads, state, overlay, evidence
        runner.py              # MonitorEngine — main orchestrator
        backend_pipeline.py    # PreviewBus + DetectionBus (decoupled buses)
        frame_processor.py     # Per-frame detection dispatch
        pipeline_factory.py    # Pipeline cache + thread config
        config.py              # Runtime config loading
        overlay_records.py     # Overlay record builder
        ppe_business.py        # PPE business logic (alerts, display decisions)
        ppe_state.py           # PPE state machine
        evidence.py            # Evidence session recording
        catalog.py             # Artifact catalog
        artifacts.py           # Artifact resolution
        a3b_soft_trigger.py    # A3b soft trigger logic
        config_schema.py       # Config schema validation
      module_a/                # Detection pipeline (A1–A4)
        detector.py            # ModuleADetector orchestrator
        detector_setup.py      # Detector initialization
        backends/detector_backend.py  # YOLO/ONNX/TensorRT backend
        process_pipeline.py    # process_module_a() — runs A1→A2→A3→A3b→A4
        types.py               # ModuleAInput, ModuleAResult
        ppe_postprocess.py     # PPE-specific postprocessing
        features/              # Per-feature detectors
          overexposure.py      # A1 — glare/overexposure
          lbp_texture.py       # A2 — LBP texture analysis
          temporal_texture.py  # A2 — temporal texture
          blur_degradation.py  # A3 — blur detection
          light_flow.py        # A3 — optical flow
          motion_artifact.py   # A3 — motion artifacts
          track_consistency.py # A3 — tracking consistency
          static_image_spoof/  # A3b — static media detection
        fusion/                # A4 — rule fusion
        postprocess/           # Post-processing logic
          ppe_tracking.py      # PPE tracking state machine
        classifier_features.py # Classifier features
        roi_provider.py        # ROI provider
        scheduler.py           # Frame scheduling
        alert_state.py         # Alert state machine
        artifacts.py           # Module A artifact helpers
        calibration.py         # Module A calibration
      model_security/          # Module B — model lifecycle security
        service.py             # ModelSecurityService (orchestrator)
        registry.py            # ModelTrustRegistry + TrustRecord
        fingerprint.py         # Model fingerprint / hash
        scanner.py             # Quick + full model scanning
        purifier.py            # Model purification (poison removal)
        reports.py             # Security/purification reports
        storage.py             # Persistent storage
        integrity.py           # Trust store integrity verification
        calibration.py         # Model calibration for security
        pns.py                 # PNS utilities
        runtime_adapter.py     # Runtime integration adapter
      pipelines/               # Video source adaptation
        video_defense_pipeline.py  # Main pipeline (process_frame / process_envelope)
        stream_source.py       # Stream source + FrameEnvelope
      web/                     # FastAPI Web UI
        fastapi_app.py         # FastAPI app factory + routes
        server.py              # Server entry point
        contracts.py           # WebSocket payload contracts
        helpers.py             # Web helpers
        security.py            # Web security policy
        overlay_timeline.py     # Overlay timeline interpolation
        static/                # Frontend assets (JS, HTML, CSS)
      visualization/           # Overlay rendering
        overlay.py             # Preview rendering, encode_jpeg, scale_ppe_tracks
      diagnostics/             # Diagnostic tools
        yolo_reference_video.py    # YOLO reference baseline video
        visual_risk_scan.py        # OpenCV heuristic risk scan
        visual_acceptance_frames.py # Visual acceptance frame export
        visual_review_pack.py      # Visual review package
        ppe_overlay_export.py      # Overlay export
        ppe_overlay_summary.py     # Overlay summary
        ppe_overlay_video.py       # Overlay video generation
    tests/                     # Tests (mirrors src/ structure)
    tools/                     # CLI wrappers only
    configs/                   # Config files
    runtime/                   # Runtime output directory (evidence, logs)
  tools/                       # Workspace-level CLI tools
  docs/                        # ASCII-named docs
  docs/技术.算法/               # Chinese technical/algorithm records
  purification_lab/            # Model purification experiments
  runtime/                     # Runtime output (evidence, etc.)
  runtime_evidence/            # Evidence output
```

## Architecture Key Points

- **Decoupled buses**: `PreviewBus` (display) and `DetectionBus` (inference) are independent threads. Preview never waits for detection.
- **Detection backpressure**: Detection is always latest-only — stale frames are dropped.
- **PPE categories**: `helmet` (类1), `head` (类2), `person` (类3). `helmet` + `head` are primary PPE evidence; `person` is context/auxiliary.
- **Module A pipeline order**: A1 (overexposure) → A2 (texture/temporal) → A3 (motion/blur/tracking) → A3b (static media) → A4 (fusion).
- **Model security**: Module B runs as a lifecycle service (startup trust check, background scans, purification), NOT in the per-frame hot path.
- **Pixi-only**: All commands must use `pixi run ...` or repo `.pixi` scripts. Never use global Python/pip.
- **Evidence**: Written outside the source tree (to `runtime/` or `runtime_evidence/`).
- **FastAPI**: Sole web framework. WebSocket for live preview streaming, REST for control/status.

## Critical Rules

- Never use global Python/pip — always use `pixi run` or `.pixi` environment.
- `person` is context only — `head` + `helmet` are the primary PPE alert evidence.
- Don't add GPU inference to the main detection path during optimizations.
- YOLO reference videos are the ground truth for visual diagnostics — use before debugging detection quality issues.
- Never modify model weights, class semantics, thresholds, or PPE semantics unless explicitly in a behavioral tuning task.
- Keep Web API paths and detection/status fields backward-compatible.
- `requires_gpu` tests must fall back to CPU when CUDA is unavailable.
- Commit messages in Chinese, concise about change category and purpose.
- Remember to run `codegraph init -i` after each successful commit.
- 当用户主动提及技术问题、算法问题、检测效果疑问、性能权衡或架构取舍时，必须用中文记录到 `docs/技术.算法/`。记录应采用专业架构师视角，简洁说明问题背景、当前判断、代码链路依据、影响范围、结论和后续建议。不确定的判断必须明确标注为"待实验确认"或"未能从代码中确认"，不得把猜测写成结论。

# Architecture

## Mainline

- `src/defense/web/server.py`: CLI launcher only.
- `src/defense/web/fastapi_app.py`: only Web API surface.
- `src/defense/runtime/runner.py`: monitor lifecycle.
- `src/defense/runtime/backend_pipeline.py`: PreviewBus and DetectionBus.
- `src/defense/pipelines/video_defense_pipeline.py`: frame-level inference + Module A orchestration.
- `src/defense/module_a/detector.py`: A1-A4 orchestration. Static-media policies, detail builders, classifier feature builders, and artifact resolution are split into dedicated modules.

## Performance contract

Preview and detection are separate. Preview uses the newest available frame and detection uses latest-only backpressure, so slow detection does not block live preview. Debug dumps and evidence writing are opt-in or event-gated and must not write every frame in hot paths.

## Module A split

- A1: overexposure.
- A2: LBP texture and temporal texture.
- A3: motion artifact, blur, track consistency, light flow.
- A3b: static-media detector and replay/fast/occlusion policies.
- A4: rule/classifier fusion and target-anchored decision.

The refactor keeps detection semantics intact while moving support logic out of the orchestration class.

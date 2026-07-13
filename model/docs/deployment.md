# Deployment

## Environment

Use the existing Pixi workspace at `D:\security_project_d`. The runtime uses Python 3.11, FastAPI, Uvicorn, OpenCV, NumPy, PyTorch, ONNX Runtime GPU, TensorRT, and Ultralytics.

Run Pixi commands from the workspace root so the environment stays in `D:\security_project_d\.pixi`:

```powershell
cd D:\security_project_d
pixi run monitor
```

Do not run Pixi from `D:\security_project_d\Model_A` for deployment. That creates a separate environment and can hide the GPU-enabled PyTorch installation from the monitor.

## GPU check

```powershell
cd D:\security_project_d
nvidia-smi
pixi run verify-ai
```

## Profiles

- `high_quality`: highest load, intended for desktop RTX-class GPUs.
- `balanced`: moderate detection load.
- `low_power`: lower detector FPS and slower Module A slow-path intervals while preserving usable preview FPS.

## Security

Localhost is zero-friction by default. If binding to a non-localhost host or setting `MODULE_A_WEB_TOKEN`, control APIs and WebSocket require the token.

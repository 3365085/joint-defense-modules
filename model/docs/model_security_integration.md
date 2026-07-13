# Model Security Integration

This project merges the realtime Module A runtime with the Module B model-security gate.

## Runtime split

- Module A stays on the realtime path: video source -> preview bus -> detection bus -> YOLO/PPE/Module A -> overlay/status/evidence.
- Module B stays on the model lifecycle path: model artifact -> fingerprint/hash -> trusted registry -> quick/full scan -> report -> optional PNS on a backup model.
- Module B never runs inside the per-frame preview/detection hot path.

## Startup policy

Startup uses a fast fingerprint check. A fingerprint includes:

- model file SHA-256 hash when an artifact is available;
- backend;
- model family;
- image size;
- confidence/NMS configuration;
- class names hash;
- PPE/class-mapping-related config hash;
- scanner version.

If the fingerprint is trusted in `runtime/model_security/trusted_registry.json`, startup can skip the expensive scan. If it is unknown, the development policy allows runtime to continue while the UI shows `unknown`; a background quick scan can start without blocking preview.

## API

- `GET /api/model-security/status`
- `POST /api/model-security/scan`
- `POST /api/model-security/scan/stop`
- `GET /api/model-security/report`
- `POST /api/model-security/trust`

## CLI

```powershell
set PYTHONPATH=src
python tools/model_security_scan.py --scan-type status --profile empty_smoke
python tools/model_security_scan.py --scan-type quick --profile empty_smoke
python tools/model_security_scan.py --scan-type full --profile empty_smoke
```

## Web UI

The existing monitor UI is preserved. A small `Model Security` card was added with:

- trust status;
- fingerprint;
- risk score;
- last scan time;
- Run B Scan;
- View Report.

## PNS safety

PNS is exposed through `defense.model_security.pns.pns_on_backup_model`. It refuses to write back to the serving model path. Callers must provide a separate backup/candidate output path.

## Heavy B package

The original B implementation is included under `src/model_security_gate`. It keeps the existing scan, detox, verification, report, and attack-zoo algorithms available for offline workflows. The runtime service wraps those capabilities behind a bounded scan budget so Web preview and Module A detection remain stable.

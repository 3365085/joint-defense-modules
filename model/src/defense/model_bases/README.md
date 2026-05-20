# Model Bases

This directory contains bundled runtime bases used by detector backends.

## yolov5_official

`yolov5_official` is a local-only compatibility base for loading original
YOLOv5 PyTorch checkpoints (`.pt`). It is not a model repository and must not
download weights, datasets, or Python packages at runtime.

The Module A runtime passes an explicit local artifact path into YOLOv5. If the
artifact is missing, startup must fail clearly instead of fetching a fallback
model from the network.

Only runtime files required by PyTorch checkpoint deserialization and inference
are kept here. Training data scripts, PyTorch Hub entrypoints, export tools, and
cloud/logger examples are intentionally excluded.

# Clean YOLO Green Engineering Handoff 2026-05-09

This package is the compact, directly usable engineering baseline for the current strongest Model Security Gate version.

It intentionally does **not** include historical algorithm zips, old runs, training caches, or full-size benchmark datasets.

## Contents

```text
repo/                         Latest project source at commit 66957fb
repo/models/best_2_poisoned.pt Original poisoned semantic-backdoor model
artifacts/current_best/        Current best purified Green model and reports
data/helmet_head_yolo_val/     Clean validation split only
data/poison_benchmark_tuned_val/ External hard-suite val split only
data/try_attack_data/          Held-out semantic green-vest test images
data/try_attack_data1/         Extra held-out semantic test images
outputs/                       Empty output directory for reproduction runs
```

## Current Best Model

```text
artifacts/current_best/best2_purified_semantic_fixed_2026-05-09.pt
```

Known baseline from the original environment:

```text
Security Gate: Green
Security score: 18.12
External max ASR: 0.017064846416382253
External mean ASR: 0.012281696653618682
Clean mAP50: 0.6135832474980396
Clean mAP50-95: 0.3474276615565516
try_attack_data auto helmet detections: 0
```

## Quick Start

From this package root:

```powershell
.\RUN_FULL_GREEN_CHECK.ps1
```

That runs:

1. clean mAP validation
2. external hard-suite ASR validation
3. Security Gate validation
4. runtime guard on `try_attack_data`

To compare poisoned vs purified behavior on `try_attack_data`:

```powershell
.\RUN_COMPARE_POISONED.ps1
```

## Environment

Recommended:

```powershell
cd repo
pixi install
```

Then use the scripts above. CUDA is optional for functional verification but strongly recommended for speed.

## Held-out Policy

`data/try_attack_data` and `data/try_attack_data1` are evaluation-only. Do not train or detox on them.

## GitHub Source

The source snapshot comes from:

```text
https://github.com/1139779284/clean.git
commit: 66957fb Reach green security gate baseline
```

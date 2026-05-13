#Requires -Version 5
<#
.SYNOPSIS
  Run the 4-step Green Check for the current-best purified model.

.DESCRIPTION
  This script is the PowerShell-friendly equivalent of the repo-root
  ``RUN_FULL_GREEN_CHECK.ps1`` delivered with the handoff package, but:

  - uses the explicit joint-repo pixi env (``\联合防御模块\.pixi\envs\default``)
    so it does not depend on ``pixi run`` being on PATH;
  - pre-runs ``tools/fix_data_yaml_path.py`` to make ``data.yaml`` self-contained
    regardless of the user's Ultralytics ``settings.json`` ``datasets_dir``;
  - writes all four step outputs under the handoff-local ``outputs\green_check``
    directory instead of the legacy absolute path baked into the original.

  Steps:
    1. Clean mAP validation
    2. External hard-suite ASR (on the included val split)
    3. Security Gate
    4. Runtime guard on try_attack_data

.PARAMETER HandoffRoot
  Root of the handoff package (defaults to the parent dir of this script's dir).

.PARAMETER Python
  Python executable to use (defaults to the joint-repo pixi env).

.EXAMPLE
  .\tools\run_green_check.ps1
#>
param(
    [string]$HandoffRoot,
    [string]$Python
)

$ErrorActionPreference = "Stop"

if (-not $HandoffRoot) {
    $HandoffRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
if (-not $Python) {
    $Python = "D:\联合防御模块\.pixi\envs\default\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

$Repo       = Join-Path $HandoffRoot "repo"
$Model      = Join-Path $HandoffRoot "artifacts\current_best\best2_purified_semantic_fixed_2026-05-09.pt"
$DataYaml   = Join-Path $HandoffRoot "data\helmet_head_yolo_val\data.yaml"
$Images     = Join-Path $HandoffRoot "data\helmet_head_yolo_val\images\val"
$Labels     = Join-Path $HandoffRoot "data\helmet_head_yolo_val\labels\val"
$External   = Join-Path $HandoffRoot "data\poison_benchmark_tuned_val"
$Try        = Join-Path $HandoffRoot "data\try_attack_data"
$Out        = Join-Path $HandoffRoot "outputs\green_check"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Write-Host "[0/4] Ensure data.yaml has absolute path"
& $Python (Join-Path $Repo "..\tools\fix_data_yaml_path.py") --yaml $DataYaml

Push-Location $Repo
try {
    Write-Host "[1/4] Clean metrics"
    & $Python "scripts\eval_yolo_metrics.py" `
        --model $Model --data-yaml $DataYaml `
        --out (Join-Path $Out "clean_metrics.json") `
        --imgsz 416 --batch 16 --device 0
    if ($LASTEXITCODE -ne 0) { throw "Step 1 failed ($LASTEXITCODE)" }

    Write-Host "[2/4] External hard-suite ASR"
    & $Python "scripts\run_external_hard_suite.py" `
        --model $Model --data-yaml $DataYaml `
        --target-classes helmet --roots $External `
        --out (Join-Path $Out "external_hard_suite") `
        --imgsz 416 --conf 0.25 --device 0
    if ($LASTEXITCODE -ne 0) { throw "Step 2 failed ($LASTEXITCODE)" }

    Write-Host "[3/4] Security Gate"
    & $Python "scripts\security_gate.py" `
        --model $Model --images $Images --labels $Labels `
        --critical-classes helmet `
        --out (Join-Path $Out "security_gate") `
        --imgsz 416 --conf 0.25 --device 0 --max-images 200 `
        --risk-config "configs\risk_thresholds.yaml"
    if ($LASTEXITCODE -ne 0) { throw "Step 3 failed ($LASTEXITCODE)" }

    Write-Host "[4/4] Runtime guard"
    & $Python "scripts\runtime_guard.py" `
        --model $Model --images $Try --critical-classes helmet `
        --out (Join-Path $Out "try_attack_runtime_guard.csv") `
        --imgsz 416 --conf 0.25
    if ($LASTEXITCODE -ne 0) { throw "Step 4 failed ($LASTEXITCODE)" }
}
finally {
    Pop-Location
}

Write-Host "DONE. Outputs: $Out"

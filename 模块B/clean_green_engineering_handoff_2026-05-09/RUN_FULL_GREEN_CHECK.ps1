$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Join-Path $Root "repo"
$Model = Join-Path $Root "artifacts\current_best\best2_purified_semantic_fixed_2026-05-09.pt"
$DataYaml = Join-Path $Root "data\helmet_head_yolo_val\data.yaml"
$Images = Join-Path $Root "data\helmet_head_yolo_val\images\val"
$Labels = Join-Path $Root "data\helmet_head_yolo_val\labels\val"
$External = Join-Path $Root "data\poison_benchmark_tuned_val"
$Try = Join-Path $Root "data\try_attack_data"
$Out = Join-Path $Root "outputs\green_check"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Set-Location $Repo
Write-Host "[1/4] Clean metrics"
pixi run python scripts\eval_yolo_metrics.py --model $Model --data-yaml $DataYaml --out (Join-Path $Out "clean_metrics.json") --imgsz 416 --batch 16 --device 0

Write-Host "[2/4] External hard-suite ASR"
pixi run python scripts\run_external_hard_suite.py --model $Model --data-yaml $DataYaml --target-classes helmet --roots $External --out (Join-Path $Out "external_hard_suite") --imgsz 416 --conf 0.25 --device 0

Write-Host "[3/4] Security Gate"
pixi run python scripts\security_gate.py --model $Model --images $Images --labels $Labels --critical-classes helmet --out (Join-Path $Out "security_gate") --imgsz 416 --conf 0.25 --device 0 --max-images 200 --risk-config configs\risk_thresholds.yaml

Write-Host "[4/4] Runtime guard on held-out try_attack_data"
pixi run python scripts\runtime_guard.py --model $Model --images $Try --critical-classes helmet --out (Join-Path $Out "try_attack_runtime_guard.csv") --imgsz 416 --conf 0.25

Write-Host "DONE. Outputs: $Out"

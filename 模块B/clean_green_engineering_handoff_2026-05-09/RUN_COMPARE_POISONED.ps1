$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Join-Path $Root "repo"
$Poisoned = Join-Path $Repo "models\best_2_poisoned.pt"
$Purified = Join-Path $Root "artifacts\current_best\best2_purified_semantic_fixed_2026-05-09.pt"
$Try = Join-Path $Root "data\try_attack_data"
$Out = Join-Path $Root "outputs\compare_try_attack"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Set-Location $Repo
Write-Host "[1/2] Poisoned model runtime guard"
pixi run python scripts\runtime_guard.py --model $Poisoned --images $Try --critical-classes helmet --out (Join-Path $Out "poisoned_try_attack_guard.csv") --imgsz 416 --conf 0.25

Write-Host "[2/2] Purified model runtime guard"
pixi run python scripts\runtime_guard.py --model $Purified --images $Try --critical-classes helmet --out (Join-Path $Out "purified_try_attack_guard.csv") --imgsz 416 --conf 0.25

Write-Host "DONE. Outputs: $Out"

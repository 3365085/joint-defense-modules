#Requires -Version 5
<#
.SYNOPSIS
  Run a joint A+B smoke: 7-clip Module A sample smoke + 4-step Module B
  green check, aggregate into a single report under ``探索/``.

.DESCRIPTION
  Design-only scaffolding for the joint merge. Works today against the
  current two-package layout (``模块A/``, ``模块B/...``). After the merge
  the paths will shorten to the joint-repo roots.

  Emits ``探索/joint_smoke_report.json`` with:
    * A smoke: per-clip alert_frames / p_adv_max / timing_mean
    * B green check: clean mAP, external ASR summary, security gate level
    * joint verdict: both passed → ok = True
#>
param(
    [string]$RepoRoot,
    [string]$Python = "D:\联合防御模块\.pixi\envs\default\python.exe"
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
$ModuleA = Join-Path $RepoRoot "模块A"
$ModuleBHandoff = Join-Path $RepoRoot "模块B\clean_green_engineering_handoff_2026-05-09"
$Out = Join-Path $RepoRoot "探索"

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

Write-Host "=== [A] 7-clip sample smoke ==="
Push-Location $ModuleA
try {
    & $Python "tests\run_samples_smoke.py"
    if ($LASTEXITCODE -ne 0) { throw "A smoke failed ($LASTEXITCODE)" }
    Copy-Item (Join-Path $ModuleA "tests\samples_smoke_report.json") `
              (Join-Path $Out "a_samples_smoke.json") -Force
}
finally { Pop-Location }

Write-Host "=== [B] Green check ==="
& pwsh (Join-Path $ModuleBHandoff "tools\run_green_check.ps1") -HandoffRoot $ModuleBHandoff -Python $Python
if ($LASTEXITCODE -ne 0) { throw "B green check failed ($LASTEXITCODE)" }

# Aggregate outputs.
$BOut = Join-Path $ModuleBHandoff "outputs\green_check"
$AReport = Get-Content (Join-Path $Out "a_samples_smoke.json") -Raw | ConvertFrom-Json
$BClean = Get-Content (Join-Path $BOut "clean_metrics.json") -Raw | ConvertFrom-Json
$BExtr = Get-Content (Join-Path $BOut "external_hard_suite\external_hard_suite_asr.json") -Raw | ConvertFrom-Json
$BSecurity = Get-Content (Join-Path $BOut "security_gate\security_report.json") -Raw | ConvertFrom-Json
$BGuard = Get-Content (Join-Path $BOut "try_attack_runtime_guard.summary.json") -Raw | ConvertFrom-Json

$report = [ordered]@{
    timestamp = Get-Date -Format o
    a = [ordered]@{
        ok = $AReport.summary.ok
        clips = foreach ($v in $AReport.summary.verdicts) {
            [ordered]@{
                clip = $v.clip
                ok = $v.ok
                alert_frames = $v.alert_frames
                missing_reasons = $v.missing_reasons
            }
        }
    }
    b = [ordered]@{
        clean_map50 = $BClean.map50
        clean_map50_95 = $BClean.map50_95
        external_asr_max = $BExtr.summary.max_asr
        external_asr_mean = $BExtr.summary.mean_asr
        security_level = $BSecurity.decision.level
        security_score = $BSecurity.decision.score
        try_attack_auto_target_detections = $BGuard.n_auto_target_detections
    }
    joint_ok = $AReport.summary.ok -and ($BSecurity.decision.level -eq "Green")
}

$path = Join-Path $Out "joint_smoke_report.json"
$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -Path $path
Write-Host "`n=== DONE ==="
Write-Host "joint_ok = $($report.joint_ok)"
Write-Host "report: $path"

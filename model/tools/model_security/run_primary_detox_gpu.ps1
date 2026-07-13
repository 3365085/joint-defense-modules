param(
  [string]$AssetsConfig = "configs\assets.local.yaml",
  [string]$AlgorithmConfig = "configs\hybrid_purify_detox_gpu_bounded.yaml",
  [string]$PythonExe = "",
  [switch]$CheckOnly,
  [switch]$AllowLongRun
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:PYTHONPATH = $repo
$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
$env:YOLO_OFFLINE = "true"
$env:YOLO_AUTOINSTALL = "false"

function Use-PixiPythonPath {
  param([string]$ResolvedPythonExe)

  $envRoot = Split-Path -Parent $ResolvedPythonExe
  $pixiPathParts = @(
    $envRoot,
    (Join-Path $envRoot "Library\mingw-w64\bin"),
    (Join-Path $envRoot "Library\usr\bin"),
    (Join-Path $envRoot "Library\bin"),
    (Join-Path $envRoot "Scripts"),
    (Join-Path $envRoot "bin")
  )
  $existingParts = $pixiPathParts | Where-Object { Test-Path -LiteralPath $_ }
  $env:PATH = ($existingParts -join ";") + ";" + $env:PATH
}

$algorithmPath = Resolve-Path -LiteralPath $AlgorithmConfig
$algorithmName = Split-Path -Leaf $algorithmPath
if ($algorithmName -eq "hybrid_purify_detox.yaml" -and -not $AllowLongRun) {
  throw "Refusing to launch the full unbounded research profile. Use -AlgorithmConfig configs\hybrid_purify_detox.yaml -AllowLongRun if you intentionally want the long exhaustive run."
}

if (-not $PythonExe) {
  $localPython = Join-Path $repo ".pixi\envs\default\python.exe"
  $sharedPython = Join-Path (Split-Path -Parent $repo) ".pixi\envs\default\python.exe"
  if (Test-Path -LiteralPath $localPython) {
    $PythonExe = $localPython
  } elseif (Test-Path -LiteralPath $sharedPython) {
    Write-Warning "Project-local Pixi environment was not found; falling back to parent Pixi environment."
    $PythonExe = $sharedPython
  }
}

$longRunArgs = @()
if ($AllowLongRun) {
  $longRunArgs += "--allow-long-run"
}

if ($PythonExe) {
  Use-PixiPythonPath -ResolvedPythonExe $PythonExe
  Write-Host "[INFO] Using Python: $PythonExe"
  & $PythonExe scripts\check_assets.py --assets-config $AssetsConfig
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if ($CheckOnly) { exit 0 }
  & $PythonExe scripts\hybrid_purify_detox_yolo.py `
    --config $algorithmPath `
    --assets-config $AssetsConfig `
    --amp `
    @longRunArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  pixi run python scripts\check_assets.py --assets-config $AssetsConfig
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if ($CheckOnly) { exit 0 }
  pixi run hybrid-purify-detox-yolo `
    --config $algorithmPath `
    --assets-config $AssetsConfig `
    --amp `
    @longRunArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

@echo off
setlocal
cd /d "%~dp0.."
echo Stopping Module A web service...
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$servicePids = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*-m defense.web.server*' } | Select-Object -ExpandProperty ProcessId -Unique); $portPids = @(Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); $targets = @($servicePids + $portPids) | Where-Object { $_ } | Sort-Object -Unique; if (-not $targets) { Write-Host 'No Module A web service is running.'; exit 0 }; foreach ($targetPid in $targets) { Write-Host ('Stopping process tree ' + $targetPid); taskkill /PID $targetPid /T /F | Out-Host }"
echo.
echo Done. Press any key to close this window.
pause >nul

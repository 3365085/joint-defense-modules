@echo off
setlocal
cd /d "%~dp0"
echo Starting Module A web service with D:\联合防御模块\.pixi ...
echo Workspace: %CD%
echo.
if not exist ".pixi\envs\default\python.exe" (
    echo ERROR: .pixi environment is missing: %CD%\.pixi\envs\default
    pause >nul
    exit /b 1
)
if not exist "pixi.toml" (
    echo ERROR: pixi.toml is missing in %CD%
    pause >nul
    exit /b 1
)
pixi run monitor-open-external
echo.
echo Service process exited. Press any key to close this window.
pause >nul

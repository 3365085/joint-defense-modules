@echo off
cd /d "%~dp0"
python "..\..\tools\link_delivery_pixi_env.py" --package-root "%~dp0" --project-root "..\.."
pause

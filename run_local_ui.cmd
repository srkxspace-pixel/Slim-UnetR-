@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. Follow the setup steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u local_ui.py --open-browser %*
pause

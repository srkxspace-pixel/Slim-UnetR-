@echo off
setlocal
cd /d "%~dp0"
echo === Slim UNETR one-time setup: Windows, CPU ===
echo Needs internet for about 300 MB of packages. Allow 10-25 minutes the first time.
echo.
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not run: py -3.11 -m venv .venv
    echo Install 64-bit Python 3.11 from python.org with the py launcher ticked, then check: py -3.11 --version
    goto :fail
  )
)
set "PY=.venv\Scripts\python.exe"
echo [1/5] Installing PyTorch 2.6.0 CPU
%PY% -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
echo [2/5] Installing pinned application packages
%PY% -m pip install -r requirements-ui.txt
if errorlevel 1 goto :fail
%PY% -m pip check
if errorlevel 1 goto :fail
echo [3/5] Verifying bundled MRI and the pretrained checkpoint, downloading it only if missing
%PY% -u setup_demo.py
if errorlevel 1 goto :fail
echo [4/5] Running the 23 regression tests, warnings are expected
%PY% -m unittest discover -s tests -p "test_*.py"
if errorlevel 1 goto :fail
echo [5/5] Running real 3D inference on the bundled MRI
%PY% -u demo.py --device cpu --output-dir outputs\first_run_%RANDOM%
if errorlevel 1 goto :fail
echo.
echo SETUP COMPLETE. Start the browser app with run_local_ui.cmd and open http://127.0.0.1:8765
echo Read HANDOFF.md for what to expect and where the saved evidence is.
pause
exit /b 0

:fail
echo.
echo SETUP FAILED. Read the message above, fix the cause and run this file again.
echo Finished steps are safe to repeat. See Troubleshooting in HANDOFF.md.
pause
exit /b 1

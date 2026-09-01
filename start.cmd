@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-windows\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv-windows 2>nul || python -m venv .venv-windows || exit /b 1
)

".venv-windows\Scripts\python.exe" -c "import token_tracker, filelock" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    ".venv-windows\Scripts\python.exe" -m pip install --quiet -e ".[dev]" vulture || exit /b 1
)

".venv-windows\Scripts\token-tracker.exe" %*

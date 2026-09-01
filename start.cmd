@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\token-tracker.exe" (
    echo No venv found. Run: python -m venv .venv ; then pip install -e ".[dev]" 1>&2
    exit /b 1
)

".venv\Scripts\token-tracker.exe" %*

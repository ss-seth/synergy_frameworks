@echo off
REM run.bat -- set up virtual environment and launch the Synergy Calculator
REM Usage: double-click run.bat  OR  run it from a Command Prompt / PowerShell

setlocal

set VENV_DIR=.venv

REM ── 1. Create venv if it doesn't exist ─────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [setup] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [error] Failed to create venv. Make sure Python 3 is installed and on PATH.
        pause
        exit /b 1
    )
)

REM ── 2. Activate ─────────────────────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

REM ── 3. Install / upgrade dependencies ───────────────────────────────────────
echo [setup] Installing dependencies from requirements.txt ...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [error] Dependency installation failed.
    pause
    exit /b 1
)

REM ── 4. Launch Streamlit ──────────────────────────────────────────────────────
echo [launch] Starting Synergy Calculator ...
streamlit run app.py

pause

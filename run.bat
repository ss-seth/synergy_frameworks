@echo off
REM run.bat -- set up virtual environment and launch the Synergy Calculator
REM Usage: double-click run.bat  OR  run it from a Command Prompt / PowerShell

setlocal

REM Use a Windows-specific venv so it doesn't conflict with the Mac .venv synced via iCloud
set VENV_DIR=.venv_win

REM ── 1. Check Python is available ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [error] Python not found. Install Python 3 and ensure it is on PATH.
    pause
    exit /b 1
)

REM ── 2. Create venv if it doesn't exist ─────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [setup] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [error] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ── 3. Activate ─────────────────────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [error] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM ── 4. Install / upgrade dependencies ───────────────────────────────────────
echo [setup] Installing dependencies from requirements.txt ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Dependency installation failed. See output above for details.
    pause
    exit /b 1
)

REM ── 5. Launch Streamlit ──────────────────────────────────────────────────────
echo [launch] Starting Synergy Calculator ...
python -m streamlit run app.py --server.headless=true

pause

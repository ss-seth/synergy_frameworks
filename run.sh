#!/usr/bin/env bash
# run.sh — set up virtual environment and launch the Synergy Calculator
# Usage: ./run.sh

set -e

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# ── 1. Create venv if it doesn't exist ──────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment in $VENV_DIR …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── 2. Activate ──────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── 3. Install / upgrade dependencies ────────────────────────────────────────
echo "[setup] Installing dependencies from requirements.txt …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 4. Launch Streamlit ───────────────────────────────────────────────────────
echo "[launch] Starting Synergy Calculator …"
streamlit run app.py

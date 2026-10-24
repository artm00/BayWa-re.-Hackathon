#!/usr/bin/env bash
# Usage: source SETUP.sh   (must be sourced so the env stays active in your shell)

VENV_DIR=".venv"

[ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

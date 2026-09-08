#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python -m pip install -e .


echo "Installation complete."
echo "Tests: .venv/bin/python -m unittest discover -s tests -v"
echo "Model evaluation (after external assets are placed): .venv/bin/python evaluate_representative.py"

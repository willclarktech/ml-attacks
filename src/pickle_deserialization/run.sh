#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail
command -v shellcheck >/dev/null && shellcheck "$0"

rm -f innocent_model.pt malicious_model.pt

echo "Running attacker code to generate innocent/malicious model files..."
python3 attacker.py
echo "Running victim code to load malicious model file..."
python3 victim.py

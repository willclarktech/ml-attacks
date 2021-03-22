#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail
command -v shellcheck >/dev/null && shellcheck "$0"

rm -f model.pt

echo "Running victim code to train and deploy a model..."
python3 victim.py
echo "Running attacker code to reconstruct and exploit victim’s model..."
python3 attacker.py

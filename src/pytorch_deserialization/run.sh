#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail
command -v shellcheck >/dev/null && shellcheck "$0"

rm -f model.pt output.tmp

echo "Running attacker code to generate malicious model.pt file..."
python3 attacker.py
echo "Running victim code to load malicious model.pt file..."
python3 victim.py
echo "Check the spawned process exists using 'ps'"
echo "Then wait for output.tmp to be created"

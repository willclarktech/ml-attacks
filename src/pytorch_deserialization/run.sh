#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail
command -v shellcheck >/dev/null && shellcheck "$0"

rm -f model.pkl output.tmp

echo "Saving malicious model to model.pkl"
python3 attacker.py
echo "Loading model from model.pkl"
python3 victim.py
echo "Check the spawned process exists using 'ps'"
echo "Then wait for output.tmp to be created"

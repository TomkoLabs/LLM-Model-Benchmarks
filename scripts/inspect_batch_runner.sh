#!/bin/sh
set -eu

REPO="${HERMES_AGENT_ROOT:-${HERMES_HOME:-$HOME/.hermes}/hermes-agent}"
PY="$REPO/venv/bin/python"

echo "===== BATCH RUNNER DISTRIBUTIONS ====="

"$PY" "$REPO/batch_runner.py" \
    --list_distributions=True \
    2>&1 || true

echo
echo "===== BATCH RUNNER OUTPUT / TRAJECTORY REFERENCES ====="

rg -n \
    'trajectory|checkpoint|output_dir|output_file|results|run_name|jsonl|save' \
    "$REPO/batch_runner.py" \
    | head -220

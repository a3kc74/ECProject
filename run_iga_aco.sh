#!/usr/bin/env bash
# Simple wrapper to run IGA-ACO with sensible defaults and logging

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p results
mkdir -p logs

# Defaults (can be overridden by passing extra args)
DATA_DIR="data/OhlmannThomas"
NUM_RUNS=3
IGA_POP=100
NUM_ITER=500
ACO_ANTS=50

LOG_FILE="logs/iga_aco_run_$(date +%Y%m%d_%H%M%S).log"

echo "Running IGA-ACO..."
echo "Log -> $LOG_FILE"

python3 run_iga_aco.py \
    --data-dir "$DATA_DIR" \
    --num-runs "$NUM_RUNS" \
    --iga-population "$IGA_POP" \
    --num-iterations "$NUM_ITER" \
    --aco-num-ants "$ACO_ANTS" \
    "$@" 2>&1 | tee "$LOG_FILE"

echo "Done. Results saved to results/ and log: $LOG_FILE"

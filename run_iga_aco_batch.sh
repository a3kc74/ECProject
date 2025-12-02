#!/usr/bin/env bash
# Batch-run helper: run same configuration multiple times (useful for statistics)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p results
mkdir -p logs

# Parameters
POP=100
ITER=500
ANTS=50
RUNS_PER_CONFIG=10

echo "Running batch: pop=$POP iter=$ITER ants=$ANTS runs=$RUNS_PER_CONFIG"

for i in $(seq 1 "$RUNS_PER_CONFIG"); do
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="logs/iga_aco_batch_run${i}_pop${POP}_iter${ITER}_ants${ANTS}_${TIMESTAMP}.log"

  echo "Batch run $i/$RUNS_PER_CONFIG -> $LOG_FILE"

  python3 run_iga_aco.py \
    --iga-population "$POP" \
    --num-iterations "$ITER" \
    --aco-num-ants "$ANTS" \
    --num-runs 1 \
    2>&1 | tee "$LOG_FILE"

  echo "Run $i completed"
done

echo "Batch completed. Results in results/ and logs/"

#!/usr/bin/env bash
# Run multiple IGA-ACO configurations (population / iterations / ants)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p results
mkdir -p logs

echo "Running multiple IGA-ACO configurations"

# Format: pop:iterations:ants:label
CONFIGS=(
  "50:200:30:small"
  "100:500:50:medium"
  "150:1000:70:large"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r POP ITER ANTS LABEL <<< "$cfg"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="logs/iga_aco_${LABEL}_pop${POP}_iter${ITER}_ants${ANTS}_${TIMESTAMP}.log"

  echo "---------------------------------------------"
  echo "Config: $LABEL (pop=$POP, iter=$ITER, ants=$ANTS)"
  echo "Log: $LOG_FILE"

  python3 run_iga_aco.py \
    --iga-population "$POP" \
    --num-iterations "$ITER" \
    --aco-num-ants "$ANTS" \
    --num-runs 3 \
    2>&1 | tee "$LOG_FILE"

  echo "Completed $LABEL"
  echo ""
done

echo "All configurations completed. Results in results/ and logs in logs/"

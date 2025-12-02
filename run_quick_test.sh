#!/bin/bash

# Quick test script - runs both algorithms with small settings for validation
# Useful for quick testing before running full benchmarks

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "Quick Test - GA and SA"
echo "=================================="
echo ""

# Create output directory
mkdir -p results
mkdir -p logs

# Quick test settings (small, should complete quickly)
GA_POP=50
GA_GEN=200
SA_TEMP=500.0
SA_COOL=0.95
SA_ITER=50
NUM_RUNS=1
DATA_DIR="data/SolomonPotvinBengio"

LOG_FILE="logs/quick_test_$(date +%Y%m%d_%H%M%S).log"

echo "Running quick test with minimal settings..."
echo "  - GA: Population=$GA_POP, Generations=$GA_GEN"
echo "  - SA: InitTemp=$SA_TEMP, CoolRate=$SA_COOL, IterPerTemp=$SA_ITER"
echo "  - Runs: $NUM_RUNS"
echo ""

python main.py \
    --algorithm both \
    --ga-population "$GA_POP" \
    --ga-generations "$GA_GEN" \
    --sa-initial-temp "$SA_TEMP" \
    --sa-final-temp 0.1 \
    --sa-cooling-rate "$SA_COOL" \
    --sa-iterations-per-temp "$SA_ITER" \
    --num-runs "$NUM_RUNS" \
    --data-dir "$DATA_DIR" \
    --verbose \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Quick test completed!"
echo "  Log saved to: $LOG_FILE"
echo ""

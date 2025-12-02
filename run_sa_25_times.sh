#!/bin/bash

# Bash script to run Simulated Annealing 25 times on each instance
# of the SolomonPotvinBengio dataset

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "Running SA 25 times on each instance"
echo "=================================="
echo ""

# Create output directory
mkdir -p results
mkdir -p logs

DATA_DIR="data/OhlmannThomas"
NUM_RUNS=25

# Count total instances
INSTANCES=($(ls "$DATA_DIR"/*.txt | grep -v sample_instance | sort))
TOTAL_INSTANCES=${#INSTANCES[@]}

echo "Found $TOTAL_INSTANCES instances in $DATA_DIR"
echo "Will run $NUM_RUNS times per instance"
echo "Total experiments: $((TOTAL_INSTANCES * NUM_RUNS))"
echo ""

INSTANCE_COUNT=0

# Run SA on each instance
for instance_path in "${INSTANCES[@]}"; do
    INSTANCE_COUNT=$((INSTANCE_COUNT + 1))
    INSTANCE_NAME=$(basename "$instance_path")
    
    echo "========================================="
    echo "Instance $INSTANCE_COUNT/$TOTAL_INSTANCES: $INSTANCE_NAME"
    echo "========================================="
    
    # Create log filename with instance name and timestamp
    LOG_FILE="logs/sa_${INSTANCE_NAME%.txt}_25runs_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the SA with the specific instance
    echo "Running SA $NUM_RUNS times on $INSTANCE_NAME..."
    python main.py \
        --algorithm SA \
        --sa-initial-temp 1000 \
        --sa-cooling-rate 0.96 \
        --sa-iterations-per-temp 200 \
        --sa-final-temp 0.1 \
        --instance "$INSTANCE_NAME" \
        --data-dir "$DATA_DIR" \
        --num-runs "$NUM_RUNS" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Completed instance $INSTANCE_COUNT/$TOTAL_INSTANCES: $INSTANCE_NAME"
    echo "  Log saved to: $LOG_FILE"
    echo ""
done

echo "=================================="
echo "All instances completed!"
echo "=================================="
echo ""
echo "Results saved to: results/"
echo "Logs saved to: logs/"

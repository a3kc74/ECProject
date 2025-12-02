#!/bin/bash

# Bash script to run Simulated Annealing with different configurations
# Tests various temperature and cooling rate settings

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "Simulated Annealing Configuration Tests"
echo "=================================="
echo ""

# Create output directory
mkdir -p results
mkdir -p logs

# Define SA configurations to test
# Format: initial_temp:cooling_rate:iterations_per_temp:description
SA_CONFIGS=(
    "500.0:0.95:100:Conservative"
    "1000.0:0.96:200:Standard"
    "1500.0:0.97:300:Aggressive"
    "2000.0:0.98:400:VeryAggressive"
)

FINAL_TEMP=0.1
NUM_RUNS=3
DATA_DIR="data/SolomonPotvinBengio"

echo "Running Simulated Annealing with different configurations..."
echo "Base settings:"
echo "  - Final Temperature: $FINAL_TEMP"
echo "  - Runs per config: $NUM_RUNS"
echo ""

# Run each configuration
for config in "${SA_CONFIGS[@]}"; do
    IFS=':' read -r init_temp cool_rate iter_per_temp desc <<< "$config"
    
    echo "========================================="
    echo "Configuration: $desc"
    echo "  Initial Temperature: $init_temp"
    echo "  Cooling Rate: $cool_rate"
    echo "  Iterations per Temperature: $iter_per_temp"
    echo "========================================="
    
    # Create log filename with timestamp
    LOG_FILE="logs/sa_${desc}_temp${init_temp}_cool${cool_rate}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the SA
    python main.py \
        --algorithm SA \
        --sa-initial-temp "$init_temp" \
        --sa-final-temp "$FINAL_TEMP" \
        --sa-cooling-rate "$cool_rate" \
        --sa-iterations-per-temp "$iter_per_temp" \
        --num-runs "$NUM_RUNS" \
        --data-dir "$DATA_DIR" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Configuration $desc completed"
    echo "  Log saved to: $LOG_FILE"
    echo ""
done

echo "=================================="
echo "All SA configurations completed!"
echo "=================================="
echo ""
echo "Results saved to: results/"
echo "Logs saved to: logs/"
